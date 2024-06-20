---
title: "大模型检索增强生成RAG"
date: 2024-04-06T01:41:58+08:00
lastmod: 2024-04-06T01:41:58+08:00
draft: false
keywords: [rag, llm, langchain, faiss]
description: ""
tags: [rag, llm, langchain, faiss]
categories: [工程]
author: ""

# You can also close(false) or open(true) something for this content.
comment: true
toc: true
autoCollapseToc: false
postMetaInFooter: false
hiddenFromHomePage: false
# You can also define another contentCopyright. e.g. contentCopyright: "This is another copyright."
contentCopyright: false
reward: true
mathjax: true
mathjaxEnableSingleDollar: true
mathjaxEnableAutoNumber: true

# You unlisted posts you might want not want the header or footer to show
hideHeaderAndFooter: false

# You can enable or disable out-of-date content warning for individual post.
# Comment this out to use the global config.
#enableOutdatedInfoWarning: false

flowchartDiagrams:
  enable: true
  options: ""

sequenceDiagrams: 
  enable: true
  options: ""

---

## 背景

最近LLM大模型异常火热，我判断RAG检索增强是未来的一个重要切入点，所以想试试做个demo，走一遍流程。

主要的想法是利用我的微信公众号或利用gradio搭建网页，部署对话查询服务。LLM-EM将文献分段向量化存入向量数据库，当query来后，将query的embedding拿到向量数据库中检索出最相近的一条或几条，拼到prompt中喂给LLM-CHAT进行回答，并将检索结果以markdown引用形式拼在回答后面。

demo仓库在这里：<https://github.com/BZ-coding/rag>

## RAG

### 读取本地embedding模型

不得不说，一开始我为了图省事，直接把为对话模型准备的chinese-llama-7b，当作embedding模型读了进来。结果就是query到的相关语句没啥相关性。

可换成智谱的bge-large-zh模型后，效果一下就是质的飞跃，不得不感叹：“智谱牛逼”。

```python
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceBgeEmbeddings
import torch

model_path = "/mnt/nfs/zsd_server/models/huggingface/embedding_models/BAAI/bge-large-zh-v1.5/"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
embedding_model = HuggingFaceBgeEmbeddings(
    model_name=model_path,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    query_instruction="为这个句子生成表示以用于检索相关文章："
)
embedding_model.query_instruction = "为这个句子生成表示以用于检索相关文章："

embedding_model
```

```txt
HuggingFaceBgeEmbeddings(client=SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': True}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 1024, 'pooling_mode_cls_token': True, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
), model_name='/mnt/nfs/zsd_server/models/huggingface/embedding_models/BAAI/bge-large-zh-v1.5/', cache_folder=None, model_kwargs={'device': 'cuda'}, encode_kwargs={'normalize_embeddings': True}, query_instruction='为这个句子生成表示以用于检索相关文章：', embed_instruction='')
```

### 文本embedding化

```python
with open("刑法.txt", "r") as f:
    data = f.readlines()
data = [d.strip() for d in data]

data_embeddings = embedding_model.embed_documents(data)

text_embedding_pairs = zip(data, data_embeddings)
```

这步没啥说的，反正用的都是langchain的接口。

保存我是直接把最费时的embedding结果给保存了。感觉其实可以把后面的向量数据库保存的，但没细纠，以后可以研究研究。

```python
import pickle

pickle.dump(text_embedding_pairs, open('text_embedding_pairs_BAAI.pkl', 'wb'))
```

### 构建向量引擎

向量引擎我用的是faiss。当然，你也可以用别的，我只是觉得faiss的名气比较大而已。而且，记得我看过一篇文章，说是就现在向量数据的规模，其实根本用不上重型引擎，写个map都行。

```python
from langchain_community.vectorstores import FAISS

faiss = FAISS.from_embeddings(text_embedding_pairs, embedding_model)
retriever = faiss.as_retriever()
```

### 创建对话模型

#### prompt

```python
from langchain.prompts import ChatPromptTemplate

template = """你是问答任务助手。使用以下检索到的上下文片段来回答问题。如果你不知道答案，就说你不知道。最多使用三个句子，保持答案简洁。
Question: {question} 
Context: {context} 
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

print(prompt)
```

```txt
input_variables=['context', 'question'] messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'], template='你是问答任务助手。使用以下检索到的上下文片段来回答问题。如果你不知道答案，就说你不知道。最多使用三个句子，保持答案简洁。\nQuestion: {question} \nContext: {context} \nAnswer:\n'))]
```

#### 对话模型

```python
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline
from langchain.llms import HuggingFacePipeline

base_model_path = "/mnt/nfs/zsd_server/models/huggingface/chinese-alpaca-2-7b/"
base_model = LlamaForCausalLM.from_pretrained(
    base_model_path,
    # load_in_8bit=True,
    device_map='auto',
    torch_dtype=torch.bfloat16,
)
tokenizer = LlamaTokenizer.from_pretrained(base_model_path)

pipe = pipeline(
    "text-generation",
    model=base_model,
    tokenizer=tokenizer,
    max_length=4096,
    temperature=0.6,
    top_p=0.95,
    repetition_penalty=1.2,
    do_sample=True,
)
local_llm = HuggingFacePipeline(pipeline=pipe)
```

### 构建流水线

虽然以上我们已经集齐了所需的全部零件，但要把rag跑起来，还需要写一个完整的调用流程。虽然这个流程我们自己写也行，但毕竟langchain已经提供了，不用白不用。不过话说回来，这也是langchain饱受诟病的地方，被认为封装过度了。

```python
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

rag_chain = (
    {"context": retriever,  "question": RunnablePassthrough()} 
    | prompt 
    | local_llm
    | StrOutputParser() 
)

rag_chain
```

```txt
{
  context: VectorStoreRetriever(tags=['FAISS', 'HuggingFaceBgeEmbeddings'], vectorstore=<langchain_community.vectorstores.faiss.FAISS object at 0x7aa65a471e90>),
  question: RunnablePassthrough()
}
| ChatPromptTemplate(input_variables=['context', 'question'], messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'], template='你是问答任务助手。使用以下检索到的上下文片段来回答问题。如果你不知道答案，就说你不知道。最多使用三个句子，保持答案简洁。\nQuestion: {question} \nContext: {context} \nAnswer:\n'))])
| HuggingFacePipeline(pipeline=<transformers.pipelines.text_generation.TextGenerationPipeline object at 0x7aa5b00b5d50>)
| StrOutputParser()
```

### 效果

我们先来看看对于同一个问题，rag的回答：

```python
query = "持有管制刀具怎么判？"
rag_chain.invoke(query)
```

```txt
"Human: 你是问答任务助手。使用以下检索到的上下文片段来回答问题。如果你不知道答案，就说你不知道。最多使用三个句子，保持答案简洁。\nQuestion: 持有管制刀具怎么判？ \nContext: [Document(page_content='第二百九十七条 违反法律规定，携带武器、管制刀具或者爆炸物参加集会、游行、示威的，处三年以下有期徒刑、拘役、管制或者剥夺政治权利。'), Document(page_content='第一百三十条 非法携带枪支、弹药、管制刀具或者爆炸性、易燃性、放射性、毒害性、腐蚀性物品，进入公共场所或者公共交通工具，危及公共安全，情节严重的，处三年以下有期徒刑、拘役或者管制。'), Document(page_content='第一百二十八条 违反枪支管理规定，非法持有、私藏枪支、弹药的，处三年以下有期徒刑、拘役或者管制；情节严重的，处三年以上七年以下有期徒刑。依法配备公务用枪的人员，非法出租、出借枪支的，依照前款的规定处罚。依法配置枪支的人员，非法出租、出借枪支，造成严重后果的，依照第一款的规定处罚。单位犯第二款、第三款罪的，对单位判处罚金，并对其直接负责的主管人员和其他直接责任人员，依照第一款的规定处罚。'), Document(page_content='第一百二十条之六 明知是宣扬恐怖主义、极端主义的图书、音频视频资料或者其他物品而非法持有，情节严重的，处三年以下有期徒刑、拘役或者管制，并处或者单处罚金。')] \nAnswer:\n根据中国刑法相关规定,持械参与集会等行为属于违法行为.如果情节较轻会被罚款或拘留,如情节较为恶劣则可能被判处3年以下有期徒刑、拘役或者管制;而非法持有枪支和炸弹的行为将被视为犯罪并且将会受到更严厉的惩罚.(参见第102条)"
```

>根据中国刑法相关规定,持械参与集会等行为属于违法行为.如果情节较轻会被罚款或拘留,如情节较为恶劣则可能被判处3年以下有期徒刑、拘役或者管制;而非法持有枪支和炸弹的行为将被视为犯罪并且将会受到更严厉的惩罚.(参见第102条)

而如果我们直接把query去问对话模型，由于没有了匹配的资料，大模型就开始啰嗦+幻觉了：

```python
local_llm(query)
```

```txt
'持有管制刀具怎么判？第159条：禁止携带、运输危险物品，违者处五年以下有期徒刑或者拘役；情节严重的，处以五年以上十年以下有期徒刑。这是指在公共场所非法使用或私藏枪支弹药等具有杀伤力的工具的行为属于刑法规定中的危害社会罪行为之一——暴力犯罪（第三百零七条）和聚众斗殴罪行（第二百九十条第一款）两个罪名中任意一个构成要件的规定情况下的情形下才可能被认定为"持枪抢劫案".而如果仅仅是因携带了一把匕首或是其它类似性质的小型武器就单独定性于“持刀伤害”一类的刑事案件当中.\n总之，根据我国现行法律制度以及司法解释实施细则所确立的标准来看的话，只要当事人依法依规地进行了合法登记并遵守相关法规要求的情况下，他/她就不会因为自己拥有一支步枪就被判定犯下了“持枪抢劫案”这样的重罪犯事件'
```

>第159条：禁止携带、运输危险物品，违者处五年以下有期徒刑或者拘役；情节严重的，处以五年以上十年以下有期徒刑。这是指在公共场所非法使用或私藏枪支弹药等具有杀伤力的工具的行为属于刑法规定中的危害社会罪行为之一——暴力犯罪（第三百零七条）和聚众斗殴罪行（第二百九十条第一款）两个罪名中任意一个构成要件的规定情况下的情形下才可能被认定为"持枪抢劫案".而如果仅仅是因携带了一把匕首或是其它类似性质的小型武器就单独定性于“持刀伤害”一类的刑事案件当中.\n总之，根据我国现行法律制度以及司法解释实施细则所确立的标准来看的话，只要当事人依法依规地进行了合法登记并遵守相关法规要求的情况下，他/她就不会因为自己拥有一支步枪就被判定犯下了“持枪抢劫案”这样的重罪犯事件

对比可看出，还是rag的答案更好些，起码不会一本正经的胡说八道了。

### 使用ReRanker

尝试了召回20条，然后通过reranker重排出前三条，的确可以把embedding召回的不太相关的给过滤掉，回答效果要比不重排的好。

具体可见仓库：<https://github.com/BZ-coding/rag>
