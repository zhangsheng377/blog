---
title: "pytorch使用bert进行多分类"
date: 2021-08-17T23:04:58+08:00
lastmod: 2021-08-17T23:04:58+08:00
draft: false
keywords: []
description: ""
tags: [pytorch, bert, 多分类, 拼接特征]
categories: [算法]
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

之前虽然会用keras来搭建模型，但后来发现用keras修改具体的底层细节实现比较麻烦。又不想用tensorflow写那么多代码，所以近日正好来试试传说中自由灵活、用户体验好的pytorch~~

## 关键点

```python
# 查看gpu是否可用
torch.cuda.is_available()
```

```python
# 据说打开这两句可以加速并减小显存占用
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
```

```python
# 按照pytorch的Dataset格式定义数据集
class MyDataset(Dataset):
    def __init__(self, input_path, dataset_type):
        self.dataset_type = dataset_type
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_bert_path)
        self.data_list = self.load_data(input_path)

    def __getitem__(self, index):
        item = self.data_list[index]
        
        token = self.tokenizer(item['text'], 
                               add_special_tokens=True,
                               max_length=512,
                               truncation=True,
                               padding='max_length',
                               return_tensors="pt")
        del item['text']
        item['input_ids'] = token['input_ids'][0]
        item['token_type_ids'] = token['token_type_ids'][0]
        item['attention_mask'] = token['attention_mask'][0]
        
        item['category'] = category_encoder.transform(np.array([item['category']]).reshape(-1, 1)).toarray()[0]
        item['paragraphs_num'] = paragraphs_num_encoder.transform(np.array([item['paragraphs_num']]).reshape(-1, 1))[0]
        item['words_len'] = words_len_encoder.transform(np.array([item['words_len']]).reshape(-1, 1))[0]
        del item['pic_num']
        item['source'] = source_encoder.transform(np.array([item['source']]).reshape(-1, 1)).toarray()[0]
        
        del item['id']

        if self.dataset_type == 'test':
            item['doctype'] = -1
        else:
            item['doctype'] = doctype_list.index(item['doctype'])
        label = item['doctype']
        
        del item['doctype']
        
        return item, label

    def __len__(self):
        return len(self.data_list)

    def load_data(self, input_path):
        data_list = []
        with open(input_path, 'r', encoding='utf-8') as input_file:
            for line in tqdm(input_file):
                json_data = json.loads(line)
                data_list.append(json_data)
        if self.dataset_type != 'test':
            random.shuffle(data_list)
        return data_list
```

```python
# 利用预训练bert模型构建自己的模型
class BertClassificationModel(nn.Module):
    # 定义每层网络
    def __init__(self, hidden_size=768):
        super(BertClassificationModel, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_bert_path)
        
        category_size = len(category_encoder.categories_[0])
        paragraphs_num_size = 1
        words_len_size = 1
        source_size = len(source_encoder.categories_[0])
        linear_size = hidden_size + category_size + paragraphs_num_size + source_size + words_len_size
        # 可以用Sequential定义串行子网络，forward使用时更方便
        self.net = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(linear_size, 512),
            # 激活函数指定inplace=True据说省显存
            nn.ReLU(inplace=True),
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Linear(64, len(doctype_list))

    # 实现每层网络的前向传播逻辑，重点就是这
    def forward(self, batch_data):
        # 由于用dataloader传过来的就是tensor格式，所以如果要用新变量接的话，需要用.clone().detach()
        # 为了使input、output、indices都在一个设备上，所以在这里input的数据需要.cuda()
        # 当然，其实不用中间变量接，直接写在bert的入参里，加上.cuda()也是可以的
        input_ids = batch_data['input_ids'].clone().detach().cuda()
        token_type_ids = batch_data['token_type_ids'].clone().detach().cuda()
        attention_mask = batch_data['attention_mask'].clone().detach().cuda()
        bert_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        '''BaseModelOutputWithPoolingAndCrossAttentions(last_hidden_state=tensor([[[-0.1029,  0.9732, -0.5482,  ...,  0.5311, -0.2025,  0.2088],
         [ 0.0189,  0.5685, -0.5773,  ...,  0.0142, -0.3346,  0.0128],
         [ 0.2679, -0.1117, -0.6152,  ...,  1.0253,  0.6540,  0.2070],
         ...,
         [ 0.8424, -0.6507, -0.6250,  ...,  0.1572,  0.2110, -0.2585],
         [ 0.8012,  0.6836,  0.0201,  ..., -0.9195, -0.0715,  0.3281],
         [ 0.4075,  0.3863, -0.0741,  ...,  0.5316, -0.0412, -0.0857]],

        [[-0.2499,  0.2224, -0.1780,  ...,  0.7905, -0.5384,  0.2773],
         [-0.1736,  0.6806,  0.1101,  ..., -0.5826, -0.4254,  0.1009],
         [ 0.1549, -0.3790, -0.0912,  ...,  0.9467,  0.4750,  0.3511],
         ...,
         [-0.1728,  0.3743,  0.0098,  ...,  0.5561, -0.4878, -0.1095],
         [-0.6088,  0.5164,  0.2149,  ...,  0.7675, -0.5470,  0.0869],
         [-0.3542,  0.5069,  0.0714,  ...,  0.5809, -0.5175, -0.1571]]],
       grad_fn=<NativeLayerNormBackward>), pooler_output=tensor([[ 0.9976,  0.9978,  0.9992,  ..., -0.9999, -0.9993,  0.4574],
        [ 0.9767,  0.9997,  0.9953,  ..., -0.9999, -0.9716, -0.0691]],
       grad_fn=<TanhBackward>), hidden_states=None, past_key_values=None, attentions=None, cross_attentions=None)'''
        # 第一个0是取隐含层last_hidden_state，第二个0是取每层的cls向量
        bert_cls_hidden_state = bert_output[0][:, 0, :]
        category = batch_data['category'].clone().detach().cuda()
        paragraphs_num = batch_data['paragraphs_num'].clone().detach().cuda()
        source = batch_data['source'].clone().detach().cuda()
        words_len = batch_data['words_len'].clone().detach().cuda()
        # 按照维度1拼接，就是拼成1行。按哪个维度，就是shape其他的维度不变，在那个维度上累计。
        cat_layer = torch.cat((bert_cls_hidden_state, category, paragraphs_num, source, words_len), 1)
        # 解决float和double类型不一致的问题，而不用按照提示信息设定double为默认，因为这样会撑爆显存。
        output = self.net(cat_layer.to(torch.float32))
        output = self.out(output)
        
        return output
```

```python
# 网上找的早停控制器
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
#         torch.save(model.state_dict(), model_train_first_path)	# 这里会存储迄今最优模型的参数
        torch.save(model, model_train_first_path)
        self.val_loss_min = val_loss
```

```python
batch_size = 2

# torch.cuda.set_device(0)
model = BertClassificationModel()
# 打开这行会撑爆显存。如果它报float和double类型不一致，那只要把linear层的输入转为float32即可。
# torch.set_default_tensor_type(torch.DoubleTensor)
model = model.cuda()
# print(model)

# 不同子网络设定不同的学习率
Bert_model_param = []
Bert_downstream_param = []
for items, _ in model.named_parameters():
    if "bert" in items:
        Bert_model_param.append(_)
    else:
        Bert_downstream_param.append(_)
param_groups = [{"params": Bert_model_param, "lr": 1e-5},
                {"params": Bert_downstream_param, "lr": 1e-4}]
optimizer = optim.Adam(param_groups, eps=1e-7, weight_decay=0.001)
# 初始化 early_stopping 对象
patience = 2 # 当验证集损失在连续n次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
early_stopping = EarlyStopping(patience, verbose=True)
criterion = nn.CrossEntropyLoss()
```

```python
epoch_num = 5
batch_loss_num = 100
for epoch in range(epoch_num):
    model.train() # 设置模型为训练模式
    train_dataset = MyDataset(processed_expose_train_labeled_train_path, 'train')
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    with tqdm(total=len(train_loader)) as t:
        loss_sum = 0.0
        batch_loss = 1.0
        for batch_idx,data_ in enumerate(train_loader, 0):
            data, label = data_
            # 清空梯度
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label.cuda())
            loss.backward()

            # 更新模型参数
            optimizer.step()
            
            loss_sum += loss.item()
            if batch_idx % batch_loss_num == batch_loss_num - 1:
                batch_loss = loss_sum / batch_loss_num
                loss_sum = 0.0
          
            t.set_postfix_str(f'train_loss={loss.item():.6f} batch_loss={batch_loss:.6f}')
            t.update()
            
            del data, label, output 
            gc.collect()
            torch.cuda.empty_cache()
    #----------------------------------------------------
    model.eval() # 设置模型为评估/测试模式
    valid_dataset = MyDataset(processed_expose_train_labeled_valid_path, 'valid')
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loss_sum = 0.0
    with tqdm(total=len(valid_loader)) as t:
        with torch.no_grad():
            for data, label in valid_loader:
                # 一般如果验证集不是很大的话，模型验证就不需要按批量进行了，但要注意输入参数的维度不能错
                output = model(data)
                loss = criterion(output, label.cuda())
                valid_loss_sum += loss.item()
                t.set_postfix_str(f'valid loss={loss.item()}')
                t.update()
    early_stopping(valid_loss_sum, model)
    # 若满足 early stopping 要求
    if early_stopping.early_stop:
        print("Early stopping")
        # 结束模型训练
        break
```

## 附录

### 源代码

```python
#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('nvidia-smi')


# In[ ]:





# In[2]:


import json
import random
import os
import pickle
import time
import gc

from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


# In[3]:


torch.__version__


# In[4]:


torch.cuda.is_available()


# In[5]:


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


# In[6]:


processed_expose_test_path = "data/processed_test_expose.json"
processed_expose_train_labeled_path = "data/processed_train_expose_labeled.json"
processed_expose_train_labeled_train_path = "data/processed_train_expose_labeled_train.json"
processed_expose_train_labeled_valid_path = "data/processed_train_expose_labeled_valid.json"

pretrained_bert_path = "bert-base-chinese/"

category_encoder_path = 'model/category_encoder.pickle'
paragraphs_num_encoder_path = 'model/paragraphs_num_encoder.pickle'
source_encoder_path = 'model/source_encoder.pickle'
doctype_encoder_path = 'model/doctype_encoder.pickle'
words_len_encoder_path = 'model/words_len_encoder.pickle'
model_train_first_path = 'model/model_train_first.pt'

num_workers = 0


# In[ ]:





# In[7]:


# with open(processed_expose_train_path, 'r', encoding="utf-8") as input_file, \
#      open(processed_expose_train_labeled_path, 'w', encoding="utf-8") as labeled_file, \
#      open(processed_expose_train_unlabel_path, 'w', encoding="utf-8") as unlabel_file:
#     for line in tqdm(input_file):
#         json_data = json.loads(line)
#         if json_data['doctype'] != '':
#             labeled_file.write(f"{json.dumps(json_data, ensure_ascii=False)}\n")
#         else:
#             unlabel_file.write(f"{json.dumps(json_data, ensure_ascii=False)}\n")


# In[ ]:





# In[8]:


def get_feature_list(input_path, feature_name):
    feature_list = []
    with open(input_path, 'r', encoding="utf-8") as input_file:
        for line in tqdm(input_file):
            json_data = json.loads(line)
            feature_list.append(json_data[feature_name])
    return feature_list


# In[9]:


def get_category_encoder():
    if os.path.exists(category_encoder_path):
        with open (category_encoder_path, 'rb') as category_encoder_file: 
            return pickle.load(category_encoder_file)
    
    category_list = get_feature_list(processed_expose_train_labeled_path, 'category') + get_feature_list(processed_expose_test_path, 'category')
    category_list = np.array(category_list).reshape(-1, 1)
    encoder = OneHotEncoder(categories='auto', handle_unknown='ignore').fit(category_list)
    
    with open (category_encoder_path, 'wb') as category_encoder_file:
        pickle.dump(encoder, category_encoder_file)
    return encoder


# In[10]:


category_encoder = get_category_encoder()
category_encoder.transform(np.array([1]).reshape(-1, 1)).toarray()[0]


# In[11]:


len(category_encoder.categories_[0])


# In[12]:


def get_paragraphs_num_encoder():
    if os.path.exists(paragraphs_num_encoder_path):
        with open (paragraphs_num_encoder_path, 'rb') as paragraphs_num_encoder_file: 
            return pickle.load(paragraphs_num_encoder_file)
    
    paragraphs_num_list = get_feature_list(processed_expose_train_labeled_path, 'paragraphs_num') + get_feature_list(processed_expose_test_path, 'paragraphs_num')
    paragraphs_num_list = np.array(paragraphs_num_list).reshape(-1, 1)
    encoder = StandardScaler().fit(paragraphs_num_list)
    
    with open (paragraphs_num_encoder_path, 'wb') as paragraphs_num_encoder_file:
        pickle.dump(encoder, paragraphs_num_encoder_file)
    return encoder


# In[13]:


paragraphs_num_encoder = get_paragraphs_num_encoder()
paragraphs_num_encoder.transform(np.array([100]).reshape(-1, 1))[0][0]


# In[14]:


paragraphs_num_encoder.transform(np.array([99999999999]).reshape(-1, 1))[0][0]


# In[15]:


def get_words_len_encoder():
    if os.path.exists(words_len_encoder_path):
        with open (words_len_encoder_path, 'rb') as words_len_encoder_file: 
            return pickle.load(words_len_encoder_file)
    
    words_len_list = get_feature_list(processed_expose_train_labeled_path, 'words_len') + get_feature_list(processed_expose_test_path, 'words_len')
    words_len_list = np.array(words_len_list).reshape(-1, 1)
    encoder = StandardScaler().fit(words_len_list)
    
    with open (words_len_encoder_path, 'wb') as words_len_encoder_file:
        pickle.dump(encoder, words_len_encoder_file)
    return encoder


# In[16]:


words_len_encoder = get_words_len_encoder()
words_len_encoder.transform(np.array([1000]).reshape(-1, 1))[0][0]


# In[17]:


words_len_encoder.transform(np.array([2000]).reshape(-1, 1))[0][0]


# In[18]:


def get_source_encoder():
    if os.path.exists(source_encoder_path):
        with open (source_encoder_path, 'rb') as source_encoder_file: 
            return pickle.load(source_encoder_file)
    
    source_list = get_feature_list(processed_expose_train_labeled_path, 'source') + get_feature_list(processed_expose_test_path, 'source')
    source_list = np.array(source_list).reshape(-1, 1)
    encoder = OneHotEncoder(categories='auto', handle_unknown='ignore').fit(source_list)
    
    with open (source_encoder_path, 'wb') as source_encoder_file:
        pickle.dump(encoder, source_encoder_file)
    return encoder


# In[19]:


source_encoder = get_source_encoder()
source_encoder.transform(np.array(['中国经济周刊']).reshape(-1, 1)).toarray()[0]


# In[20]:


source_encoder.transform(np.array(['hg']).reshape(-1, 1)).toarray()[0]


# In[21]:


len(source_encoder.categories_[0])


# In[22]:


def get_doctype_encoder():
    if os.path.exists(doctype_encoder_path):
        with open (doctype_encoder_path, 'rb') as doctype_encoder_file: 
            return pickle.load(doctype_encoder_file)
    
    doctype_list = get_feature_list(processed_expose_train_labeled_path, 'doctype')
    doctype_set = set(doctype_list)
    doctype_list = list(doctype_set)
    
    with open (doctype_encoder_path, 'wb') as doctype_encoder_file:
        pickle.dump(doctype_list, doctype_encoder_file)
    return doctype_list


# In[23]:


doctype_list = get_doctype_encoder()
doctype_list, len(doctype_list)


# In[ ]:





# In[24]:


# with open(processed_expose_train_labeled_path, 'r', encoding="utf-8") as input_file, \
#      open(processed_expose_train_labeled_train_path, 'w', encoding="utf-8") as train_file, \
#      open(processed_expose_train_labeled_valid_path, 'w', encoding="utf-8") as valid_file:
#     for line in tqdm(input_file):
#         json_data = json.loads(line)
#         if random.random() < 0.1:
#             valid_file.write(f"{json.dumps(json_data, ensure_ascii=False)}\n")
#         else:
#             train_file.write(f"{json.dumps(json_data, ensure_ascii=False)}\n")


# In[ ]:





# In[25]:


class MyDataset(Dataset):
    def __init__(self, input_path, dataset_type):
        self.dataset_type = dataset_type
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_bert_path)
        self.data_list = self.load_data(input_path)

    def __getitem__(self, index):
        item = self.data_list[index]
        
        token = self.tokenizer(item['text'], add_special_tokens=True,
                                              max_length=512,
                                              truncation=True,
                                              padding='max_length',
                                              return_tensors="pt")
        del item['text']
#         item = dict(item, **token)
        item['input_ids'] = token['input_ids'][0]
        item['token_type_ids'] = token['token_type_ids'][0]
        item['attention_mask'] = token['attention_mask'][0]
        
        item['category'] = category_encoder.transform(np.array([item['category']]).reshape(-1, 1)).toarray()[0]
        item['paragraphs_num'] = paragraphs_num_encoder.transform(np.array([item['paragraphs_num']]).reshape(-1, 1))[0]
        item['words_len'] = words_len_encoder.transform(np.array([item['words_len']]).reshape(-1, 1))[0]
        del item['pic_num']
        item['source'] = source_encoder.transform(np.array([item['source']]).reshape(-1, 1)).toarray()[0]
        
        del item['id']

        if self.dataset_type == 'test':
            item['doctype'] = -1
        else:
            item['doctype'] = doctype_list.index(item['doctype'])
        label = item['doctype']
        
        del item['doctype']
        
        #         print(item)
        
        return item, label

    def __len__(self):
        return len(self.data_list)

    def load_data(self, input_path):
        data_list = []
        with open(input_path, 'r', encoding='utf-8') as input_file:
            for line in tqdm(input_file):
                json_data = json.loads(line)
                data_list.append(json_data)
        if self.dataset_type != 'test':
            random.shuffle(data_list)
        return data_list


# In[26]:


# train_dataset = MyDataset(processed_expose_train_labeled_path, 'train')


# In[27]:


# train_loader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True, num_workers=num_workers)
# for data, label in tqdm(train_loader):
#     print(data)
#     print(label)

# #     bert_output = bert(input_ids=data['input_ids'], token_type_ids=data['token_type_ids'], attention_mask=data['attention_mask'])
# #     print(bert_output)
# #     bert_cls_hidden_state = bert_output[0][:, 0, :]
# #     print(bert_cls_hidden_state)
# #     print(bert_cls_hidden_state.shape)
#     break


# In[28]:


class BertClassificationModel(nn.Module):
    """Bert分类器模型"""
    def __init__(self, hidden_size=768):
        super(BertClassificationModel, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_bert_path)
        
        category_size = len(category_encoder.categories_[0])
        paragraphs_num_size = 1
        words_len_size = 1
        source_size = len(source_encoder.categories_[0])
        linear_size = hidden_size + category_size + paragraphs_num_size + source_size + words_len_size
        self.net = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(linear_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Linear(64, len(doctype_list))
#         self.out = nn.Linear(linear_size, len(doctype_list))

    def forward(self, batch_data):
        input_ids = batch_data['input_ids'].clone().detach().cuda()
        token_type_ids = batch_data['token_type_ids'].clone().detach().cuda()
        attention_mask = batch_data['attention_mask'].clone().detach().cuda()
        bert_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        bert_cls_hidden_state = bert_output[0][:, 0, :]
#         print(bert_cls_hidden_state.shape)
#         print(batch_data['category'].shape)
#         print(batch_data['paragraphs_num'].shape)
#         print(batch_data['source'].shape)
        category = batch_data['category'].clone().detach().cuda()
        paragraphs_num = batch_data['paragraphs_num'].clone().detach().cuda()
        source = batch_data['source'].clone().detach().cuda()
        words_len = batch_data['words_len'].clone().detach().cuda()
        cat_layer = torch.cat((bert_cls_hidden_state, category, paragraphs_num, source, words_len), 1)
#         print(cat_layer.shape)
        output = self.net(cat_layer.to(torch.float32))
        output = self.out(output)
        
        return output


# In[29]:


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
#         torch.save(model.state_dict(), model_train_first_path)	# 这里会存储迄今最优模型的参数
        torch.save(model, model_train_first_path)
        self.val_loss_min = val_loss


# In[30]:


batch_size = 2

# torch.cuda.set_device(0)
model = BertClassificationModel()
# torch.set_default_tensor_type(torch.DoubleTensor)
model = model.cuda()
# print(model)

# 不同子网络设定不同的学习率
Bert_model_param = []
Bert_downstream_param = []
for items, _ in model.named_parameters():
    if "bert" in items:
        Bert_model_param.append(_)
    else:
        Bert_downstream_param.append(_)
param_groups = [{"params": Bert_model_param, "lr": 1e-5},
                {"params": Bert_downstream_param, "lr": 1e-4}]
optimizer = optim.Adam(param_groups, eps=1e-7, weight_decay=0.001)
# 初始化 early_stopping 对象
patience = 2	# 当验证集损失在连续n次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
early_stopping = EarlyStopping(patience, verbose=True)
criterion = nn.CrossEntropyLoss()


# In[ ]:


epoch_num = 5
batch_loss_num = 100
for epoch in range(epoch_num):
    model.train()	# 设置模型为训练模式
    train_dataset = MyDataset(processed_expose_train_labeled_train_path, 'train')
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    with tqdm(total=len(train_loader)) as t:
        loss_sum = 0.0
        batch_loss = 1.0
        for batch_idx,data_ in enumerate(train_loader, 0):
            data, label = data_
            # 清空梯度
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label.cuda())
#             output = model(data)
#             loss = criterion(output, label)
            loss.backward()

            # 更新模型参数
            optimizer.step()
            
            loss_sum += loss.item()
            if batch_idx % batch_loss_num == batch_loss_num - 1:
                batch_loss = loss_sum / batch_loss_num
                loss_sum = 0.0
          
            t.set_postfix_str(f'train_loss={loss.item():.6f} batch_loss={batch_loss:.6f}')
            t.update()
            
            del data, label, output 
            gc.collect()
            torch.cuda.empty_cache()
    #----------------------------------------------------
    model.eval() # 设置模型为评估/测试模式
    valid_dataset = MyDataset(processed_expose_train_labeled_valid_path, 'valid')
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loss_sum = 0.0
    with tqdm(total=len(valid_loader)) as t:
        with torch.no_grad():
            for data, label in valid_loader:
                # 一般如果验证集不是很大的话，模型验证就不需要按批量进行了，但要注意输入参数的维度不能错
                output = model(data)
                loss = criterion(output, label.cuda())
#                 output = model(data)
#                 loss = criterion(output, label)
                valid_loss_sum += loss.item()
                t.set_postfix_str(f'valid loss={loss.item()}')
                t.update()
    early_stopping(valid_loss_sum, model)
    # 若满足 early stopping 要求
    if early_stopping.early_stop:
        print("Early stopping")
        # 结束模型训练
        break
    # # 保存完整的 BERT 分类器模型
    # torch.save(model, model_train_first_path)


# In[ ]:


# torch.save(model, model_train_first_path)


# In[ ]:


# 获得 early stopping 时的模型参数
# model.load_state_dict(torch.load(model_train_first_path))


# In[ ]:





```
