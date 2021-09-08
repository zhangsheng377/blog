---
title: "pytorch版bert多目标多分类"
date: 2021-09-04T23:22:58+08:00
lastmod: 2021-09-04T23:22:58+08:00
draft: false
keywords: []
description: ""
tags: [pytorch, bert, 多分类, 多目标]
categories: []
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

基于上次把pytorch使用bert给调成功后，就想试试多目标，或者叫专家网络的结构。

题目是这样的，有10个正常分类和一个其他分类，但是标注的数据集只有这10个正常分类的，还有剩下大批未标注的数据。

所以就准备这样设计网络：先弄一个专家网络，专门使用有标注的数据训练那10个正常分类；然后，把专家网络的这10个输出，拼上之前的输入，再进一个正常的全部类别的分类网络，训练那带其他类别的11个分类。这样也就相当于是，最后的全类别网络可以参考之前的10类别专家的结论。

当然，后来又想了一个结构，可以在前面再加一个训练是不是其他类别的专家网络，然后把它的2分类结果，和之后的10类别专家的结果，以及正常输入，一起拼起来进最终的分类网络。

原本准备每个专家网络都用bert的，但发现6g显存的2060(在此感谢我的师弟)无法放下包含2个bert的模型，所以只好让这几个专家都使用同一个bert，这样就变成了多目标的那一套共享层了，也算是异曲同工。

最后的loss还是按照每个专家网络，包括他们各自的输出层，独自更新设计的。并且针对非全部类别的专家网络，在计算他们的loss时，所使用的predict_label和label也都取了相应的子集。

但这样其实共享层的bert就会更新多次了，这的确是当时没设计好的地方，应该把共享层的bert的参数更新单独取出，使用上层各网络loss的平均值才对。

## 关键点

模型定义：
```python
class MutilTargetModel(nn.Module):
    """Bert分类器模型"""
    def __init__(self, hidden_size=768):
        super(MutilTargetModel, self).__init__()
        
#         self.expert = AutoModel.from_pretrained(pretrained_bert_path)
        
        self.bert = AutoModel.from_pretrained(pretrained_bert_path)
        
        category_size = len(category_encoder.categories_[0])
        paragraphs_num_size = 1
        words_len_size = 1
        source_size = len(source_encoder.categories_[0])
        expert_size = 10
        linear_size = hidden_size + category_size + paragraphs_num_size + source_size + words_len_size + expert_size
        self.all_net = nn.Sequential(
            nn.Linear(linear_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, len(doctype_list)),
        )
        
        self.expert_net = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, expert_size),
        )

    def forward(self, batch_data):
        input_ids = batch_data['input_ids'].cuda()
        token_type_ids = batch_data['token_type_ids'].cuda()
        attention_mask = batch_data['attention_mask'].cuda()
        
        bert_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        bert_cls_hidden_state = bert_output[0][:, 0, :]
        
#         expert = self.expert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0][:, 0, :]
        expert_output = self.expert_net(bert_cls_hidden_state)
        
        cat_layer = torch.cat((bert_cls_hidden_state, 
                               batch_data['category'].cuda(), 
                               batch_data['paragraphs_num'].cuda(), 
                               batch_data['source'].cuda(), 
                               batch_data['words_len'].cuda(),
                               expert_output), 
                              1)
        all_output = self.all_net(cat_layer.to(torch.float32))

        return all_output, expert_output
    
    def freeze(self, layer):
        for child in layer.children():
            for param in child.parameters():
                param.requires_grad = False
```

给不同的网络单独设置优化器：
```python
# 不同子网络设定不同的学习率
Bert_model_param = []
Bert_downstream_param = []
for items, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if items.startswith('expert.') or items.startswith('expert_net.'):
        continue
    if items.startswith('bert.'):
        Bert_model_param.append(param)
    else:
        Bert_downstream_param.append(param)
param_groups = [{"params": Bert_model_param, "lr": 1e-5},
                {"params": Bert_downstream_param, "lr": 1e-4}]
all_optimizer = optim.Adam(param_groups, eps=1e-7, weight_decay=0.001)

Bert_model_param = []
Bert_downstream_param = []
for items, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if not items.startswith('expert.') or items.startswith('expert_net.'):
        continue
    if items.startswith('bert.'):
        Bert_model_param.append(param)
    else:
        Bert_downstream_param.append(param)
param_groups = [{"params": Bert_model_param, "lr": 1e-5},
                {"params": Bert_downstream_param, "lr": 1e-4}]
expert_optimizer = optim.Adam(param_groups, eps=1e-7, weight_decay=0.001)

# 初始化 early_stopping 对象
patience = 2	# 当验证集损失在连续n次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
early_stopping = EarlyStopping(patience, verbose=True)
criterion = nn.CrossEntropyLoss()
```

训练：
```python
epoch_num = 5
batch_loss_num = 100
for epoch in range(epoch_num):
    model.train()	# 设置模型为训练模式
    train_dataset = MyDataset([processed_expose_train_train_path], 'train')
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    with tqdm(total=len(train_loader)) as t:
        loss_sum = 0.0
        batch_loss = 1.0
        for batch_idx,data_ in enumerate(train_loader, 0):
            data, label = data_
            # 清空梯度
            all_optimizer.zero_grad()
            expert_optimizer.zero_grad()
            
            all_output, expert_output = model(data)
            
            all_loss = criterion(all_output, label.cuda())
#             all_loss = all_loss / 2
            
#             expert_predict = np.argmax(expert_output.cpu(), 1)
            other_index = doctype_list.index('其他')
            expert_predict = torch.tensor(np.argmax(expert_output.cpu().detach().numpy(), 1))
            expert_index = label != other_index
            expert_label = label[expert_index]
            expert_output = expert_output[expert_index]
            expert_loss = criterion(expert_output, expert_label.cuda())
#             expert_loss = expert_loss / 2
    
            all_loss.backward(retain_graph=True)
#             all_loss.backward()
            expert_loss.backward()
            # 更新模型参数
            all_optimizer.step()
            expert_optimizer.step()
            
            loss_sum += all_loss.item()
            if batch_idx % batch_loss_num == batch_loss_num - 1:
                batch_loss = loss_sum / batch_loss_num
                localtime = time.asctime( time.localtime(time.time()) )
                with open("early_stop.log", 'a+', encoding="utf-8") as log_file:
                    log_file.write(f'{localtime} train_loss={all_loss.item():.6f} batch_loss={batch_loss:.6f}\n')
                loss_sum = 0.0
          
            t.set_postfix_str(f'train_loss={all_loss.item():.6f} batch_loss={batch_loss:.6f}')
            t.update()
            
            del data, label, all_output, expert_output, other_index, expert_predict, expert_index, expert_label
            gc.collect()
            torch.cuda.empty_cache()
            
    #----------------------------------------------------
    model.eval() # 设置模型为评估/测试模式
    valid_dataset = MyDataset([processed_expose_train_valid_path], 'valid')
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loss_sum = 0.0
    with tqdm(total=len(valid_loader)) as t:
        with torch.no_grad():
            for data, label in valid_loader:
                # 一般如果验证集不是很大的话，模型验证就不需要按批量进行了，但要注意输入参数的维度不能错
                all_output, _ = model(data)
                all_loss = criterion(all_output, label.cuda())
                valid_loss_sum += all_loss.item()
                t.set_postfix_str(f'valid loss={all_loss.item()}')
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
# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
get_ipython().system('nvidia-smi')


# %%



# %%
import json
import random
import os
import pickle
import time
import gc
import copy

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
import pandas as pd


# %%
torch.__version__


# %%
torch.cuda.is_available()


# %%
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enabled = False


# %%
processed_expose_test_path = "data/processed_test_expose.json"
processed_expose_train_labeled_path = "data/processed_train_expose_labeled.json"
processed_expose_train_unlabel_predict_other_path = "data/processed_train_expose_unlabel_predict_other_0.5.json"
processed_expose_train_valid_path = "data/processed_train_valid.json"
processed_expose_train_train_path = "data/processed_train_train.json"

pretrained_bert_path = "bert-base-chinese/"

category_encoder_path = 'model/category_encoder_second.pickle'
paragraphs_num_encoder_path = 'model/paragraphs_num_encoder_second.pickle'
source_encoder_path = 'model/source_encoder_second.pickle'
doctype_encoder_path = 'model/doctype_encoder_second.pickle'
words_len_encoder_path = 'model/words_len_encoder_second.pickle'
model_train_mutil_model_path = 'model/model_train_mutil_target.pt'

submission_path = "submission_train_mutil_target_predict.csv"

num_workers = 0


# %%



# %%
def get_feature_list(input_path, feature_name):
    feature_list = []
    with open(input_path, 'r', encoding="utf-8") as input_file:
        for line in tqdm(input_file):
            json_data = json.loads(line)
            feature_list.append(json_data[feature_name])
    return feature_list


# %%
def get_category_encoder():
    if os.path.exists(category_encoder_path):
        with open (category_encoder_path, 'rb') as category_encoder_file: 
            return pickle.load(category_encoder_file)
    
    category_list = get_feature_list(processed_expose_train_labeled_path, 'category') +                     get_feature_list(processed_expose_test_path, 'category') +                     get_feature_list(processed_expose_train_unlabel_predict_other_path, 'category')
    category_list = np.array(category_list).reshape(-1, 1)
    encoder = OneHotEncoder(categories='auto', handle_unknown='ignore').fit(category_list)
    
    with open (category_encoder_path, 'wb') as category_encoder_file:
        pickle.dump(encoder, category_encoder_file)
    return encoder


# %%
category_encoder = get_category_encoder()
category_encoder.transform(np.array([1]).reshape(-1, 1)).toarray()[0]


# %%
len(category_encoder.categories_[0])


# %%
def get_paragraphs_num_encoder():
    if os.path.exists(paragraphs_num_encoder_path):
        with open (paragraphs_num_encoder_path, 'rb') as paragraphs_num_encoder_file: 
            return pickle.load(paragraphs_num_encoder_file)
    
    paragraphs_num_list = get_feature_list(processed_expose_train_labeled_path, 'paragraphs_num') +                           get_feature_list(processed_expose_test_path, 'paragraphs_num') +                           get_feature_list(processed_expose_train_unlabel_predict_other_path, 'paragraphs_num')
    paragraphs_num_list = np.array(paragraphs_num_list).reshape(-1, 1)
    encoder = StandardScaler().fit(paragraphs_num_list)
    
    with open (paragraphs_num_encoder_path, 'wb') as paragraphs_num_encoder_file:
        pickle.dump(encoder, paragraphs_num_encoder_file)
    return encoder


# %%
paragraphs_num_encoder = get_paragraphs_num_encoder()
paragraphs_num_encoder.transform(np.array([100]).reshape(-1, 1))[0][0]


# %%
paragraphs_num_encoder.transform(np.array([99999999999]).reshape(-1, 1))[0][0]


# %%
def get_words_len_encoder():
    if os.path.exists(words_len_encoder_path):
        with open (words_len_encoder_path, 'rb') as words_len_encoder_file: 
            return pickle.load(words_len_encoder_file)
    
    words_len_list = get_feature_list(processed_expose_train_labeled_path, 'words_len') +                      get_feature_list(processed_expose_test_path, 'words_len') +                      get_feature_list(processed_expose_train_unlabel_predict_other_path, 'words_len')
    words_len_list = np.array(words_len_list).reshape(-1, 1)
    encoder = StandardScaler().fit(words_len_list)
    
    with open (words_len_encoder_path, 'wb') as words_len_encoder_file:
        pickle.dump(encoder, words_len_encoder_file)
    return encoder


# %%
words_len_encoder = get_words_len_encoder()
words_len_encoder.transform(np.array([1000]).reshape(-1, 1))[0][0]


# %%
words_len_encoder.transform(np.array([2000]).reshape(-1, 1))[0][0]


# %%
def get_source_encoder():
    if os.path.exists(source_encoder_path):
        with open (source_encoder_path, 'rb') as source_encoder_file: 
            return pickle.load(source_encoder_file)
    
    source_list = get_feature_list(processed_expose_train_labeled_path, 'source') +                   get_feature_list(processed_expose_test_path, 'source') +                   get_feature_list(processed_expose_train_unlabel_predict_other_path, 'source')
    source_list = np.array(source_list).reshape(-1, 1)
    encoder = OneHotEncoder(categories='auto', handle_unknown='ignore').fit(source_list)
    
    with open (source_encoder_path, 'wb') as source_encoder_file:
        pickle.dump(encoder, source_encoder_file)
    return encoder


# %%
source_encoder = get_source_encoder()
source_encoder.transform(np.array(['中国经济周刊']).reshape(-1, 1)).toarray()[0]


# %%
source_encoder.transform(np.array(['hg']).reshape(-1, 1)).toarray()[0]


# %%
len(source_encoder.categories_[0])


# %%
def get_doctype_encoder():
    if os.path.exists(doctype_encoder_path):
        with open (doctype_encoder_path, 'rb') as doctype_encoder_file: 
            return pickle.load(doctype_encoder_file)
    
    doctype_list = get_feature_list(processed_expose_train_labeled_path, 'doctype') +                   get_feature_list(processed_expose_train_unlabel_predict_other_path, 'doctype')
    doctype_set = set(doctype_list)
    doctype_set.remove('其他')
    doctype_list = list(doctype_set)
    doctype_list.append('其他')
    
    with open (doctype_encoder_path, 'wb') as doctype_encoder_file:
        pickle.dump(doctype_list, doctype_encoder_file)
    return doctype_list


# %%
doctype_list = get_doctype_encoder()
doctype_list, len(doctype_list)


# %%



# %%
# with open(processed_expose_train_labeled_path, 'r', encoding="utf-8") as input_file, \
#      open(processed_expose_train_unlabel_predict_other_path, 'r', encoding="utf-8") as input1_file, \
#      open(processed_expose_train_train_path, 'w', encoding="utf-8") as train_file, \
#      open(processed_expose_train_valid_path, 'w', encoding="utf-8") as valid_file:
#     for line in tqdm(input_file):
#         json_data = json.loads(line)
#         if random.random() < 0.1:
#             valid_file.write(f"{json.dumps(json_data, ensure_ascii=False)}\n")
#         else:
#             train_file.write(f"{json.dumps(json_data, ensure_ascii=False)}\n")
#     for line in tqdm(input1_file):
#         json_data = json.loads(line)
#         if random.random() < 0.1:
#             valid_file.write(f"{json.dumps(json_data, ensure_ascii=False)}\n")
#         else:
#             train_file.write(f"{json.dumps(json_data, ensure_ascii=False)}\n")


# %%



# %%
class MyDataset(Dataset):
    def __init__(self, input_paths, dataset_type):
        self.dataset_type = dataset_type
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_bert_path)
        self.data_list = self.load_data(input_paths)
        
    @classmethod
    def get_data_label(cls, tokenizer, item, dataset_type):
        token = tokenizer(item['text'], add_special_tokens=True,
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
        
#         del item['id']

        if dataset_type == 'test':
            item['doctype'] = -1
        else:
            item['doctype'] = doctype_list.index(item['doctype'])
        label = item['doctype']
        
        del item['doctype']
        
        #         print(item)
        
        return item, label

    def __getitem__(self, index):
        item = self.data_list[index]
        item, label = MyDataset.get_data_label(self.tokenizer, item, self.dataset_type)
        return item, label
        

    def __len__(self):
        return len(self.data_list)

    def load_data(self, input_paths):
        if type(input_paths) != list:
            print("input_paths is not list!")
            return
        data_list = []
        for input_path in input_paths:
            with open(input_path, 'r', encoding='utf-8') as input_file:
                for line in tqdm(input_file):
                    json_data = json.loads(line)
                    data_list.append(json_data)
        if self.dataset_type != 'test':
            random.shuffle(data_list)
        return data_list


# %%
# train_dataset = MyDataset([processed_expose_train_train_path], 'train')
# train_loader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True, num_workers=num_workers)
# for batch_idx,data_ in enumerate(train_loader, 0):
#     print(data_)
#     break


# %%
# torch.tensor([])


# %%
train_dataset = MyDataset([processed_expose_train_train_path], 'train')
train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=num_workers)
data_tmp = {'id':[],
       'category':torch.tensor([]),
       'paragraphs_num':torch.tensor([]),
       'source':torch.tensor([]),
       'words_len':torch.tensor([]),
       'input_ids':torch.tensor([]),
       'token_type_ids':torch.tensor([]),
       'attention_mask':torch.tensor([]),}
label_tmp = torch.tensor([], dtype=torch.int)
count = 0
for data_ in train_loader:
#     print(data_)
    data, label = data_
#     print(label)
    data_tmp['id'].extend(data['id'])
    data_tmp['category'] = torch.cat((data_tmp['category'], data['category']), 0)
    data_tmp['paragraphs_num'] = torch.cat((data_tmp['paragraphs_num'], data['paragraphs_num']), 0)
    data_tmp['source'] = torch.cat((data_tmp['source'], data['source']), 0)
    data_tmp['words_len'] = torch.cat((data_tmp['words_len'], data['words_len']), 0)
    data_tmp['input_ids'] = torch.cat((data_tmp['input_ids'], data['input_ids']), 0)
    data_tmp['token_type_ids'] = torch.cat((data_tmp['token_type_ids'], data['token_type_ids']), 0)
    data_tmp['attention_mask'] = torch.cat((data_tmp['attention_mask'], data['attention_mask']), 0)
    label_tmp = torch.cat((label_tmp, label), 0)
    count += 1
    if count > 1:
        print(data_tmp)
        print(label_tmp)
        print(label_tmp.shape[0])
        print(torch.where(label_tmp == 1, 2, label_tmp))
        data_tmp = {'id':[],
               'category':torch.tensor([]),
               'paragraphs_num':torch.tensor([]),
               'source':torch.tensor([]),
               'words_len':torch.tensor([]),
               'input_ids':torch.tensor([]),
               'token_type_ids':torch.tensor([]),
               'attention_mask':torch.tensor([]),}
        label_tmp = torch.tensor([], dtype=torch.int64)
        break


# %%



# %%
class MutilTargetModel(nn.Module):
    """Bert分类器模型"""
    def __init__(self, hidden_size=768):
        super(MutilTargetModel, self).__init__()
        
#         self.expert = AutoModel.from_pretrained(pretrained_bert_path)
        
        self.bert = AutoModel.from_pretrained(pretrained_bert_path)
        
        category_size = len(category_encoder.categories_[0])
        paragraphs_num_size = 1
        words_len_size = 1
        source_size = len(source_encoder.categories_[0])
        expert_size = 10
        linear_size = hidden_size + category_size + paragraphs_num_size + source_size + words_len_size + expert_size
        self.all_net = nn.Sequential(
            nn.Linear(linear_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, len(doctype_list)),
        )
        
        self.expert_net = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, expert_size),
        )

    def forward(self, batch_data):
        input_ids = batch_data['input_ids'].cuda()
        token_type_ids = batch_data['token_type_ids'].cuda()
        attention_mask = batch_data['attention_mask'].cuda()
        
        bert_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        bert_cls_hidden_state = bert_output[0][:, 0, :]
        
#         expert = self.expert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0][:, 0, :]
        expert_output = self.expert_net(bert_cls_hidden_state)
        
        cat_layer = torch.cat((bert_cls_hidden_state, 
                               batch_data['category'].cuda(), 
                               batch_data['paragraphs_num'].cuda(), 
                               batch_data['source'].cuda(), 
                               batch_data['words_len'].cuda(),
                               expert_output), 
                              1)
#         print(cat_layer.shape)
        all_output = self.all_net(cat_layer.to(torch.float32))

        return all_output, expert_output
    
    def freeze(self, layer):
        for child in layer.children():
#             print(child)
            for param in child.parameters():
                param.requires_grad = False
#                 print(param)


# %%
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
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}. score:{-score}')
            with open("early_stop.log", 'a+', encoding="utf-8") as log_file:
                log_file.write(f'{localtime} EarlyStopping counter: {self.counter} out of {self.patience}. score:{-score}\n')
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
            localtime = time.asctime( time.localtime(time.time()) )
            with open("early_stop.log", 'a+', encoding="utf-8") as log_file:
                log_file.write(f'{localtime} Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...\n')
#         torch.save(model.state_dict(), model_train_mutil_model_path)	# 这里会存储迄今最优模型的参数
        torch.save(model, model_train_mutil_model_path)
        self.val_loss_min = val_loss


# %%
batch_size = 3

# torch.cuda.set_device(0)
model = MutilTargetModel()
# torch.set_default_tensor_type(torch.DoubleTensor)
model = model.cuda()
# print(model)

# 不同子网络设定不同的学习率
Bert_model_param = []
Bert_downstream_param = []
for items, param in model.named_parameters():
    if not param.requires_grad:
#         print(items)
        continue
    if items.startswith('expert.') or items.startswith('expert_net.'):
        continue
    if items.startswith('bert.'):
        Bert_model_param.append(param)
    else:
        Bert_downstream_param.append(param)
param_groups = [{"params": Bert_model_param, "lr": 1e-5},
                {"params": Bert_downstream_param, "lr": 1e-4}]
all_optimizer = optim.Adam(param_groups, eps=1e-7, weight_decay=0.001)

Bert_model_param = []
Bert_downstream_param = []
for items, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if not items.startswith('expert.') or items.startswith('expert_net.'):
        continue
    if items.startswith('bert.'):
        Bert_model_param.append(param)
    else:
        Bert_downstream_param.append(param)
param_groups = [{"params": Bert_model_param, "lr": 1e-5},
                {"params": Bert_downstream_param, "lr": 1e-4}]
expert_optimizer = optim.Adam(param_groups, eps=1e-7, weight_decay=0.001)

# 初始化 early_stopping 对象
patience = 2	# 当验证集损失在连续n次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
early_stopping = EarlyStopping(patience, verbose=True)
criterion = nn.CrossEntropyLoss()


# %%
epoch_num = 5
batch_loss_num = 100
for epoch in range(epoch_num):
    model.train()	# 设置模型为训练模式
    train_dataset = MyDataset([processed_expose_train_train_path], 'train')
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    with tqdm(total=len(train_loader)) as t:
        loss_sum = 0.0
        batch_loss = 1.0
        for batch_idx,data_ in enumerate(train_loader, 0):
            data, label = data_
            # 清空梯度
            all_optimizer.zero_grad()
            expert_optimizer.zero_grad()
            
            all_output, expert_output = model(data)
            
            all_loss = criterion(all_output, label.cuda())
#             all_loss = all_loss / 2
            
#             print(f"label:{label}")
#             print(f"expert_output:{expert_output}")
#             print(all_output)
#             expert_predict = np.argmax(expert_output.cpu(), 1)
            other_index = doctype_list.index('其他')
            expert_predict = torch.tensor(np.argmax(expert_output.cpu().detach().numpy(), 1))
#             print(f"expert_predict:{expert_predict}")
#             expert_label = torch.where(label > other_index, label - 1, label)
#             print(expert_label)
#             expert_label = torch.where(label == other_index, expert_predict, expert_label)
            expert_index = label != other_index
#             print(f"expert_index:{expert_index}")
            expert_label = label[expert_index]
# #             expert_label = label.clone().detach().float().requires_grad_()
# #             for i in range(label.shape[0]):
# #                 if label[i] == doctype_list.index('其他'):
# #                     expert_label[i] = torch.max(expert_output[i])
#             print(f"expert_label:{expert_label}")
            expert_output = expert_output[expert_index]
#             print(f"expert_output:{expert_output}")
            expert_loss = criterion(expert_output, expert_label.cuda())
#             expert_loss = expert_loss / 2
    
            all_loss.backward(retain_graph=True)
#             all_loss.backward()
            expert_loss.backward()
            # 更新模型参数
            all_optimizer.step()
            expert_optimizer.step()
            
            loss_sum += all_loss.item()
            if batch_idx % batch_loss_num == batch_loss_num - 1:
                batch_loss = loss_sum / batch_loss_num
                localtime = time.asctime( time.localtime(time.time()) )
                with open("early_stop.log", 'a+', encoding="utf-8") as log_file:
                    log_file.write(f'{localtime} train_loss={all_loss.item():.6f} batch_loss={batch_loss:.6f}\n')
                loss_sum = 0.0
          
            t.set_postfix_str(f'train_loss={all_loss.item():.6f} batch_loss={batch_loss:.6f}')
            t.update()
            
            del data, label, all_output, expert_output, other_index, expert_predict, expert_index, expert_label
            gc.collect()
            torch.cuda.empty_cache()
            
    #----------------------------------------------------
    model.eval() # 设置模型为评估/测试模式
    valid_dataset = MyDataset([processed_expose_train_valid_path], 'valid')
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loss_sum = 0.0
    with tqdm(total=len(valid_loader)) as t:
        with torch.no_grad():
            for data, label in valid_loader:
                # 一般如果验证集不是很大的话，模型验证就不需要按批量进行了，但要注意输入参数的维度不能错
                all_output, _ = model(data)
                all_loss = criterion(all_output, label.cuda())
                valid_loss_sum += all_loss.item()
                t.set_postfix_str(f'valid loss={all_loss.item()}')
                t.update()
    early_stopping(valid_loss_sum, model)
    # 若满足 early stopping 要求
    if early_stopping.early_stop:
        print("Early stopping")
        # 结束模型训练
        break


# %%
# 保存完整的 BERT 分类器模型
# torch.save(model, model_train_mutil_model_path)


# %%
# 获得 early stopping 时的模型参数
# model.load_state_dict(torch.load(model_train_mutil_model_path))


# %%
epoch_num = 1
batch_loss_num = 100
for epoch in range(epoch_num):
    model.train()	# 设置模型为训练模式
    train_dataset = MyDataset([processed_expose_train_labeled_path, processed_expose_train_unlabel_predict_other_path], 'train')
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    with tqdm(total=len(train_loader)) as t:
        loss_sum = 0.0
        batch_loss = 1.0
        for batch_idx,data_ in enumerate(train_loader, 0):
            data, label = data_
             # 清空梯度
            all_optimizer.zero_grad()
            expert_optimizer.zero_grad()
            
            all_output, expert_output = model(data)
            
            all_loss = criterion(all_output, label.cuda())
#             all_loss = all_loss / 2
            
            other_index = doctype_list.index('其他')
#             print(expert_output)
            expert_predict = torch.tensor(np.argmax(expert_output.cpu().detach().numpy(), 1))
#             print(expert_predict)
#             print(label)
#             expert_label = torch.where(label > other_index, label - 1, label)
#             expert_label = torch.where(label == other_index, expert_predict, expert_label)
            expert_label = label.clone().detach()
            for i in range(label.shape[0]):
                if label[i] == other_index:
                    expert_label[i] = expert_predict[i]
                elif label[i] > other_index:
                    expert_label[i] -= 1
#             print(expert_label)
            expert_loss = criterion(expert_output, expert_label.cuda())
#             expert_loss = expert_loss / 2
    
            all_loss.backward(retain_graph=True)
            expert_loss.backward()
            # 更新模型参数
            all_optimizer.step()
            expert_optimizer.step()
            
            loss_sum += all_loss.item()
            if batch_idx % batch_loss_num == batch_loss_num - 1:
                batch_loss = loss_sum / batch_loss_num
                localtime = time.asctime( time.localtime(time.time()) )
                with open("early_stop.log", 'a+', encoding="utf-8") as log_file:
                    log_file.write(f'{localtime} train_loss={all_loss.item():.6f} batch_loss={batch_loss:.6f}\n')
                loss_sum = 0.0
          
            t.set_postfix_str(f'train_loss={all_loss.item():.6f} batch_loss={batch_loss:.6f}')
            t.update()
            
            del data, label, all_output, expert_output 
            gc.collect()
            torch.cuda.empty_cache()

    # 保存完整的 BERT 分类器模型
    torch.save(model, model_train_mutil_model_path)
    localtime = time.asctime( time.localtime(time.time()) )
    with open("early_stop.log", 'a+', encoding="utf-8") as log_file:
        log_file.write(f'{localtime} Saving model ...\n')
        
    model.eval() # 设置模型为评估/测试模式
    test_dataset = MyDataset([processed_expose_test_path], 'test')
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True, num_workers=num_workers)
    predict_doctype = {}
    with torch.no_grad():
        for data, label in tqdm(test_loader):
            id = data['id'][0]
            output, _ = model(data)
            predict_list = F.softmax(output, dim=1).tolist()[0]
            predict_index = np.argmax(predict_list)
            predict_label = doctype_list[predict_index]
            predict_doctype[id] = predict_label
    predict_data = {'predict_doctype' : predict_doctype}
    df = pd.DataFrame(predict_data)
    df.index.name = 'id'
    df.to_csv(f"submission_train_mutil_target_predict_{epoch}.csv")


# %%



# %%
# model = torch.load(model_train_mutil_model_path)
# model = model.cuda()


# %%



# %%
model.eval() # 设置模型为评估/测试模式
valid_dataset = MyDataset([processed_expose_train_valid_path], 'valid')
valid_loader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=True, num_workers=num_workers)
with torch.no_grad():
    for data, label in tqdm(valid_loader):
        print(data, label)
        print(data['id'][0])
        output, _ = model(data)
#         print(output)
#         print(torch.sigmoid(output))
#         sigmoid = nn.Sigmoid()
#         print(sigmoid(output))
        predict_list = F.softmax(output, dim=1).tolist()[0]
        predict_index = np.argmax(predict_list)
        predict_label = doctype_list[predict_index]
        print(predict_label)
        break


# %%
# with open("data/processed_train_expose_labeled.json", "r", encoding='utf-8') as input_file:
#     for line in tqdm(input_file):
#         json_data = json.loads(line)
#         if json_data['id'] == 'ee137ac3-c2a2-4aba-a517-36840ffd2f1a':
#             print(line)
#             break


# %%
# model.eval() # 设置模型为评估/测试模式
# valid_dataset = MyDataset("data/test.txt", 'test')
# valid_loader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=True, num_workers=num_workers)
# with torch.no_grad():
#     for data, label in tqdm(valid_loader):
# #         print(data, label)
#         output = model(data)
# #         print(output)
# #         print(torch.sigmoid(output))
#         print(F.softmax(output, dim=1).tolist()[0])
#         break


# %%



# %%
model.eval() # 设置模型为评估/测试模式
test_dataset = MyDataset([processed_expose_test_path], 'test')
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True, num_workers=num_workers)
predict_doctype = {}
with torch.no_grad():
    for data, label in tqdm(test_loader):
        id = data['id'][0]
        output, _ = model(data)
        predict_list = F.softmax(output, dim=1).tolist()[0]
        predict_index = np.argmax(predict_list)
        predict_label = doctype_list[predict_index]
        predict_doctype[id] = predict_label
predict_data = {'predict_doctype' : predict_doctype}
df = pd.DataFrame(predict_data)
df.head()


# %%
df.index.name = 'id'
df.head()


# %%
df.to_csv("submission_train_mutil_target_predict.csv")


# %%
# 看起来，复杂、大模型是有效的(在验证数据集上，第二个epoch的loss还在降低)，所以下一步准备试一下cnn和albert(或roberta)


# %%




```
