---
title: "用自定义损失函数实现选择启用不同子网络"
date: 2021-03-31T01:14:58+08:00
lastmod: 2021-03-31T01:14:58+08:00
draft: false
keywords: []
description: ""
tags: [深度学习, 自定义, 损失函数, 子网络, keras]
categories: []
author: ""

# You can also close(false) or open(true) something for this content.
# P.S. comment can only be closed
comment: false
toc: true
autoCollapseToc: false
postMetaInFooter: false
hiddenFromHomePage: false
# You can also define another contentCopyright. e.g. contentCopyright: "This is another copyright."
contentCopyright: false
reward: false
mathjax: false
mathjaxEnableSingleDollar: false
mathjaxEnableAutoNumber: false

# You unlisted posts you might want not want the header or footer to show
hideHeaderAndFooter: false

# You can enable or disable out-of-date content warning for individual post.
# Comment this out to use the global config.
#enableOutdatedInfoWarning: false

flowchartDiagrams:
  enable: false
  options: ""

sequenceDiagrams: 
  enable: false
  options: ""

---

## 背景

最近发现有一道题，还挺有意思的。题目大意是，每条训练样本是一个文章对，labelA标签标识这两篇文章相似，labelB标签标识这两篇文章属于同一事件(即紧相似)，但这个文章对不会同时拥有两个标签，即要么有A标签，要么有B标签，且A、B标签的文章对不重合。

面对这道题，一般的思路是建立两个模型。但因为标签A、B其实是有相似程度上的联系的，单独训练两个模型就失去了标签的相关性，感觉比较亏。

但如果要训练多任务的单模型，也比较麻烦。因为一个样本不能同时拥有这两个标签，而且比如不属于同一事件，没办法推出是否相似，即无法构造出缺失的标签。

所以想到，利用一个类似mask的特殊标签取值，来控制启用哪个子网络去训练。具体实现措施准备把另一个子网络的loss置0，以此不让其训练。

## 自定义loss函数

由于标签值是0、1，所以将 -1 设置为mask值，将y_true中的 -1 值，替换为y_pred中的对应值，以此使其对应样本的loss为0。

最后，由于标签是0、1，所以采用 binary_crossentropy 作为损失函数：

```python
def mycrossentropy(y_true, y_pred):
    assert len(y_true.shape) == 2
    y_true_np = y_true.numpy()
    y_pred_np = y_pred.numpy()
    tmp = y_pred_np.copy()
    for i in range(y_true.shape[0]):
        for j in range(y_true.shape[1]):
            if y_true_np[i][j] != -1:
                tmp[i][j] = y_true_np[i][j]
    
    tmp_tensor = tf.Variable(tmp)

    loss = binary_crossentropy(tmp_tensor, y_pred)
    return loss
```

## 模型定义

在编译模型时指定自定义loss函数，keras会自动对两个输出目标分别使用该自定义loss函数，最后模型算的是这两个loss之和：

```python
def get_model():
    K.clear_session()
    
    bert_model = TFBertForPreTraining.from_pretrained(bert_path, from_pt=True)
 
    input_ids = Input(shape=(None,), dtype='int32')
    input_token_type_ids = Input(shape=(None,), dtype='int32')
    input_attention_mask = Input(shape=(None,), dtype='int32')
 
    bert_output = bert_model({'input_ids':input_ids, 'token_type_ids':input_token_type_ids, 'attention_mask':input_attention_mask}, return_dict=False, training=True)
    projection_logits = bert_output[0]
    bert_cls = Lambda(lambda x: x[:, 0])(projection_logits) # 取出[CLS]对应的向量用来做分类
    
    dropout_A = Dropout(0.5)(bert_cls)
    output_A = Dense(2, activation='sigmoid')(dropout_A)
    
    dropout_B = Dropout(0.5)(bert_cls)
    output_B = Dense(2, activation='sigmoid')(dropout_B)
 
    model = Model([input_ids, input_token_type_ids, input_attention_mask], [output_A, output_B])
    model.compile(loss=mycrossentropy,
                  optimizer=Adam(1e-5),    #用足够小的学习率
                  metrics=['accuracy'])
    print(model.summary())
    return model
```





-----------------------

## 附录

### 全部源码

~~~markdown
# 导包


```python
import os
import sys
import re
from collections import Counter
import random

from tqdm import tqdm
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
tf.config.run_functions_eagerly(True)
from keras.metrics import top_k_categorical_accuracy
from keras.layers import *
from keras.callbacks import *
from keras.models import Model, load_model
import keras.backend as K
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.losses import SparseCategoricalCrossentropy, binary_crossentropy
from transformers import (
    BertTokenizer,
    TFBertForPreTraining,
    TFBertModel,
)
```


```python
tf.__version__
```




    '2.3.0'




```python

```


```python
data_path = "sohu2021_open_data/"
text_max_length = 512
bert_path = r"../chinese_L-12_H-768_A-12"
```


```python

```


```python

```


```python

```


```python

```

# 构建标签表


```python
label_to_id = {'0':0, '1':1}
```


```python
labels = [0, 1]
```


```python

```


```python

```


```python

```


```python

```

# 构建原数据文本迭代器


```python
def _transform_text(text):
   text = text.strip().replace('\n', '。').replace('\t', '').replace('\u3000', '')
   return re.sub(r'。+', '。', text)
```


```python
def get_data_iterator(data_path, file_name):
    # TODO: 随机取
    for category in os.listdir(data_path):
        category_path = os.path.join(data_path, category)
        if not os.path.isdir(category_path):
            continue
            
        file_path = os.path.join(category_path, file_name)
        if not os.path.isfile(file_path):
            continue
        
#         print(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                
                data['source'] = _transform_text(data['source'])
                if len(data['source']) == 0:
                    print('source:', line, data)
                    break
#                     continue
                    
                data['target'] = _transform_text(data['target'])
                if len(data['target']) == 0:
                    print('target:', line, data)
                    break
#                     continue
                
                label_name_list = list(key for key in data.keys() if key[:5]=='label')
                if len(label_name_list) != 1:
                    print('label_name_list:', line, data)
                    break
#                     continue
                label_name = label_name_list[0]
                if data[label_name] not in label_to_id.keys():
                    print('label_name:', line, data, label_name)
                    break
#                     continue
                    
                yield data['source'], data['target'], label_to_id[data[label_name]]
```


```python
it = get_data_iterator(data_path, "train.txt")
```


```python
next(it)
```




    ('谁能打破科比81分纪录？奥尼尔给出5个候选人，补充利拉德比尔！', 'NBA现役能入名人堂的球星很多，但是能被立铜像只有2人', 0)




```python

```


```python

```


```python

```


```python

```

# 获取数据集样本个数


```python
def get_sample_num(data_path, file_name):
    count = 0
    it = get_data_iterator(data_path, file_name)
    for data in tqdm(it):
        count += 1
    return count
```


```python
train_sample_count = get_sample_num(data_path, "train.txt")
```

    59638it [00:04, 12249.30it/s]



```python
dev_sample_count = get_sample_num(data_path, "valid.txt")
```

    9940it [00:00, 12439.17it/s]



```python
train_sample_count, dev_sample_count
```




    (59638, 9940)




```python

```


```python

```


```python

```


```python

```

# 构建数据迭代器


```python
tokenizer = BertTokenizer.from_pretrained(bert_path)
```


```python
def _get_indices(text, text_pair=None):
    return tokenizer.encode(text=text,
                            text_pair=text_pair,
                            max_length=text_max_length, 
                            add_special_tokens=True, 
                            padding='max_length', 
                            truncation_strategy='only_first', 
#                                          return_tensors='tf'
                            )
```


```python
def get_keras_bert_iterator(data_path, file_name, tokenizer):
    while True:
        data_it = get_data_iterator(data_path, file_name)
        for source, target, label in data_it:
            indices = _get_indices(text=source, 
                                   text_pair=target)
            yield indices, label
```


```python
it = get_keras_bert_iterator(data_path, "train.txt", tokenizer)
```


```python
# next(it)
```


```python

```


```python

```


```python

```


```python

```

# 构建批次数据迭代器


```python
def batch_iter(data_path, file_name, tokenizer, batch_size=64, shuffle=True):
    """生成批次数据"""
    keras_bert_iter = get_keras_bert_iterator(data_path, file_name, tokenizer)
    while True:
        data_list = []
        for _ in range(batch_size):
            data = next(keras_bert_iter)
            data_list.append(data)
        if shuffle:
            random.shuffle(data_list)
        
        indices_list = []
        label_list = []
        for data in data_list:
            indices, label = data
            indices_list.append(indices)
            label_list.append(label)

        yield np.array(indices_list), np.array(label_list)
```


```python
it = batch_iter(data_path, "train.txt", tokenizer, batch_size=1)
```


```python
# next(it)
```


```python
it = batch_iter(data_path, "train.txt", tokenizer, batch_size=2)
```


```python
next(it)
```

    /home/zsd-server/miniconda3/envs/my/lib/python3.8/site-packages/transformers/tokenization_utils_base.py:2162: FutureWarning: The `truncation_strategy` argument is deprecated and will be removed in a future version, use `truncation=True` to truncate examples to a max length. You can give a specific length with `max_length` (e.g. `max_length=45`) or leave max_length to None to truncate to the maximal input size of the model (e.g. 512 for Bert).  If you have pairs of inputs, you can give a specific truncation strategy selected among `truncation='only_first'` (will only truncate the first sentence in the pairs) `truncation='only_second'` (will only truncate the second sentence in the pairs) or `truncation='longest_first'` (will iteratively remove tokens from the longest sentence in the pairs).
      warnings.warn(





    (array([[ 101, 6435, 2810, ...,    0,    0,    0],
            [ 101, 6443, 5543, ...,    0,    0,    0]]),
     array([0, 0]))




```python

```


```python

```


```python

```


```python

```

# 定义base模型


```python
# !transformers-cli convert --model_type bert \
#   --tf_checkpoint chinese_L-12_H-768_A-12/bert_model.ckpt \
#   --config chinese_L-12_H-768_A-12/bert_config.json \
#   --pytorch_dump_output chinese_L-12_H-768_A-12/pytorch_model.bin
```


```python
# bert_model = TFBertForPreTraining.from_pretrained("./chinese_L-12_H-768_A-12/", from_pt=True)
```


```python
# # it = get_keras_bert_iterator(r"data/keras_bert_train.txt", cat_to_id, tokenizer)
# it = batch_iter(r"data/keras_bert_train.txt", cat_to_id, tokenizer, batch_size=1)
# out = bert_model(next(it)[0])
# out[0]
```


```python
def get_model(label_list):
    K.clear_session()
    
    bert_model = TFBertForPreTraining.from_pretrained(bert_path, from_pt=True)
 
    input_indices = Input(shape=(None,), dtype='int32')
 
    bert_output = bert_model(input_indices)
    projection_logits = bert_output[0]
    bert_cls = Lambda(lambda x: x[:, 0])(projection_logits) # 取出[CLS]对应的向量用来做分类
    
    dropout = Dropout(0.5)(bert_cls)
    output = Dense(len(label_list), activation='softmax')(dropout)
 
    model = Model(input_indices, output)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(1e-5),    #用足够小的学习率
                  metrics=['accuracy'])
    print(model.summary())
    return model
```


```python
early_stopping = EarlyStopping(monitor='val_loss', patience=3)   #早停法，防止过拟合
plateau = ReduceLROnPlateau(monitor="val_accuracy", verbose=1, mode='max', factor=0.5, patience=2) #当评价指标不在提升时，减少学习率
checkpoint = ModelCheckpoint('trained_model/keras_bert_sohu.hdf5', monitor='val_loss',verbose=2, save_best_only=True, mode='max', save_weights_only=True) #保存最好的模型
```

## 模型训练


```python
def get_step(sample_count, batch_size):
    step = sample_count // batch_size
    if sample_count % batch_size != 0:
        step += 1
    return step
```


```python
# batch_size = 2
# train_step = get_step(train_sample_count, batch_size)
# dev_step = get_step(dev_sample_count, batch_size)

# train_dataset_iterator = batch_iter(data_path, "train.txt", tokenizer, batch_size)
# dev_dataset_iterator = batch_iter(data_path, "valid.txt", tokenizer, batch_size)

# model = get_model(labels)

# #模型训练
# model.fit(
#     train_dataset_iterator,
#     steps_per_epoch=10,
# #     steps_per_epoch=train_step,
#     epochs=5,
#     validation_data=dev_dataset_iterator,
#     validation_steps=2,
# #     validation_steps=dev_step,
#     callbacks=[early_stopping, plateau, checkpoint],
#     verbose=1
# )

# model.save_weights("trained_model/keras_bert_sohu_final.weights")
# model.save("trained_model/keras_bert_sohu_final.model")
```


```python

```


```python

```


```python

```


```python

```

# 多任务分支模型

## 构建数据迭代器


```python
label_type_to_id = {'labelA':0, 'labelB':1}
```


```python
def get_text_iterator(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield line
```


```python
def get_data_iterator(data_path, file_name):
    # TODO: 随机取
    file_iters = []
    for category in os.listdir(data_path):
        category_path = os.path.join(data_path, category)
        if not os.path.isdir(category_path):
            continue
            
        file_path = os.path.join(category_path, file_name)
        if not os.path.isfile(file_path):
            continue
            
        
        file_iter = get_text_iterator(file_path)
        file_iters.append(file_iter)
        
    while len(file_iters) > 0:
        i = random.randrange(len(file_iters))
        line = next(file_iters[i], None)
        if line is None:
            del file_iters[i]
            continue
            
        data = json.loads(line)

        data['source'] = _transform_text(data['source'])
        if len(data['source']) == 0:
            print('source:', line, data)
            break
#                     continue

        data['target'] = _transform_text(data['target'])
        if len(data['target']) == 0:
            print('target:', line, data)
            break
#                     continue

        label_name_list = list(key for key in data.keys() if key[:5]=='label')
        if len(label_name_list) != 1:
            print('label_name_list:', line, data)
            break
#                     continue
        label_name = label_name_list[0]
        if data[label_name] not in label_to_id.keys():
            print('label_name:', line, data, label_name)
            break
#                     continue
        
        label_dict = {key:-1 for key in label_type_to_id.keys()}
        label_dict[label_name] = label_to_id[data[label_name]]

        yield data['source'], data['target'], label_dict['labelA'], label_dict['labelB']
```


```python
it = get_data_iterator(data_path, "train.txt")
```


```python
next(it)
```




    ('陈立农、小鬼亮相腾讯音乐娱乐盛典红毯金发贵公子+红发脏辫鬼少帅出特色',
     '今日，腾讯音乐娱乐盛典在澳门举行，虽然坤坤今天没有走红毯，但依旧存在感满分，前辈任贤齐谈及欣赏的后辈时cue到了坤坤，“蔡徐坤，他这么帅气，他也努力在创作。”新生代团体Boystory也表示期待“蔡徐坤哥哥”的舞台。 从乐坛前辈到新生代团体，坤坤简直老少通杀，国民度爆表，他不在江湖，江湖却处处有他的传说，期待今晚蔡徐坤的精彩舞台！ 直播【戳这里】   下载「爱豆App」了解偶像的最新动态 下载「爱豆App」了解偶像的最新动态',
     -1,
     0)




```python
get_sample_num(data_path, "train.txt")
```

    59638it [00:04, 12109.11it/s]





    59638




```python
def _get_indices(text, text_pair=None):
    return tokenizer.encode_plus(text=text,
                            text_pair=text_pair,
                            max_length=text_max_length, 
                            add_special_tokens=True, 
                            padding='max_length', 
                            truncation_strategy='longest_first', 
#                                          return_tensors='tf',
                            return_token_type_ids=True
                            )
```


```python
def get_keras_bert_iterator(data_path, file_name, tokenizer):
    while True:
        data_it = get_data_iterator(data_path, file_name)
        for source, target, labelA, labelB in data_it:
            data = _get_indices(text=source, 
                                   text_pair=target)
#             print(indices, type(indices), len(indices))
            yield data['input_ids'], data['token_type_ids'], data['attention_mask'], labelA, labelB
```


```python
it = get_keras_bert_iterator(data_path, "train.txt", tokenizer)
```


```python
# next(it)
```


```python
def batch_iter(data_path, file_name, tokenizer, batch_size=64, shuffle=True):
    """生成批次数据"""
    keras_bert_iter = get_keras_bert_iterator(data_path, file_name, tokenizer)
    while True:
        data_list = []
        for _ in range(batch_size):
            data = next(keras_bert_iter)
            data_list.append(data)
        if shuffle:
            random.shuffle(data_list)
        
        input_ids_list = []
        token_type_ids_list = []
        attention_mask_list = []
        labelA_list = []
        labelB_list = []
        for data in data_list:
            input_ids, token_type_ids, attention_mask, labelA, labelB = data
#             print(indices, type(indices))
            input_ids_list.append(input_ids)
            token_type_ids_list.append(token_type_ids)
            attention_mask_list.append(attention_mask)
            labelA_list.append(labelA)
            labelB_list.append(labelB)

        yield [np.array(input_ids_list), np.array(token_type_ids_list), np.array(attention_mask_list)], [np.array(labelA_list, dtype=np.int32), np.array(labelB_list, dtype=np.int32)]
```


```python
it = batch_iter(data_path, "train.txt", tokenizer, batch_size=2)
```


```python
next(it)
```




    ([array([[ 101,  686, 4518, ..., 4507, 1102,  102],
             [ 101,  122, 3299, ..., 8024, 1506,  102]]),
      array([[0, 0, 0, ..., 1, 1, 1],
             [0, 0, 0, ..., 1, 1, 1]]),
      array([[1, 1, 1, ..., 1, 1, 1],
             [1, 1, 1, ..., 1, 1, 1]])],
     [array([ 0, -1], dtype=int32), array([-1,  0], dtype=int32)])



## 定义模型


```python
def mycrossentropy(y_true, y_pred):
#     print(y_true)
    assert len(y_true.shape) == 2
    y_true_np = y_true.numpy()
    y_pred_np = y_pred.numpy()
    tmp = y_pred_np.copy()
    for i in range(y_true.shape[0]):
        for j in range(y_true.shape[1]):
            if y_true_np[i][j] != -1:
                tmp[i][j] = y_true_np[i][j]
#     print(y_true_np, y_pred_np, tmp)
    
    tmp_tensor = tf.Variable(tmp)
#     print(tmp_tensor)

    loss = binary_crossentropy(tmp_tensor, y_pred)
    return loss
```


```python
def get_model():
    K.clear_session()
    
    bert_model = TFBertForPreTraining.from_pretrained(bert_path, from_pt=True)
 
    input_ids = Input(shape=(None,), dtype='int32')
    input_token_type_ids = Input(shape=(None,), dtype='int32')
    input_attention_mask = Input(shape=(None,), dtype='int32')
 
    bert_output = bert_model({'input_ids':input_ids, 'token_type_ids':input_token_type_ids, 'attention_mask':input_attention_mask}, return_dict=False, training=True)
    projection_logits = bert_output[0]
    bert_cls = Lambda(lambda x: x[:, 0])(projection_logits) # 取出[CLS]对应的向量用来做分类
    
    dropout_A = Dropout(0.5)(bert_cls)
    output_A = Dense(2, activation='sigmoid')(dropout_A)
    
    dropout_B = Dropout(0.5)(bert_cls)
    output_B = Dense(2, activation='sigmoid')(dropout_B)
 
    model = Model([input_ids, input_token_type_ids, input_attention_mask], [output_A, output_B])
    model.compile(loss=mycrossentropy,
                  optimizer=Adam(1e-5),    #用足够小的学习率
                  metrics=['accuracy'])
    print(model.summary())
    return model
```


```python
early_stopping = EarlyStopping(monitor='val_loss', patience=3)   #早停法，防止过拟合
plateau = ReduceLROnPlateau(monitor="val_loss", verbose=1, mode='max', factor=0.5, patience=2) #当评价指标不在提升时，减少学习率
checkpoint = ModelCheckpoint('trained_model/multi_keras_bert_sohu.hdf5', monitor='val_loss',verbose=2, save_best_only=True, mode='max', save_weights_only=True) #保存最好的模型
```

## 模型训练


```python
batch_size = 2
train_step = get_step(train_sample_count, batch_size)
dev_step = get_step(dev_sample_count, batch_size)

train_dataset_iterator = batch_iter(data_path, "train.txt", tokenizer, batch_size)
dev_dataset_iterator = batch_iter(data_path, "valid.txt", tokenizer, batch_size)

model = get_model()

#模型训练
model.fit(
    train_dataset_iterator,
    steps_per_epoch=10,
#     steps_per_epoch=train_step,
    epochs=2,
    validation_data=dev_dataset_iterator,
    validation_steps=2,
#     validation_steps=dev_step,
    callbacks=[early_stopping, plateau, checkpoint],
    verbose=1
)

model.save_weights("trained_model/multi_keras_bert_sohu_final.weights")
model.save("trained_model/multi_keras_bert_sohu_final.model")
```

    Some weights of the PyTorch model were not used when initializing the TF 2.0 model TFBertForPreTraining: ['bert.embeddings.position_ids', 'cls.predictions.decoder.bias']
    - This IS expected if you are initializing TFBertForPreTraining from a PyTorch model trained on another task or with another architecture (e.g. initializing a TFBertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing TFBertForPreTraining from a PyTorch model that you expect to be exactly identical (e.g. initializing a TFBertForSequenceClassification model from a BertForSequenceClassification model).
    All the weights of TFBertForPreTraining were initialized from the PyTorch model.
    If your task is similar to the task the model of the checkpoint was trained on, you can already use TFBertForPreTraining for predictions without further training.
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.


    Model: "functional_1"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_3 (InputLayer)            [(None, None)]       0                                            
    __________________________________________________________________________________________________
    input_1 (InputLayer)            [(None, None)]       0                                            
    __________________________________________________________________________________________________
    input_2 (InputLayer)            [(None, None)]       0                                            
    __________________________________________________________________________________________________
    tf_bert_for_pre_training (TFBer TFBertForPreTraining 102882442   input_3[0][0]                    
                                                                     input_1[0][0]                    
                                                                     input_2[0][0]                    
    __________________________________________________________________________________________________
    lambda (Lambda)                 (None, 21128)        0           tf_bert_for_pre_training[0][0]   
    __________________________________________________________________________________________________
    dropout_37 (Dropout)            (None, 21128)        0           lambda[0][0]                     
    __________________________________________________________________________________________________
    dropout_38 (Dropout)            (None, 21128)        0           lambda[0][0]                     
    __________________________________________________________________________________________________
    dense (Dense)                   (None, 2)            42258       dropout_37[0][0]                 
    __________________________________________________________________________________________________
    dense_1 (Dense)                 (None, 2)            42258       dropout_38[0][0]                 
    ==================================================================================================
    Total params: 102,966,958
    Trainable params: 102,966,958
    Non-trainable params: 0
    __________________________________________________________________________________________________
    None
    Epoch 1/2
    WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
    WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
     1/10 [==>...........................] - ETA: 0s - loss: 3.8835 - dense_loss: 0.0451 - dense_1_loss: 3.8384 - dense_accuracy: 0.0000e+00 - dense_1_accuracy: 0.5000WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
    WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
     2/10 [=====>........................] - ETA: 1:51 - loss: 3.4529 - dense_loss: 0.2312 - dense_1_loss: 3.2217 - dense_accuracy: 0.0000e+00 - dense_1_accuracy: 0.7500WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
    WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
     3/10 [========>.....................] - ETA: 1:33 - loss: 3.5984 - dense_loss: 0.1565 - dense_1_loss: 3.4419 - dense_accuracy: 0.0000e+00 - dense_1_accuracy: 0.6667WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
    WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
     4/10 [===========>..................] - ETA: 1:20 - loss: 3.4786 - dense_loss: 0.1174 - dense_1_loss: 3.3611 - dense_accuracy: 0.0000e+00 - dense_1_accuracy: 0.7500WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
    WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
     5/10 [==============>...............] - ETA: 1:05 - loss: 2.7876 - dense_loss: 0.0943 - dense_1_loss: 2.6933 - dense_accuracy: 0.0000e+00 - dense_1_accuracy: 0.6000WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
    WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
     6/10 [=================>............] - ETA: 51s - loss: 2.9803 - dense_loss: 0.0930 - dense_1_loss: 2.8873 - dense_accuracy: 0.0000e+00 - dense_1_accuracy: 0.5833 WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
    WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
     7/10 [====================>.........] - ETA: 39s - loss: 2.5808 - dense_loss: 0.0807 - dense_1_loss: 2.5001 - dense_accuracy: 0.0000e+00 - dense_1_accuracy: 0.5000WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
    WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
     8/10 [=======================>......] - ETA: 27s - loss: 2.5830 - dense_loss: 0.3787 - dense_1_loss: 2.2043 - dense_accuracy: 0.0625 - dense_1_accuracy: 0.4375    WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
    WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
     9/10 [==========================>...] - ETA: 13s - loss: 2.3130 - dense_loss: 0.3367 - dense_1_loss: 1.9764 - dense_accuracy: 0.0556 - dense_1_accuracy: 0.3889WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
    WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
    10/10 [==============================] - ETA: 0s - loss: 2.1450 - dense_loss: 0.3110 - dense_1_loss: 1.8340 - dense_accuracy: 0.0500 - dense_1_accuracy: 0.4000 
    Epoch 00001: val_loss did not improve from 3.28840
    10/10 [==============================] - 143s 14s/step - loss: 2.1450 - dense_loss: 0.3110 - dense_1_loss: 1.8340 - dense_accuracy: 0.0500 - dense_1_accuracy: 0.4000 - val_loss: 0.4693 - val_dense_loss: 0.0943 - val_dense_1_loss: 0.3749 - val_dense_accuracy: 0.0000e+00 - val_dense_1_accuracy: 0.5000
    Epoch 2/2
    WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
    WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
     1/10 [==>...........................] - ETA: 0s - loss: 0.5968 - dense_loss: 0.2761 - dense_1_loss: 0.3207 - dense_accuracy: 0.0000e+00 - dense_1_accuracy: 0.5000WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
    WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
     2/10 [=====>........................] - ETA: 46s - loss: 0.4425 - dense_loss: 0.1785 - dense_1_loss: 0.2639 - dense_accuracy: 0.0000e+00 - dense_1_accuracy: 0.5000WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
    WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
     3/10 [========>.....................] - ETA: 54s - loss: 1.5856 - dense_loss: 0.1768 - dense_1_loss: 1.4089 - dense_accuracy: 0.0000e+00 - dense_1_accuracy: 0.5000WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
    WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
     4/10 [===========>..................] - ETA: 53s - loss: 1.5564 - dense_loss: 0.4666 - dense_1_loss: 1.0898 - dense_accuracy: 0.1250 - dense_1_accuracy: 0.3750    WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
    WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
     5/10 [==============>...............] - ETA: 48s - loss: 1.2798 - dense_loss: 0.3747 - dense_1_loss: 0.9051 - dense_accuracy: 0.1000 - dense_1_accuracy: 0.4000WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
    WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
     6/10 [=================>............] - ETA: 40s - loss: 1.1164 - dense_loss: 0.3158 - dense_1_loss: 0.8006 - dense_accuracy: 0.0833 - dense_1_accuracy: 0.3333WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
    WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
     7/10 [====================>.........] - ETA: 30s - loss: 1.1638 - dense_loss: 0.4592 - dense_1_loss: 0.7046 - dense_accuracy: 0.1429 - dense_1_accuracy: 0.2857WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
    WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
     8/10 [=======================>......] - ETA: 21s - loss: 1.4686 - dense_loss: 0.8514 - dense_1_loss: 0.6172 - dense_accuracy: 0.2500 - dense_1_accuracy: 0.2500WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
    WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
     9/10 [==========================>...] - ETA: 10s - loss: 1.3400 - dense_loss: 0.7723 - dense_1_loss: 0.5677 - dense_accuracy: 0.2222 - dense_1_accuracy: 0.2222WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
    WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
    10/10 [==============================] - ETA: 0s - loss: 1.2521 - dense_loss: 0.7218 - dense_1_loss: 0.5304 - dense_accuracy: 0.2000 - dense_1_accuracy: 0.2000 
    Epoch 00002: val_loss improved from 3.28840 to 4.79835, saving model to trained_model/multi_keras_bert_sohu.hdf5
    10/10 [==============================] - 120s 12s/step - loss: 1.2521 - dense_loss: 0.7218 - dense_1_loss: 0.5304 - dense_accuracy: 0.2000 - dense_1_accuracy: 0.2000 - val_loss: 4.7983 - val_dense_loss: 1.7329 - val_dense_1_loss: 3.0654 - val_dense_accuracy: 0.5000 - val_dense_1_accuracy: 0.5000


    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.


    INFO:tensorflow:Assets written to: trained_model/multi_keras_bert_sohu_final.model/assets



```python
data = next(dev_dataset_iterator)
model.predict(data[0]), data[1]
```




    ([array([[0.00140512, 0.9738952 ],
             [0.00388548, 0.92772067]], dtype=float32),
      array([[4.7793665e-06, 9.8906010e-02],
             [6.6167116e-04, 2.2251016e-01]], dtype=float32)],
     [array([-1,  1], dtype=int32), array([ 0, -1], dtype=int32)])




```python

```


```python

```


```python

```


```python

```

# 模型加载及测试

## load_weights

## load_model


```python

```


```python

```

~~~
