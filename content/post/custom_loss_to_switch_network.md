---
title: "用自定义损失函数实现选择启用不同子网络"
date: 2021-03-31T01:14:58+08:00
lastmod: 2021-03-31T01:14:58+08:00
draft: false
keywords: []
description: ""
tags: [深度学习, 自定义, 损失函数, 子网络, keras]
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

由于标签值是0、1，所以将 -1 设置为mask值。若y_true中为 -1 ，则将对应位置的y_true和y_pred替换为0，以此使其对应样本的loss为0。

最后，由于标签是0、1，所以采用 binary_crossentropy 作为损失函数：

```python
def transform_y(y_true, y_pred):
    mask_value = tf.constant(-1)
    mask_y_true = tf.not_equal(tf.cast(y_true, dtype=tf.int32), tf.cast(mask_value, dtype=tf.int32))
#     print(f"mask_y_true:{mask_y_true}")
#     y_true_ = tf.cond(tf.equal(y_true, mask_value), lambda: 0, lambda: y_true)
    y_true_ = tf.cast(y_true, dtype=tf.int32) * tf.cast(mask_y_true, dtype=tf.int32)
    y_pred_ = tf.cast(y_pred, dtype=tf.float32) * tf.cast(mask_y_true, dtype=tf.float32)
#     print(f"y_true_:{y_true_}, y_pred_:{y_pred_}")
    
    return y_true_, y_pred_


def my_binary_crossentropy(y_true, y_pred):
#     print(f"y_true:{y_true}, y_pred:{y_pred}")
    
    y_true, y_pred = transform_y(y_true, y_pred)
#     print(f"y_true_:{y_true}, y_pred_:{y_pred}")

    loss = binary_crossentropy(y_true, y_pred)
#     print(f"loss:{loss}")
    return loss


def tarnsform_metrics(y_true, y_pred):
    y_true_, y_pred_ = y_true.numpy(), y_pred.numpy()
    for i in range(y_true_.shape[0]):
        for j in range(y_true_.shape[1]):
            if y_true_[i][j] == -1:
                y_true_[i][j] = 0
                y_pred_[i][j] = 0
            if y_pred_[i][j] > 0.5:
                y_pred_[i][j] = 1
            else:
                y_pred_[i][j] = 0
    return y_true_, y_pred_


def my_binary_accuracy(y_true, y_pred):
#     print("my_binary_accuracy")
#     print(f"y_true:{y_true}, y_pred:{y_pred}")
    
    y_true_, y_pred_ = tarnsform_metrics(y_true, y_pred)
#     print(f"y_true_:{y_true_}, y_pred_:{y_pred_}")

    accuracy = binary_accuracy(y_true_, y_pred_)
    return accuracy


def my_f1_score(y_true, y_pred):
#     print("my_f1_score")
#     print(f"y_true:{y_true}, y_pred:{y_pred}")
    
    y_true_, y_pred_ = tarnsform_metrics(y_true, y_pred)
#     print(f"y_true_:{y_true_}, y_pred_:{y_pred_}")

    return f1_score(y_true_, y_pred_, average='macro')
```



## 模型定义

在编译模型时指定自定义loss函数，keras 会自动对两个输出目标分别使用该自定义loss函数，最后模型算的是这两个loss之和：

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
    output_A = Dense(1, activation='sigmoid')(dropout_A)
    
    dropout_B = Dropout(0.5)(bert_cls)
    output_B = Dense(1, activation='sigmoid')(dropout_B)
 
    model = Model([input_ids, input_token_type_ids, input_attention_mask], [output_A, output_B])
    model.compile(
                  loss=my_binary_crossentropy,
#                   loss='binary_crossentropy',
#                   loss=binary_crossentropy,
                  optimizer=Adam(1e-5),    #用足够小的学习率
                  metrics=[my_binary_accuracy, my_f1_score]
#                   metrics='accuracy'
                 )
    print(model.summary())
    return model
```



## TIP：

1. 在处理数据时，若labelB\=\=1，则labelA=1；若labelA\=\=0，则labelB=0 。





-----------------------

## 附录

### 全部源码

~~~markdown
# 导包


```python
import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
import re
from collections import Counter
import random
import json

from tqdm import tqdm
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
tf.config.run_functions_eagerly(True)
tf.get_logger().setLevel(tf.compat.v1.logging.ERROR)
from keras.metrics import top_k_categorical_accuracy, binary_accuracy
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
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
```


```python
tf.__version__
```




    '2.3.0'




```python

```


```python
data_path = "sohu2021_open_data_clean/"
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

    59638it [00:04, 13354.07it/s]



```python
dev_sample_count = get_sample_num(data_path, "valid.txt")
```

    9940it [00:00, 13041.43it/s]



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
        if label_dict['labelA'] == 0:
            label_dict['labelB'] = 0
        if label_dict['labelB'] == 1:
            label_dict['labelA'] = 1

        yield data['source'], data['target'], label_dict['labelA'], label_dict['labelB']
```


```python
it = get_data_iterator(data_path, "train.txt")
```


```python
next(it)
```




    ('谁能打破科比81分纪录？奥尼尔给出5个候选人，补充利拉德比尔！', 'NBA现役能入名人堂的球星很多，但是能被立铜像只有2人', 0, 0)




```python
get_sample_num(data_path, "train.txt")
```

    59638it [00:04, 11996.58it/s]





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




    ([array([[ 101, 5381, 5273, ...,    0,    0,    0],
             [ 101, 3297, 6818, ..., 8024, 4125,  102]]),
      array([[0, 0, 0, ..., 0, 0, 0],
             [0, 0, 0, ..., 1, 1, 1]]),
      array([[1, 1, 1, ..., 0, 0, 0],
             [1, 1, 1, ..., 1, 1, 1]])],
     [array([-1,  1], dtype=int32), array([ 0, -1], dtype=int32)])



## 定义模型


```python
def transform_y(y_true, y_pred):
    mask_value = tf.constant(-1)
    mask_y_true = tf.not_equal(tf.cast(y_true, dtype=tf.int32), tf.cast(mask_value, dtype=tf.int32))
#     print(f"mask_y_true:{mask_y_true}")
#     y_true_ = tf.cond(tf.equal(y_true, mask_value), lambda: 0, lambda: y_true)
    y_true_ = tf.cast(y_true, dtype=tf.int32) * tf.cast(mask_y_true, dtype=tf.int32)
    y_pred_ = tf.cast(y_pred, dtype=tf.float32) * tf.cast(mask_y_true, dtype=tf.float32)
#     print(f"y_true_:{y_true_}, y_pred_:{y_pred_}")
    
    return y_true_, y_pred_
```


```python
def my_binary_crossentropy(y_true, y_pred):
#     print(f"y_true:{y_true}, y_pred:{y_pred}")
    
    y_true, y_pred = transform_y(y_true, y_pred)
#     print(f"y_true_:{y_true}, y_pred_:{y_pred}")

    loss = binary_crossentropy(y_true, y_pred)
#     print(f"loss:{loss}")
    return loss
```


```python
def tarnsform_metrics(y_true, y_pred):
    y_true_, y_pred_ = y_true.numpy(), y_pred.numpy()
    for i in range(y_true_.shape[0]):
        for j in range(y_true_.shape[1]):
            if y_true_[i][j] == -1:
                y_true_[i][j] = 0
                y_pred_[i][j] = 0
            if y_pred_[i][j] > 0.5:
                y_pred_[i][j] = 1
            else:
                y_pred_[i][j] = 0
    return y_true_, y_pred_
```


```python
def my_binary_accuracy(y_true, y_pred):
#     print("my_binary_accuracy")
#     print(f"y_true:{y_true}, y_pred:{y_pred}")
    
    y_true_, y_pred_ = tarnsform_metrics(y_true, y_pred)
#     print(f"y_true_:{y_true_}, y_pred_:{y_pred_}")

    accuracy = binary_accuracy(y_true_, y_pred_)
    return accuracy
```


```python
def my_f1_score(y_true, y_pred):
#     print("my_f1_score")
#     print(f"y_true:{y_true}, y_pred:{y_pred}")
    
    y_true_, y_pred_ = tarnsform_metrics(y_true, y_pred)
#     print(f"y_true_:{y_true_}, y_pred_:{y_pred_}")

    return f1_score(y_true_, y_pred_, average='macro')
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
    output_A = Dense(1, activation='sigmoid')(dropout_A)
    
    dropout_B = Dropout(0.5)(bert_cls)
    output_B = Dense(1, activation='sigmoid')(dropout_B)
 
    model = Model([input_ids, input_token_type_ids, input_attention_mask], [output_A, output_B])
    model.compile(
                  loss=my_binary_crossentropy,
#                   loss='binary_crossentropy',
#                   loss=binary_crossentropy,
                  optimizer=Adam(1e-5),    #用足够小的学习率
                  metrics=[my_binary_accuracy, my_f1_score]
#                   metrics='accuracy'
                 )
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


    WARNING: AutoGraph could not transform <bound method Socket.send of <zmq.sugar.socket.Socket object at 0x7f74107b3460>> and will run it as-is.
    Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
    Cause: module, class, method, function, traceback, frame, or code object was expected, got cython_function_or_method
    To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert


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
    dense (Dense)                   (None, 1)            21129       dropout_37[0][0]                 
    __________________________________________________________________________________________________
    dense_1 (Dense)                 (None, 1)            21129       dropout_38[0][0]                 
    ==================================================================================================
    Total params: 102,924,700
    Trainable params: 102,924,700
    Non-trainable params: 0
    __________________________________________________________________________________________________
    None
    Epoch 1/2


    /home/zsd-server/miniconda3/envs/my/lib/python3.8/site-packages/transformers/tokenization_utils_base.py:2162: FutureWarning: The `truncation_strategy` argument is deprecated and will be removed in a future version, use `truncation=True` to truncate examples to a max length. You can give a specific length with `max_length` (e.g. `max_length=45`) or leave max_length to None to truncate to the maximal input size of the model (e.g. 512 for Bert).  If you have pairs of inputs, you can give a specific truncation strategy selected among `truncation='only_first'` (will only truncate the first sentence in the pairs) `truncation='only_second'` (will only truncate the second sentence in the pairs) or `truncation='longest_first'` (will iteratively remove tokens from the longest sentence in the pairs).
      warnings.warn(
    /home/zsd-server/miniconda3/envs/my/lib/python3.8/site-packages/tensorflow/python/data/ops/dataset_ops.py:3349: UserWarning: Even though the tf.config.experimental_run_functions_eagerly option is set, this option does not apply to tf.data functions. tf.data functions are still traced and executed as graphs.
      warnings.warn(


    10/10 [==============================] - ETA: 0s - loss: 2.9102 - dense_loss: 0.7665 - dense_1_loss: 2.1437 - dense_my_binary_accuracy: 0.8500 - dense_my_f1_score: 0.8000 - dense_1_my_binary_accuracy: 0.6500 - dense_1_my_f1_score: 0.5667
    Epoch 00001: val_loss improved from -inf to 1.98809, saving model to trained_model/multi_keras_bert_sohu.hdf5
    10/10 [==============================] - 111s 11s/step - loss: 2.9102 - dense_loss: 0.7665 - dense_1_loss: 2.1437 - dense_my_binary_accuracy: 0.8500 - dense_my_f1_score: 0.8000 - dense_1_my_binary_accuracy: 0.6500 - dense_1_my_f1_score: 0.5667 - val_loss: 1.9881 - val_dense_loss: 1.9870 - val_dense_1_loss: 0.0011 - val_dense_my_binary_accuracy: 0.7500 - val_dense_my_f1_score: 0.6667 - val_dense_1_my_binary_accuracy: 1.0000 - val_dense_1_my_f1_score: 1.0000
    Epoch 2/2
    10/10 [==============================] - ETA: 0s - loss: 3.0176 - dense_loss: 2.8052 - dense_1_loss: 0.2125 - dense_my_binary_accuracy: 0.7500 - dense_my_f1_score: 0.7000 - dense_1_my_binary_accuracy: 0.9000 - dense_1_my_f1_score: 0.8667 
    Epoch 00002: val_loss improved from 1.98809 to 2.56778, saving model to trained_model/multi_keras_bert_sohu.hdf5
    10/10 [==============================] - 114s 11s/step - loss: 3.0176 - dense_loss: 2.8052 - dense_1_loss: 0.2125 - dense_my_binary_accuracy: 0.7500 - dense_my_f1_score: 0.7000 - dense_1_my_binary_accuracy: 0.9000 - dense_1_my_f1_score: 0.8667 - val_loss: 2.5678 - val_dense_loss: 2.5678 - val_dense_1_loss: 7.7785e-06 - val_dense_my_binary_accuracy: 0.5000 - val_dense_my_f1_score: 0.3333 - val_dense_1_my_binary_accuracy: 1.0000 - val_dense_1_my_f1_score: 1.0000


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



```python
dev_dataset_iterator = batch_iter(data_path, "valid.txt", tokenizer, batch_size=1)
data = next(dev_dataset_iterator)
model.predict(data[0]), data[1]
```




    ([array([[0.00284165]], dtype=float32),
      array([[3.6964306e-05]], dtype=float32)],
     [array([1], dtype=int32), array([-1], dtype=int32)])




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
