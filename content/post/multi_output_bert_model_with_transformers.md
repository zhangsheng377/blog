---
title: "用transformers实现多输出、参数共享的bert模型"
date: 2021-03-27T00:57:58+08:00
lastmod: 2021-03-27T00:57:58+08:00
draft: false
keywords: []
description: ""
tags: [深度学习, 多输出, 参数共享, 模型, keras, bert]
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

在nlp领域，预训练模型bert可谓是红得发紫。

但现在能搜到的大多数都是pytorch写的框架，而且大多都是单输出模型。

所以，本文以 有相互关系的多层标签分类 为背景，用keras设计了多输出、参数共享的模型。



以上，是之前一篇文章的开头。那篇文章读取bert使用的是keras_bert第三方库，后来发现，兼容性等不是太好。

所以这篇准备采用transformers实现相同功能。（当然，兼容性要比keras_bert第三方库好一点，但限制是必须要tensorflow2）

## transformers基础应用

首先，可以在命令行里，使用如下命令，将tensorflow的检查点文件转成pytorch模型文件：

```shell
transformers-cli convert --model_type bert \
  --tf_checkpoint chinese_L-12_H-768_A-12/bert_model.ckpt \
  --config chinese_L-12_H-768_A-12/bert_config.json \
  --pytorch_dump_output chinese_L-12_H-768_A-12/pytorch_model.bin
```

随后，可以使用如下代码，读取bert模型：

```python
bert_model = TFBertForPreTraining.from_pretrained("./chinese_L-12_H-768_A-12/", from_pt=True)
```

最后，可以使用如下方式读取bert输出结果，并搭建模型：

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
from keras.metrics import top_k_categorical_accuracy
from keras.layers import *
from keras.callbacks import *
from keras.models import Model, load_model
import keras.backend as K
from keras.optimizers import Adam
from keras.utils import to_categorical
from transformers import (
    BertTokenizer,
    TFBertForPreTraining,
    TFBertModel,
)
```


```python

```


```python
data_path = "000_text_classifier_tensorflow_textcnn/THUCNews/"
text_max_length = 512
bert_path = r"chinese_L-12_H-768_A-12"
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
def _read_file(filename):
    """读取一个文件并转换为一行"""
    with open(filename, 'r', encoding='utf-8') as f:
        s = f.read().strip().replace('\n', '。').replace('\t', '').replace('\u3000', '')
        return re.sub(r'。+', '。', s)
```


```python
def get_data_iterator(data_path):
    for category in os.listdir(data_path):
        category_path = os.path.join(data_path, category)
        for file_name in os.listdir(category_path):
            yield _read_file(os.path.join(category_path, file_name)), category
```


```python
it = get_data_iterator(data_path)
```


```python
next(it)
```




    ('竞彩解析：日本美国争冠死磕 两巴相逢必有生死。周日受注赛事，女足世界杯决赛、美洲杯两场1/4决赛毫无疑问是全世界球迷和彩民关注的焦点。本届女足世界杯的最大黑马日本队能否一黑到底，创造亚洲奇迹？女子足坛霸主美国队能否再次“灭黑”成功，成就三冠伟业？巴西、巴拉圭冤家路窄，谁又能笑到最后？诸多谜底，在周一凌晨就会揭晓。日本美国争冠死磕。本届女足世界杯，是颠覆与反颠覆之争。夺冠大热门东道主德国队1/4决赛被日本队加时赛一球而“黑”，另一个夺冠大热门瑞典队则在半决赛被日本队3:1彻底打垮。而美国队则捍卫着女足豪强的尊严，在1/4决赛，她们与巴西女足苦战至点球大战，最终以5:3淘汰这支迅速崛起的黑马球队，而在半决赛，她们更是3:1大胜欧洲黑马法国队。美日两队此次世界杯进程惊人相似，小组赛前两轮全胜，最后一轮输球，1/4决赛同样与对手90分钟内战成平局，半决赛竟同样3:1大胜对手。此次决战，无论是日本还是美国队夺冠，均将创造女足世界杯新的历史。两巴相逢必有生死。本届美洲杯，让人大跌眼镜的事情太多。巴西、巴拉圭冤家路窄似乎更具传奇色彩。两队小组赛同分在B组，原本两个出线大热门，却双双在前两轮小组赛战平，两队直接交锋就是2:2平局，结果双双面临出局危险。最后一轮，巴西队在下半场终于发威，4:2大胜厄瓜多尔后来居上以小组第一出线，而巴拉圭最后一战还是3:3战平委内瑞拉获得小组第三，侥幸凭借净胜球优势挤掉A组第三名的哥斯达黎加，获得一个八强席位。在小组赛，巴西队是在最后时刻才逼平了巴拉圭，他们的好运气会在淘汰赛再显神威吗？巴拉圭此前3轮小组赛似乎都缺乏运气，此番又会否被幸运之神补偿一下呢？。另一场美洲杯1/4决赛，智利队在C组小组赛2胜1平以小组头名晋级八强；而委内瑞拉在B组是最不被看好的球队，但竟然在与巴西、巴拉圭同组的情况下，前两轮就奠定了小组出线权，他们小组3战1胜2平保持不败战绩，而入球数跟智利一样都是4球，只是失球数比智利多了1个。但既然他们面对强大的巴西都能保持球门不失，此番再创佳绩也不足为怪。',
     '彩票')




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
def read_category(data_path):
    """读取分类目录，固定"""
    categories = os.listdir(data_path)

    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id
```


```python
categories, cat_to_id = read_category(data_path)
```


```python
cat_to_id
```




    {'彩票': 0,
     '家居': 1,
     '游戏': 2,
     '股票': 3,
     '科技': 4,
     '社会': 5,
     '财经': 6,
     '时尚': 7,
     '星座': 8,
     '体育': 9,
     '房产': 10,
     '娱乐': 11,
     '时政': 12,
     '教育': 13}




```python
categories
```




    ['彩票',
     '家居',
     '游戏',
     '股票',
     '科技',
     '社会',
     '财经',
     '时尚',
     '星座',
     '体育',
     '房产',
     '娱乐',
     '时政',
     '教育']




```python

```


```python

```


```python

```


```python

```

# 构建训练、验证、测试集


```python
def build_dataset(data_path, train_path, dev_path, test_path):
    data_iter = get_data_iterator(data_path)
    with open(train_path, 'w', encoding='utf-8') as train_file, \
         open(dev_path, 'w', encoding='utf-8') as dev_file, \
         open(test_path, 'w', encoding='utf-8') as test_file:
        
        for text, label in tqdm(data_iter):
            radio = random.random()
            if radio < 0.8:
                train_file.write(text + "\t" + label + "\n")
            elif radio < 0.9:
                dev_file.write(text + "\t" + label + "\n")
            else:
                test_file.write(text + "\t" + label + "\n")
```


```python
# build_dataset(data_path, r"data/keras_bert_train.txt", r"data/keras_bert_dev.txt", r"data/keras_bert_test.txt")
```


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
def get_sample_num(data_path):
    count = 0
    with open(data_path, 'r', encoding='utf-8') as data_file:
        for line in tqdm(data_file):
            count += 1
    return count
```


```python
train_sample_count = get_sample_num(r"data/keras_bert_train.txt")
```

    668858it [00:09, 71362.95it/s]



```python
dev_sample_count = get_sample_num(r"data/keras_bert_dev.txt")
```

    83721it [00:01, 72723.97it/s]



```python
test_sample_count = get_sample_num(r"data/keras_bert_test.txt")
```

    83496it [00:01, 72445.72it/s]



```python
train_sample_count, dev_sample_count, test_sample_count
```




    (668858, 83721, 83496)




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
def get_text_iterator(data_path):
    with open(data_path, 'r', encoding='utf-8') as data_file:
        for line in data_file:
            data_split = line.strip().split('\t')
            if len(data_split) != 2:
                print(line)
                continue
            yield data_split[0], data_split[1]
```


```python
it = get_text_iterator(r"data/keras_bert_train.txt")
```


```python
next(it)
```




    ('竞彩解析：日本美国争冠死磕 两巴相逢必有生死。周日受注赛事，女足世界杯决赛、美洲杯两场1/4决赛毫无疑问是全世界球迷和彩民关注的焦点。本届女足世界杯的最大黑马日本队能否一黑到底，创造亚洲奇迹？女子足坛霸主美国队能否再次“灭黑”成功，成就三冠伟业？巴西、巴拉圭冤家路窄，谁又能笑到最后？诸多谜底，在周一凌晨就会揭晓。日本美国争冠死磕。本届女足世界杯，是颠覆与反颠覆之争。夺冠大热门东道主德国队1/4决赛被日本队加时赛一球而“黑”，另一个夺冠大热门瑞典队则在半决赛被日本队3:1彻底打垮。而美国队则捍卫着女足豪强的尊严，在1/4决赛，她们与巴西女足苦战至点球大战，最终以5:3淘汰这支迅速崛起的黑马球队，而在半决赛，她们更是3:1大胜欧洲黑马法国队。美日两队此次世界杯进程惊人相似，小组赛前两轮全胜，最后一轮输球，1/4决赛同样与对手90分钟内战成平局，半决赛竟同样3:1大胜对手。此次决战，无论是日本还是美国队夺冠，均将创造女足世界杯新的历史。两巴相逢必有生死。本届美洲杯，让人大跌眼镜的事情太多。巴西、巴拉圭冤家路窄似乎更具传奇色彩。两队小组赛同分在B组，原本两个出线大热门，却双双在前两轮小组赛战平，两队直接交锋就是2:2平局，结果双双面临出局危险。最后一轮，巴西队在下半场终于发威，4:2大胜厄瓜多尔后来居上以小组第一出线，而巴拉圭最后一战还是3:3战平委内瑞拉获得小组第三，侥幸凭借净胜球优势挤掉A组第三名的哥斯达黎加，获得一个八强席位。在小组赛，巴西队是在最后时刻才逼平了巴拉圭，他们的好运气会在淘汰赛再显神威吗？巴拉圭此前3轮小组赛似乎都缺乏运气，此番又会否被幸运之神补偿一下呢？。另一场美洲杯1/4决赛，智利队在C组小组赛2胜1平以小组头名晋级八强；而委内瑞拉在B组是最不被看好的球队，但竟然在与巴西、巴拉圭同组的情况下，前两轮就奠定了小组出线权，他们小组3战1胜2平保持不败战绩，而入球数跟智利一样都是4球，只是失球数比智利多了1个。但既然他们面对强大的巴西都能保持球门不失，此番再创佳绩也不足为怪。',
     '彩票')




```python
tokenizer = BertTokenizer.from_pretrained(bert_path)
```


```python
def get_keras_bert_iterator(data_path, cat_to_id, tokenizer):
    while True:
        data_iter = get_text_iterator(data_path)
        for text, category in data_iter:
            input_ids = tokenizer.encode(text, 
                                         max_length=text_max_length, 
                                         add_special_tokens=True, 
                                         padding='max_length', 
                                         truncation_strategy='only_first', 
#                                          return_tensors='tf'
                                        )
            yield input_ids, cat_to_id[category]
```


```python
it = get_keras_bert_iterator(r"data/keras_bert_train.txt", cat_to_id, tokenizer)
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
def batch_iter(data_path, cat_to_id, tokenizer, batch_size=64, shuffle=True):
    """生成批次数据"""
    keras_bert_iter = get_keras_bert_iterator(data_path, cat_to_id, tokenizer)
    while True:
        data_list = []
        for _ in range(batch_size):
            data = next(keras_bert_iter)
            data_list.append(data)
        if shuffle:
            random.shuffle(data_list)
        
        indices_list = []
        label_index_list = []
        for data in data_list:
            indices, label_index = data
            indices_list.append(indices)
            label_index_list.append(label_index)

        yield np.array(indices_list), np.array(label_index_list)
```


```python
it = batch_iter(r"data/keras_bert_train.txt", cat_to_id, tokenizer, batch_size=1)
```


```python
# next(it)
```


```python
it = batch_iter(r"data/keras_bert_train.txt", cat_to_id, tokenizer, batch_size=2)
```


```python
next(it)
```

    /home/zsd-server/miniconda3/envs/my/lib/python3.8/site-packages/transformers/tokenization_utils_base.py:2162: FutureWarning: The `truncation_strategy` argument is deprecated and will be removed in a future version, use `truncation=True` to truncate examples to a max length. You can give a specific length with `max_length` (e.g. `max_length=45`) or leave max_length to None to truncate to the maximal input size of the model (e.g. 512 for Bert).  If you have pairs of inputs, you can give a specific truncation strategy selected among `truncation='only_first'` (will only truncate the first sentence in the pairs) `truncation='only_second'` (will only truncate the second sentence in the pairs) or `truncation='longest_first'` (will iteratively remove tokens from the longest sentence in the pairs).
      warnings.warn(





    (array([[ 101, 4993, 2506, ...,  131,  123,  102],
            [ 101, 2506, 3696, ..., 1139,  125,  102]]),
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
checkpoint = ModelCheckpoint('trained_model/keras_bert_THUCNews.hdf5', monitor='val_loss',verbose=2, save_best_only=True, mode='max', save_weights_only=True) #保存最好的模型
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
batch_size = 2
train_step = get_step(train_sample_count, batch_size)
dev_step = get_step(dev_sample_count, batch_size)

train_dataset_iterator = batch_iter(r"data/keras_bert_train.txt", cat_to_id, tokenizer, batch_size)
dev_dataset_iterator = batch_iter(r"data/keras_bert_dev.txt", cat_to_id, tokenizer, batch_size)

model = get_model(categories)

#模型训练
model.fit(
    train_dataset_iterator,
    steps_per_epoch=10,
    epochs=5,
    validation_data=dev_dataset_iterator,
    validation_steps=2,
    callbacks=[early_stopping, plateau, checkpoint],
    verbose=1
)
```

    Some weights of the PyTorch model were not used when initializing the TF 2.0 model TFBertForPreTraining: ['bert.embeddings.position_ids', 'cls.predictions.decoder.bias']
    - This IS expected if you are initializing TFBertForPreTraining from a PyTorch model trained on another task or with another architecture (e.g. initializing a TFBertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing TFBertForPreTraining from a PyTorch model that you expect to be exactly identical (e.g. initializing a TFBertForSequenceClassification model from a BertForSequenceClassification model).
    All the weights of TFBertForPreTraining were initialized from the PyTorch model.
    If your task is similar to the task the model of the checkpoint was trained on, you can already use TFBertForPreTraining for predictions without further training.


    WARNING:tensorflow:AutoGraph could not transform <bound method Socket.send of <zmq.sugar.socket.Socket object at 0x7f3e4d7e2340>> and will run it as-is.
    Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
    Cause: module, class, method, function, traceback, frame, or code object was expected, got cython_function_or_method
    To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
    WARNING: AutoGraph could not transform <bound method Socket.send of <zmq.sugar.socket.Socket object at 0x7f3e4d7e2340>> and will run it as-is.
    Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
    Cause: module, class, method, function, traceback, frame, or code object was expected, got cython_function_or_method
    To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert


    The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    /home/zsd-server/miniconda3/envs/my/lib/python3.8/site-packages/transformers/tokenization_utils_base.py:2162: FutureWarning: The `truncation_strategy` argument is deprecated and will be removed in a future version, use `truncation=True` to truncate examples to a max length. You can give a specific length with `max_length` (e.g. `max_length=45`) or leave max_length to None to truncate to the maximal input size of the model (e.g. 512 for Bert).  If you have pairs of inputs, you can give a specific truncation strategy selected among `truncation='only_first'` (will only truncate the first sentence in the pairs) `truncation='only_second'` (will only truncate the second sentence in the pairs) or `truncation='longest_first'` (will iteratively remove tokens from the longest sentence in the pairs).
      warnings.warn(
    The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.


    Model: "functional_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, None)]            0         
    _________________________________________________________________
    tf_bert_for_pre_training (TF TFBertForPreTrainingOutpu 102882442 
    _________________________________________________________________
    lambda (Lambda)              (None, 21128)             0         
    _________________________________________________________________
    dropout_37 (Dropout)         (None, 21128)             0         
    _________________________________________________________________
    dense (Dense)                (None, 14)                295806    
    =================================================================
    Total params: 103,178,248
    Trainable params: 103,178,248
    Non-trainable params: 0
    _________________________________________________________________
    None
    Epoch 1/5
    WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
    WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.


    /home/zsd-server/miniconda3/envs/my/lib/python3.8/site-packages/tensorflow/python/framework/indexed_slices.py:431: UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
      warnings.warn(
    The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.


    WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
    WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
    10/10 [==============================] - ETA: 0s - loss: 13.3638 - accuracy: 0.1000    

    The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.


    
    Epoch 00001: val_loss improved from -inf to 0.80593, saving model to trained_model/keras_bert_THUCNews.hdf5
    10/10 [==============================] - 89s 9s/step - loss: 13.3638 - accuracy: 0.1000 - val_loss: 0.8059 - val_accuracy: 1.0000
    Epoch 2/5
    10/10 [==============================] - ETA: 0s - loss: 2.2765 - accuracy: 0.3500
    Epoch 00002: val_loss did not improve from 0.80593
    10/10 [==============================] - 87s 9s/step - loss: 2.2765 - accuracy: 0.3500 - val_loss: 0.6079 - val_accuracy: 0.7500
    Epoch 3/5
    10/10 [==============================] - ETA: 0s - loss: 0.3474 - accuracy: 0.8500
    Epoch 00003: ReduceLROnPlateau reducing learning rate to 4.999999873689376e-06.
    
    Epoch 00003: val_loss did not improve from 0.80593
    10/10 [==============================] - 88s 9s/step - loss: 0.3474 - accuracy: 0.8500 - val_loss: 2.5302e-05 - val_accuracy: 1.0000
    Epoch 4/5
    10/10 [==============================] - ETA: 0s - loss: 0.0018 - accuracy: 1.0000
    Epoch 00004: val_loss did not improve from 0.80593
    10/10 [==============================] - 86s 9s/step - loss: 0.0018 - accuracy: 1.0000 - val_loss: 6.5565e-07 - val_accuracy: 1.0000
    Epoch 5/5
    10/10 [==============================] - ETA: 0s - loss: 0.0372 - accuracy: 1.0000
    Epoch 00005: ReduceLROnPlateau reducing learning rate to 2.499999936844688e-06.
    
    Epoch 00005: val_loss did not improve from 0.80593
    10/10 [==============================] - 86s 9s/step - loss: 0.0372 - accuracy: 1.0000 - val_loss: 5.9605e-07 - val_accuracy: 1.0000





    <tensorflow.python.keras.callbacks.History at 0x7f3dc062d820>




```python

```


```python

```


```python

```


```python

```

# 多输出模型

## 构建数据迭代器


```python
second_label_list = [0, 1, 2]
```


```python
def get_keras_bert_iterator(data_path, cat_to_id, tokenizer, second_label_list):
    while True:
        data_iter = get_text_iterator(data_path)
        for text, category in data_iter:
            input_ids = tokenizer.encode(text, 
                                         max_length=text_max_length, 
                                         add_special_tokens=True, 
                                         padding='max_length', 
                                         truncation_strategy='only_first', 
#                                          return_tensors='tf'
                                        )
            yield input_ids, cat_to_id[category], random.choice(second_label_list)
```


```python
it = get_keras_bert_iterator(r"data/keras_bert_train.txt", cat_to_id, tokenizer, second_label_list)
```


```python
# next(it)
```


```python
def batch_iter(data_path, cat_to_id, tokenizer, second_label_list, batch_size=64, shuffle=True):
    """生成批次数据"""
    keras_bert_iter = get_keras_bert_iterator(data_path, cat_to_id, tokenizer, second_label_list)
    while True:
        data_list = []
        for _ in range(batch_size):
            data = next(keras_bert_iter)
            data_list.append(data)
        if shuffle:
            random.shuffle(data_list)
        
        indices_list = []
        label_index_list = []
        second_label_list = []
        for data in data_list:
            indices, label_index, second_label = data
            indices_list.append(indices)
            label_index_list.append(label_index)
            second_label_list.append(second_label)

        yield np.array(indices_list), [np.array(label_index_list), np.array(second_label_list)]
```


```python
it = batch_iter(r"data/keras_bert_train.txt", cat_to_id, tokenizer, second_label_list, batch_size=2)
```


```python
next(it)
```




    (array([[ 101, 2506, 3696, ..., 1139,  125,  102],
            [ 101, 4993, 2506, ...,  131,  123,  102]]),
     [array([0, 0]), array([1, 2])])



## 定义模型


```python
def get_model(label_list, second_label_list):
    K.clear_session()
    
    bert_model = TFBertForPreTraining.from_pretrained(bert_path, from_pt=True)
 
    input_indices = Input(shape=(None,), dtype='int32')
 
    bert_output = bert_model(input_indices)
    projection_logits = bert_output[0]
    bert_cls = Lambda(lambda x: x[:, 0])(projection_logits) # 取出[CLS]对应的向量用来做分类
    
    dropout = Dropout(0.5)(bert_cls)
    output = Dense(len(label_list), activation='softmax')(dropout)

    dropout_second = Dropout(0.5)(bert_cls)
    output_second = Dense(len(second_label_list), activation='softmax')(dropout_second)
 
    model = Model(input_indices, [output, output_second])
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(1e-5),    #用足够小的学习率
                  metrics=['accuracy'])
    print(model.summary())
    return model
```


```python
early_stopping = EarlyStopping(monitor='val_loss', patience=3)   #早停法，防止过拟合
plateau = ReduceLROnPlateau(monitor="val_loss", verbose=1, mode='max', factor=0.5, patience=2) #当评价指标不在提升时，减少学习率
checkpoint = ModelCheckpoint('trained_model/muilt_keras_bert_THUCNews.hdf5', monitor='val_loss',verbose=2, save_best_only=True, mode='max', save_weights_only=True) #保存最好的模型
```

## 模型训练


```python
batch_size = 2
train_step = get_step(train_sample_count, batch_size)
dev_step = get_step(dev_sample_count, batch_size)

train_dataset_iterator = batch_iter(r"data/keras_bert_train.txt", cat_to_id, tokenizer, second_label_list, batch_size)
dev_dataset_iterator = batch_iter(r"data/keras_bert_dev.txt", cat_to_id, tokenizer, second_label_list, batch_size)

model = get_model(categories, second_label_list)

#模型训练
model.fit(
    train_dataset_iterator,
    steps_per_epoch=10,
    epochs=5,
    validation_data=dev_dataset_iterator,
    validation_steps=2,
    callbacks=[early_stopping, plateau, checkpoint],
    verbose=1
)
```

    Some weights of the PyTorch model were not used when initializing the TF 2.0 model TFBertForPreTraining: ['bert.embeddings.position_ids', 'cls.predictions.decoder.bias']
    - This IS expected if you are initializing TFBertForPreTraining from a PyTorch model trained on another task or with another architecture (e.g. initializing a TFBertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing TFBertForPreTraining from a PyTorch model that you expect to be exactly identical (e.g. initializing a TFBertForSequenceClassification model from a BertForSequenceClassification model).
    All the weights of TFBertForPreTraining were initialized from the PyTorch model.
    If your task is similar to the task the model of the checkpoint was trained on, you can already use TFBertForPreTraining for predictions without further training.
    The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.


    Model: "functional_1"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)            [(None, None)]       0                                            
    __________________________________________________________________________________________________
    tf_bert_for_pre_training (TFBer TFBertForPreTraining 102882442   input_1[0][0]                    
    __________________________________________________________________________________________________
    lambda (Lambda)                 (None, 21128)        0           tf_bert_for_pre_training[0][0]   
    __________________________________________________________________________________________________
    dropout_37 (Dropout)            (None, 21128)        0           lambda[0][0]                     
    __________________________________________________________________________________________________
    dropout_38 (Dropout)            (None, 21128)        0           lambda[0][0]                     
    __________________________________________________________________________________________________
    dense (Dense)                   (None, 14)           295806      dropout_37[0][0]                 
    __________________________________________________________________________________________________
    dense_1 (Dense)                 (None, 3)            63387       dropout_38[0][0]                 
    ==================================================================================================
    Total params: 103,241,635
    Trainable params: 103,241,635
    Non-trainable params: 0
    __________________________________________________________________________________________________
    None
    Epoch 1/5
    WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
    WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.


    The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.


    WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
    WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_pre_training/bert/pooler/dense/kernel:0', 'tf_bert_for_pre_training/bert/pooler/dense/bias:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/kernel:0', 'tf_bert_for_pre_training/nsp___cls/seq_relationship/bias:0'] when minimizing the loss.
    10/10 [==============================] - ETA: 0s - loss: 13.2166 - dense_loss: 7.6030 - dense_1_loss: 5.6137 - dense_accuracy: 0.3000 - dense_1_accuracy: 0.3000 

    The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.


    
    Epoch 00001: val_loss improved from -inf to 1.83895, saving model to trained_model/muilt_keras_bert_THUCNews.hdf5
    10/10 [==============================] - 92s 9s/step - loss: 13.2166 - dense_loss: 7.6030 - dense_1_loss: 5.6137 - dense_accuracy: 0.3000 - dense_1_accuracy: 0.3000 - val_loss: 1.8389 - val_dense_loss: 0.0022 - val_dense_1_loss: 1.8367 - val_dense_accuracy: 1.0000 - val_dense_1_accuracy: 0.5000
    Epoch 2/5
    10/10 [==============================] - ETA: 0s - loss: 3.4398 - dense_loss: 0.6309 - dense_1_loss: 2.8089 - dense_accuracy: 0.8000 - dense_1_accuracy: 0.3500
    Epoch 00002: val_loss improved from 1.83895 to 2.41962, saving model to trained_model/muilt_keras_bert_THUCNews.hdf5
    10/10 [==============================] - 84s 8s/step - loss: 3.4398 - dense_loss: 0.6309 - dense_1_loss: 2.8089 - dense_accuracy: 0.8000 - dense_1_accuracy: 0.3500 - val_loss: 2.4196 - val_dense_loss: 3.2663e-04 - val_dense_1_loss: 2.4193 - val_dense_accuracy: 1.0000 - val_dense_1_accuracy: 0.2500
    Epoch 3/5
    10/10 [==============================] - ETA: 0s - loss: 2.4220 - dense_loss: 0.0156 - dense_1_loss: 2.4063 - dense_accuracy: 1.0000 - dense_1_accuracy: 0.3000
    Epoch 00003: val_loss did not improve from 2.41962
    10/10 [==============================] - 86s 9s/step - loss: 2.4220 - dense_loss: 0.0156 - dense_1_loss: 2.4063 - dense_accuracy: 1.0000 - dense_1_accuracy: 0.3000 - val_loss: 1.3294 - val_dense_loss: 2.6792e-05 - val_dense_1_loss: 1.3294 - val_dense_accuracy: 1.0000 - val_dense_1_accuracy: 0.2500
    Epoch 4/5
    10/10 [==============================] - ETA: 0s - loss: 2.2683 - dense_loss: 0.0068 - dense_1_loss: 2.2615 - dense_accuracy: 1.0000 - dense_1_accuracy: 0.3500
    Epoch 00004: ReduceLROnPlateau reducing learning rate to 4.999999873689376e-06.
    
    Epoch 00004: val_loss did not improve from 2.41962
    10/10 [==============================] - 89s 9s/step - loss: 2.2683 - dense_loss: 0.0068 - dense_1_loss: 2.2615 - dense_accuracy: 1.0000 - dense_1_accuracy: 0.3500 - val_loss: 1.1061 - val_dense_loss: 0.0025 - val_dense_1_loss: 1.1036 - val_dense_accuracy: 1.0000 - val_dense_1_accuracy: 0.2500
    Epoch 5/5
    10/10 [==============================] - ETA: 0s - loss: 3.7829 - dense_loss: 0.6761 - dense_1_loss: 3.1068 - dense_accuracy: 0.9500 - dense_1_accuracy: 0.3000
    Epoch 00005: val_loss did not improve from 2.41962
    10/10 [==============================] - 89s 9s/step - loss: 3.7829 - dense_loss: 0.6761 - dense_1_loss: 3.1068 - dense_accuracy: 0.9500 - dense_1_accuracy: 0.3000 - val_loss: 1.3932 - val_dense_loss: 0.0022 - val_dense_1_loss: 1.3909 - val_dense_accuracy: 1.0000 - val_dense_1_accuracy: 0.0000e+00





    <tensorflow.python.keras.callbacks.History at 0x7f3d7ce29cd0>




```python
dev_dataset_iterator = batch_iter(r"data/keras_bert_dev.txt", cat_to_id, tokenizer, second_label_list, batch_size)
model.evaluate_generator(generator=dev_dataset_iterator, steps=2)
```

    WARNING:tensorflow:From <ipython-input-47-b8b146ad62ac>:2: Model.evaluate_generator (from tensorflow.python.keras.engine.training) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use Model.evaluate, which supports generators.





    [1.1637728214263916, 0.003070184262469411, 1.1607024669647217, 1.0, 0.0]




```python
model.save_weights("trained_model/muilt_keras_bert_THUCNews_final.weights")
```


```python
model.save("trained_model/muilt_keras_bert_THUCNews_final.model")
```

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


    WARNING:tensorflow:From /home/zsd-server/miniconda3/envs/my/lib/python3.8/site-packages/tensorflow/python/training/tracking/tracking.py:111: Model.state_updates (from tensorflow.python.keras.engine.training) is deprecated and will be removed in a future version.
    Instructions for updating:
    This property should not be used in TensorFlow 2.0, as updates are applied automatically.
    WARNING:tensorflow:From /home/zsd-server/miniconda3/envs/my/lib/python3.8/site-packages/tensorflow/python/training/tracking/tracking.py:111: Layer.updates (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
    Instructions for updating:
    This property should not be used in TensorFlow 2.0, as updates are applied automatically.


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


    INFO:tensorflow:Assets written to: trained_model/muilt_keras_bert_THUCNews_final.model/assets



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


```python
model_test = get_model(categories, second_label_list)
```

    Some weights of the PyTorch model were not used when initializing the TF 2.0 model TFBertForPreTraining: ['bert.embeddings.position_ids', 'cls.predictions.decoder.bias']
    - This IS expected if you are initializing TFBertForPreTraining from a PyTorch model trained on another task or with another architecture (e.g. initializing a TFBertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing TFBertForPreTraining from a PyTorch model that you expect to be exactly identical (e.g. initializing a TFBertForSequenceClassification model from a BertForSequenceClassification model).
    All the weights of TFBertForPreTraining were initialized from the PyTorch model.
    If your task is similar to the task the model of the checkpoint was trained on, you can already use TFBertForPreTraining for predictions without further training.
    The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.


    Model: "functional_1"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)            [(None, None)]       0                                            
    __________________________________________________________________________________________________
    tf_bert_for_pre_training (TFBer TFBertForPreTraining 102882442   input_1[0][0]                    
    __________________________________________________________________________________________________
    lambda (Lambda)                 (None, 21128)        0           tf_bert_for_pre_training[0][0]   
    __________________________________________________________________________________________________
    dropout_37 (Dropout)            (None, 21128)        0           lambda[0][0]                     
    __________________________________________________________________________________________________
    dropout_38 (Dropout)            (None, 21128)        0           lambda[0][0]                     
    __________________________________________________________________________________________________
    dense (Dense)                   (None, 14)           295806      dropout_37[0][0]                 
    __________________________________________________________________________________________________
    dense_1 (Dense)                 (None, 3)            63387       dropout_38[0][0]                 
    ==================================================================================================
    Total params: 103,241,635
    Trainable params: 103,241,635
    Non-trainable params: 0
    __________________________________________________________________________________________________
    None



```python
batch_size = 2
dev_dataset_iterator = batch_iter(r"data/keras_bert_dev.txt", cat_to_id, tokenizer, second_label_list, batch_size)
model_test.evaluate_generator(generator=dev_dataset_iterator, steps=2)
```

    The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.





    [31.814546585083008, 23.794151306152344, 8.02039623260498, 0.0, 0.25]




```python
model_test.load_weights("trained_model/muilt_keras_bert_THUCNews_final.weights")
```




    <tensorflow.python.training.tracking.util.CheckpointLoadStatus at 0x7f3d53816700>




```python
batch_size = 2
dev_dataset_iterator = batch_iter(r"data/keras_bert_dev.txt", cat_to_id, tokenizer, second_label_list, batch_size)
model_test.evaluate_generator(generator=dev_dataset_iterator, steps=2)
```




    [0.815148115158081, 0.003070184262469411, 0.8120779395103455, 1.0, 0.25]



## load_model


```python
model_test = load_model("trained_model/muilt_keras_bert_THUCNews_final.model")
```


```python

```


```python

```

~~~
