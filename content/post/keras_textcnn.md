---
title: "用keras实现textcnn"
date: 2021-03-08T00:56:58+08:00
lastmod: 2021-03-08T00:56:58+08:00
draft: false
keywords: []
description: ""
tags: [深度学习, textcnn, 模型, keras]
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

本文主要参考是的：

> https://blog.csdn.net/asialee_bird/article/details/88813385



## 基础版CNN

```python
def get_model():
    K.clear_session()
    
    model = Sequential()
    model.add(Embedding(len(vocab) + 1, 300, input_length=50)) #使用Embeeding层将每个词编码转换为词向量
    model.add(Conv1D(256, 5, padding='same'))
    model.add(MaxPooling1D(3, 3, padding='same'))
    model.add(Conv1D(128, 5, padding='same'))
    model.add(MaxPooling1D(3, 3, padding='same'))
    model.add(Conv1D(64, 3, padding='same'))
    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(BatchNormalization())  # (批)规范化层
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(3, activation='softmax'))
    
    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    print(model.summary())
    
    return model
```



## 简单版TextCNN

```python
def get_model():
    K.clear_session()
    
    main_input = Input(shape=(50,), dtype='float64')
    # 词嵌入（使用预训练的词向量）
    embedder = Embedding(len(vocab) + 1, 300, input_length=50, trainable=False)
    embed = embedder(main_input)
    # 词窗大小分别为3,4,5
    cnn1 = Conv1D(256, 3, padding='same', strides=1, activation='relu')(embed)
    cnn1 = MaxPooling1D(pool_size=48)(cnn1)
    cnn2 = Conv1D(256, 4, padding='same', strides=1, activation='relu')(embed)
    cnn2 = MaxPooling1D(pool_size=47)(cnn2)
    cnn3 = Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
    cnn3 = MaxPooling1D(pool_size=46)(cnn3)
    # 合并三个模型的输出向量
    cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
    flat = Flatten()(cnn)
    drop = Dropout(0.2)(flat)
    main_output = Dense(3, activation='softmax')(drop)
    model = Model(inputs=main_input, outputs=main_output)
    
    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    print(model.summary())
    
    return model
```

## 附录

### 全部源码

#### 导包

```python
import os
import random
from joblib import load, dump

from sklearn.model_selection import train_test_split
import pandas as pd
import jieba
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, Dense, Input, concatenate
from keras import backend as K
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
```



#### 构建文本迭代器

```python
def get_text_label_iterator(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_split = line.strip().split('\t')
            if len(line_split) != 2:
                print(line)
                continue
            yield line_split[0], line_split[1]

it = get_text_label_iterator(r"data/keras_bert_train.txt")
next(it)
```

  ('竞彩解析：日本美国争冠死磕 两巴相逢必有生死。周日受注赛事，女足世界杯决赛、美洲杯两场1/4决赛毫无疑问是全世界球迷和彩民关注的焦点。本届女足世界杯的最大黑马日本队能否一黑到底，创造亚洲奇迹？女子足坛霸主美国队能否再次“灭黑”成功，成就三冠伟业？巴西、巴拉圭冤家路窄，谁又能笑到最后？诸多谜底，在周一凌晨就会揭晓。日本美国争冠死磕。本届女足世界杯，是颠覆与反颠覆之争。夺冠大热门东道主德国队1/4决赛被日本队加时赛一球而“黑”，另一个夺冠大热门瑞典队则在半决赛被日本队3:1彻底打垮。而美国队则捍卫着女足豪强的尊严，在1/4决赛，她们与巴西女足苦战至点球大战，最终以5:3淘汰这支迅速崛起的黑马球队，而在半决赛，她们更是3:1大胜欧洲黑马法国队。美日两队此次世界杯进程惊人相似，小组赛前两轮全胜，最后一轮输球，1/4决赛同样与对手90分钟内战成平局，半决赛竟同样3:1大胜对手。此次决战，无论是日本还是美国队夺冠，均将创造女足世界杯新的历史。两巴相逢必有生死。本届美洲杯，让人大跌眼镜的事情太多。巴西、巴拉圭冤家路窄似乎更具传奇色彩。两队小组赛同分在B组，原本两个出线大热门，却双双在前两轮小组赛战平，两队直接交锋就是2:2平局，结果双双面临出局危险。最后一轮，巴西队在下半场终于发威，4:2大胜厄瓜多尔后来居上以小组第一出线，而巴拉圭最后一战还是3:3战平委内瑞拉获得小组第三，侥幸凭借净胜球优势挤掉A组第三名的哥斯达黎加，获得一个八强席位。在小组赛，巴西队是在最后时刻才逼平了巴拉圭，他们的好运气会在淘汰赛再显神威吗？巴拉圭此前3轮小组赛似乎都缺乏运气，此番又会否被幸运之神补偿一下呢？。另一场美洲杯1/4决赛，智利队在C组小组赛2胜1平以小组头名晋级八强；而委内瑞拉在B组是最不被看好的球队，但竟然在与巴西、巴拉圭同组的情况下，前两轮就奠定了小组出线权，他们小组3战1胜2平保持不败战绩，而入球数跟智利一样都是4球，只是失球数比智利多了1个。但既然他们面对强大的巴西都能保持球门不失，此番再创佳绩也不足为怪。',

   '彩票')



#### 获得词汇表vocab

```python
def get_segment_iterator(data_path):
    data_iter = get_text_label_iterator(data_path)
    for text, label in data_iter:
        yield list(jieba.cut(text)), label
        
it = get_segment_iterator(r"data/keras_bert_train.txt")
# next(it)

def get_only_segment_iterator(data_path):
    segment_iter = get_segment_iterator(data_path)
    for segment, label in tqdm(segment_iter):
        yield segment
# tokenizer=Tokenizer()  #创建一个Tokenizer对象
# # fit_on_texts函数可以将输入的文本中的每个词编号，编号是根据词频的，词频越大，编号越小
# tokenizer.fit_on_texts(get_only_segment_iterator(r"data/keras_bert_train.txt"))

# dump(tokenizer, r"data/keras_textcnn_tokenizer.bin")

tokenizer = load(r"data/keras_textcnn_tokenizer.bin")
vocab = tokenizer.word_index #得到每个词的编号
```



#### 获取样本个数

```python
def get_sample_count(data_path):
    data_iter = get_text_label_iterator(data_path)
    count = 0
    for text, label in tqdm(data_iter):
        count += 1
    return count

train_sample_count = get_sample_count(r"data/keras_bert_train.txt")
dev_sample_count = get_sample_count(r"data/keras_bert_dev.txt")
```



#### 构建标签表

```python
def read_category(data_path):
    """读取分类目录，固定"""
    categories = os.listdir(data_path)

    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id

categories, cat_to_id = read_category("000_text_classifier_tensorflow_textcnn/THUCNews")
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



#### 构建输入数据迭代器

```python
def get_data_iterator(data_path):
    while True:
        segment_iter = get_segment_iterator(data_path)
        for segment, label in segment_iter:
            word_ids = tokenizer.texts_to_sequences([segment])
            padded_seqs = pad_sequences(word_ids,maxlen=50)[0] #将超过固定值的部分截掉，不足的在最前面用0填充
            yield padded_seqs, cat_to_id[label]

it = get_data_iterator(r"data/keras_bert_train.txt")
next(it)
```

  Building prefix dict from the default dictionary ...

  Loading model from cache /tmp/jieba.cache

  Loading model cost 1.039 seconds.

  Prefix dict has been built succesfully.

  (array([  69,  2160,   57,  3010,   55,  828,   68,  1028,

​        456,  3712,  2130,   1,   36, 116604,  361,  7019,

​        377,   26,   8,   76,  539,   1,  346,  7323,

​       89885,  7019,   73,   7,   55,   84,   3,   33,

​       3199,   69,  579,  1366,   2,  1526,   26,   89,

​        456,  5741,  8256,   1,  6163,  7253, 10831,   14,

​       77404,   3], dtype=int32),

   0)

```python
def get_batch_data_iterator(data_path, batch_size=64, shuffle=True):
    data_iter = get_data_iterator(data_path)
    while True:
        data_list = []
        for _ in range(batch_size):
            data = next(data_iter)
            data_list.append(data)
        if shuffle:
            random.shuffle(data_list)
        
        pad_sequences_list = []
        label_index_list = []
        for data in data_list:
            pad_sequences, label_index = data
            pad_sequences_list.append(pad_sequences.tolist())
            label_index_list.append(label_index)

        yield np.array(pad_sequences_list), np.array(label_index_list)

it = get_batch_data_iterator(r"data/keras_bert_train.txt", batch_size=1)
next(it)
```

  (array([[  69,  2160,   57,  3010,   55,  828,   68,  1028,

​        456,  3712,  2130,   1,   36, 116604,  361,  7019,

​        377,   26,   8,   76,  539,   1,  346,  7323,

​       89885,  7019,   73,   7,   55,   84,   3,   33,

​        3199,   69,  579,  1366,   2,  1526,   26,   89,

​        456,  5741,  8256,   1,  6163,  7253, 10831,   14,

​       77404,   3]]),

   array([0]))

```python
it = get_batch_data_iterator(r"data/keras_bert_train.txt", batch_size=1)
next(it)
```

  (array([[   5,  5013, 14313,  601, 15377, 23499,   13,  493,

​        1541,  247,   5, 35557, 21529, 15377,   5,  1764,

​         11,  2774, 15377,   5,  279,  1764,  430,   5,

​        4742, 36921, 24090,  6387, 23499,   13,  5013,  8319,

​        6387,   5,  2370,  1764,  6387,   5, 16122,  1764,

​        6387,   5, 14313,  3707,  6387,   5,   11,  2774,

​        247,  6387],

​      [  69,  2160,   57,  3010,   55,  828,   68,  1028,

​        456,  3712,  2130,   1,   36, 116604,  361,  7019,

​        377,   26,   8,   76,  539,   1,  346,  7323,

​       89885,  7019,   73,   7,   55,   84,   3,   33,

​        3199,   69,  579,  1366,   2,  1526,   26,   89,

​        456,  5741,  8256,   1,  6163,  7253, 10831,   14,

​       77404,   3]]),

   array([0, 0]))



#### 定义 基础版CNN

```python
def get_model():
    K.clear_session()
    
    model = Sequential()
    model.add(Embedding(len(vocab) + 1, 300, input_length=50)) #使用Embeeding层将每个词编码转换为词向量
    model.add(Conv1D(256, 5, padding='same'))
    model.add(MaxPooling1D(3, 3, padding='same'))
    model.add(Conv1D(128, 5, padding='same'))
    model.add(MaxPooling1D(3, 3, padding='same'))
    model.add(Conv1D(64, 3, padding='same'))
    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(BatchNormalization())  # (批)规范化层
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(3, activation='softmax'))
    
    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    print(model.summary())
    
    return model

early_stopping = EarlyStopping(monitor='val_acc', patience=3)   #早停法，防止过拟合
plateau = ReduceLROnPlateau(monitor="val_acc", verbose=1, mode='max', factor=0.5, patience=2) #当评价指标不在提升时，减少学习率
# checkpoint = ModelCheckpoint('trained_model/keras_bert_THUCNews.hdf5', monitor='val_acc',verbose=2, save_best_only=True, mode='max', save_weights_only=True) #保存最好的模型

def get_step(sample_count, batch_size):
    step = sample_count // batch_size
    if sample_count % batch_size != 0:
        step += 1
    return step

batch_size = 8
train_step = get_step(train_sample_count, batch_size)
dev_step = get_step(dev_sample_count, batch_size)

train_dataset_iterator = get_batch_data_iterator(r"data/keras_bert_train.txt", batch_size)
dev_dataset_iterator = get_batch_data_iterator(r"data/keras_bert_dev.txt", batch_size)

model = get_model()

#模型训练
model.fit(
    train_dataset_iterator,
    steps_per_epoch=train_step,
    epochs=10,
    validation_data=dev_dataset_iterator,
    validation_steps=dev_step,
    callbacks=[early_stopping, plateau],
    verbose=1
)
```

  Model: "sequential"

  _________________________________________________________________

  Layer (type)         Output Shape       Param #  

  =================================================================

  embedding (Embedding)    (None, 50, 300)      454574700 

  _________________________________________________________________

  conv1d (Conv1D)       (None, 50, 256)      384256  

  _________________________________________________________________

  max_pooling1d (MaxPooling1D) (None, 17, 256)      0     

  _________________________________________________________________

  conv1d_1 (Conv1D)      (None, 17, 128)      163968  

  _________________________________________________________________

  max_pooling1d_1 (MaxPooling1 (None, 6, 128)      0     

  _________________________________________________________________

  conv1d_2 (Conv1D)      (None, 6, 64)       24640   

  _________________________________________________________________

  flatten (Flatten)      (None, 384)        0     

  _________________________________________________________________

  dropout (Dropout)      (None, 384)        0     

  _________________________________________________________________

  batch_normalization (BatchNo (None, 384)        1536   

  _________________________________________________________________

  dense (Dense)        (None, 256)        98560   

  _________________________________________________________________

  dropout_1 (Dropout)     (None, 256)        0     

  _________________________________________________________________

  dense_1 (Dense)       (None, 3)         771    

  =================================================================

  Total params: 455,248,431

  Trainable params: 455,247,663

  Non-trainable params: 768

  _________________________________________________________________

  None

  Epoch 1/10

​    1/83608 [..............................] - ETA: 3:28 - loss: 1.1427 - accuracy: 0.3750



#### 定义 简单版TextCNN

```python
def get_model():
    K.clear_session()
    
    main_input = Input(shape=(50,), dtype='float64')
    # 词嵌入（使用预训练的词向量）
    embedder = Embedding(len(vocab) + 1, 300, input_length=50, trainable=False)
    embed = embedder(main_input)
    # 词窗大小分别为3,4,5
    cnn1 = Conv1D(256, 3, padding='same', strides=1, activation='relu')(embed)
    cnn1 = MaxPooling1D(pool_size=48)(cnn1)
    cnn2 = Conv1D(256, 4, padding='same', strides=1, activation='relu')(embed)
    cnn2 = MaxPooling1D(pool_size=47)(cnn2)
    cnn3 = Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
    cnn3 = MaxPooling1D(pool_size=46)(cnn3)
    # 合并三个模型的输出向量
    cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
    flat = Flatten()(cnn)
    drop = Dropout(0.2)(flat)
    main_output = Dense(3, activation='softmax')(drop)
    model = Model(inputs=main_input, outputs=main_output)
    
    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    print(model.summary())
    
    return model

batch_size = 8
train_step = get_step(train_sample_count, batch_size)
dev_step = get_step(dev_sample_count, batch_size)

train_dataset_iterator = get_batch_data_iterator(r"data/keras_bert_train.txt", batch_size)
dev_dataset_iterator = get_batch_data_iterator(r"data/keras_bert_dev.txt", batch_size)

model = get_model()

#模型训练
model.fit(
    train_dataset_iterator,
    steps_per_epoch=train_step,
    epochs=10,
    validation_data=dev_dataset_iterator,
    validation_steps=dev_step,
    callbacks=[early_stopping, plateau],
    verbose=1
)
```

  Model: "functional_1"

  __________________________________________________________________________________________________

  Layer (type)          Output Shape     Param #   Connected to           

  ==================================================================================================

  input_1 (InputLayer)      [(None, 50)]     0                      

  __________________________________________________________________________________________________

  embedding (Embedding)      (None, 50, 300)   454574700  input_1[0][0]          

  __________________________________________________________________________________________________

  conv1d (Conv1D)         (None, 50, 256)   230656   embedding[0][0]         

  __________________________________________________________________________________________________

  conv1d_1 (Conv1D)        (None, 50, 256)   307456   embedding[0][0]         

  __________________________________________________________________________________________________

  conv1d_2 (Conv1D)        (None, 50, 256)   384256   embedding[0][0]         

  __________________________________________________________________________________________________

  max_pooling1d (MaxPooling1D)  (None, 1, 256)    0      conv1d[0][0]           

  __________________________________________________________________________________________________

  max_pooling1d_1 (MaxPooling1D) (None, 1, 256)    0      conv1d_1[0][0]          

  __________________________________________________________________________________________________

  max_pooling1d_2 (MaxPooling1D) (None, 1, 256)    0      conv1d_2[0][0]          

  __________________________________________________________________________________________________

  concatenate (Concatenate)    (None, 1, 768)    0      max_pooling1d[0][0]       

​                                   max_pooling1d_1[0][0]      

​                                   max_pooling1d_2[0][0]      

  __________________________________________________________________________________________________

  flatten (Flatten)        (None, 768)     0      concatenate[0][0]        

  __________________________________________________________________________________________________

  dropout (Dropout)        (None, 768)     0      flatten[0][0]          

  __________________________________________________________________________________________________

  dense (Dense)          (None, 3)      2307    dropout[0][0]          

  ==================================================================================================

  Total params: 455,499,375

  Trainable params: 924,675

  Non-trainable params: 454,574,700

  __________________________________________________________________________________________________

  None

  Epoch 1/10

   238/83608 [..............................] - ETA: 2:31:07 - loss: 0.0308 - accuracy: 0.9979