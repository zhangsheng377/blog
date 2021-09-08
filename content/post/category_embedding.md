---
title: "将类别特征通过Embedding层映射并进行拼接"
date: 2021-04-28T02:53:58+08:00
lastmod: 2021-04-28T02:53:58+08:00
draft: false
keywords: []
description: ""
tags: [深度学习, 类别特征, embedding, 拼接, keras]
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
  enable: false
  options: ""

sequenceDiagrams: 
  enable: false
  options: ""

---

## 背景

最近有一道题，想把类别特征也放到模型里去，跟bert输出拼接到一起。

所以便设计了，使用Embedding层，将类别特征的词表映射到X维向量空间里。

    keras.layers.Embedding(input_dim, output_dim, embeddings_initializer='uniform', embeddings_regularizer=None, activity_regularizer=None, embeddings_constraint=None, mask_zero=False, input_length=None)

其中，input_dim 是词表长度。

同时，因为Embedding层的输出会带上扩展的层数，变成(batch_size, input_length, output_dim)。不方便拼接，所以还需要Flatten层，帮忙拉平。

## 代码实现

```python
bert_path = r"chinese_L-12_H-768_A-12"

oriSource_count = 5574
category_count = 30
sorted_label_list = range(30)
```

```python
def get_model():
    K.clear_session()
    
    bert_model = TFBertModel.from_pretrained(bert_path, from_pt=True, trainable=True)
    for l in bert_model.layers:
        l.trainable = True
    
    input_bert_indices = Input(shape=(None,), dtype=tf.int32)
    input_bert_segments = Input(shape=(None,), dtype=tf.int32)
    bert_output = bert_model([input_bert_indices, input_bert_segments])[0]
    bert_cls = Lambda(lambda x: x[:, 0])(bert_output)
    
    input_num_feature = Input(shape=(5,))
    
    input_oriSource_index = Input(shape=(1,))
    oriSource_embedding = Embedding(oriSource_count, 256)(input_oriSource_index)
    oriSource_emb = Flatten()(oriSource_embedding)
    print(oriSource_emb)
    
    input_category_index = Input(shape=(1,))
    category_embedding = Embedding(category_count, 32)(input_category_index)
    category_emb = Flatten()(category_embedding)
    print(category_emb)
    
    emb = concatenate([bert_cls, input_num_feature, oriSource_emb, category_emb])
    print(emb)
    
    d0 = Dense(1024, activation='relu')(emb)
    d0_ = Dropout(0.2)(d0)
    
    d1 = Dense(256, activation='relu')(d0_)
    d1_ = Dropout(0.2)(d1)
    
    output = Dense(len(sorted_label_list), activation='softmax')(d1_)
    
    model = Model([input_bert_indices, input_bert_segments, input_num_feature, input_oriSource_index, input_category_index], output)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(1e-5),
                  metrics=['accuracy']
                 )
    return model
```

```python
model = get_model()
model.summary()
```

    KerasTensor(type_spec=TensorSpec(shape=(None, 256), dtype=tf.float32, name=None), name='flatten/Reshape:0', description="created by layer 'flatten'")
    KerasTensor(type_spec=TensorSpec(shape=(None, 32), dtype=tf.float32, name=None), name='flatten_1/Reshape:0', description="created by layer 'flatten_1'")
    KerasTensor(type_spec=TensorSpec(shape=(None, 1061), dtype=tf.float32, name=None), name='concatenate/concat:0', description="created by layer 'concatenate'")
    Model: "model"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)            [(None, None)]       0                                            
    __________________________________________________________________________________________________
    input_2 (InputLayer)            [(None, None)]       0                                            
    __________________________________________________________________________________________________
    input_4 (InputLayer)            [(None, 1)]          0                                            
    __________________________________________________________________________________________________
    input_5 (InputLayer)            [(None, 1)]          0                                            
    __________________________________________________________________________________________________
    tf_bert_model (TFBertModel)     TFBaseModelOutputWit 102267648   input_1[0][0]                    
                                                                    input_2[0][0]                    
    __________________________________________________________________________________________________
    embedding (Embedding)           (None, 1, 256)       1426944     input_4[0][0]                    
    __________________________________________________________________________________________________
    embedding_1 (Embedding)         (None, 1, 32)        960         input_5[0][0]                    
    __________________________________________________________________________________________________
    lambda (Lambda)                 (None, 768)          0           tf_bert_model[0][0]              
    __________________________________________________________________________________________________
    input_3 (InputLayer)            [(None, 5)]          0                                            
    __________________________________________________________________________________________________
    flatten (Flatten)               (None, 256)          0           embedding[0][0]                  
    __________________________________________________________________________________________________
    flatten_1 (Flatten)             (None, 32)           0           embedding_1[0][0]                
    __________________________________________________________________________________________________
    concatenate (Concatenate)       (None, 1061)         0           lambda[0][0]                     
                                                                    input_3[0][0]                    
                                                                    flatten[0][0]                    
                                                                    flatten_1[0][0]                  
    __________________________________________________________________________________________________
    dense (Dense)                   (None, 1024)         1087488     concatenate[0][0]                
    __________________________________________________________________________________________________
    dropout_37 (Dropout)            (None, 1024)         0           dense[0][0]                      
    __________________________________________________________________________________________________
    dense_1 (Dense)                 (None, 256)          262400      dropout_37[0][0]                 
    __________________________________________________________________________________________________
    dropout_38 (Dropout)            (None, 256)          0           dense_1[0][0]                    
    __________________________________________________________________________________________________
    dense_2 (Dense)                 (None, 30)           7710        dropout_38[0][0]                 
    ==================================================================================================
    Total params: 105,053,150
    Trainable params: 105,053,150
    Non-trainable params: 0
    __________________________________________________________________________________________________
