---
title: "keras_tf2数据并行"
date: 2021-05-07T20:19:58+08:00
lastmod: 2021-05-07T20:19:58+08:00
draft: false
keywords: []
description: ""
tags: [并行, 数据并行, 多gpu, keras, tf]
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

一直都想去搭一个并行模型，即使用多gpu共同计算的模型。今天终于把keras+tf2下的数据并行模式调通啦~~

## 关键点

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import tensorflow as tf
''' 不可以打开边缘计算模式 '''
# tf.config.run_functions_eagerly(True)
```

```python
from tensorflow.python.client import device_lib
''' 查看tf能看到的gpu '''
device_lib.list_local_devices()
```

    [name: "/device:CPU:0"
    device_type: "CPU"
    memory_limit: 268435456
    locality {
    }
    incarnation: 12467642772062802858,
    name: "/device:GPU:0"
    device_type: "GPU"
    memory_limit: 10770692224
    locality {
    bus_id: 1
    links {
        link {
        device_id: 1
        type: "StreamExecutor"
        strength: 1
        }
    }
    }
    incarnation: 18298986829211475306
    physical_device_desc: "device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:05:00.0, compute capability: 6.1",
    name: "/device:GPU:1"
    device_type: "GPU"
    memory_limit: 10770692224
    locality {
    bus_id: 1
    links {
        link {
        type: "StreamExecutor"
        strength: 1
        }
    }
    }
    incarnation: 13220649671921966797
    physical_device_desc: "device: 1, name: GeForce GTX 1080 Ti, pci bus id: 0000:08:00.0, compute capability: 6.1"]

```python
''' 创建分布策略 '''
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
```

    Number of devices: 2

```python
def get_model():
    ''' 定义模型时不可以清除session '''
#     K.clear_session()

    bert_model = TFBertModel.from_pretrained(bert_path, from_pt=True, trainable=True)
    for l in bert_model.layers:
        l.trainable = True

    input_ids_texta = Input(shape=(None,), dtype='int32', name='input_ids_texta')
    input_token_type_ids_texta = Input(shape=(None,), dtype='int32', name='input_token_type_ids_texta')
    input_attention_mask_texta = Input(shape=(None,), dtype='int32', name='input_attention_mask_texta')
    input_ids_textb = Input(shape=(None,), dtype='int32', name='input_ids_textb')
    input_token_type_ids_textb = Input(shape=(None,), dtype='int32', name='input_token_type_ids_textb')
    input_attention_mask_textb = Input(shape=(None,), dtype='int32', name='input_attention_mask_textb')
    input_token_type_ids_textb = Input(shape=(None,), dtype='int32', name='input_token_type_ids_textb')
    input_bm25 = Input(shape=(1), dtype='float32', name='input_bm25')
    input_tf_cosine = Input(shape=(1), dtype='float32', name='input_tf_cosine')
    input_tfidf_cosine = Input(shape=(1), dtype='float32', name='input_tfidf_cosine')
    input_cat_texta = Input(shape=(1), dtype='float32', name='input_cat_texta')
    input_cat_textb = Input(shape=(1), dtype='float32', name='input_cat_textb')

    bert_output_texta = bert_model({'input_ids':input_ids_texta, 'token_type_ids':input_token_type_ids_texta, 'attention_mask':input_attention_mask_texta}, return_dict=False, training=True)
    projection_logits_texta = bert_output_texta[0]
    bert_cls_texta = Lambda(lambda x: x[:, 0])(projection_logits_texta) # 取出[CLS]对应的向量用来做分类

    bert_output_textb = bert_model({'input_ids':input_ids_textb, 'token_type_ids':input_token_type_ids_textb, 'attention_mask':input_attention_mask_textb}, return_dict=False, training=True)
    projection_logits_textb = bert_output_textb[0]
    bert_cls_textb = Lambda(lambda x: x[:, 0])(projection_logits_textb) # 取出[CLS]对应的向量用来做分类

    subtracted = Subtract()([bert_cls_texta, bert_cls_textb])
    cos = Dot(axes=1, normalize=True)([bert_cls_texta, bert_cls_textb]) # dot=1按行点积，normalize=True输出余弦相似度

    bert_cls = concatenate([bert_cls_texta, bert_cls_textb, subtracted, cos, input_bm25, input_tf_cosine, input_tfidf_cosine, input_cat_texta, input_cat_textb], axis=-1)

    dense_A_0 = Dense(256, activation='relu')(bert_cls)
    dropout_A_0 = Dropout(0.2)(dense_A_0)
    dense_A_1 = Dense(32, activation='relu')(dropout_A_0)
    dropout_A_1 = Dropout(0.2)(dense_A_1)
    output_A = Dense(1, activation='sigmoid', name='output_A')(dropout_A_1)

    dense_B_0 = Dense(256, activation='relu')(bert_cls)
    dropout_B_0 = Dropout(0.2)(dense_B_0)
    dense_B_1 = Dense(32, activation='relu')(dropout_B_0)
    dropout_B_1 = Dropout(0.2)(dense_B_1)
    output_B = Dense(1, activation='sigmoid', name='output_B')(dropout_B_1)

    input_data = {
        'ids_texta':input_ids_texta,
        'token_type_ids_texta':input_token_type_ids_texta,
        'attention_mask_texta':input_attention_mask_texta,
        'ids_textb':input_ids_textb,
        'token_type_ids_textb':input_token_type_ids_textb,
        'attention_mask_textb':input_attention_mask_textb,
        'bm25':input_bm25,
        'tf_cosine':input_tf_cosine,
        'tfidf_cosine':input_tfidf_cosine,
        'cat_texta':input_cat_texta,
        'cat_textb':input_cat_textb,
    }
    output_data = {
        'labelA':output_A,
        'labelB':output_B,
    }
    model = Model(input_data, output_data)
    model.compile(
                  loss={
                      'labelA':my_binary_crossentropy_A,
                      'labelB':my_binary_crossentropy_B,
                  },
                  optimizer=Adam(1e-5),    #用足够小的学习率
                  metrics='accuracy'
                 )
    print(model.summary())
    return model
```

```python
batch_size = 2 * strategy.num_replicas_in_sync
epochs = 10

train_dataset_iterator = get_dataset("data/shuffle_total_file.tfrecord", epochs=epochs, batch_size=batch_size)
train_step = get_step(train_dataset_len, batch_size)

with strategy.scope():
    model = get_model()
    plot_model(model, "keras_bert_transformers_two_text_input_SubStract_bm25cosine_1.png", show_shapes=True)
    
    model.load_weights(check_point_path)

''' 不可以把训练部分也放到策略scope中，因为它的本质就相当于是用for遍历gpus，在每个gpu上创建模型，如果把训练也放进去，就变成在第一块gpu上训练了 '''
for epoch in range(epochs):
    model.fit(
        train_dataset_iterator,
        steps_per_epoch=train_step,
        epochs=1,
        callbacks=[early_stopping, plateau, checkpoint, tensorboard_callback],
        verbose=1
    )

    model.save_weights(weights_path)

    save_test_result(model, f"{result_path}.epoch_{epoch}.csv")
```

    +-------------------------------+----------------------+----------------------+
    |   1  GeForce GTX 108...  Off  | 00000000:05:00.0 Off |                  N/A |
    | 38%   67C    P2   249W / 250W |  10639MiB / 11178MiB |     92%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
    |   2  GeForce GTX 108...  Off  | 00000000:08:00.0 Off |                  N/A |
    | 37%   67C    P2   254W / 250W |  10639MiB / 11178MiB |     92%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+

## 附录

### 源代码

    # 导包


    ```python
    import sys 
    sys.version
    ```




        '3.7.10 | packaged by conda-forge | (default, Feb 19 2021, 16:07:37) \n[GCC 9.3.0]'




    ```python
    # from google.colab import drive
    # drive.mount('/content/drive')
    ```


    ```python
    # !unzip sohu2021_open_data_clean.zip
    # !unzip chinese_L-12_H-768_A-12.zip
    ```


    ```python
    # !pip install transformers
    ```


    ```python
    import os
    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    import sys
    import re
    from collections import Counter
    import random
    import json
    from joblib import dump, load
    from functools import partial
    from datetime import datetime
    import multiprocessing

    from tqdm import tqdm
    import numpy as np
    # import tensorflow.keras as keras
    import tensorflow as tf
    # tf.config.run_functions_eagerly(True)
    tf.get_logger().setLevel(tf.compat.v1.logging.ERROR)
    import keras
    from keras.metrics import top_k_categorical_accuracy, binary_accuracy
    from keras.layers import *
    from keras.callbacks import *
    from keras.models import Model, load_model, model_from_json
    import keras.backend as K
    from keras.optimizers import Adam
    from keras.utils import to_categorical, plot_model
    from keras.losses import SparseCategoricalCrossentropy, binary_crossentropy
    # from keras.utils import multi_gpu_model
    # from keras.utils.training_utils import multi_gpu_model
    # from tensorflow.keras.utils import multi_gpu_model
    from transformers import (
        BertTokenizer,
        TFBertForPreTraining,
        TFBertModel,
    )
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
    from sklearn.utils import class_weight
    import torch
    from pyhanlp import *
    import jieba

    from my_utils import calculate_bm25_similarity, calculate_tf_cosine_similarity, calculate_tfidf_cosine_similarity
    ```


    ```python
    tf.__version__
    ```




        '2.4.1'




    ```python
    keras.__version__
    ```




        '2.4.3'




    ```python
    from tensorflow.python.client import device_lib
    device_lib.list_local_devices()
    ```




        [name: "/device:CPU:0"
        device_type: "CPU"
        memory_limit: 268435456
        locality {
        }
        incarnation: 4234606634315361239,
        name: "/device:GPU:0"
        device_type: "GPU"
        memory_limit: 10770692224
        locality {
        bus_id: 1
        links {
            link {
            device_id: 1
            type: "StreamExecutor"
            strength: 1
            }
        }
        }
        incarnation: 17812997849526279671
        physical_device_desc: "device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:05:00.0, compute capability: 6.1",
        name: "/device:GPU:1"
        device_type: "GPU"
        memory_limit: 10770692224
        locality {
        bus_id: 1
        links {
            link {
            type: "StreamExecutor"
            strength: 1
            }
        }
        }
        incarnation: 5271932875471281872
        physical_device_desc: "device: 1, name: GeForce GTX 1080 Ti, pci bus id: 0000:08:00.0, compute capability: 6.1"]




    ```python
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    ```

        Number of devices: 2



    ```python

    ```


    ```python
    data_path = "sohu2021_open_data_clean/"
    # train_file_names = ["train.txt", "valid.txt", "round2.txt", "round3.txt"]
    train_file_name = "data/shuffle_total_file.json"
    text_max_length = 512
    bert_path = r"chinese_L-12_H-768_A-12"

    check_point_path = 'trained_model_substract_1/multi_keras_bert_sohu.weights'
    weights_path = "trained_model_substract_1/multi_keras_bert_sohu_final.weights"
    config_path = "trained_model_substract_1/multi_keras_bert_sohu_final.model_config.json"
    result_path = "trained_model_substract_1/multi_keras_bert_sohu_test_result_final.csv"
    ```


    ```python
    # bm25Model = load("bm25.bin")
    # bm25Model
    ```


    ```python

    ```


    ```python

    ```


    ```python

    ```


    ```python

    ```


    ```python
    # 转换bert模型，到pytorch的pd格式
    ```


    ```python
    # !transformers-cli convert --model_type bert \
    #   --tf_checkpoint chinese_L-12_H-768_A-12/bert_model.ckpt \
    #   --config chinese_L-12_H-768_A-12/bert_config.json \
    #   --pytorch_dump_output chinese_L-12_H-768_A-12/pytorch_model.bin
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
    label_to_id = {'0':0, '1':1}
    ```


    ```python
    # def get_text_iterator(file_path):
    #     with open(file_path, 'r', encoding='utf-8') as f:
    #         for line in f:
    #             yield line
    ```


    ```python
    def _transform_text(text):
    text = text.strip().replace('\n', '。').replace('\t', '').replace('\u3000', '')
    return re.sub(r'。+', '。', text)
    ```


    ```python
    # def get_summary(text, senc_num=20):
    #     a = HanLP.extractSummary(text, 20)
    #     a_ = str(a)
    #     return a_[1:-1]
    ```


    ```python
    # def get_data_iterator(data_path, file_names):
    #     # TODO: 随机取
    #     file_iters = []
    #     for file_name in file_names:
    #       for category in os.listdir(data_path):
    #           category_path = os.path.join(data_path, category)
    #           if not os.path.isdir(category_path):
    #               continue
                
    #           file_path = os.path.join(category_path, file_name)
    #           if not os.path.isfile(file_path):
    #               continue
                
            
    #           file_iter = get_text_iterator(file_path)
    #           cat_source = 0
    #           if category[0] == '长':
    #             cat_source = 1
    #           cat_target = 0
    #           if category[1] == '长':
    #             cat_target = 1
    #           file_iters.append((file_iter, cat_source, cat_target))
            
    #     while len(file_iters) > 0:
    #         i = random.randrange(len(file_iters))
    #         line = next(file_iters[i][0], None)
    #         cat_source = file_iters[i][1]
    #         cat_target = file_iters[i][2]
    #         if line is None:
    #             del file_iters[i]
    #             continue
                
    #         data = json.loads(line)

    #         data['source'] = _transform_text(data['source'])
    #         if len(data['source']) == 0:
    #             print('source:', line, data)
    #             break
    # #                     continue

    #         data['target'] = _transform_text(data['target'])
    #         if len(data['target']) == 0:
    #             print('target:', line, data)
    #             break
    # #                     continue

    #         label_name_list = list(key for key in data.keys() if key[:5]=='label')
    #         if len(label_name_list) != 1:
    #             print('label_name_list:', line, data)
    #             break
    # #                     continue
    #         label_name = label_name_list[0]
    #         if data[label_name] not in label_to_id.keys():
    #             print('label_name:', line, data, label_name)
    #             break
    # #                     continue
            
    #         label_dict = {key: -1 for key in label_type_to_id.keys()}
    #         label_dict[label_name] = label_to_id[data[label_name]]
    #         if label_dict['labelA'] == 0:
    #             label_dict['labelB'] = 0
    #         if label_dict['labelB'] == 1:
    #             label_dict['labelA'] = 1

    #         yield data['source'], data['target'], cat_source, cat_target, label_dict['labelA'], label_dict['labelB']
    ```


    ```python
    # it = get_data_iterator(data_path, train_file_names)
    ```


    ```python
    # next(it)
    ```


    ```python
    def get_sample_num(data_path, file_names):
        count = 0
        it = get_data_iterator(data_path, file_names)
        for data in tqdm(it):
            count += 1
        return count
    ```


    ```python
    # sample_count = get_sample_num(data_path, train_file_names)
    # sample_count
    ```


    ```python
    # def get_shuffle_total_file(data_path, file_names, output_file_path):
    #     data_list = []
    #     it = get_data_iterator(data_path, file_names)
    #     for source, target, cat_source, cat_target, labelA, labelB in tqdm(it):
    #         json_data = {
    #             'source':source,
    #             'target':target,
    #             'cat_source':cat_source,
    #             'cat_target':cat_target,
    #             'labelA':labelA,
    #             'labelB':labelB
    #         }
    #         data_list.append(json_data)
    #     random.shuffle(data_list)
        
    #     with open(output_file_path, 'w', encoding='utf-8') as output_file:
    #         for json_data in data_list:
    #             output_file.write(f"{json.dumps(json_data)}\n")
    ```


    ```python
    # get_shuffle_total_file(data_path, train_file_names, train_file_name)
    ```


    ```python
    def get_data_iterator(data_path, file_name):
        with open(file_name, 'r', encoding='utf-8') as f:
            for line in f:
                json_data = json.loads(line)
                yield json_data['source'], json_data['target'], json_data['cat_source'], json_data['cat_target'], json_data['labelA'], json_data['labelB']
    ```


    ```python
    it = get_data_iterator(data_path, train_file_name)
    ```


    ```python
    next(it)
    ```




        ('湖人120-102火箭 一觉醒来，湖人队又是一场大胜。  毫不夸张的说，这两支球队可能都是全NBA球迷最多队伍之一，两队一交手，其收视率，堪比篮圈春晚。 既然是春晚，那就不能少了节目，今儿个双方为大家伙儿准备了以下几个节目： 其一，洛杉矶小伍德，对阵休斯顿真伍德， 其二，洛杉矶詹姆斯，对阵休斯顿詹姆斯， 前三，洛杉矶塔克，对阵休斯顿塔克， 其四，理智考辛斯，对阵暴走莫里斯。  第一节末端，考神与莫里斯率先整活儿，引爆全场气氛。 话说当时詹老汉正持球背打，莫里斯使出一套坦克突进，直接将火箭球员杰肖恩-泰特老哥撵出球场。 考辛斯一看泰特倒地，大喝一声：你敢欺负我队友，这还得了？ 上前一套马保国式接化发，将莫里斯推出二米开外，并回头打算拉起倒地的泰特。 事实证明，人在生气的时候最会被激发出无限的潜能，莫里斯从被推倒地上到站起来推开考辛斯，仅仅用时不到两秒， 这么冷静就走开了！？ 这还是我们认识的考辛斯吗？？ 那速度，简直比韦德、艾弗森、TJ-福特上篮速度更快，詹老汉可能也没想到莫里斯还有这爆发力，本想拉一把，却愣是没追上。 恕技巧君直言，要是莫里斯老哥打球也能有这速度，那么顶掉小黑，出任湖人队首发控卫简直绰绰有余。  判罚结果是，双方各吃一T，大莫被驱逐。 当然，考神在场上也没待多久。 第二节刚开打，老詹突破内线，考神下手切球，一巴掌不小心切到老詹头上。 从老詹的表情大家伙儿可以看到这记如来神掌是有多痛。 尽管考神及时弥补，上前试图拉起老詹，并努力告诉裁判：“那可是我詹大爷，我怎么可能恶犯他？”，依然没逃出被驱逐的命运。  好在浓眉的发挥，又把大家伙儿的目光吸引了回来。 平日里浓眉划划水，一旦遇到自己的模板，那叫一个放开抡绝不含糊，光凭一个上半场，小伍德已经8投8中，入账21分3盖帽。 戴维斯：听说你是我的模板？来个8中8致敬一下。 另一边真伍德整个人都是崩溃的：谁特么说他是小伍德？这不是给我找麻烦么！碰别人就划水碰我就认真，这以后可咋整啊？！ 毫不夸张的说，今儿这场大胜，洛杉矶小伍德得占首功，一人直接打崩火箭队内线。  至于两个詹姆斯，反倒是手牵手好朋友，你划一桨我划一桨，洛杉矶gigi怪入账18+7+7，休斯顿大胡子拿到20+6+9，顺便带走7个失误。 泰特梦幻脚步戏耍老詹 区别在于，湖人赢球了，詹姆斯数据怎样都无所谓，火箭队输球，场边观众直呼：“哈登！醒醒！”  最后一组对位，塔克之间的对决。 火箭老塔克本以为欺负小年轻，不过是手到擒来洒洒水，谁料还是大意了。这家伙居然上场就暴走，出战短短20分钟，便8投7中有17分5篮板3助攻4抢断入账。 反观老塔克，上场30分钟，4投1中4分入账。在老塔克的衬托之下，小塔克的形象显的无比高大。  湖人队在丰田中心，带走第一场比赛的胜利，重新登顶西部第一。',
        '湖人作为上赛季的卫冕冠军以及本赛季最有希望夺冠的球队，他们的每场比赛都会被球迷关注，联盟也对湖人的比赛格外重视。可能是为了让湖人的比赛变得更加公正，执法湖人比赛的裁判也格外严厉。之前在湖人和火箭的比赛中，半场比赛没结束双方就被吹罚了五次技术犯规，本以为这已经算是告一段落了，没想到联盟还进一步发布了有关考辛斯和大莫里斯的追加处罚。  平时有关注NBA的朋友应该都清楚NBA现在的规则，一般情况下只有球员在场外发表了一些不当言论才会被罚款，一个赛季或许都不会出现一次因为犯规而被罚款的情况。然而火箭和湖人这一场比赛就诞生了两个因为犯规而被罚款的情况，合起来一共被罚款4.5万美元，这足够他们在场外被处罚两次了。可为什么大莫里斯的罚款会被考辛斯多那么多呢？一个3.5万美金，另一个只被罚了1万美金。  在当时那场比赛中，考辛斯和大莫里斯都是因为一次技术犯规以及一次恶意犯规而被罚出场外，可他们的恶意犯规也存在区别。大莫里斯当时被吹罚了一级恶意犯规，而考辛斯对詹姆斯头部击打的行为只是被吹罚了二级恶意犯规。明明大莫里斯和考辛斯的行为都差不多，而且大莫里斯只是将对手推倒，可考辛斯是直接击打对手头部，那为什么大莫里斯还会比考辛斯被罚得更多呢？  这个恶意犯规一级和二级的区别其实不能只看球员犯规的严重性，这还需要看球员犯规的意图，很明显大莫里斯就是故意将对手推倒在地，而考辛斯很可能是无意中打到詹姆斯的头部。如果考辛斯之前没有和大莫里斯发生冲突，或许这一次击打詹姆斯的头部只会被判罚一个违体犯规。当然考辛斯本来就是联盟中恶意犯规的专业户，裁判或许早就对他有意见了，吹一个二级恶意犯规也正常。  现在的各种体育联盟似乎都有点太尊重裁判了，球员们无法用自己的方式去影响比赛，大莫里斯和考辛斯这种吵架放在上世纪八十年代可能都不会被吹罚犯规，因为那时候乔丹和坏孩子军团的对碰要可怕很多。不仅是NBA存在这种情况，CBA其实也有很多这些问题。裁判可以直接影响比赛，只要裁判愿意的话，那么他就可以操控任意一场比赛的结果。  湖人和火箭很可能会因为这一次比赛而产生恩怨，一直都以总冠军为目标的火箭肯定想要击败湖人，只要击败了湖人，那火箭就有信心和机会冲击总冠军了。不过考辛斯和大莫里斯只是队内的角色球员，他们的矛盾很难影响到两支球队的胜负结果。当然比赛还是要存在这些火药味的，不然的话比赛就毫无趣味了。  不知道大家对于这件事情有没有什么别的看法和意见呢？欢迎在下面评论交流一下。',
        1,
        1,
        0,
        0)




    ```python
    # sample_count = get_sample_num(data_path, train_file_name)
    # sample_count
    ```


    ```python
    def get_sample_y(data_path, file_names):
        labelA_list = []
        labelB_list = []
        it = get_data_iterator(data_path, file_names)
        for source, target, cat_source, cat_target, labelA, labelB in tqdm(it):
            if labelA != -1:
            labelA_list.append(labelA)
            if labelB != -1:
            labelB_list.append(labelB)
        return labelA_list, labelB_list
    ```


    ```python
    # np.unique(labelA_list), labelA_list
    ```


    ```python
    labelA_list, labelB_list = get_sample_y(data_path, train_file_name)
    labelA_class_weights = class_weight.compute_class_weight('balanced', np.unique(labelA_list), np.array(labelA_list))
    labelB_class_weights = class_weight.compute_class_weight('balanced', np.unique(labelB_list), np.array(labelB_list))
    labelA_class_weights, labelB_class_weights
    ```

        168714it [00:22, 7509.62it/s]
        /data1/wangchenyue/zsd/code/sohu2021/conda_env/lib/python3.7/site-packages/sklearn/utils/validation.py:72: FutureWarning: Pass classes=[0 1], y=[0 0 1 ... 1 0 1] as keyword args. From version 1.0 (renaming of 0.25) passing these as positional arguments will result in an error
        "will result in an error", FutureWarning)
        /data1/wangchenyue/zsd/code/sohu2021/conda_env/lib/python3.7/site-packages/sklearn/utils/validation.py:72: FutureWarning: Pass classes=[0 1], y=[0 0 1 ... 1 0 0] as keyword args. From version 1.0 (renaming of 0.25) passing these as positional arguments will result in an error
        "will result in an error", FutureWarning)





        (array([0.94664122, 1.05973338]), array([0.56295201, 4.47127937]))




    ```python
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    ```


    ```python
    def _get_indices(text, text_pair=None):
        return tokenizer.encode_plus(text=text,
                                text_pair=text_pair,
                                max_length=text_max_length, 
                                add_special_tokens=True, 
                                padding='max_length', 
    #                             truncation_strategy='longest_first', 
                                truncation=True,
    #                                          return_tensors='tf',
                                return_token_type_ids=True
                                )
    ```


    ```python
    def get_keras_bert_iterator_notwhile(data_path, file_names, tokenizer):
        data_it = get_data_iterator(data_path, file_names)
        for source, target, cat_source, cat_target, labelA, labelB in data_it:
            data_source = _get_indices(text=source)
            data_target = _get_indices(text=target)
    #             print(indices, type(indices), len(indices))
            seg_source = jieba.lcut(source)
            seg_target = jieba.lcut(target)
            bm25 = calculate_bm25_similarity(bm25Model, seg_source, seg_target)
            tf_cosine = calculate_tf_cosine_similarity(seg_source, seg_target)
            tfidf_cosine = calculate_tfidf_cosine_similarity(seg_source, seg_target, bm25Model.idf)
            id = ""
            yield data_source['input_ids'], data_source['token_type_ids'], data_source['attention_mask'], \
                data_target['input_ids'], data_target['token_type_ids'], data_target['attention_mask'], \
                bm25, tf_cosine, tfidf_cosine, \
                cat_source, cat_target, \
                labelA, labelB, id
    ```


    ```python
    it = get_keras_bert_iterator_notwhile(data_path, train_file_name, tokenizer)
    ```


    ```python
    # next(it)
    ```


    ```python
    def to_tfrecord(it, output_path):
    #     it = get_keras_bert_iterator_notwhile(data_path, train_file_name, tokenizer)
        with tf.io.TFRecordWriter(output_path) as tfrecord_writer:
            for ids_texta, token_type_ids_texta, attention_mask_texta, \
                ids_textb, token_type_ids_textb, attention_mask_textb, \
                bm25, tf_cosine, tfidf_cosine, \
                cat_texta, cat_textb, \
                labelA, labelB, id in tqdm(it):

                """ 2. 定义features """
                example = tf.train.Example(
                    features = tf.train.Features(
                        feature = {
                            'ids_texta': tf.train.Feature(
                                int64_list=tf.train.Int64List(value=ids_texta)),
                            'token_type_ids_texta': tf.train.Feature(
                                int64_list=tf.train.Int64List(value=token_type_ids_texta)),
                            'attention_mask_texta': tf.train.Feature(
                                int64_list=tf.train.Int64List(value=attention_mask_texta)),
                            'ids_textb': tf.train.Feature(
                                int64_list=tf.train.Int64List(value=ids_textb)),
                            'token_type_ids_textb': tf.train.Feature(
                                int64_list=tf.train.Int64List(value=token_type_ids_textb)),
                            'attention_mask_textb': tf.train.Feature(
                                int64_list=tf.train.Int64List(value=attention_mask_textb)),
                            'bm25': tf.train.Feature(
                                float_list=tf.train.FloatList(value=[bm25])),
                            'tf_cosine': tf.train.Feature(
                                float_list=tf.train.FloatList(value=[tf_cosine])),
                            'tfidf_cosine': tf.train.Feature(
                                float_list=tf.train.FloatList(value=[tfidf_cosine])),
                            'cat_texta': tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[cat_texta])),
                            'cat_textb': tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[cat_textb])),
                            'labelA': tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[labelA])),
                            'labelB': tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[labelB])),
                            'id': tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[id.encode('utf-8')])),
                        }))

                """ 3. 序列化,写入"""
                serialized = example.SerializeToString()
                tfrecord_writer.write(serialized)
    ```


    ```python
    # it = get_keras_bert_iterator_notwhile(data_path, train_file_name, tokenizer)
    # to_tfrecord(it, "data/shuffle_total_file.tfrecord")
    ```


    ```python
    # it = get_test_keras_bert_iterator(data_path, "test_with_id.txt")
    # to_tfrecord(it, "data/test_file.tfrecord")
    ```


    ```python
    def parse_from_single_example(example_proto, need_id):
        """ 从example message反序列化得到当初写入的内容 """
        # 描述features
        desc = {
            'ids_texta': tf.io.FixedLenFeature([512], dtype=tf.int64),
            'token_type_ids_texta': tf.io.FixedLenFeature([512], dtype=tf.int64),
            'attention_mask_texta': tf.io.FixedLenFeature([512], dtype=tf.int64),
            'ids_textb': tf.io.FixedLenFeature([512], dtype=tf.int64),
            'token_type_ids_textb': tf.io.FixedLenFeature([512], dtype=tf.int64),
            'attention_mask_textb': tf.io.FixedLenFeature([512], dtype=tf.int64),
            'bm25': tf.io.FixedLenFeature([1], dtype=tf.float32),
            'tf_cosine': tf.io.FixedLenFeature([1], dtype=tf.float32),
            'tfidf_cosine': tf.io.FixedLenFeature([1], dtype=tf.float32),
            'cat_texta': tf.io.FixedLenFeature([1], dtype=tf.int64),
            'cat_textb': tf.io.FixedLenFeature([1], dtype=tf.int64),
            'labelA': tf.io.FixedLenFeature([1], dtype=tf.int64),
            'labelB': tf.io.FixedLenFeature([1], dtype=tf.int64),
            'id': tf.io.FixedLenFeature([], dtype=tf.string),
        }
        # 使用tf.io.parse_single_example反序列化
        example = tf.io.parse_single_example(example_proto, desc)
        
        data = {
            'ids_texta': example['ids_texta'],
            'token_type_ids_texta': example['token_type_ids_texta'],
            'attention_mask_texta': example['attention_mask_texta'],
            'ids_textb': example['ids_textb'],
            'token_type_ids_textb': example['token_type_ids_textb'],
            'attention_mask_textb': example['attention_mask_textb'],
            'bm25': example['bm25'],
            'tf_cosine': example['tf_cosine'],
            'tfidf_cosine': example['tfidf_cosine'],
            'cat_texta': example['cat_texta'],
            'cat_textb': example['cat_textb'],
        }
        label = {
            'labelA': example['labelA'],
            'labelB': example['labelB'],
        }
        if not need_id:
            return data, label
        return data, label, example['id']
    ```


    ```python
    dataset = tf.data.TFRecordDataset(["data/test_file.tfrecord"])
    ```


    ```python
    data_iter = iter(dataset)
    first_example = next(data_iter)
    ```


    ```python
    data = parse_from_single_example(first_example, need_id=False)
    ```


    ```python
    # data['ids_texta'].numpy(), data['ids_texta'].numpy().shape
    ```


    ```python
    def get_dataset(file_list, epochs, batch_size, need_id=False):
        return tf.data.TFRecordDataset(file_list) \
            .map(partial(parse_from_single_example, need_id=need_id), multiprocessing.cpu_count()-2) \
            .repeat(epochs) \
            .shuffle(buffer_size=batch_size, reshuffle_each_iteration=True) \
            .batch(batch_size) \
            .prefetch(buffer_size=batch_size*2)
    ```


    ```python
    dataset = get_dataset("data/shuffle_total_file.tfrecord", epochs=1, batch_size=1)
    dataset
    ```




        <PrefetchDataset shapes: ({ids_texta: (None, 512), token_type_ids_texta: (None, 512), attention_mask_texta: (None, 512), ids_textb: (None, 512), token_type_ids_textb: (None, 512), attention_mask_textb: (None, 512), bm25: (None, 1), tf_cosine: (None, 1), tfidf_cosine: (None, 1), cat_texta: (None, 1), cat_textb: (None, 1)}, {labelA: (None, 1), labelB: (None, 1)}), types: ({ids_texta: tf.int64, token_type_ids_texta: tf.int64, attention_mask_texta: tf.int64, ids_textb: tf.int64, token_type_ids_textb: tf.int64, attention_mask_textb: tf.int64, bm25: tf.float32, tf_cosine: tf.float32, tfidf_cosine: tf.float32, cat_texta: tf.int64, cat_textb: tf.int64}, {labelA: tf.int64, labelB: tf.int64})>




    ```python
    # next(iter(dataset))
    ```


    ```python
    def get_dataset_length(file_list):
        dataset = get_dataset(file_list, epochs=1, batch_size=1)
        count = 0
        for _ in tqdm(iter(dataset)):
            count += 1
        return count
    ```


    ```python
    train_dataset_len = get_dataset_length("data/shuffle_total_file.tfrecord")
    train_dataset_len
    ```

        168714it [01:10, 2403.46it/s]





        168714




    ```python

    ```


    ```python

    ```


    ```python
    def save_test_result(model, result_file):
        it = get_dataset("data/test_file.tfrecord", epochs=1, batch_size=1, need_id=True)
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(f"id,label\n")
            for data, _, id in tqdm(it):
            predict = model.predict(data)
            if id[0].numpy()[-1] == 'a':
                predict_cls = 1 if predict['labelA'][0][0] > 0.5 else 0
            elif id[0].numpy()[-1] == 'b':
                predict_cls = 1 if predict['labelB'][0][0] > 0.5 else 0
            else:
                print(id)
                continue
            f.write(f"{id},{predict_cls}\n")
            #       count += 1
            #       print(f"\b\b\b\b\b\b{count}", end="")
    ```


    ```python

    ```


    ```python

    ```


    ```python

    ```


    ```python

    ```

    ## 定义模型


    ```python
    def variant_focal_loss(gamma=2., alpha=0.5, rescale = False):

        gamma = float(gamma)
        alpha = float(alpha)

        def focal_loss_fixed(y_true, y_pred):
            # print(y_true)
            """
            Focal loss for bianry-classification
            FL(p_t)=-rescaled_factor*alpha_t*(1-p_t)^{gamma}log(p_t)
            
            Notice: 
            y_pred is probability after sigmoid

            Arguments:
                y_true {tensor} -- groud truth label, shape of [batch_size, 1]
                y_pred {tensor} -- predicted label, shape of [batch_size, 1]

            Keyword Arguments:
                gamma {float} -- (default: {2.0})  
                alpha {float} -- (default: {0.5})

            Returns:
                [tensor] -- loss.
            """
            epsilon = 1.e-9  
            y_true = tf.convert_to_tensor(y_true, tf.float32)
            y_pred = tf.convert_to_tensor(y_pred, tf.float32)
            model_out = tf.clip_by_value(y_pred, epsilon, 1.-epsilon)  # to advoid numeric underflow
            
            # compute cross entropy ce = ce_0 + ce_1 = - (1-y)*log(1-y_hat) - y*log(y_hat)
            ce_0 = tf.multiply(tf.subtract(1., y_true), -tf.math.log(tf.subtract(1., model_out)))
            ce_1 = tf.multiply(y_true, -tf.math.log(model_out))

            # compute focal loss fl = fl_0 + fl_1
            # obviously fl < ce because of the down-weighting, we can fix it by rescaling
            # fl_0 = -(1-y_true)*(1-alpha)*((y_hat)^gamma)*log(1-y_hat) = (1-alpha)*((y_hat)^gamma)*ce_0
            fl_0 = tf.multiply(tf.pow(model_out, gamma), ce_0)
            fl_0 = tf.multiply(1.-alpha, fl_0)
            # fl_1= -y_true*alpha*((1-y_hat)^gamma)*log(y_hat) = alpha*((1-y_hat)^gamma*ce_1
            fl_1 = tf.multiply(tf.pow(tf.subtract(1., model_out), gamma), ce_1)
            fl_1 = tf.multiply(alpha, fl_1)
            fl = tf.add(fl_0, fl_1)
            f1_avg = tf.reduce_mean(fl)
            
            if rescale:
                # rescale f1 to keep the quantity as ce
                ce = tf.add(ce_0, ce_1)
                ce_avg = tf.reduce_mean(ce)
                rescaled_factor = tf.divide(ce_avg, f1_avg + epsilon)
                f1_avg = tf.multiply(rescaled_factor, f1_avg)
            
            return f1_avg
        
        return focal_loss_fixed
    ```


    ```python
    def f1_loss(y_true, y_pred):
        # y_true:真实标签0或者1；y_pred:为正类的概率
        loss = 2 * tf.reduce_sum(y_true * y_pred) / tf.reduce_sum(y_true + y_pred) + K.epsilon()
        return -loss
    ```


    ```python
    def transform_y(y_true, y_pred):
        mask_value = tf.constant(-1)
        mask_y_true = tf.not_equal(tf.cast(y_true, dtype=tf.int32), tf.cast(mask_value, dtype=tf.int32))
    #     print(f"mask_y_true:{mask_y_true}")
    #     y_true_ = tf.cond(tf.equal(y_true, mask_value), lambda: 0, lambda: y_true)
        y_true_ = tf.cast(y_true, dtype=tf.int32) * tf.cast(mask_y_true, dtype=tf.int32)
        y_pred_ = tf.cast(y_pred, dtype=tf.float32) * tf.cast(mask_y_true, dtype=tf.float32)
    #     print(f"y_true_:{y_true_}, y_pred_:{y_pred_}")
        
        return tf.cast(y_true_, dtype=tf.float32), tf.cast(y_pred_, dtype=tf.float32)
    ```


    ```python
    def my_binary_crossentropy(y_true, y_pred, class_weight_0, class_weight_1):
    #     print(f"y_true: {y_true}")
        mask_value = tf.constant(-1)
    #     mask_y_true = tf.not_equal(tf.cast(y_true, dtype=tf.int32), tf.cast(mask_value, dtype=tf.int32))
        
    #     mask = tf.zeros(shape=y_true.shape)
        zero_value = tf.constant(0)
    #     print(f"cond0: {tf.equal(y_true, mask_value)}")
    #     print(f"cond1: {tf.equal(y_true, zero_value)}")
    #     weight = [tf.cond(tf.equal(x, mask_value), lambda: 0, tf.cond(tf.equal(x, zero_value), lambda: class_weights[0], lambda: class_weights[1])) for x in y_true]
    #     weight = [0 if x[0]==-1 else class_weights[x[0]] for x in y_true]
        y_true_0 = tf.equal(tf.cast(y_true, dtype=tf.int32), tf.cast(tf.constant(0), dtype=tf.int32))
        weight_0 = tf.cast(y_true_0, dtype=tf.float32) * tf.cast(tf.constant(class_weight_0), dtype=tf.float32)
        y_true_1 = tf.equal(tf.cast(y_true, dtype=tf.int32), tf.cast(tf.constant(1), dtype=tf.int32))
        weight_1 = tf.cast(y_true_1, dtype=tf.float32) * tf.cast(tf.constant(class_weight_1), dtype=tf.float32)
        weight = weight_0 + weight_1
    #     print(f"weight: {weight}")
        
        bin_loss = binary_crossentropy(y_true, y_pred)
    #     print(f"bin_loss: {bin_loss}")
    #     f1_loss = f1_loss(y_true, y_pred)
    #     loss = bin_loss + f1_loss
        
        loss_ = tf.cast(bin_loss, dtype=tf.float32) * tf.cast(weight, dtype=tf.float32)
    #     print(f"loss_: {loss_}")
        loss_abs = tf.abs(loss_)
    #     print(f"loss_abs: {loss_abs}")

    return loss_abs
    ```


    ```python
    def my_binary_crossentropy_A(y_true, y_pred):
        return my_binary_crossentropy(y_true, y_pred, labelA_class_weights[0], labelA_class_weights[1])

    def my_binary_crossentropy_B(y_true, y_pred):
        return my_binary_crossentropy(y_true, y_pred, labelB_class_weights[0], labelB_class_weights[1])
    ```


    ```python
    def tarnsform_metrics(y_true, y_pred):
        y_true_, y_pred_ = y_true.numpy(), y_pred.numpy()
        for i in range(y_true_.shape[0]):
            for j in range(y_true_.shape[1]):
                if y_true_[i][j] == -1:
                    y_true_[i][j] = 0
                    y_pred_[i][j] = random.choice([0, 1])
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
    #     K.clear_session()

        bert_model = TFBertModel.from_pretrained(bert_path, from_pt=True, trainable=True)
        for l in bert_model.layers:
            l.trainable = True

        input_ids_texta = Input(shape=(None,), dtype='int32', name='input_ids_texta')
        input_token_type_ids_texta = Input(shape=(None,), dtype='int32', name='input_token_type_ids_texta')
        input_attention_mask_texta = Input(shape=(None,), dtype='int32', name='input_attention_mask_texta')
        input_ids_textb = Input(shape=(None,), dtype='int32', name='input_ids_textb')
        input_token_type_ids_textb = Input(shape=(None,), dtype='int32', name='input_token_type_ids_textb')
        input_attention_mask_textb = Input(shape=(None,), dtype='int32', name='input_attention_mask_textb')
        input_token_type_ids_textb = Input(shape=(None,), dtype='int32', name='input_token_type_ids_textb')
        input_bm25 = Input(shape=(1), dtype='float32', name='input_bm25')
        input_tf_cosine = Input(shape=(1), dtype='float32', name='input_tf_cosine')
        input_tfidf_cosine = Input(shape=(1), dtype='float32', name='input_tfidf_cosine')
        input_cat_texta = Input(shape=(1), dtype='float32', name='input_cat_texta')
        input_cat_textb = Input(shape=(1), dtype='float32', name='input_cat_textb')

        bert_output_texta = bert_model({'input_ids':input_ids_texta, 'token_type_ids':input_token_type_ids_texta, 'attention_mask':input_attention_mask_texta}, return_dict=False, training=True)
        projection_logits_texta = bert_output_texta[0]
        bert_cls_texta = Lambda(lambda x: x[:, 0])(projection_logits_texta) # 取出[CLS]对应的向量用来做分类

        bert_output_textb = bert_model({'input_ids':input_ids_textb, 'token_type_ids':input_token_type_ids_textb, 'attention_mask':input_attention_mask_textb}, return_dict=False, training=True)
        projection_logits_textb = bert_output_textb[0]
        bert_cls_textb = Lambda(lambda x: x[:, 0])(projection_logits_textb) # 取出[CLS]对应的向量用来做分类

        subtracted = Subtract()([bert_cls_texta, bert_cls_textb])
        cos = Dot(axes=1, normalize=True)([bert_cls_texta, bert_cls_textb]) # dot=1按行点积，normalize=True输出余弦相似度

        bert_cls = concatenate([bert_cls_texta, bert_cls_textb, subtracted, cos, input_bm25, input_tf_cosine, input_tfidf_cosine, input_cat_texta, input_cat_textb], axis=-1)

        dense_A_0 = Dense(256, activation='relu')(bert_cls)
        dropout_A_0 = Dropout(0.2)(dense_A_0)
        dense_A_1 = Dense(32, activation='relu')(dropout_A_0)
        dropout_A_1 = Dropout(0.2)(dense_A_1)
        output_A = Dense(1, activation='sigmoid', name='output_A')(dropout_A_1)

        dense_B_0 = Dense(256, activation='relu')(bert_cls)
        dropout_B_0 = Dropout(0.2)(dense_B_0)
        dense_B_1 = Dense(32, activation='relu')(dropout_B_0)
        dropout_B_1 = Dropout(0.2)(dense_B_1)
        output_B = Dense(1, activation='sigmoid', name='output_B')(dropout_B_1)

        input_data = {
            'ids_texta':input_ids_texta,
            'token_type_ids_texta':input_token_type_ids_texta,
            'attention_mask_texta':input_attention_mask_texta,
            'ids_textb':input_ids_textb,
            'token_type_ids_textb':input_token_type_ids_textb,
            'attention_mask_textb':input_attention_mask_textb,
            'bm25':input_bm25,
            'tf_cosine':input_tf_cosine,
            'tfidf_cosine':input_tfidf_cosine,
            'cat_texta':input_cat_texta,
            'cat_textb':input_cat_textb,
        }
        output_data = {
            'labelA':output_A,
            'labelB':output_B,
        }
        model = Model(input_data, output_data)
        model.compile(
    #                   loss=my_binary_crossentropy,
    #                   loss={
    #                       'output_A':my_binary_crossentropy_A,
    #                       'output_B':my_binary_crossentropy_B,
    #                   },
                    loss={
                        'labelA':my_binary_crossentropy_A,
                        'labelB':my_binary_crossentropy_B,
                    },
    #                   loss='binary_crossentropy',
    #                   loss=binary_crossentropy,
                    optimizer=Adam(1e-5),    #用足够小的学习率
    #                   metrics=[my_binary_accuracy, my_f1_score]
                    metrics='accuracy'
                    )
        print(model.summary())
        return model
    ```


    ```python
    early_stopping = EarlyStopping(monitor='loss', patience=3)   #早停法，防止过拟合
    plateau = ReduceLROnPlateau(monitor="loss", verbose=1, factor=0.5, patience=2) #当评价指标不在提升时，减少学习率
    checkpoint = ModelCheckpoint(check_point_path, monitor='loss', verbose=2, save_best_only=True, save_weights_only=True) #保存最好的模型
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f"./logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}", update_freq=50)
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
    # model = get_model()
    # plot_model(model, "keras_bert_transformers_two_text_input_SubStract_bm25cosine_1.png", show_shapes=True)
    ```


    ```python
    # model.load_weights(check_point_path)
    ```


    ```python
    # batch_size = 2
    # epochs = 10

    # train_dataset_iterator = batch_iter(data_path, train_file_name, tokenizer, batch_size)
    # train_step = get_step(sample_count, batch_size)

    # model.fit(
    #     train_dataset_iterator,
    #     # steps_per_epoch=10,
    #     steps_per_epoch=train_step,
    #     epochs=epochs,
    # #       validation_data=dev_dataset_iterator,
    #   # validation_steps=2,
    # #       validation_steps=dev_step,
    # #     validation_split=0.2,
    # #     class_weight={
    # #         'output_A':labelA_class_weights,
    # #         'output_B':labelB_class_weights,
    # #     },
    #     callbacks=[early_stopping, plateau, checkpoint, tensorboard_callback],
    #     verbose=1
    # )

    # model.save_weights(weights_path)
    # # model_json = model.to_json()
    # # with open(config_path, 'w', encoding='utf-8') as file:
    # #     file.write(model_json)

    # save_test_result(model, result_path)
    ```


    ```python
    # model = get_model()
    # # with open(config_path, 'r', encoding='utf-8') as json_file:
    # #     loaded_model_json = json_file.read()
    # # model = model_from_json(loaded_model_json)
    # model.load_weights(check_point_path)
    # save_test_result(model, "trained_model_substract_1/multi_keras_bert_sohu_test_result_epoch6.csv")
    ```


    ```python
    batch_size = 2 * strategy.num_replicas_in_sync
    epochs = 10

    train_dataset_iterator = get_dataset("data/shuffle_total_file.tfrecord", epochs=epochs, batch_size=batch_size)
    train_step = get_step(train_dataset_len, batch_size)

    with strategy.scope():
        model = get_model()
        plot_model(model, "keras_bert_transformers_two_text_input_SubStract_bm25cosine_1.png", show_shapes=True)
        
        model.load_weights(check_point_path)

    for epoch in range(epochs):
        model.fit(
            train_dataset_iterator,
            steps_per_epoch=train_step,
            epochs=1,
            callbacks=[early_stopping, plateau, checkpoint, tensorboard_callback],
            verbose=1
        )

        model.save_weights(weights_path)

        save_test_result(model, f"{result_path}.epoch_{epoch}.csv")
    ```

        Some weights of the PyTorch model were not used when initializing the TF 2.0 model TFBertModel: ['cls.predictions.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.dense.weight', 'cls.predictions.decoder.weight', 'bert.embeddings.position_ids', 'cls.predictions.transform.LayerNorm.bias', 'cls.seq_relationship.bias', 'cls.seq_relationship.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.decoder.bias']
        - This IS expected if you are initializing TFBertModel from a PyTorch model trained on another task or with another architecture (e.g. initializing a TFBertForSequenceClassification model from a BertForPreTraining model).
        - This IS NOT expected if you are initializing TFBertModel from a PyTorch model that you expect to be exactly identical (e.g. initializing a TFBertForSequenceClassification model from a BertForSequenceClassification model).
        All the weights of TFBertModel were initialized from the PyTorch model.
        If your task is similar to the task the model of the checkpoint was trained on, you can already use TFBertModel for predictions without further training.


        Model: "model"
        __________________________________________________________________________________________________
        Layer (type)                    Output Shape         Param #     Connected to                     
        ==================================================================================================
        input_attention_mask_texta (Inp [(None, None)]       0                                            
        __________________________________________________________________________________________________
        input_ids_texta (InputLayer)    [(None, None)]       0                                            
        __________________________________________________________________________________________________
        input_token_type_ids_texta (Inp [(None, None)]       0                                            
        __________________________________________________________________________________________________
        input_attention_mask_textb (Inp [(None, None)]       0                                            
        __________________________________________________________________________________________________
        input_ids_textb (InputLayer)    [(None, None)]       0                                            
        __________________________________________________________________________________________________
        input_token_type_ids_textb (Inp [(None, None)]       0                                            
        __________________________________________________________________________________________________
        tf_bert_model (TFBertModel)     TFBaseModelOutputWit 102267648   input_attention_mask_texta[0][0] 
                                                                        input_ids_texta[0][0]            
                                                                        input_token_type_ids_texta[0][0] 
                                                                        input_attention_mask_textb[0][0] 
                                                                        input_ids_textb[0][0]            
                                                                        input_token_type_ids_textb[0][0] 
        __________________________________________________________________________________________________
        lambda (Lambda)                 (None, 768)          0           tf_bert_model[0][0]              
        __________________________________________________________________________________________________
        lambda_1 (Lambda)               (None, 768)          0           tf_bert_model[1][0]              
        __________________________________________________________________________________________________
        subtract (Subtract)             (None, 768)          0           lambda[0][0]                     
                                                                        lambda_1[0][0]                   
        __________________________________________________________________________________________________
        dot (Dot)                       (None, 1)            0           lambda[0][0]                     
                                                                        lambda_1[0][0]                   
        __________________________________________________________________________________________________
        input_bm25 (InputLayer)         [(None, 1)]          0                                            
        __________________________________________________________________________________________________
        input_tf_cosine (InputLayer)    [(None, 1)]          0                                            
        __________________________________________________________________________________________________
        input_tfidf_cosine (InputLayer) [(None, 1)]          0                                            
        __________________________________________________________________________________________________
        input_cat_texta (InputLayer)    [(None, 1)]          0                                            
        __________________________________________________________________________________________________
        input_cat_textb (InputLayer)    [(None, 1)]          0                                            
        __________________________________________________________________________________________________
        concatenate (Concatenate)       (None, 2310)         0           lambda[0][0]                     
                                                                        lambda_1[0][0]                   
                                                                        subtract[0][0]                   
                                                                        dot[0][0]                        
                                                                        input_bm25[0][0]                 
                                                                        input_tf_cosine[0][0]            
                                                                        input_tfidf_cosine[0][0]         
                                                                        input_cat_texta[0][0]            
                                                                        input_cat_textb[0][0]            
        __________________________________________________________________________________________________
        dense (Dense)                   (None, 256)          591616      concatenate[0][0]                
        __________________________________________________________________________________________________
        dense_2 (Dense)                 (None, 256)          591616      concatenate[0][0]                
        __________________________________________________________________________________________________
        dropout_37 (Dropout)            (None, 256)          0           dense[0][0]                      
        __________________________________________________________________________________________________
        dropout_39 (Dropout)            (None, 256)          0           dense_2[0][0]                    
        __________________________________________________________________________________________________
        dense_1 (Dense)                 (None, 32)           8224        dropout_37[0][0]                 
        __________________________________________________________________________________________________
        dense_3 (Dense)                 (None, 32)           8224        dropout_39[0][0]                 
        __________________________________________________________________________________________________
        dropout_38 (Dropout)            (None, 32)           0           dense_1[0][0]                    
        __________________________________________________________________________________________________
        dropout_40 (Dropout)            (None, 32)           0           dense_3[0][0]                    
        __________________________________________________________________________________________________
        output_A (Dense)                (None, 1)            33          dropout_38[0][0]                 
        __________________________________________________________________________________________________
        output_B (Dense)                (None, 1)            33          dropout_40[0][0]                 
        ==================================================================================================
        Total params: 103,467,394
        Trainable params: 103,467,394
        Non-trainable params: 0
        __________________________________________________________________________________________________
        None
        491/42179 [..............................] - ETA: 4:45:12 - loss: 0.2088 - output_A_loss: -0.0998 - output_B_loss: 0.3086 - output_A_accuracy: 0.4839 - output_B_accuracy: 0.7174


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


    ```python

    ```


    ```python

    ```


    ```python

    ```


    ```python

    ```

