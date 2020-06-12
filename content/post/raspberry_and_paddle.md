---
title: "树莓派上使用paddle预训练模型"
date: 2020-05-17T01:47:08+08:00
lastmod: 2020-05-17T01:47:08+08:00
draft: false
keywords: []
description: ""
tags: [树莓派, paddle, python]
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

树莓派安装paddlelite;
x86电脑上安装paddlehub，并将paddlehub中的预训练模型转换为paddlelite格式，使之能在树莓派上运行。

## **树莓派编译安装paddlelite**

```shell
sudo apt install patchelf cmake
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 150

git clone https://github.com/PaddlePaddle/Paddle-Lite.git
cd Paddle-Lite
sudo ./lite/tools/build.sh \
  --build_extra=ON \
  --arm_os=armlinux \
  --arm_abi=armv7hf \
  --arm_lang=gcc \
  --build_python=ON \
  full_publish

cd build.lite.armlinux.armv7hf.gcc/inference_lite_lib.armlinux.armv7hf/python/install
sudo python3 setup.py install
```

具体源码编译时的参数说明，大家可参考：<https://paddlepaddle.github.io/Paddle-Lite/v2.2.0/source_compile/#本地编译直接在rk3399或树莓派上编译>

## **x86电脑上将paddlehub的预训练模型转为paddlelite格式**

### 安装paddlehub、paddlepaddle

```shell
python -m pip install paddlehub
python -m pip install paddlepaddle
```

### 测试paddlehub预训练模型

```python
import paddlehub as hub

senta = hub.Module(name="senta_gru")
test_text = ["这家餐厅很好吃", "这部电影真的很差劲"]

results = senta.sentiment_classify(texts=test_text, use_gpu=False, batch_size=1)

for result in results:
    print(result['text'])
    print(result['sentiment_label'])
    print(result['sentiment_key'])
    print(result['positive_probs'])
    print(result['negative_probs'])
```

### 下载opt转换工具

<https://paddlepaddle.github.io/Paddle-Lite/v2.2.0/model_optimize_tool/>

### 下载paddlehub预训练模型并转换

下载预训练模型压缩包：

```shell
hub download senta_gru
```

解压，放置在 saved_models/senta_gru 下，

转换模型：

```shell
./opt --model_dir=senta_gru/infer_model --valid_targets=arm --optimize_out_type=naive_buffer --optimize_out=saved_models/senta_gru
```

PS: 查看opt转换工具所支持的模型算子：

```shell
./opt --print_model_ops=true --valid_targets=arm --model_dir=senta_gru/infer_model
```

## **在树莓派上测试转换后的预训练模型**

```python
from paddlelite.lite import *

config = MobileConfig()
config.set_model_dir("model")

# (2) 创建predictor
predictor = create_paddle_predictor(config)
'''
......
'''
```

然后就出现段错误了。。。

-------------------------------------------

## **后记**

之后开始各种找原因，也想到可能是由于漏转换了什么而导致的。。。

也反复找了资料，但由于paddle还尚未稳定，甚至连官方网站里的demo都已过时，接口都已变动得消失不见了。。。

所以最终只好暂时放弃。

可能 paddlepaddle 和 paddlehub ，作为百度paddle框架的主力，迭代速度快、完成度也还行，属于比较可用。但其他的组件可能优先级就比较低了，文档也不同步，实在是造成了很大的困扰。

毕竟已经占用了一整天的时间了，还要抽空学习推荐系统，和完成我的與情预测股票项目，所以paddlelite项目就只能等百度paddle稳定了再启动吧。
