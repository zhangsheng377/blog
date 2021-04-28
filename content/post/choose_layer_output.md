---
title: "获取模型的中间层输出"
date: 2021-04-29T01:52:58+08:00
lastmod: 2021-04-29T01:57:58+08:00
draft: false
keywords: []
description: ""
tags: [深度学习, 模型, 中间层, 输出, keras]
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

最近在疯狂搭模型，突然有人来问我要我模型的中间层输出的embedding，于是，我就研究了一下怎么获取模型的中间层输出。

## 代码实现

```python
# 载入模型
full_model = get_model()
full_model.load_weights("xxxx")

# 查看模型各层
full_model.layers
```

    [....]

```python
# 抽取中间层输出，组建新模型
model = Model(full_model.input, full_model.layers[3].output)

# 构建数据，获取中间层输出
data, label = next(data_iterator)
bert_output = model.predict([[data[0]], [data[1]]])
bert_output.shape
```

    (1, 768)
