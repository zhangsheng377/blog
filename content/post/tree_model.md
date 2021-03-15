---
title: "树模型"
date: 2021-03-16T02:09:58+08:00
lastmod: 2021-03-16T02:09:58+08:00
draft: false
keywords: []
description: ""
tags: [树模型, 决策树]
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

# 树模型

​		在各种机器学习的算法中，树模型可以说是最贴近于人类思维的模型，一切的树模型都是一种基于特征空间划分的具有树形分支结构的模型。举个直观的例子：决策树。

## 决策树

​		决策树可以说是树模型中最基础，也是最有名的模型，它可以被认为是一堆if-then的规则集合。

![决策树](/images/c2cec3fdfc0392456a6ac4258694a4c27d1e2538.png)

比如这张图，人们该如何去判断是否应该出去玩。

1. 首先，我们可以看天气预报，如果是多云，那我们就可以出去玩；
2. 但如果是晴天，那我们还需要考虑湿度，如果湿度小于等于70%，也就是人体感觉舒适的湿度，那我们就可以出去玩，反之，则不行；
3. 那么另一种情况，下雨天，我们还需要去看是否在刮大风，如果已经起风了，则说明马上就要下雨了，我们也不能出去玩；但是如果还没起风，我们还是可以出去试试运气的，兴许是天气预报错了呢~

​       看到这里，大家可能会觉得，决策树模型这么简单呀，就是人类的正常思维嘛。的确，决策树模型的原理就是这么的简单，但还有几个问题我们需要上升到理论的高度去解决：

1. 上图，人类在推理的时候，所看到的数据特征其实有 天气预报，湿度，还有是否刮风等特征，那么我们该选择哪个特征建立根节点呢？这就是特征选择问题。
2. 还有决策树的生成，
3. 及决策树的修剪的问题。
