---
title: "股票量化监控系统"
date: 2021-05-06T00:04:58+08:00
lastmod: 2021-05-06T00:04:58+08:00
draft: false
keywords: []
description: ""
tags: [股票, 量化, 监控, 推送, 微信, redis, 爬虫]
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
  enable: true
  options: ""

sequenceDiagrams: 
  enable: true
  options: ""

---

## 背景

一直都想去搭一个量化交易的平台。从一开始搭建的 [與情预测股票系统](https://github.com/zhangsheng377/emotion-predict-stock) ，锻炼了爬虫技术、数据库技术和机器学习技术；到后来的 [股票价格时序分解系统](https://github.com/zhangsheng377/stats_stock) 锻炼了出图、docker部署、内网穿透和微信公众号后台技术。到现在，搭建出了一个 可以实时对股票进行量化指标监控，并通过微信公众号提醒的系统。

## 流程图

```mermaid
graph TB
A[Apple]-->B{Boy}
A---C(Cat)
B.->D((Dog))
C==喵==>D
style A fill:#2ff,fill-opacity:0.1,stroke:#faa,stroke-width:4px
style D stroke:#000,stroke-width:8px;
```

```mermaid
flowchat
st=>start: 开始
e=>end: 结束
op1=>operation: 操作1 | past
op2=>operation: 操作2 | current
op3=>operation: 操作3 | future
pa=>parallel: 多输出操作4 | approved
cond=>condition: 确认？ | rejected

st->op1->cond
cond(true)->e	
cond(no)->op2(right)->op3->pa(path1,right)->op1
pa(path2,left) ->e
st@>op1({"stroke":"Blue"})@>cond({"stroke":"Green"})@>e({"stroke":"Red","stroke-width":6,"arrow-end":"classic-wide-long"})
```

