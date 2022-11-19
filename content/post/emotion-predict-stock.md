---
title: "與情预测股票"
date: 2020-04-05T16:33:37+08:00
draft: false
keywords: []
description: ""
tags: [python, ml, stock]
categories: [工程]
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
## 项目地址: <https://github.com/zhangsheng377/emotion-predict-stock>

## 已完成项

1. 使用爬虫爬取股民对于股票的评论。(目前是爬取雪球网上人们对于小米公司的评论)

2. 对人们的评论做情感分析，得到评论的情感得分。(使用工厂模式获取情感分析模型，目前使用的是snownlp)

3. 将爬取到的数据，以及情感得分，存入数据库。(使用工厂模式，目前对接的是monogodb)

---------------------------------------------------------

## 未完成项

1. 爬取每日股票的涨跌幅等，作为lable的数据。

2. 将每个发言人id作为key，当天发言的平均情感得分作为value；按天为粒度，将所有id的得分打成一行存入数据库。

3. 将前3天的数据拼成一行作为一条x，将历史数据放入时间序列预测模型(也可能采用传统机器学习，树模型xgboost之类的)，预测第二天股票的涨跌幅。

4. 对接微信公众号等(或微信机器人)，支持用户订阅股票，每日开盘前自动推送预测结果。
