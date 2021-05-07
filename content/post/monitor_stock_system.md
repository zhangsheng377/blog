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

![量化股票交易系统](/images/量化股票监控系统.png)

简单说来就是：

1. 我们先用一个爬虫模块，监控数据库里存储的订阅股票信息，采用多线程的方式爬取所有订阅的股票数据，并把数据存储到数据库里，同时缓存一份当天的数据到redis里。
2. 数据库采取简单工厂设计模式提供生成接口，支持不同的数据库后端，在前端提供一致的服务。具体的后端数据库，包括 mongodb和sqlite等。
3. 算法策略模块负责，拉起多线程，对每一只股票，用每一种算法策略进行处理，将生成的报警点存储进redis缓存，若没有报警点则忽略。
4. 订阅分发模块负责监控每个股票的算法结果有无变化，若有，则从数据库中查找所有同时订阅该股票以及该算法策略的用户，并取出他们的微信id，主动发送报警。并且，报警信息中会利用该股票的历史数据，以及历史报警信息，出图，并存储至七牛云图床，让微信用户在打开该报警信息时，可以从最近的七牛云cdn访问到图片。![股票消息列表](/images/stock_message_list.png) ![股票消息](/images/stock_message.png)
5. 同时部署了微信服务器程序，负责处理微信用户发来的消息，包括绑定微信号、订阅股票、取消订阅股票、查询股票、查询已订阅的股票等功能。该服务器由于部署在家庭内网，所以还需要额外部署一个花生壳的phddns服务来提供内网穿透的能力。
6. 以上这些模块全部使用docker部署，并且为了统一管理，编写了docker-compose.xml，并采用portainer，来进行统一部署和管理。![容器列表](/images/container_list.png)
