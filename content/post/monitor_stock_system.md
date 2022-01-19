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

一直都想去搭一个量化交易的平台。从一开始搭建的 [與情预测股票系统](https://github.com/zhangsheng377/emotion-predict-stock) ，锻炼了爬虫技术、数据库技术和机器学习技术；到后来的 [股票价格时序分解系统](https://github.com/zhangsheng377/stats_stock) 锻炼了出图、docker部署、内网穿透和微信公众号后台技术。到现在，搭建出了一个 可以实时对股票进行量化指标监控，并通过微信公众号提醒的系统。

## 流程图

![股票监控系统——模块关系图](/images/股票监控系统——模块关系图.png)

简单说来就是：

1. 爬虫模块，监控数据库里存储的订阅股票信息，采用多线程的方式爬取所有订阅的股票数据，并把数据存储到数据库里，同时缓存一份当天的数据到redis里；并作为生产者，把股票数据更新的消息发送至mq。
2. 其中，数据库采取简单工厂设计模式提供生成接口，支持不同的数据库后端，在前端提供一致的服务。具体的后端数据库，包括 mongodb和sqlite等。
3. 算法策略模块作为mq的消费者，收到股票的更新消息后，会从redis中读取股票数据，并用每一种算法策略进行处理。若策略结果有更新，则将结果缓存进redis，同时作为生产者，将策略结果更新的消息发送至mq。
4. 推送模块从mq监听策略结果的更新，若有，则从redis读取策略结果，以及所有订阅该股票和该算法策略的用户，取出他们的微信id，主动发送报警。而报警信息中会利用该股票的历史数据，以及历史报警信息，画出图像，并存储至七牛云图床，让微信用户在打开该报警信息时，可以从最近的七牛云cdn访问到图片。![股票消息列表](/images/stock_message_list.png) ![股票消息](/images/stock_message.jpg)
5. 另外，还部署了微信服务器程序，负责处理微信用户发来的消息，包括绑定微信号、订阅股票、取消订阅股票、查询股票、查询已订阅的股票等功能。
6. 以上这些模块全部使用docker部署，并且为了统一管理，编写了docker-compose.xml，并采用portainer，来进行统一部署和管理。![容器列表](/images/container_list.png)

## 感想

其实，原来我的这个项目，为了图省事，是采用定时器轮询的方式，消息都是存在redis里。。。直到最近，才改成了用消息队列来传递。

这一改，最大的感受就是，变得微服务化了。消息队列带来的，不光是一个组件，或是交互方式上的改进，更多的，是编程思想的改变。

自从微服务化了，每个组件都只聚焦于自身的业务逻辑，代码也变得更简洁了。（再也不需要之前的一堆带锁定时器，代码相互调用，深耦合了）
