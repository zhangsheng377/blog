---
title: "远程打印文件"
date: 2022-11-18T22:49:58+08:00
lastmod: 2022-11-18T22:49:58+08:00
draft: false
keywords: [远程, 打印机, 上传, lpr]
description: ""
tags: [远程, 打印机, 上传, lpr]
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
  enable: true
  options: ""

sequenceDiagrams: 
  enable: true
  options: ""

---

## 背景

之前弄了个网络打印机服务器cups，是可以添加打印机然后发起打印的，而手机、电脑却又只支持在局域网内添加打印机，所以一旦我在外面的话，就无法利用家里的打印机打印了。
