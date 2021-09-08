---
title: "docker定义时区"
date: 2020-11-15T02:28:13+08:00
lastmod: 2020-11-15T02:28:13+08:00
draft: false
keywords: []
description: ""
tags: [docker, timezone]
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
## 起因

由于我们用的docker镜像基本都是utc时间，而中国是+8时区，所以在本地化时就很比较麻烦。这样，就需要一种可以定义docker时区的方法。

## linux如何控制时区

在 Linux 系统中，控制时区和时间的主要是两个地方：
> /etc/timezone 主要代表当前时区设置，一般链接指向/usr/share/zoneinfo目录下的具体时区。
>
> /etc/localtime 主要代表当前时区设置下的本地时间。

所以，我们只需要把这两个文件挂载到docker容器中，即可定义容器的时区了。

## docker容器启动命令

```shell
docker run -d -p 5000:5000 -v /etc/timezone:/etc/timezone:ro -v /etc/localtime:/etc/localtime:ro --restart always zhangsheng377/stats_stock
```

其中:

* -d 为后台运行
* -p 为开放的端口范围
* --restart always 为重启策略
* -v 为挂载
