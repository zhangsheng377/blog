---
title: "在树莓派上搭建docker"
date: 2020-07-28T23:58:13+08:00
lastmod: 2020-07-28T23:58:13+08:00
draft: false
keywords: []
description: ""
tags: [树莓派, docker]
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
## 起因

总是听人说在树莓派上使用docker多好多好，近日，自己推演了一番，发现的确不错，起码隔离效果挺好。要知道，树莓派由于是arm架构，所以更多pip包只能以apt python3-xxx的形式安装在系统里(用venv会各种错，安装错、使用时错，痛不欲生)。所以，就想试试用docker隔离，把我之前的[时序分解股票](/post/stats_stock)，做成微服务放上去。

## 安装过程

1. 使用[清华源](https://mirrors.tuna.tsinghua.edu.cn/help/docker-ce/)，安装docker-ce。
2. 使用[阿里镜像](https://cr.console.aliyun.com/cn-hangzhou/instances/mirrors)，代理docker-hub。

## 安装监控

```shell
sudo docker run -d -p 9000:9000 --name portainer --restart always -v /var/run/docker.sock:/var/run/docker.sock -v portainer_data:/data portainer/portainer
```

## 添加用户进组

```shell
sudo usermod -aG docker $USER
```

把用户加进docker组里，之后新终端里就不需要用sudo docker了。

## 制作自己的base镜像

```dockerfile
FROM resin/rpi-raspbian:latest
ENTRYPOINT []

RUN rm /etc/apt/sources.list
COPY sources.list /etc/apt/sources.list
RUN rm /etc/apt/sources.list.d/raspi.list
COPY raspi.list /etc/apt/sources.list.d/raspi.list

RUN apt update && \
    apt upgrade
RUN apt install libcurl4

CMD ["/bin/bash"]
```

```shell
docker build -t zhangsheng377/raspberry_base .
```

## 进入自己制作的镜像的bash界面

若是上述dockerfile中没有加
> CMD ["/bin/bash"]

则需要在 docker run 时指定command:

```shell
docker run -ti zhangsheng377/raspberry_base /bin/bash
```

否则，只需:

```shell
docker run -ti zhangsheng377/raspberry_base
```

## 更新镜像

```dockerfile
FROM zhangsheng377/raspberry_base
ENTRYPOINT []

RUN apt install apt-utils

CMD ["/bin/bash"]
```

```shell
docker build -t zhangsheng377/raspberry_base -f Dockerfile .
```

## 未完待续
