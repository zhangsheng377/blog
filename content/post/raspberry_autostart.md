---
title: "树莓派自启动"
date: 2020-06-06T23:45:21+08:00
lastmod: 2020-06-06T23:45:21+08:00
draft: false
keywords: []
description: ""
tags: [树莓派, linux, shell]
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
## 新建服务文件
```shell
sudo nano /usr/lib/systemd/system/xx_net.service
```
---------------------------------------

```shell
[Unit]

Description=xx_net

[Service]

Type=oneshot

ExecStart=/home/pi/Desktop/XX-Net/start

[Install]

WantedBy=multi-user.target

```
---------------------------------------

## 指定服务自启动
```shell
sudo systemctl enable xx_net.service
```
