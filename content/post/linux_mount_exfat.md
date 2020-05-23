---
title: "Linux挂载exfat格式U盘"
date: 2020-05-23T10:39:27+08:00
lastmod: 2020-05-23T10:39:27+08:00
draft: false
keywords: []
description: ""
tags: []
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

## 安装exfat组件

```shell
sudo apt install exfat-utils
```

## 查看U盘位置

```shell
sudo fdisk -l
```

## 挂载U盘

```shell
sudo mount.exfat-fuse /dev/sdb1 /mnt/usb
```

## 卸载U盘

```shell
sudo umount /dev/sdb1
```
