---
title: "在linux上装中文字体"
date: 2020-07-22T01:28:50+08:00
lastmod: 2020-07-22T01:28:50+08:00
draft: false
keywords: []
description: ""
tags: [linux, 树莓派, ttf, python]
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

最近要在树莓派上用python matplotlib画图，但发现显示不了中文。一指定字体才发现，simhei字体没装。

## 解决方案

1. 从windows上拷贝simhei.ttf字体文件至树莓派(被我存了一份在GitHub上：<https://github.com/zhangsheng377/stats_stock/blob/master/simhei.ttf>)，存到 /usr/share/fonts 目录下，可新建文件夹。

2. 然后刷新字体：

```shell
sudo fc-cache -f -v
```

可从回显中检查有无载入刚才的字体文件。

3. 然后删除matplotlib字体缓存：

```python
import matplotlib as plt
plt.get_cachedir()
```

```shell
rm -rf /home/xxx/.cache/matplotlib
```

4. 然后在python中指定字体，即可正常显示中文：

```python
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
```

## 输出的图片

![时序分解股票](/images/stats_stock.png)
