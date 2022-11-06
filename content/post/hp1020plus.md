---
title: "hp1020plus网络打印机服务器"
date: 2022-11-06T22:49:58+08:00
lastmod: 2022-11-06T22:49:58+08:00
draft: false
keywords: [hp1020plus, 打印机, cups, foo2zjs]
description: ""
tags: [hp1020plus, 打印机, cups, foo2zjs]
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

之前就想着，弄个打印机，一方面，我们自己平常可以打印些文件；另一方面，等宝宝长大了，给他打印试卷、错题啥的，也是极好的。

原本趁着双11，在网上看了hp的136w和奔图的一款网络打印机，激光的，带复印、扫描功能，价钱1000左右，觉得稍微有些小贵，但也还好。

后来正好跟我父母提了一嘴，他们表示，家里还有一台旧打印机，他们也不用了，让我把它搬到新家去。我想想也行，还省钱了，就搬了来。一看，是hp1020plus，没有网络功能，所以为了方便使用，需要做一个网络打印机服务器，让手机和电脑可以添加远程打印机，直接打印。

## 传统cups+foo2zjs方案

其实，之所以这回直接把旧打印机搬来用，一点都不慌的原因是，我早在几年前，就跟这台打印机做过网络服务器。

![xx](/images/20221106234123.jpg)

当年，我才用的方案就是cups+foo2zjs驱动。

于是，先来试一遍当年的方案。

先sudo apt install cups，再wget -O foo2zjs.tar.gz http://foo2zjs.rkkda.com/foo2zjs.tar.gz

咦，怎么下载不下来？用浏览器试试。额，http://foo2zjs.rkkda.com/ 这个网站都已经被卖掉啦？。。。

于是，之能从github上找到别人的镜像仓，然后正常编译、测试。额，网页可以打开，任务也能正常下发，但打印机就是没反应。。。

之后，按照动作的排列组合试了很多种方案，都不行。突然灵光一现，发现原有在/dev/usb下的lp0端口不见了。怀疑可能是热插拔插件导致的，但又排列组合了一堆方案，可还是不对。。。

## hplip+‘printer-driver-foo2zjs’方案

又在上文的一系列测试中，发现了hplip的存在。寻思着，既然有官方的驱动软件了，那干嘛不用呢。

于是，在把系统清理干净后，先sudo apt install hplip，发现里面自带了cups。但不幸的是，cups直接添加打印机，执行打印任务，报驱动失败的错误。

但这回有报错就简单多了，把报错信息上网搜一下，发现是要装一下hplip的插件：hp-plugin -i。安装后再试一下，这回现象又回去了，即没有报错，但打印机没反应。好吧，看来还是要从foo2zjs驱动入手。

在尝试各种编译foo2zjs，都没效果后，终于发现了一套可行的方案：

首先用hp-check，把缺少的依赖，能装上的都装上；

然后，不要自己编译foo2zjs，直接在apt里搜索foo2zjs，发现，哈，竟然有两个foo2zjs的包：printer-driver-foo2zjs、printer-driver-foo2zjs-common。于是，直接装上这两个包再试试，哈，成功啦！

## 效果

服务器网页上：
![xx](/images/20221107000734.png)

pc上添加远端打印机：
![xx](/images/20221107000841.png)

苹果、安卓手机上直接打印：
![xx](/images/20221107001029.jpg)
