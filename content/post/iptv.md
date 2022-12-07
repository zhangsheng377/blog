---
title: "任意设备观看iptv"
date: 2022-12-06T22:49:58+08:00
lastmod: 2022-12-06T22:49:58+08:00
draft: false
keywords: [iptv]
description: ""
tags: [iptv]
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

最近眼瞅着就要过年了嘛，过年一家人总是要看春晚的。但我家的光猫，上网口跟iptv口是两条口，一旦把电视接到iptv口就看不了bilibili了。。。所以我家几乎从来都没接过iptv。。。

但过年总是要看春晚的，所以前两年我家就是在各大视频网站或cntv上看春晚。但除夕夜网络总是不太好。

所以就想着，能不能把光猫的iptv口也用上，让电视不用换网线，就既可以看b站，又可以看电视呢？（当然，不用网络直播源，不清晰又卡。。。）

## 获取电视机顶盒信息

把一直没用的机顶盒翻出来，直连接在光猫的iptv口上，将机顶盒连接电视，开机。

点击遥控器的设置键，要求输入密码。别慌，电信的是通用的10000 。别的品牌的百度一下也有。

然后我们调到设备信息的页签，记下IP地址、子网掩码、网关、主备DNS、有线MAC，无线MAC也可以记一下，不过好像没用。

然后就可以关机，把机顶盒继续收在柜子里啦。

## 同时接入网络口和iptv口

### 主流方案：单路复用

现在主流的方案就是修改光猫配置，将网络流量和iptv流量不绑定出端口，从而实现一条网线单路复用，同时传输网络流量和iptv。

### 我的方案：双wan口

虽然电信用一条光纤单路复用同时传输网络流量和iptv到家，但网线的带宽还是要远远小于光纤的。而且我家的网口只是千兆口，让我从光猫里出来继续单路复用，总感觉会影响网速。

那有没有不用单路复用的方案呢？当然是有的啦。我家的工控机软路由有6个网口呢。

所以我从原先的聚合lan口中，拆了一个网口，作为iptv的接入wan口。

在iKuai中配上wan2，选择基于物理网卡的混合模式，这样可以设置多个登录方式，从而让我们直连光猫。

1. 配置静态ip，192.168.1.x，网关写光猫的ip就行。这样家里的设备就可以在不同网段中，直连光猫了。此条线路命名为vwan_model。
2. 配置动态ip。这里要填写上面我们记录的机顶盒mac地址，因为iptv都是绑定设备的，其他mac地址分配不到ip。（当然，蒙对了说不定也行）至于这边貌似也可以填写同网段下的静态ip，但我觉得还是伪装mac来自动dhcp获取ip会更保险一些。此条线路命名为vwan_iptv。

## 伪装截取组播流

这时候，虽然我们的软路由通过伪装mac，同时接入了网络流和iptv流，但iptv使用的是组播协议，如果不运行组播协议的话那还是看不到电视。

所以我们在iKuai中打开IGMP代理，选择IGMPv3协议版本，上联端口选择我们刚才设置动态IP的线路vwan_iptv，下联端口选择lan1线路，这样家里的所有设备就都可以通过组播代理，获取到组播流了。

## 组播流转单播流

到上一步，我们获取到的依然是组播流，但是大多数家庭设备并不支持组播流，所以我们还需要把组播流转为单播流，才能无障碍的在任意设备观看iptv。

这里我们在iKuai路由器系统中设置UDPXY，信号源接口选择我们代理的线路vwan_iptv，服务端口（也就是给内网设备观看iptv服务使用的端口）随便写一个吧 。

## 获取iptv节目地址（组播订阅地址）

其实在上一步做完时，我们就已经可以看iptv了。但是，我们的串流地址写啥呢？

这其实是因为组播建立了连接之后，还需要订阅一个源，才会有数据传来。但我们不知道这个订阅地址是啥呀。。。

### 主流方案：机顶盒抓包

一般的主流方案就是，既然没有订阅地址，那直接对机顶盒抓包就行了。一般他们需要一个双端口网卡，我虽然有交换机，也可以进行抓包，但终究还是烦了点，所以只是当后备方案预备着。

### 我的方案：找现成的iptv源

那既然是抓包，就肯定有人抓过并共享了，而且iptv讲究稳定性，订阅地址肯定不会经常更换，所以我只需要找到人们共享出来的iptv源就行啦。

但尝试了几个失败后才发现，原来iptv并不是我想象中的全国一个巨大的组播树。而是借鉴了CDN的思路，先把组播流缓存到各个地级缓存服务器，再从该服务器向周边人群建立组播树。

所以我找了一个南京电信的iptv组播源 <https://github.com/shawze/IPTV> ，把里面的ip和端口改成我自己路由器的ip和设置的udpxy服务端口，得到m3u文件，比如：

```m3u
#EXTINF:-1, CCTV1
http://x.x.x.x:y/udp/239.49.8.19:9614
```

亲测好用。

## 播放器

### pc：vlc等

把上面的m3u节目单，即iptv源，拖进vlc的播放列表，再随便选择一个节目（频道）：
![vlc](/images/iptv_vlc.png)

### 安卓：vlc等

我用的小米8，也可以安装vlc哈。亲测很好用。而且发现可以多个不同的节目同时看，一个节目差不多要流量1.多MB，我在pc和手机上看不同的节目，总流量速度就是2.28MB/s。

### iphone：nplayer

![vlc](/images/iptv_iphone_0.jpg)

![vlc](/images/iptv_iphone_1.jpg)

### TV：Kodi

我用的是海信的电视，自己装了蚂蚁市场，里面直接就有Kodi，安装。

进入Kodi后，点击设置，选择Interface -> Skin，将font改为Arial based，不然中文显示的都是乱码。

如果需要将界面都给改成中文，可以进入插件，选择look and feel -> language，安装简体中文，选择确认切换语言，即可。

接下来，我们安装iptv客户端。

依然是进入插件页，选择PVR客户端，安装PVR IPTV Simple Client插件。然后进入插件配置，将m3u位置改为本地，然后选择nfs。。。

导入后，即可正常观看iptv啦。

![vlc](/images/iptv_tv.jpg)
