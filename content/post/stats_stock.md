---
title: "时序分解股票数据并部署在微信公众号上"
date: 2020-07-20T23:30:06+08:00
lastmod: 2020-07-20T23:30:06+08:00
draft: false
keywords: []
description: ""
tags: [时序分解, 股票, tushare, 微信公众号]
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
  enable: false
  options: ""

sequenceDiagrams: 
  enable: false
  options: ""

---
## 目的

将股票价格进行时序分解，得到趋势图、周期图和误差图。然后放到微信公众号上，让用户输入"002581.SZ"等股票代码，即可自动回复以上的图片。

## 主要思路

1. 用tushare获得股票的历史数据。
2. 用statsmodels的STL进行时序分解。
3. 用matplotlib出趋势图、周期图和误差图 的三合一图片。
4. 将以上功能部署到树莓派上。
5. 树莓派利用花生壳作内网穿透，对外提供服务。
6. 树莓派上部署微信公众号服务器，对用户提供便捷服务。

## 输出的图片

![时序分解股票](/images/stats_stock.png)

## 进度

现在已完成使用tushare获取指定股票历史数据、使用statsmodels进行时序分解、使用matplotlib出三合一图。

但想要接上微信公众号，则必须要有一台公网80端口的服务器。准备用花生壳进行内网穿透，但64位的树莓派4好像装不了花生壳客户端。而经过一晚上实验，发现局域网内必须要有一台电脑运行着花生壳客户端，否则无法解析。

~~**好像在PC上运行花生壳客户端，也只能访问树莓派一会儿，之后还是不行。需要进一步验证，同时验证32位树莓派上同时运行花生壳客户端和服务器程序，能不能稳定解析。**~~

~~所以接下来准备用之前旧的树莓派搭个32位的arm服务器，把项目迁过去，运行花生壳客户端，同时运行服务器程序。~~

由于我家是电信光猫的内网ip，光猫只有一个lan口没有wifi，所以只能先接路由器，再接各设备。但这样花生壳的内网穿透会报"域名IP地址指向与转发服务器IP不一致"。解决方案是，先直连光猫，从 <http://192.168.1.1/romfile.cfg> 上下载配置文件，找到超级管理员telecomadmin和其密码，登上光猫后台，配置DMZ到路由器。然后再上路由器后台，配置DMZ到32位旧树莓派。

在服务器程序的一开始去执行"sudo phddns start"来运行花生壳客户端。另外，可以使用"sudo phddns status"来查看SN号，以此可以在花生壳网页管理台上看客户端在不在线。然后在花生壳网页管理台上配置域名映射。

忙了快一天，花生壳映射终于成功了！

另外，由于树莓派上官方的statsmodels版本太低，所以只能去git clone官方最新代码到本地，然后 python3 setup.py install 。再另外，由于github的速度极慢，所以只能先clone到国内的gitee上，再从gitee上clone到本地。

然后就是微信客户端的流程。应该是收到消息后，取股票数据，时序分解，出图，上传临时图片获取资源ID，发送资源ID给用户完成回复图片。整个流程必须控制在5秒内，否则微信服务器算超时。

使用 re.fullmatch 进行全匹配。

在非交互式后端，若只要matplotlib生成图片不需要显示窗口，需要:

```python
import matplotlib
matplotlib.use('Agg')
```

使用python的dict，作为cache，保存media_id。这样就可以不用每次查询都重新拉数据、出图、上传图片了。

- [x] 使用tushare获取指定股票历史数据
- [x] 使用statsmodels进行时序分解
- [x] 使用matplotlib出三合一图
- [x] 旧树莓派搭建32位的arm服务器，运行花生壳客户端
- [x] 项目迁移至旧树莓派32位的arm服务器，不用定时运行更新公网地址的代码了
- [x] 搭建微信客户端的流程，注意时限
- [x] 使用cache缓存media_id
- [x] 增加定时器去更新day和access_token

## 微信公众号

该项目已部署至个人微信公众号上，由一台旧32位树莓派充当服务器。

只需把 002581.SZ 这样的股票代码发送至公众号，即可获得该只股票股价的时序分解图，分别是趋势图、周期图和误差图。

![微信公众号二维码](/images/qrcode_for_gh_52186fb6ad9e_258.jpg)

## 代码开源地址

<https://github.com/zhangsheng377/stats_stock>

## docker

目前，已把该树莓派镜像部署至docker，这样就不用screen暴力的在后台运行及监控了。

<https://github.com/zhangsheng377/docker/tree/master/stats_stock>

## 打赏

如果各位觉得该项目帮到了您的话，还请不吝打赏，谢谢！

![支付宝收款二维码](/images/alipay_20200801211208.png)

![微信收款二维码](/images/money_weixin_20200719212002.png)
