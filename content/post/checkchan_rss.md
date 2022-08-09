---
title: "利用check酱监控网页并生成rss"
date: 2022-08-09T23:33:58+08:00
lastmod: 2022-08-09T23:33:58+08:00
draft: false
keywords: []
description: ""
tags: [check酱, 监控网页, rss]
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

最近发现了一个很好玩的开源项目：Check酱。

<https://github.com/easychen/checkchan-dist>

它可以通过保存cookie的方式，监控任意网页上任意元素的变化。然后通过使用Server酱或webhook的方式，对这些元素的变化进行时时提醒。

## 部署Check酱实时监控

Check酱的本体是一个chrome插件，但如果只用插件的话，则只能在开机打开浏览器时进行监控。

好在官方提供了docker镜像，于是我首先使用Docker-compose的方式将其部署在我的服务器上。

```docker
version: '3'
services:
  chrome:
    image: easychen/checkchan:latest
    container_name: checkchan_checkchan
    restart: always
    logging:
      options:
        max-size: "10m"
    volumes:
      - /etc/timezone:/etc/timezone:ro
      - /etc/localtime:/etc/localtime:ro
      - "/xxx/data/checkchan:/checkchan/data"
    environment:
      - "CKC_PASSWD=xxx"
      - "VDEBUG=OFF"
      - "VNC=ON"
      #- "WIN_WIDTH=414"
      #- "WIN_HEIGHT=896"
      #- "XVFB_WHD=500x896x16"
      - "API_KEY=xxx"
      - "ERROR_IMAGE=NORMAL" # NONE,NORMAL,FULL
      #- "SNAP_URL_BASE=http://xxx:xx"
      #- "SNAP_FULL=0"
      - "TZ=Asia/Nanjing"
      #- "WEBHOOK_URL=" # 云端 Webhook地址，不需要则不用设置
    ports:
      - "x:x" # 远程桌面(VNC)
      - "x:x" # 远程桌面的Web界面(NoVNC)
      - "x:x" # 云端
```

于是，我用Check酱监控了b站up主的更新(主要是虽然b站有动态-投稿，可以只看关注的up主更新的视频，但等我发现这个功能时，我已经把2000个关注都点满了。。。)，以及京东商品的价格，还有微信公众号的更新。

这边借用官方的图：
![Check酱浏览器插件](https://github.com/easychen/checkchan-dist/blob/master/image/20220526194209.png)

然后配置Server酱，将监控信息发送到钉钉机器人通道：
![Check酱配置Server酱](https://github.com/easychen/checkchan-dist/blob/master/image/20220521224002.png)
![钉钉机器人通道](https://github.com/easychen/checkchan-dist/blob/master/image/20220521224356.png)
![钉钉机器人](/images/checkchan_dingding.jpg)

现在看起来一切都很完美，但很快我就发现cookie是有时效的，所以我隔几天就要去登录一下。这样显然就麻烦了。

## 万物皆可Rss

所以很快，我就发现了RssHub。

<https://docs.rsshub.app/>

<https://github.com/DIYgod/RSSHub>

它是一个由广大网友共同编写规则，从而可以将大多数你想要的东西（包括那些不支持rss的）都变成rss订阅源的工具。

所以，现在的思路就从直接监控网页，变成了监控rss的变化。（因为一般的rss阅读器提醒不是那么及时，并且我又喜欢把app都冰冻锁起来，所以我需要一个可以在聊天软件里提醒我的功能）

当然，由于一些不可说的原因，rsshub的官方站在大陆使用会有点不顺畅。不过好在这依然是个开源软件，所以我们继续用docker部署在服务器上：

```docker
version: '3'

services:
    rsshub:
        # two ways to enable puppeteer:
        # * comment out marked lines, then use this image instead: diygod/rsshub:chromium-bundled
        # * (consumes more disk space and memory) leave everything unchanged
        image: diygod/rsshub
        container_name: rsshub_rsshub
        restart: always
        logging:
          options:
            max-size: "10m"
        volumes:
          - /etc/timezone:/etc/timezone:ro
          - /etc/localtime:/etc/localtime:ro
        ports:
            - '1200:1200'
        environment:
            NODE_ENV: production
            CACHE_TYPE: redis
            REDIS_URL: 'redis://redis:6379/'
            PUPPETEER_WS_ENDPOINT: 'ws://browserless:3000'  # marked
        depends_on:
            - browserless  # marked

    browserless:  # marked
        image: browserless/chrome  # marked
        container_name: rsshub_chrome
        restart: always
        logging:
          options:
            max-size: "10m"
        volumes:
          - /etc/timezone:/etc/timezone:ro
          - /etc/localtime:/etc/localtime:ro

        ulimits:  # marked
          core:  # marked
            hard: 0  # marked
            soft: 0  # marked
```

所以当前的架构就是这样：

![架构0](/images/checkchan_flowchart.png)

我把网页的变化点做成rss发布，然后使用Check酱去监控rss的变化，一旦有变化就会通过Server酱向我的钉钉推送提醒；同时，我使用The Old Reader去订阅并管理我感兴趣的Rss源，这样当我的钉钉接收到提醒时，我就可以用安卓的FeedMe或网页端去查看更新的内容了，并且已读的条目也会双向同步到The Old Reader。

完美。

但。。。我老婆又向我提出了一个新的需求：她也想要去查看更新。。。

这个需求的麻烦点在于，我不想让她的已读动作影响到我。看似用两个The Old Reader账号就能解决，但这样我就需要每次修改订阅都要在两个账号同步修改，太麻烦了。。。

## 将监控动态发布成Rss

各种搜索解决方案，发现新版Check酱刚增加了一项功能，可以将监控的动态上传、发布成Rss。

简单说来就是：<https://github.com/easychen/checkchan-dist/issues/39>

当正常的php网站部署， 然后填上对应的rss_upload.php地址：

![上行地址](https://user-images.githubusercontent.com/27722028/183260354-623d6109-49c1-4551-836d-f20088b7fd63.png)

然后->“动态”， 点击"RSS“上传：

![Rss上传](https://user-images.githubusercontent.com/27722028/183260386-71177de9-3cb8-4721-82c7-7ee4149a7307.png)

最后打开https://example.com/rss.php 即可：

![Rss](https://user-images.githubusercontent.com/27722028/183260465-98fe0cbe-4ee3-4fa1-ba76-a6d0f2e627ac.png)

rss_upload.php:

```php
<?php

header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: GET, POST');
header("Access-Control-Allow-Headers: X-Requested-With");

if (isset($_POST['rss']) && strlen($_POST['rss']) > 0) {
    // sae
    // $kv = new SaeKV();
    // $kv->set('rss', $_POST['rss']);

    // 保存到文件
    file_put_contents("rss.xml", $_POST['rss']);
    //die("rss saved 202208072310");
    die($_POST['rss']);
} else {
    die("rss not found");
}

print_r($_POST);
```

rss.php:

```php
<?php

// $kv = new SaeKV();
// $rss = $kv->get( 'rss' );
$rss = file_get_contents("rss.xml");

header('Content-Type: application/rss+xml; charset=utf-8');
echo $rss;
```

利用这个功能，我就可以把Check酱监控的动态发布成Rss源，这样不论给多少人订阅都不会相互干扰了，而且监控的条目也方便管理，不用多次反复添加修改了。

![架构1](/images/checkchan_flowchart_1.png)

但是，大家肯定发现了一个问题，那就是Rss的上传是要手动触发的啊。那岂不是没有一点可用性？

幸好Check酱镜像支持了NoVNC，所以我通过NoVNC连接到服务器上部署的Check酱里，打开正在被自动化程序操控的浏览器，按F12进入开发者模式，在命令行中输入：

```js
setInterval(function () {document.querySelector("#root > div > div.solo-center.body.wrapped > div > div.side.md\\:show.hiden > div.mt-2 > div > button:nth-child(2)").click(); console.log('rss upload clicked.')},1000*60*10);
```

这段js代码，就可以让浏览器每隔10分钟，自动去点击一下上传按钮。关闭NoVNC也可以生效。

这下就全流程自动化了。

这是在iPhone的ReadOn软件上的效果：

![苹果](/images/checkchan_ios.jpg)

这是在安卓的FeedMe软件上的效果：

![苹果](/images/checkchan_android.jpg)
