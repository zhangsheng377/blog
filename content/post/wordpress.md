---
title: "使用sqlite搭建wordpress博客"
date: 2024-01-03T01:41:58+08:00
lastmod: 2024-01-03T01:41:58+08:00
draft: false
keywords: [wordpress, blog, docker, sqlite]
description: ""
tags: [wordpress, blog, docker, sqlite]
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

最近正好我的域名到期续费，就想着干脆也买个宝宝名字的域名吧，先搭个博客把域名养起来。

## 使用sqlite搭建wordpress博客

说到博客，脑海里第一个想到的就是wordpress。不过还是本着货比三家的思想调研了一圈cms，发现20+年过去了，还是wordpress最好用（只是想要个让我老婆能随时随地随手写文章的地，所以我自用的hugo+github.io的模式第一个排除了）。

但是接下来就发现，wordpress到现在还是默认mysql做数据库的，但我一个已经跑docker集群的家用小服务器，实在不想运行那么重型的mysql了。

原本想的是复用我家的mongodb，但搜了一圈貌似没有现成的方案。只好想到了文件数据库sqlite。

主要搭建过程参考的是：<https://www.locol.media/blog/2023/09/14/running-wordpress-without-a-mysql-server-the-future-of-simplicity-and-portability/>

### sqlite插件

<https://github.com/WordPress/sqlite-database-integration>

### docker-compose

```yml
version: '3.1'
services:
  wordpress:
    image: wordpress:latest
    restart: always
    environment:
      - TZ=Asia/Shanghai
    volumes:
      - /mnt/nfs/zsd_server/docker/data/blog/wordpress_data:/var/www/html
    ports:
      - 80:80
    logging:
      options:
        max-size: "10m"
```

### 启动后手动修改

```shell
sudo mkdir /mnt/nfs/zsd_server/docker/data/blog/wordpress_data/wp-content/
sudo mkdir /mnt/nfs/zsd_server/docker/data/blog/wordpress_data/wp-content/plugins
sudo cp -r /home/zsd/sqlite-database-integration /mnt/nfs/zsd_server/docker/data/blog/wordpress_data/wp-content/plugins/
sudo cp /mnt/nfs/zsd_server/docker/data/blog/wordpress_data/wp-content/plugins/sqlite-database-integration/db.copy /mnt/nfs/zsd_server/docker/data/blog/wordpress_data/wp-content/db.php
sudo sed -i "s+{SQLITE_IMPLEMENTATION_FOLDER_PATH}+/var/www/html/wp-content/plugins/sqlite-database-integration+g" /mnt/nfs/zsd_server/docker/data/blog/wordpress_data/wp-content/db.php
sudo sed -i "s+{SQLITE_PLUGIN}+sqlite-database-integration/load.php+g" /mnt/nfs/zsd_server/docker/data/blog/wordpress_data/wp-content/db.php
sudo mkdir /mnt/nfs/zsd_server/docker/data/blog/wordpress_data/wp-content/database
sudo touch /mnt/nfs/zsd_server/docker/data/blog/wordpress_data/wp-content/database/.ht.sqlite
sudo chown -R www-data:www-data /mnt/nfs/zsd_server/docker/data/blog/wordpress_data/wp-content/database
sudo chmod -R 644 /mnt/nfs/zsd_server/docker/data/blog/wordpress_data/wp-content/database
sudo chmod 766 /mnt/nfs/zsd_server/docker/data/blog/wordpress_data/wp-content/database
sudo echo 'DENY FROM ALL' | sudo tee /mnt/nfs/zsd_server/docker/data/blog/wordpress_data/wp-content/database/.htaccess > /dev/null

sudo cp /mnt/nfs/zsd_server/docker/data/blog/wordpress_data/wp-config-sample.php /mnt/nfs/zsd_server/docker/data/blog/wordpress_data/wp-config.php
sudo sed -i '1i \
<?php \
putenv("WORDPRESS_DB_HOST=localhost"); \
putenv("WORDPRESS_DB_USER=not-used"); \
putenv("WORDPRESS_DB_PASSWORD=not-used"); \
putenv("WORDPRESS_DB_NAME=not-used"); \
?>' /mnt/nfs/zsd_server/docker/data/blog/wordpress_data/index.php
```

### 使用aliyun ddns解析域名

配合ikuai软路由，解析到家庭服务器的ipv6地址。

### wordpress开启缓存

装好后试了几把，发现有时网页打开会卡一会，估计还是家庭服务器（工控机）太弱了，所以就想着各种加速手段。

一开始尝试了wordpress官方的jetpack，但是启用设置时怎么都不成功。一查，有人说国内最好别打开jetpack，会导到国外的cdn去，更慢。

好吧，所以又查了一款也是官方的cache插件：WP Super Cache。安装需要手动改改，具体方案一查就有。总体来说还不错，感觉是快了些。
