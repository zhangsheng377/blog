---
title: "docker log太占空间"
date: 2021-01-31T22:25:58+08:00
lastmod: 2021-01-31T22:25:58+08:00
draft: false
keywords: []
description: ""
tags: [docker]
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

最近发现服务器的空间总是不够，把exsi的所有资源都给了server虚拟机后，才一周磁盘就又被占满了。

## 查找大文件、大目录

从根目录开始，使用

```shell
du -h --max-depth=1
```

逐层查找大目录、大文件，最终定位到 /var/lib/docker/containers/ 这个目录占了44G。查看，发现是有个容器的log文件太大。

## 查看docker容器log大小的脚本

```shell
#!/bin/sh 
echo "======== docker containers logs file size ========"  

logs=$(find /var/lib/docker/containers/ -name *-json.log)  

for log in $logs  
    do  
        ls -lh $log
    done
```

```shell
chmod +x docker_log_size.sh

./docker_log_size.sh
```

## 限制docker的log大小

简单的删除log文件只是治标，过段时间还会生成这么多的，所以我们需要治本。

如果使用docker-compose，那么简单的在compose文件中加上

```xml
logging:
  driver: "json-file"
  options:
  max-size: "500m"
```

这个配置项，然后更新stack，即可。

或者，我们可以配置docker的全局设置：修改或新建 /etc/docker/daemon.json ：

```json
"log-driver":"json-file",
"log-opts": {"max-size":"500m", "max-file":"1"}
```

然后重启docker守护进程：

```shell
systemctl daemon-reload
systemctl restart docker
```

但需要注意，配置全局设置的方法只对新部署的容器生效。所以对于原有的容器，我们需要重新部署。
