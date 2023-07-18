---
title: "在esxi虚拟机上部署docker"
date: 2023-07-19T00:50:58+08:00
lastmod: 2023-07-19T00:50:58+08:00
draft: false
keywords: [docker, esxi]
description: ""
tags: [docker, esxi]
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

我家的服务器是弄了一台工控机，装esxi，虚拟出软路由、nas和Ubuntu，最后在Ubuntu上部署了docker集群。

但这样就会有个问题：docker的storage-driver好像对于esxi的支持并不好。具体表现为：几乎每次重启或升级，docker-engine总是会无法启动：

```log
docker.service: Scheduled restart job, restart counter is at 3.
```

## 定位

### 常见查看原因方法

一般用systemctl status docker.service或另一个，来查看报错。

或者直接systemctl start docker.service来看看启动会报什么错。

### 命令行启动docker-engine

但是这次的报错里没有任何有价值的信息，所以我们使用终极方法————直接命令行启动docker-engine：

```bash
sudo /usr/bin/dockerd
```

```log
(base) zsd@server:~$ sudo /usr/bin/dockerd
INFO[2023-07-17T00:09:12.684718002+08:00] Starting up
INFO[2023-07-17T00:09:12.688119015+08:00] detected 127.0.0.53 nameserver, assuming systemd-resolved, so using resolv.conf: /run/systemd/resolve/resolv.conf
INFO[2023-07-17T00:09:12.742826565+08:00] [graphdriver] trying configured driver: devicemapper
WARN[2023-07-17T00:09:12.742866753+08:00] [graphdriver] WARNING: the devicemapper storage-driver is deprecated and will be removed in a future release; visit https://docs.docker.com/go/storage-driver/ for more information
WARN[2023-07-17T00:09:12.883829105+08:00] Usage of loopback devices is strongly discouraged for production use. Please use `--storage-opt dm.thinpooldev` or use `man dockerd` to refer to dm.thinpooldev section.  storage-driver=devicemapper
failed to start daemon: error initializing graphdriver: devmapper: Base Device UUID and Filesystem verification failed: devmapper: Current Base Device UUID:b04917d5-7376-4598-8292-3bcca66fdef8 does not match with stored UUID:535cbaab-d64c-45cd-ad7e-0060a121f2cb. Possibly using a different thin pool than last invocation: devicemapper
```

这回就能很清楚的看到，是docker的graphdriver，也就是storage-driver，设置的不对。

具体原因是esxi的磁盘的uuid变了。但我不可能每次都去改uuid，所以在把所有storage-driver选项都给试过后，唯一能用的就只有vfs了。

## 修改docker配置

```bash
sudo vim /etc/docker/daemon.json
```

```json
{
  "storage-driver": "vfs",
  "ipv6": true,
  "fixed-cidr-v6": "2001:db8:1::/64",
  "log-driver":"json-file",
  "log-opts": {"max-size":"10m", "max-file":"1"},
  "data-root": "/mnt/nfs/zsd_server/docker",
  "registry-mirrors": ["https://1elptswk.mirror.aliyuncs.com"]
}
```

这里可以发现，我把data-root给设置到了我挂在的nfs上，这样可以不占用系统盘。

## 用portainer管理集群

### 部署portainer

```bash
docker stop portainer
docker rm portainer
docker pull portainer/portainer-ce:latest
docker run -d -p 8000:8000 -p 9000:9000 -p 9443:9443 \
    --name=portainer --restart=always \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v portainer_data:/data \
    --log-opt max-size=10m \
    portainer/portainer-ce
```

### 用docker compose部署容器

我们把docker compose文件保存在github上，然后在portainer里使用github里的compose文件路径（需要指定新分支：refs/heads/main），来部署和更新容器。

但是这样有个问题，就是部署的时间太长，容器被创建了但没启动。

所以我们在portainer的stack页面里，添加环境变量：

```bash
DOCKER_CLIENT_TIMEOUT=9999
COMPOSE_HTTP_TIMEOUT=9999
```

即可缓解超时。

还有个方法：自己先用命令行，把涉及到的镜像都docker pull下来，即可加快部署速度。
