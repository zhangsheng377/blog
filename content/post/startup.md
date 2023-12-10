---
title: "linux/windows开机自启动"
date: 2023-12-10T23:33:58+08:00
lastmod: 2023-12-10T23:33:58+08:00
draft: false
keywords: [linux, windows, 自启动]
description: ""
tags: [linux, windows, 自启动]
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

最近家里新装了一台电脑，配上3090显卡，准备当gpu服务器。所以就要装双系统，ubuntu的当服务器跑大模型，windows的装steam用来打游戏。

只是为了日常使用的方便，我把wol功能打开，grub默认直接进Ubuntu。但这样要想打游戏时，就必须要手动敲命令切换到grub的第二选项重启，有点麻烦。

所以准备用HomeAssistant来控制开关机和切换。试了半天，还是准备在服务器上自启动restful控制接口程序，HomeAssistant直接通过url控制。

## Ubuntu自启动

https://blog.csdn.net/feiying0canglang/article/details/124695749

1. 创建rc-local.service文件

```shell
sudo cp /lib/systemd/system/rc-local.service /etc/systemd/system
```

然后修改/etc/systemd/system/rc-local.service，在文件最下方添加如下两行：

```shell
[Install]   
WantedBy=multi-user.target   
Alias=rc-local.service
```

2. 创建rc.local文件

创建/etc/rc.local，里边写自己想要运行的命令。例：

```shell
#!/bin/sh
mount -t nfs 192.168.10.36:/volume1/nfs /mnt/nfs
ethtool -s eno1 wol g
/home/zsd/miniconda3/bin/python /home/zsd/gpu_server.py &>/home/zsd/gpu_server.log
exit 0
```

（wol的g模式时兼容性最好的，如果是d就说明disable）

（指明python可执行文件地址的方式可以用conda环境。）

给/etc/rc.local加上可执行权限

```shell
sudo chmod +x /etc/rc.local
```

3. systemctl命令

启动服务

```shell
sudo systemctl start rc-local.service
```

查看服务状态

```shell
sudo systemctl status rc-local.service
```

## windows自启动

1. 创建gpu_server.py
2. 创建gpu_server.bat

```bat
cd C:\ProgramData\gpu_server
call activate base
python gpu_server.py
@REM gunicorn -b 0.0.0.0:28866 gpu_server:app
```

3. 创建run.vbs

```vbs
set ws=WScript.CreateObject("WScript.Shell") 
ws.Run "C:\ProgramData\battery\server.bat /start",0
```

（这样就没有黑色cmd窗口了）

4. 创建启动快捷方式

win+r键弹出“运行”窗口，输入“shell:startup”，打开启动文件夹。

把刚才的run.vbs创建的快捷方式复制来，可以改名字。

这样，windows的自启动就弄好了，可以从任务管理器中看到。

## HomeAssistant的configuration.yaml

增加：

```yaml
sensor:
  - platform: arest
    # resource: http://192.168.10.62:28866
    resource: http://zhangsheng377.wicp.net/
    monitored_variables:
      battery:
        name: battery

switch:
  - platform: wake_on_lan
    name: "GPU_server"
    mac: "E8:9C:25:74:7F:31"
    host: "192.168.10.63"
    broadcast_address: "192.168.10.255"
    broadcast_port: "9"
    turn_off:
      service: shell_command.shutdown_gpu_server

command_line:
  - switch:
      name: "GPU_server_reboot"
      command_on: "curl -X GET http://192.168.10.63:28866/reboot.action"
  - switch:
      name: "GPU_server_reboot_to_other"
      command_on: "curl -X GET http://192.168.10.63:28866/reboot_to_other.action"

shell_command:
  shutdown_gpu_server: "curl -X GET http://192.168.10.63:28866/shutdown.action"

```
