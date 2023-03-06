---
title: "home_assistant控制surface充电与否"
date: 2023-02-23T22:49:58+08:00
lastmod: 2023-02-23T22:49:58+08:00
draft: false
keywords: [home_assistant, surface, 涂鸦]
description: ""
tags: [home_assistant, surface, 涂鸦]
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

最近我的surface电池又鼓涨啦。。。这都已经是第二次了。。。surface的电源管理真垃圾。。。。上一次我一直插电源当台式机用，按理说是不走电池的，但电池还是鼓胀了。。。

所以我就想着，能不能让电脑自动电量低于20%时充电，高于80%时断电呢？

## 涂鸦wifi开关控制充电电路

先前做了一下调研，涂鸦可以接入小米和home assistant，并且价格也低。

## HA接入涂鸦

由于我并不想额外再弄个树莓派一类的一直插电，毕竟目前就这一个surface充电的需求，所以我是部署了HA的docker。

这样的话，集成涂鸦依旧非常方便，直接在集成中搜索tuya，找一些教程，在涂鸦上注册以下开发者，在HA上填入token、用户名等，就把上文的wifi开关接入HA啦。

## surface电量上传HA

这一步其实是最难的。

### HASS.Agent

一开始我尝试了HASS.Agent，但是它自带的battery sensor读不到surface的电量。虽然它还可以自定义sensor，但是后来我已经开始尝试restful方案了，所以HASS.Agent就暂时放弃了。

### Rest

后来我在HA的集成里，发现了aRest插件，那这样我不就可以在surface自动运行一个自动上报电量的rest接口嘛。

#### HA里注册aRest设备

在HA的/config/configuration.yaml里加入：

```yaml
# Example configuration.yaml entry
sensor:
  - platform: arest
    resource: http://xxx:yyy
    monitored_variables:
      battery:
        name: battery
```

一开始发现怎么都添加不上，怒而查HA的源码，发现了他们的一个bug。逐提交pr：<https://github.com/home-assistant/core/pull/88631>

#### surface上的服务器端

```python
import json
import psutil
from flask import Flask
from gevent.pywsgi import WSGIServer

app = Flask(__name__)


@app.route("/", methods=['POST', 'GET'])
def hello_world():
    return hello_world1()


@app.route("/battery", methods=['POST', 'GET'])
def hello_world1():
    return json.dumps({
   "variables" : {
      "battery" : psutil.sensors_battery().percent,
   },
   "id" : "battery",
   "name" : "battery",
   "connected" : True
})

if __name__ == '__main__':
    http_server = WSGIServer(('0.0.0.0', yyy), app)
    http_server.serve_forever()
```

#### 启动批处理

```bat
cd zzz
call activate base
python server.py
@REM gunicorn -b 0.0.0.0:yyy server:app
```

#### 隐藏命令行窗口

```vbs
set ws=WScript.CreateObject("WScript.Shell") 
ws.Run "zzz\server.bat /start",0
```

最后，在运行里输入shell:startup，打开启动文件夹，把这个run.vbs的快捷方式放进去，就可以开机自启动啦。

## HA设置自动化

这个其实没啥好说的，就是在HA的自动化里，配置上数值状态sensor的条件，执行行为设置为wifi开关，就可以啦。

具体代码如下：
surface充电:

```yaml
alias: surface充电
description: ""
trigger:
  - platform: numeric_state
    entity_id: sensor.arest_sensor_battery
    below: 30
condition: []
action:
  - type: turn_on
    device_id: xxx
    entity_id: switch.surface_socket_1
    domain: switch
initial_state: true
mode: restart
```

surface停止充电:

```yaml
alias: surface停止充电
description: ""
trigger:
  - platform: state
    entity_id:
      - sensor.arest_sensor_battery
    to: unavailable
  - platform: numeric_state
    entity_id: sensor.arest_sensor_battery
    above: 85
condition: []
action:
  - type: turn_off
    device_id: xxx
    entity_id: switch.surface_socket_1
    domain: switch
initial_state: true
mode: restart
```
