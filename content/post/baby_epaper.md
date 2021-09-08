---
title: "彩色墨水屏早教机"
date: 2021-08-31T23:45:58+08:00
lastmod: 2021-08-31T23:45:58+08:00
draft: false
keywords: []
description: ""
tags: [墨水屏, 早教机]
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

在某次逛淘宝时，突然发现微雪家竟然有个4寸的彩色电子墨水屏！

这个是个好东西呀，就像是kindle，不伤眼呀，给宝宝用最好。所以就想着用这个墨水屏来做个早教机吧，就当是儿子的儿童节礼物了。(其实当时已经是6月20号了。。。)

## 用树莓派驱动墨水屏
![彩色墨水屏展示树莓](/images/IMG_20210620_012233.jpg)
```python
from PIL import Image, ImageDraw, ImageFont
from waveshare_epd import epd4in01f

epd = epd4in01f.EPD()
epd.init()

Himage = Image.open(pic_path)
Himage = Himage.transpose(Image.FLIP_LEFT_RIGHT)     #水平翻转
Himage = Himage.transpose(Image.FLIP_TOP_BOTTOM)    #垂直翻转
epd.display(self.epd.getbuffer(Himage))

epd4in01f.epdconfig.module_exit()
exit()
```

## 把图片转成7色
https://www.waveshare.net/wiki/4.01inch_e-Paper_HAT_(F)
由于我的PS到期了，所以我使用开源的GIMP：
1. 打开GIMP，导入需要转换的图片：
   ![彩色墨水屏展示树莓](/images/202109032322.png)
2. 裁剪图片，调整画布大小。最后缩放到640*400像素。
   ![彩色墨水屏展示树莓](/images/202109032337.png)
3. 在图像上右击，选择“图像”->“模式”->“索引”->选择之前导入的7色色板。
   ![彩色墨水屏展示树莓](/images/202109032346.png)
   ![彩色墨水屏展示树莓](/images/202109032353.png)
4. 可以放大一些看效果。最后导出成bmp即可。
   ![彩色墨水屏展示树莓](/images/202109032354.png)
   ![彩色墨水屏展示树莓](/images/202109032356.png)

## 用播放语音掩盖刷新时间
由于这款彩色墨水屏的刷新时间高达20秒。。。所以肯定要让它在刷新图像时干点什么才能等得不那么尴尬。。。
所以我就想到了用多线程来播放语音。。。
```python
from pygame import mixer

mixer.init()

class Display_pic_thread (threading.Thread):  # 继承父类threading.Thread
    def __init__(self, father, pic_path):
        threading.Thread.__init__(self)
        self.pic_path = pic_path
        self.father = father

    def run(self):  # 把要执行的代码写到run函数里面 线程在创建后会直接运行run函数
        self.father.display_pic(self.pic_path)

def display_pic_and_play_sound(self, bmp_path, mp3_path, music_time=None):
    thread1 = self.Display_pic_thread(self, bmp_path)
    thread1.start()
    mixer.music.load(mp3_path)
    while threading.activeCount() > 1:
        mixer.music.play()
        time.sleep(music_gap_time)
    display_over_time = time.time()
    while music_time and time.time()-display_over_time < music_time:
        mixer.music.play()
        time.sleep(music_gap_time)
```

## 实现按键控制前后图片播放和自动随机播放功能
![彩色墨水屏展示树莓](/images/IMG_20210703_195056.jpg)
为了有点交互功能，所以准备加入两个大按钮，分别对应上一张图片和下一张。但同时，也要支持自动随机播放功能。
```python
key_state = {KEY_LEFT: GPIO.HIGH, KEY_RIGHT: GPIO.HIGH}


def key_callback(channel):
    global last_press_time
    if (key_state[channel] == GPIO.LOW):
        key_state[channel] = GPIO.HIGH
    else:
        key_state[channel] = GPIO.LOW
    last_press_time = time.time()


# 在通道上添加临界值检测，忽略由于开关抖动引起的边缘操作
GPIO.add_event_detect(KEY_LEFT, GPIO.BOTH,
                      callback=key_callback, bouncetime=10)
GPIO.add_event_detect(KEY_RIGHT, GPIO.BOTH,
                      callback=key_callback, bouncetime=10)


class State(Enum):
    none = 0
    repeat = 1


state = State.none
last_press_time = time.time()
last_random_display_time = time.time()


if __name__ == '__main__':
    items = Items(data_path)

    while True:
        if key_state[KEY_LEFT] == GPIO.LOW:
            logging.info("KEY_LEFT low")
            items.display_up_pic()
            last_press_time = time.time()
        elif key_state[KEY_RIGHT] == GPIO.LOW:
            logging.info("KEY_RIGHT low")
            items.display_down_pic()
            last_press_time = time.time()

        if time.time()-last_press_time > random_display_start_time and time.time()-last_random_display_time > random_display_gap_time:
            logging.info("display_random_pic")
            items.display_random_pic()
            last_random_display_time = time.time()

```
![彩色墨水屏展示树莓](/images/wx_camera_1627186906454.jpg)

## 利用gitee实现ota远程更新
建立一个shell启动脚本，在里面git pull，这样就可以实现ota远程更新啦。
```shell
#!/bin/bash
git pull && python3 main.py
```

## 硬件添加ups不间断电源和锂电板
想做成手持型的，所以肯定要加入ups和电板。还要开关，不然就会一直开机到没电了。。。

## 用autocad给早教机外壳三维建模
![彩色墨水屏展示树莓](/images/IMG_20210705_000933.jpg)
![彩色墨水屏展示树莓](/images/IMG_20210718_232252.jpg)
![彩色墨水屏展示树莓](/images/IMG_20210719_001929.jpg)
![彩色墨水屏展示树莓](/images/IMG_20210723_004410.jpg)
![彩色墨水屏展示树莓](/images/IMG_20210725_001403.jpg)
![彩色墨水屏展示树莓](/images/IMG_20210725_010207.jpg)
![彩色墨水屏展示树莓](/images/IMG_20210725_014531.jpg)

## 淘宝购买模型3d打印服务
![彩色墨水屏展示树莓](/images/IMG_20210729_222301.jpg)

## 修复及组装
![彩色墨水屏展示树莓](/images/IMG_20210801_160901.jpg)
![彩色墨水屏展示树莓](/images/IMG_20210807_182123.jpg)
![彩色墨水屏展示树莓](/images/IMG_20210807_182158.jpg)
![彩色墨水屏展示树莓](/images/IMG_20210807_182234.jpg)
