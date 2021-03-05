---
title: "esp32蓝牙通信"
date: 2021-02-21T22:01:58+08:00
lastmod: 2021-02-21T22:01:58+08:00
draft: false
keywords: []
description: ""
tags: [esp32, arduino, 蓝牙]
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
最近想做一个发热垫，可以用手机控制。

一开始思考过用wifi接入米家进行控制，这样还能使用语音助手。但后来仔细思索一番，发现使用场景不对。如果使用wifi连接，那意味着只能在室内使用了。

所以，最后还是决定直接使用蓝牙连接。

## 硬件选型

虽然选择了蓝牙连接，但为了以后扩展wifi方便，所以硬件选用了esp32，同时有wifi和蓝牙连接的功能，代码又兼容arduino，使用非常方便。

## 蓝牙连接方式

1. 初步设想是把硬件的mac地址生成二维码，手机软件扫描二维码获取mac地址，进行连接及发送温度设置等指令。
2. 后来发现，貌似可以直接用设备名进行蓝牙连接，如此一来便可以把所有的硬件设备都设置为相同的设备名，又可以省去二维码，着实不错。
3. 最后是在查资料时看到一种蓝牙广播的方式，不过尚未来及做实验，日后有机会倒可以试试。

## 温控方式

使用温敏电阻即可读取温度。

1. 最简单的温控可以是直接用继电器开关进行控制。设置温度的上下区间，加热到上区间停止，低于下区间则重启加热。
2. 高阶一点的是用pwm的方式调整发热电阻的功率，离目标温度越接近则功率越小，如此即可实现平滑温度曲线。甚至于再不行，还可上pid闭环控制算法，叠加上之前的误差，实时调整。

## 手机软件

由于我不会做安卓软件，现在只是使用一款“蓝牙串口”的app直接发送指令，控制硬件。

以后还是要学一下安卓，做一套架子出来。

## esp32程序

```c
//This example code is in the Public Domain (or CC0 licensed, at your option.)
//By Evandro Copercini - 2018
//
//This example creates a bridge between Serial and Classical Bluetooth (SPP)
//and also demonstrate that SerialBT have the same functionalities of a normal Serial

#include "BluetoothSerial.h"
#include <OneWire.h>
#include <DallasTemperature.h>

#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
#error Bluetooth is not enabled! Please run `make menuconfig` to and enable it
#endif

// 数据输出脚接开发板数字引脚
#define ONE_WIRE_BUS 17
#define PWM_PIN 16

BluetoothSerial SerialBT;
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);

char START_FLAG = '$';
char END_FLAG = '#';
int TEMPERATURE_MIN = 0;
int TEMPERATURE_MAX = 50;

int target_temperature = 0; // 1精度，*10存储
int current_temperature = 0; // 0.1精度，*10存储

// use first channel of 16 channels (started from zero)
#define LEDC_CHANNEL 0
// use 8 bit precission for LEDC timer
// 分辨率，即最大可写值为2^k-1，当为8时就是255
#define LEDC_TIMER_BIT 8
// use 5000 Hz as a LEDC base frequency
#define LEDC_BASE_FREQ 5000

void setup() {
  Serial.begin(115200);
  SerialBT.begin("ESP32test"); //Bluetooth device name
  sensors.begin();
  // Setup timer and attach timer to a pin
  ledcSetup(LEDC_CHANNEL, LEDC_BASE_FREQ, LEDC_TIMER_BIT);
  ledcAttachPin(PWM_PIN, LEDC_CHANNEL);
  Serial.println("The device started, now you can pair it with bluetooth!");
}

void SerialBT_sendMsg(String msg) {
  int i = 0;
  for (i = 0; i < msg.length(); i++) {
    SerialBT.write(msg[i]);
  }
}

int NONE = 0;
int START = 1;
int pre_status = NONE;
int num = 0;
void update_target_temperature() {
  if (SerialBT.available()) {
    char msg_char = SerialBT.read();
    if (msg_char == START_FLAG) {
      num = 0;
      pre_status = START;
    } else if (msg_char == END_FLAG && pre_status == START) {
      if (num >= TEMPERATURE_MIN && num <= TEMPERATURE_MAX) {
        target_temperature = num * 10;
        String msg = String("set temperature to " + String(target_temperature / 10.0) + "\n");
        SerialBT_sendMsg(msg);
      }
      num = 0;
      pre_status = NONE;
    } else if (isDigit(msg_char) && pre_status == START) {
      num = num * 10 + (msg_char - '0');
    } else {
      num = 0;
      pre_status = NONE;
    }
  }
}

void update_current_temperature() {
  sensors.requestTemperatures(); // 发送命令获取温度
  int tmp_temperature = int(sensors.getTempCByIndex(0) * 10);
  if (tmp_temperature != current_temperature) {
    current_temperature = tmp_temperature;
    Serial.print("Temperature for the device 1 (index 0) is: ");
    Serial.println(current_temperature / 10.0);
    SerialBT_sendMsg(String("temperature is " + String(current_temperature / 10.0) + "\n"));
  }
}

void set_heater_pwm() {
  int tmp_current_temperature = min(current_temperature, target_temperature);
  tmp_current_temperature = max(tmp_current_temperature, TEMPERATURE_MIN);
  int pwm = 0;

  if (target_temperature > tmp_current_temperature && target_temperature > TEMPERATURE_MIN) {
    pwm = 1.0 * (target_temperature - tmp_current_temperature) / (target_temperature - TEMPERATURE_MIN) * 255;
  }

  ledcWrite(LEDC_CHANNEL, pwm); // LEDC_TIMER_BIT分辨率设置为8时，最大duty为2^8-1=255
  Serial.print("pwm is: ");
  Serial.println(pwm);
}

void loop() {
  update_target_temperature();

  update_current_temperature();

  set_heater_pwm();

  delay(20);
}
```
