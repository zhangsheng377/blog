---
title: "esxi虚拟机安装黑群晖,nfs挂载"
date: 2021-11-20T17:51:58+08:00
lastmod: 2021-11-20T17:51:58+08:00
draft: false
keywords: []
description: ""
tags: [esxi, 黑群晖, nfs]
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

之前用esxi安装虚拟机的时候，因为贪图性能，所以一直用的是厚置备硬盘格式。直到前不久，对我的服务器虚拟机清理了一堆冗余文件后，发现esxi不支持把磁盘缩小。。。

就想着，要规范一下虚拟机磁盘的使用了。准备以后msata盘上只保留系统盘，然后挂载sata盘上的nfs当数据盘。

所以，翻箱倒柜的找出来了一个大约是我初高中时候的sata机械硬盘，1t大小，先凑合用，双12再买固态。另外一点就是，试了一下esxi的共享磁盘，反正我的服务器虚拟机根本读不到。。。但是，单独开个虚拟机提供nfs有点浪费了，不如咱们再装个nas吧。

## nas选型和下载

说干就干，一开始其实是排除黑群晖的，因为不能升级嘛。所以最开始实验的是[u-nas](http://www.u-nas.cn/)，开源的，还不错。nfs还没试，反正smb和ftp都能连上。但有个问题是，连它自己的软件都找不到安装的nas啊。。。算了算了，还是试试久负盛名的群晖吧。

找了一圈，多数教程都是没来由直接放上网盘去下载。说实话，哪怕是引导盘我也不敢直接运行私人的镜像啊，就算非要不可，也得找个用的人多的吧。所以，一路往上溯源，发现了[XPEnology](https://xpenology.com/forum/)论坛。注册后，收到下载邮件，https://xpenology.com/forum/topic/12952-dsm-62-loader/?utm_source=newsletter_MailerLite&utm_medium=email_MailerLite&utm_campaign=welcome_to_the_xpenology_community&utm_term=2021-11-18，最新的就是2018年更新的了，唉，算啦算啦，将就试试吧。

选最新的1.04b版本的引导盘，对应6.2/6.21版本的ds918。下载，得到synoboot.img。

## 修改sn和mac

参考这篇教程https://post.smzdm.com/p/agd8l34w/。**不修改sn的话，以后群晖自带的花生壳会用不了。**

下载[OSFMount](https://www.osforensics.com/tools/mount-disk-images.html)这个软件，加载刚才的synoboot.img，

![](https://qnam.smzdm.com/202102/20/603077edd7c5a8400.jpg_e1080.jpg)

加载第一个15mb的分区。

![](https://qnam.smzdm.com/202102/20/6030785e09dc58700.jpg_e1080.jpg)

去掉readonly选项。

挂在成功后，后打开grub文件夹下的grub.cfg文件，

### 修改sn

要用https://github.com/xpenogen/serial_generator去生成sn，或者直接用1330LWN287476、1130LWN707137、1230LWN103694、1130LWN624465、1330LWN542483，任选其一。

### 修改mac

十六进制的，随便改吧。

### 修改sata参数

![](https://qnam.smzdm.com/202102/24/60352a8b842af5963.gif_e1080.jpg)

set sata_args='DiskIdxMap=0c00 SataPortMap=24'

(好像没有用。。。)

准备弄两个sata控制器，sata0上只放引导盘，sata1上才是数据盘。



修改完成后，保存文件，OFSMount软件上选择对应的盘符，dismount，就可以得到修改后的synoboot.img了。

## 转换引导文件

安装[StarWind V2V Converter](https://www.starwindsoftware.com/download-starwind-products#download)，以进出都是local的方式，转换synoboot.img，转成esxi的vmdk格式。得到\*.vmdk和\*-flat.vmdk。

## 安装黑群晖

在esxi创建新虚拟机。linux，其他或更高版本64位(其实估计无所谓，随便选个64位的就行)。

在选择配置时，删除硬盘、scsi、usb、光驱，再加上一个sata控制器，虚拟机选项中去掉uefi引导选项，上面可以选用efi。

然后点击刚才创建的虚拟机，继续编辑。

新增一个现有硬盘，选择刚才转换的vmdk文件(上传)，分配到sata0。

新增一个新硬盘，给500g吧，厚置备延迟置零，分配到sata1。

启动，选第三个启动项，vm的。

进入http://find.synology.com/。搜索较慢，可能要2-3分钟才能出来。搜到后，一路下一步，直到安装镜像。

一定要手动安装，提前选个6.2版本的918镜像下载下来。不然版本不对，安装会卡住或者无法重启。我用的是DSM_DS918+_23739.pat。

等安装自动重启后，就是正常的新建账号，注意，一定要选dms有更新手动安装。

跳过QuickConnect。

然后进入后，新增存储空间，选自定义，性能改善，RAID的类型为JBOD，不要选上50mb的引导盘，Btrfs格式。

可以启用自带的花生壳、Download Station，下载器可以配套上chrome插件[NAS Download Manager](https://chrome.google.com/webstore/detail/iaijiochiiocodhamehbpmdlobhgghgi)。

还可以启用VideoStation，貌似挺好，正在摸索。

## 开启nfs并挂载

### 开启nfs

在群晖的控制面板-文件服务中开启nfs。

在控制面板-共享文件夹中新增nfs文件夹，停用回收站，开启数据完整性检查，在这个文件夹的nfs权限中，服务器名写ip范围：192.168.10.250/24，启用异步、非特权端口、允许访问。记住装载路径：/volume1/nfs

### linux服务器上挂载nfs

sudo apt install nfs-common

df 可以查看有没有被挂载

mount -t nfs [Synology NAS IP 地址]:[共享文件夹装载路径] /[NFS 客户端装载点]

sudo mount -t nfs 192.168.10.36:/volume1/nfs /mnt/nfs

sudo chmod 777 /mnt/nfs

卸载nfs：

sudo umount /mnt/nfs