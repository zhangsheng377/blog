---
title: "使用Hugo和Netlify建立静态博客，并托管在Github上"
date: 2019-12-30T23:56:52+08:00
draft: false
keywords: []
description: ""
tags: [hugo, netlify, github]
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
## 使用Hugo建立静态博客

---------------------------------------------------------

1. sudo apt install hugo
2. hugo new site blog
3. cd blog/
4. ~~git submodule add <https://github.com/budparr/gohugo-theme-ananke.git> themes/ananke~~

   git submodule add <https://github.com/olOwOlo/hugo-theme-even> themes/even
5. ~~echo 'theme = "ananke"' >> config.toml~~

   echo 'theme = "even"' >> config.toml
6. hugo new post/my-first-post.md
    * 注意：直接这样生成的页面头，是带有 draft: true 标记的，要想正式发布需要去掉该标记。
    * PS：建议从第5步开始，就直接将theme里exampleSite里的文件拷出来，作为初始文件，后续在其上改动、新增，即可。
7. hugo server
8. 接下来就可以随便玩一会儿，反正都是用markdown写作的
9. 还可以研究研究配置文件config.toml，以及主题theme的配置

## 上传至Github

---------------------------------------------------------

1. git add .
2. git commit -a
3. git push

## 使用Netlify进行配置和管理博客

---------------------------------------------------------

1. 按照官方教程选择Github的项目
2. 自动部署
3. 设置域名CNAME解析
4. 配置域名DNS
5. 启用SSL证书
6. 启用IPV6
7. **使用高版本hugo时,需要在netlify的deploy上配置hugo版本号:HUGO_VERSION,0.68.3**

## 日常写作

---------------------------------------------------------

1. 使用hugo new post/xxx.md生成新页面之后(注意改掉页面头的 draft: true 标记)，使用 hugo 在public文件夹里生成静态页面
2. 使用git上传代码
3. Netlify自动部署，更新网站

---------------------------------------------------------
---------------------------------------------------------

### 后记

尝试使用 NetlifyCms 的 one-click-hugo-cms ，但修改原始模板就要花费大量时间，并且 NetlifyCms 的后台功能并不像 WordPress 那样强大和方便，更像只是个用来写文章的网页编辑器。。。那我还不如用 Markdown 呢。。。

所以，现在还是换回了这个直接用 Hugo 生成的博客，毕竟我要的只是个可以方便快捷写一些技术博客的地方。

当然，或许以后可能会尝试通过加admin.html的方法来添加 NetlifyCms 后台吧～～

---------------------------------------------------------
---------------------------------------------------------

### 2020.05.11后记

在把原pdf简历改成markdown格式时，发现ananke主题的about页面默认是居中对齐的，且该主题基本找不到修改对齐方式的地方(起码不像even主题那样有$$方式可以随处设置)。。。

所以，我就把主题换成了even。

然而，even主题不支持低版本hugo。我的树莓派只能手动从GitHub上下载安装。然而，我的部署网站Netlify竟然又编译失败。。。折腾了一天才发现，对于高版本的hugo，需要在netlify的deploy上配置hugo版本号:HUGO_VERSION,0.68.3才行。
