---
title: "使用Hugo和Netlify建立静态博客，并托管在Github上"
date: 2019-12-30T23:56:52+08:00
draft: false
show_reading_time: true
---
## 使用Hugo建立静态博客

---------------------------------------------------------

1. sudo apt install hugo
2. hugo new site blog
3. cd blog/
4. git submodule add <https://github.com/budparr/gohugo-theme-ananke.git> themes/ananke
5. echo 'theme = "ananke"' >> config.toml
6. hugo new posts/my-first-post.md
    * 注意：直接这样生成的页面头，是带有 draft: true 标记的，要想正式发布需要去掉该标记。
    * PS：建议从第5步开始，就直接将theme里exampleSite里的文件拷出来，作为初始文件，后续在其上改动、新增，即可。
7. hugo server
8. 接下来就可以随便玩一会儿，反正都是用markdown写作的
9. 还可以研究研究配置文件configure.toml，以及主题theme的配置

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

## 日常写作

---------------------------------------------------------

1. 使用hugo new posts/xxx.md生成新页面之后(注意去除页面头的 draft: true 标记)，使用 hugo 在public文件夹里生成静态页面
2. 使用git上传代码
3. Netlify自动部署，更新网站

---------------------------------------------------------
---------------------------------------------------------

### 后记

尝试使用 NetlifyCms 的 one-click-hugo-cms ，但修改原始模板就要花费大量时间，并且 NetlifyCms 的后台功能并不像 WordPress 那样强大和方便，更像只是个用来写文章的网页编辑器。。。那我还不如用 Markdown 呢。。。

所以，现在还是换回了这个直接用 Hugo 生成的博客，毕竟我要的只是个可以方便快捷写一些技术博客的地方。

当然，或许以后可能会尝试通过加admin.html的方法来添加 NetlifyCms 后台吧～～
