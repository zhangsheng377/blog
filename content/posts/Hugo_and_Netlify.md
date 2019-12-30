---
title: "Hugo & Netlify"
date: 2019-12-30T23:56:52+08:00
draft: true
---
## 使用Hugo建立静态博客

---------------------------------------------------------

1. sudo apt install hugo
2. hugo new site blog
3. cd blog/
4. git submodule add <https://github.com/budparr/gohugo-theme-ananke.git> themes/ananke
5. echo 'theme = "ananke"' >> config.toml
6. hugo new posts/my-first-post.md
7. hugo server -D
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

1. 使用hugo new posts/xxx.md生成新页面之后，使用hugo -D在public文件夹里生成静态页面
2. 使用git上传代码
3. Netlify自动部署，更新网站
