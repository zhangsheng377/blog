---
title: "使用Hugo建立静态博客，并托管在Github Pages上"
date: 2022-11-19T23:56:52+08:00
draft: false
keywords: []
description: ""
tags: [hugo, github]
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

## 从netfily迁移到github pages

### netfily被封

其实，早在3年前，我刚建立新博客的时候，就是托管在netfily上的，本博客的第一篇内容 《使用Hugo和Netlify建立静态博客，并托管在Github上》 <https://www.zhangshengdong.com/post/hugo_and_netlify/> ，就是在写这件事。

但很快，netfily的登陆由于用到了google的脚本，被大陆封了。只是对于像我这样的老用户，只是不能登录而已，新文章依然可以自动从github仓库里同步发布，所以也就没在意。

可是两天前，我新写了一篇博客，本地hugo编译没问题，可是netfily却一直编译不成功。但即使翻墙，也无法登录查看原因。只好开始考虑寻找新的托管网站。

### 使用github pages

在知乎上对比了一圈托管网站，发现国内的大多要钱，或者设置复杂。最方便使用的，好像也就是github pages了。但这样一来，就没有了免费cdn，网页加载速度慢了许多。

## github流水线

### 自动化流程

github pages最好是设置一个名为username.github.io的仓库。但很不巧的是，我的博客源码之前就是保存在我的blog仓库中。而github pages生成的二级域名，对于网站好像不太友好。所以只能设计一个自动化流程：

1. 在源码blog仓库中，设置github action，自动编译静态网页；
2. 将编译出的public文件夹上传至新的BZ-coding组织的BZ-coding.github.io仓库；
3. 将BZ-coding.github.io仓库选择master分支自动发布github pages网页。

### hugo github action

在blog仓新增自动编译hugo的github action：

```yaml
# Sample workflow for building and deploying a Hugo site to GitHub Pages
name: Deploy Hugo site to Pages

on:
  # Runs on pushes targeting the default branch
  push:
    branches: ["master"]

  # Allows you to run this workflow manually from the Actions tab
  workflow_dispatch:

# Sets permissions of the GITHUB_TOKEN to allow deployment to GitHub Pages
permissions:
  contents: read
  pages: write
  id-token: write

# Allow one concurrent deployment
concurrency:
  group: "pages"
  cancel-in-progress: true

# Default to bash
defaults:
  run:
    shell: bash

jobs:
  # Build job
  build:
    runs-on: ubuntu-latest
    env:
      HUGO_VERSION: 0.102.3
    steps:
      - name: Install Hugo CLI
        run: |
          wget -O ${{ runner.temp }}/hugo.deb https://github.com/gohugoio/hugo/releases/download/v${HUGO_VERSION}/hugo_extended_${HUGO_VERSION}_Linux-64bit.deb \
          && sudo dpkg -i ${{ runner.temp }}/hugo.deb
      - name: Checkout
        uses: actions/checkout@v3
        with:
          submodules: recursive
      - name: Build with Hugo
        run: |
          echo "pwd is "
          pwd
          ls
          hugo --minify --verbose
      - name: Display
        run: |
          echo "github.workspace is ${{ github.workspace }}"
          ls ${{ github.workspace }}
          echo "github.workspace public is ${{ github.workspace }}/public"
          ls ./public
      - name: Deploy
        uses: peaceiris/actions-gh-pages@v3
        with:
          deploy_key: ${{ secrets.ACTIONS_DEPLOY_KEY }} # 这里的 ACTIONS_DEPLOY_KEY 则是上面设置 Private Key的变量名
          external_repository: BZ-coding/BZ-coding.github.io # Pages 远程仓库 
          publish_dir: "./public"
          keep_files: false # remove existing files
          publish_branch: master  # deploying branch
          commit_message: ${{ github.event.head_commit.message }}
```

### 自动上传至网页发布仓

在上面脚本的最后，加入了把编译结果文件上传至另一个仓库的代码：

```yaml
- name: Deploy
    uses: peaceiris/actions-gh-pages@v3
    with:
        deploy_key: ${{ secrets.ACTIONS_DEPLOY_KEY }} # 这里的 ACTIONS_DEPLOY_KEY 则是上面设置 Private Key的变量名
        external_repository: BZ-coding/BZ-coding.github.io # Pages 远程仓库 
        publish_dir: "./public"
        keep_files: false # remove existing files
        publish_branch: master  # deploying branch
        commit_message: ${{ github.event.head_commit.message }}
```

其中，需要把目标账号里的ssh公钥，对应的私钥，配置在blog仓库的secrets里。

并且，这边的上传，现在好像不能分成两个job，否则好像会public里为空。

### 配置自动发布网站和自定义域名

把网页发布仓BZ-coding/BZ-coding.github.io，里的github pages配置成master的根目录。

然后在BZ-coding账号里验证zhangshengdong.com域名之后，就可以在github pages配置的自定义域名里配上www.zhangshengdong.com了。还可以强制打开https。

### 解决自定义域名自动消失的问题

但是随后发现，每次网页代码更新后，BZ-coding.github.io的自定义域名就会消失。

经搜索，得知，要在生成的网站根目录里添加CNAME文件，文件内容就写你想自定义的域名就行，比如www.zhangshengdong.com 。
