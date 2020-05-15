---
title: "Opencv判断颜色相似的图片"
date: 2020-05-16T02:50:01+08:00
lastmod: 2020-05-16T02:50:01+08:00
draft: false
keywords: []
description: ""
tags: []
categories: []
author: ""

# You can also close(false) or open(true) something for this content.
# P.S. comment can only be closed
comment: false
toc: false
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

<!--more-->

有一个项目，大体是要判断一下一篇文章内的配图突不突兀。

所以就从网上随便找了4张图：
![opencv_similar_color_pic_1](/images/opencv_similar_color_pic_1.png)
可以看出，前3张图片从颜色上、从阅读感受上，应该是相似的，而最后一张应该是不同的。

而当我们只对图片做缩放(为了跑得快)，然后用bgr通道出直方图算相似度时：
![opencv_similar_color_pic_2](/images/opencv_similar_color_pic_2.png)
却发现，只有第一张和第二张图片的相似度是大于0.5的，而第二、三张，以及第三、四张图片之间的相似度几乎都小于等于0.1。

于是，经过思考后我觉得，判断两张图片在颜色上相不相似，其本质在于判断其直方图分布的形状相不相似，而不应该考虑是偏左还是偏右、是偏亮还是偏暗。一个图像亮一点，但其实它们还是相似的。

基于这个思想，我先暴力的把BGR以及HLS，三个通道先相互独立的直接均衡化，验证了判断分布形状的可行性。但同时，发现相互独立的均衡化会导致对于不同图片的分辨能力降低。所以，由此推论出，应该是把亮度拉平均衡化，同时相关联的影响到其他通道的变化。

所以，最后想出的方案是：

1. 先把图片缩放至统一大小，提升运算速度。
2. 把图像从BGR通道转至HSV通道(经实验，HSV通道比HLS通道效果好)。
3. 把HSV中的V(明度)进行均衡化(equalizeHist)
4. 再把图像从HSV通道转回BGR通道，从而达到在均衡亮度的同时影响其他通道的目的。
5. 最后，利用BGR通道进行相似度计算，大于0.5的即可认为是相似。

![opencv_similar_color_pic_3](/images/opencv_similar_color_pic_3.png)
可以发现，经过处理后，第一、二张图片，以及第二、三张图片之间的相似度已经大于0.7，而第三、四张图片的相似度则只有0.4左右。已经达到了我们开始时的目标。

------------------------------------------------------

还有的不足之处：

1. 只对V通道的均衡进行了探寻，没有研究其他通道可能的关联。
2. 第三、四张图片经过处理后的相似度有点高，需要想办法降低。

------------------------------------------------------

附上代码：

    import cv2 as cv
    import numpy as np
    from matplotlib import pyplot as plt


    def create_rgb_hist(image):
        """"创建 RGB 三通道直方图（直方图矩阵）"""
        h, w, c = image.shape
        # 创建一个（16*16*16,1）的初始矩阵，作为直方图矩阵
        # 16*16*16的意思为三通道每通道有16个bins
        rgbhist = np.zeros([16 * 16 * 16, 1], np.float32)
        bsize = 256 / 16
        for row in range(h):
            for col in range(w):
                b = image[row, col, 0]
                g = image[row, col, 1]
                r = image[row, col, 2]
                # 人为构建直方图矩阵的索引，该索引是通过每一个像素点的三通道值进行构建
                index = int(b / bsize) * 16 * 16 + int(g / bsize) * 16 + int(r / bsize)
                # 该处形成的矩阵即为直方图矩阵
                rgbhist[int(index), 0] += 1
        plt.ylim([0, 10000])
        plt.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.3)
        return rgbhist


    def hist_compare(hist1, hist2):
        """直方图比较函数"""
        '''# 创建第一幅图的rgb三通道直方图（直方图矩阵）
        hist1 = create_rgb_hist(image1)
        # 创建第二幅图的rgb三通道直方图（直方图矩阵）
        hist2 = create_rgb_hist(image2)'''
        # 进行三种方式的直方图比较
        match1 = cv.compareHist(hist1, hist2, cv.HISTCMP_BHATTACHARYYA)
        match2 = cv.compareHist(hist1, hist2, cv.HISTCMP_CORREL)
        match3 = cv.compareHist(hist1, hist2, cv.HISTCMP_CHISQR)
        print("巴氏距离：%s, 相关性：%s, 卡方：%s" % (match1, match2, match3))


    def handle_img(img):
        img = cv.resize(img, (100, 100))
        img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        img[:, :, 2] = cv.equalizeHist(img[:, :, 2])
        img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
        return img


    img1 = cv.imread("1.jpg")
    img1 = handle_img(img1)
    cv.imshow("img1", img1)

    img2 = cv.imread("2.jpg")
    img2 = handle_img(img2)
    cv.imshow("img2", img2)

    img3 = cv.imread("3.jpg")
    img3 = handle_img(img3)
    cv.imshow("img3", img3)

    img4 = cv.imread("4.jpg")
    img4 = handle_img(img4)
    cv.imshow("img4", img4)

    hist1 = create_rgb_hist(img1)
    hist2 = create_rgb_hist(img2)
    hist3 = create_rgb_hist(img3)
    hist4 = create_rgb_hist(img4)

    plt.subplot(1, 4, 1)
    plt.title("hist1")
    plt.plot(hist1)
    plt.subplot(1, 4, 2)
    plt.title("hist2")
    plt.plot(hist2)
    plt.subplot(1, 4, 3)
    plt.title("hist3")
    plt.plot(hist3)
    plt.subplot(1, 4, 4)
    plt.title("hist4")
    plt.plot(hist4)

    hist_compare(hist1, hist2)
    hist_compare(hist2, hist3)
    hist_compare(hist3, hist4)

    plt.show()

    cv.waitKey(0)
    cv.destroyAllWindows()

