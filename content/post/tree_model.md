---
title: "树模型"
date: 2021-03-16T02:09:58+08:00
lastmod: 2021-03-16T02:09:58+08:00
draft: false
keywords: []
description: ""
tags: [树模型, 决策树, ID3, C4.5, CART]
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

markup: mmark

---

# 树模型

在各种机器学习的算法中，树模型可以说是最贴近于人类思维的模型，一切的树模型都是一种基于特征空间划分的具有树形分支结构的模型。举个直观的例子：决策树。

## 决策树

决策树可以说是树模型中最基础，也是最有名的模型，它可以被认为是一堆if-then的规则集合。

![决策树](/images/c2cec3fdfc0392456a6ac4258694a4c27d1e2538.png)

比如这张图，人们该如何去判断是否应该出去玩。

1. 首先，我们可以看天气预报，如果是多云，那我们就可以出去玩；
2. 但如果是晴天，那我们还需要考虑湿度，如果湿度小于等于70%，也就是人体感觉舒适的湿度，那我们就可以出去玩，反之，则不行；
3. 那么另一种情况，下雨天，我们还需要去看是否在刮大风，如果已经起风了，则说明马上就要下雨了，我们也不能出去玩；但是如果还没起风，我们还是可以出去试试运气的，兴许是天气预报错了呢~

1. 看到这里，大家可能会觉得，决策树模型这么简单呀，就是人类的正常思维嘛。的确，决策树模型的原理就是这么的简单，但还有几个问题我们需要上升到理论的高度去解决：

* 上图，人类在推理的时候，所看到的数据特征其实有 天气预报，湿度，还有是否刮风等特征，那么我们该选择哪个特征去划分特征空间，即建立节点呢？这就是`特征选择`和`决策树生成`的问题。
* 还有，我们通过递归去划分特征空间，从而建立决策树的方式，很可能会导致过拟合的问题。一般来说，这就需要我们在建立好树之后，对已生成的树做自下而上的剪枝。那么如何剪枝，也是一个我们需要思考的问题。

   

### 特征选择及决策树生成

那么，我们首先来看特征选择问题。

如何选择一个特征进行划分，其实相当于是在讨论，用哪一个特征去划分特征空间，可以最有利于我们的分类，换言之，也就是降低了子特征空间的无序程度，也就是熵。

那么，熵既然表示的是无序的程度，它就肯定是和变量的分布有关系的。于是，我们假设某个特征$X$为$x_i$时的概率，为$p_i$：

$$
P(X=x_i) = p_i
$$

于是，我们就可以把特征$X$的熵定义为：

$$
H(X) = -\sum_{i=1}^n p_i log p_i
$$

所以，到此为止，一个给定数据集的熵我们就可以计算出来了(极大似然)。但是，我们选取一个特征来划分数据集，所要比较的其实是划分前后熵的变化，所以，我们就可以定义 信息增益为数据集$D$的熵，与 在给定特征$A$的条件下数据集D的熵之差，即：

$$
g(D,A) = H(D) - H(D|A)
$$

所以，我们最重要选取的特征，也就是给定哪个特征$A$，它的信息增益最大，那它就是我们要用来划分数据集的那个特征。



#### ID3算法

ID3算法，其实就是我们刚才所说的，每次选取信息增益最大的特征进行划分，递归地执行上述的步骤，直到所有特征的信息增益很小或者没有特征了为止。



#### C4.5算法

而我们刚才说的信息增益，其实有个小问题，就是它会偏向于去选择种类较多的那个特征。比如，一个特征里所有的取值都是唯一的，那么一旦选择这个特征进行划分，则可以在子特征空间中一下减少该特征一半的种类，但其实子特征空间里的该特征依旧全部是唯一的取值，即无序程度并没有减少。所以，C4.5算法就使用信息增益比，来校正这个问题。

$$
g_R(D,A) = {g(D,A)\over H_A(D)}
$$

即，信息增益比$g_R(D,A)$，为其信息增益$g(D,A)$与数据集D关于特征A的熵$H_A(D)$ 之比。

而其余建立树的步骤，C4.5算法和ID3算法一样。



#### CART算法

分类与回归树(classification and regression tree, CART)模型，该模型假设决策树是二叉树，其内部节点相当于是在做是否的判断。

CART算法是对回归树使用平方误差最小化准则，而对分类树使用基尼指数最小化准则，来进行特征选择，从而生成二叉树的。
$$
Gini(D) = 1 - \sum_{k=1}^K({{|C_k|}\over{|D|}})^2
$$

而基尼指数$Gini(D)$是用来表示样本集合D的不确定性。



### 决策树的剪枝

那么接下来我们来讲一下决策树的剪枝。

我们刚才说到，建立决策树的时候，我们是采用递归的方法，一直建立到不能继续了为止。但是，这样产生的树往往泛化能力比较弱。所以，我们需要考虑树的复杂度，对已生成的决策树进行简化，从而提高泛化能力。

$$
C_\alpha(T) = \sum_{t=1}^{|T|}N_tH_t(T) + \alpha|T|
$$

具体的方法呢，公式我就不细说了，因为要引入一些新的概念，公式也相对复杂一些，我怕深入讲下去会有人睡着。所以就大概讲一下剪枝的思路：

首先，用之前说过的熵，去定义一个决策树的损失函数；

计算出每个节点的熵，然后递归的从叶节点向上回缩，计算回缩前后树的损失函数；

从而得到损失函数最小的决策树。







