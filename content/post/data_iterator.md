---
title: "给深度学习模型构建数据迭代器"
date: 2021-03-04T23:44:58+08:00
lastmod: 2021-03-04T23:44:58+08:00
draft: false
keywords: []
description: ""
tags: [深度学习, 迭代器, 数据]
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
  enable: false
  options: ""

sequenceDiagrams: 
  enable: false
  options: ""

---

最近在学习keras框架，不得不感叹keras的确比pytorch好用。

那么，现在就来整理一下深度学习里最常用的数据迭代器的写法吧。

```python
# 数据文件一篇就是一个文件
def _read_file(filename):
    """读取一个文件并转换为一行"""
    with open(filename, 'r', encoding='utf-8') as f:
        s = f.read().strip().replace('\n', '。').replace('\t', '').replace('\u3000', '')
        return re.sub(r'。+', '。', s)

# 文章迭代器
def get_data_iterator(data_path):
    for category in os.listdir(data_path):
        category_path = os.path.join(data_path, category)
        for file_name in os.listdir(category_path):
            yield _read_file(os.path.join(category_path, file_name)), category

it = get_data_iterator(data_path)
print(next(it))
'''
('竞彩解析：日本美国争冠死磕 两巴相逢必有生死。周日受注赛事，女足世界杯决赛、美洲杯两场1/4决赛毫无疑问是全世界球迷和彩民关注的焦点。本届女足世界杯的最大黑马日本队能否一黑到底，创造亚洲奇迹？女子足坛霸主美国队能否再次“灭黑”成功，成就三冠伟业？巴西、巴拉圭冤家路窄，谁又能笑到最后？诸多谜底，在周一凌晨就会揭晓。日本美国争冠死磕。本届女足世界杯，是颠覆与反颠覆之争。夺冠大热门东道主德国队1/4决赛被日本队加时赛一球而“黑”，另一个夺冠大热门瑞典队则在半决赛被日本队3:1彻底打垮。而美国队则捍卫着女足豪强的尊严，在1/4决赛，她们与巴西女足苦战至点球大战，最终以5:3淘汰这支迅速崛起的黑马球队，而在半决赛，她们更是3:1大胜欧洲黑马法国队。美日两队此次世界杯进程惊人相似，小组赛前两轮全胜，最后一轮输球，1/4决赛同样与对手90分钟内战成平局，半决赛竟同样3:1大胜对手。此次决战，无论是日本还是美国队夺冠，均将创造女足世界杯新的历史。两巴相逢必有生死。本届美洲杯，让人大跌眼镜的事情太多。巴西、巴拉圭冤家路窄似乎更具传奇色彩。两队小组赛同分在B组，原本两个出线大热门，却双双在前两轮小组赛战平，两队直接交锋就是2:2平局，结果双双面临出局危险。最后一轮，巴西队在下半场终于发威，4:2大胜厄瓜多尔后来居上以小组第一出线，而巴拉圭最后一战还是3:3战平委内瑞拉获得小组第三，侥幸凭借净胜球优势挤掉A组第三名的哥斯达黎加，获得一个八强席位。在小组赛，巴西队是在最后时刻才逼平了巴拉圭，他们的好运气会在淘汰赛再显神威吗？巴拉圭此前3轮小组赛似乎都缺乏运气，此番又会否被幸运之神补偿一下呢？。另一场美洲杯1/4决赛，智利队在C组小组赛2胜1平以小组头名晋级八强；而委内瑞拉在B组是最不被看好的球队，但竟然在与巴西、巴拉圭同组的情况下，前两轮就奠定了小组出线权，他们小组3战1胜2平保持不败战绩，而入球数跟智利一样都是4球，只是失球数比智利多了1个。但既然他们面对强大的巴西都能保持球门不失，此番再创佳绩也不足为怪。',
 '彩票')
 '''

'''
经过一堆处理后...
'''

# 构建循环的数据迭代器
def get_handled_data_iterator(data_path):
    pad_sequences_iter = get_pad_sequences_iterator(data_path, sequences_max_length)
    while True:
        for pad_sequences, label_one_hot in pad_sequences_iter:
            yield pad_sequences, label_one_hot

# 构建批次迭代器
def batch_iter(data_path, batch_size=64, shuffle=True):
    """生成批次数据"""
    handled_data_iter = get_handled_data_iterator(data_path)
    while True:
        data_list = []
        for _ in range(batch_size):
            data = next(handled_data_iter)
            data_list.append(data)
        if shuffle:
            random.shuffle(data_list)
        
        pad_sequences_list = []
        label_one_hot_list = []
        for data in data_list:
            pad_sequences, label_one_hot = data
            pad_sequences_list.append(pad_sequences.tolist())
            label_one_hot_list.append(label_one_hot.tolist())

        yield np.array(pad_sequences_list), np.array(label_one_hot_list)

it = batch_iter(data_path, batch_size=2)
print(next(it))
'''
(array([[ 751,  257,  223, ...,  661,  551,    8],
        [ 772,  751,  307, ...,  296, 2015, 1169]]),
 array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]))
'''
```

之后就可以用

```python
model.fit_generator(batch_iter(data_path, batch_size=64),
                    steps_per_epoch,
                    epochs=100,
                    verbose=1,
                    callbacks=None,
                    validation_data=None,
                    validation_steps=None,
                    class_weight=None)
```

来训练模型啦~
