# **tensorflow**

原文链接：[tensorflow](https://github.com/nicodjimenez/nicodjimenez.github.io/blob/master/_posts/2017-10-08-tensorflow.markdown)

# 介绍

每隔几个月我就会向Google输入以下查询：“Tensorflow糟透了”或“f*** Tensorflow”，希望能在互联网上找到志同道合的人。 不幸的是，虽然Tensorflow已经存在了大约两年，但我仍然无法找到让Tensorflow完全满意的抨击。

虽然我认为我可能使用了错误的搜索引擎，但我认为这里有不同的力量：谷歌嫉妒。 被称为“谷歌深度嫉妒”的现象是世界各地工程师做出的以下假设：
* 在Google工作的人比你自己更聪明，更有能力
* 如果您了解Tensorflow，您可以在Google上获得深入的学习工作！ （留住深思熟虑的年轻人）
* 如果你平庸的创业公司使用Tensorflow而你在博客上谈论它的优点，谷歌可能会想要购买它
* 如果你没有“得到”Tensorflow的不直观设计，那你就是愚蠢的

让我们暂时把我们的假设抛在脑后，诚实对待Tensorflow。

当Tensorflow首次出现时，我们承诺结束了设计不良或维护不善的深度学习框架的无尽噩梦。（例如https://github.com/BVLC/caffe/issues）。 我们得到的是相当于Java的深度学习框架（一次编写，随处运行），但使用起来不那么有趣，并且使用纯粹的声明范式。呸。

出了什么问题？在尝试构建满足每个人需求的工具时，Google似乎构建了一款能够满足任何人需求的工作。

对于研究人员来说，Tensorflow很难学习并且难以使用。研究是关于灵活性的，而Tensorflow在深层次上缺乏灵活性。

想要提取神经网络中间层的值吗？您需要定义一个图形，然后使用作为字典传入的数据执行它，并且不要忘记将中间层添加为图形的输出，否则您将无法检索它们值。好吧，那比较麻烦，但它是可行的。

对于像我这样的机器学习从业者来说，Tensorflow也不是一个很好的选择。框架的声明性使调试变得更加困难。在您看到框架二进制文件有多大（20MB +），或者您尝试查看几乎不存在的C++文档，或者您想要执网络执行，在移动等低资源情况下非常有用。

# 与其他框架进行比较

Tensorflow的开发人员确实是深度学习超级巨星。然而，可能最广为人知和最受尊敬的Tensorflow的原始开发人员Yangquing Jia最近离开谷歌加入Facebook，他的Caffe2项目正在悄然崛起：（https://github.com/caffe2/caffe2/ graph / contributors，https：//github.com/caffe2/caffe2/issues）。与Tensorflow不同，Caffe2允许用户在一行代码中对一段数据执行一个层。基！

此外，Pytorch正迅速在顶级AI研究人员中受到欢迎。火炬用户，虽然通过编写Lua代码来执行简单的字符串操作来护理RSI伤害，但他们并没有成群结队地离开Tensorflow - 他们正在转向Pytorch。看起来Tensorflow对于顶级AI实验室来说还不够好。对不起，谷歌。

对我来说最有趣的问题是，尽管这种方法有明显的缺点，谷歌为Tensorflow选择了纯粹的声明范式。他们是否觉得将所有计算封装在一个计算图中会简化他们TPU上的执行模型，这样他们就可以从数百万美元的云端托管深度学习驱动的应用程序中削减Nvidia？这很难说。总的来说，Tensorflow并不像纯粹的开源项目那样。如果他们的设计合理，我会毫无问题。与美国谷歌开源项目（如Protobuf，Golang和Kubernetes）相比，Tensorflow大幅缩短。

虽然声明范式对于UI编程很有用，但是有很多原因使它成为深度学习的一个有问题的选择。

以React Javascript库为例，这是交互式Web应用程序的标准选择。在React中，数据流经应用程序的复杂性对于隐藏在开发人员中是有意义的，因为Javascript执行通常比对DOM的更新更快。 React开发人员不想担心状态如何传播的机制，只要最终用户体验“足够好”。

另一方面，在深度学习中，单个层可以逐字执行数十亿个FLOP！深度学习研究人员非常关注计算如何完成的机制，并希望得到精细控制，因为他们不断推动可能的边缘（例如动态网络）并希望轻松访问中间结果。

# 一个具体的例子

让我们看一个简单的示例，训练模型将其输入乘以3。

首先，让我们看一下Tensorflow示例：

```python
import tensorflow as tf 
import numpy as np 
X = tf.placeholder("float") 
Y = tf.placeholder("float") 
W = tf.Variable(np.random.random(), name="weight") 
pred = tf.multiply(X, W) 
cost = tf.reduce_sum(tf.pow(pred-Y, 2)) 
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost) 
init = tf.global_variables_initializer() 
with tf.Session() as sess: 
    sess.run(init) 
    for t in range(10000): 
        x = np.array(np.random.random()).reshape((1, 1, 1, 1)) 
        y = x * 3
        (_, c) = sess.run([optimizer, cost], feed_dict={X: x, Y: y}) 
        print(c)
```

现在让我们看一下做同样事情的Pytorch示例：

```python
import numpy as np
import torch from torch.autograd
import Variable
model = torch.nn.Linear(1, 1)
loss_fn = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
for t in range(10000):
    x = Variable(torch.from_numpy(np.random.random((1,1)).astype(np.float32))) 
    y = x * 3 
    y_pred = model(x) 
    loss = loss_fn(y_pred, y) 
    optimizer.zero_grad() 
    loss.backward()
    optimizer.step()
    print(loss.data[0])
```

虽然Pytorch示例代码更少一行，但操作更加明确，语法在训练循环中更紧密地遵循实际的学习过程：

1. 输入的正向传递
2. 产生损失
3. 计算渐变
4. Backprop

而在Tensorflow中，核心操作是一个神奇的sess.run调用。

为什么要编写更多的代码行来最终得到更难以理解和维护的东西？ Pytorch的界面客观上比Tensorflow好得多。它甚至都不是很接近。

通过Tensorflow，Google创建了一个同时处于低水平的框架，无法轻松地用于快速原型设计，但对于在前沿研究或资源有限的生产环境的使用，水平又显得太高。

说实话，当你有大约六个开源高级库在你已经高级的库之上构建以使你的库可用时，你会知道有些事情非常严重：

http://tflearn.org/
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim
https://github.com/fchollet/keras
https://github.com/tensorflow/skflow

注意：我会承认Tensorboard（Tensorflow的监控工具）是一个非常好的想法。如果您想为您的机器学习项目提供一个漂亮的监控解决方案，其中包括高级模型比较功能，请查看Losswise（https://losswise.com）。我开发它是为了让像我这样的机器学习开发人员能够将他们模型的性能从他们使用的任何机器学习库中分离出来，并实现我想要的Tensorboard不提供的许多很棒的功能。
