# 我在两个月前是如何开始学习AI的

原文链接：[How I started with learning AI in the last 2 months](https://hackernoon.com/how-i-started-with-learning-ai-in-the-last-2-months-251d19b23597?from=hackcv&hmsr=hackcv.com&utm_medium=hackcv.com&utm_source=hackcv.com)

所有人这段日子都十分的忙碌。我们的个人生活和职业生涯都是如此。除此之外，冉工智能之类的东西开始兴起，你了解到你的技能在接下来的两年即将过时。

当我关笔我的启动Zead,我猛然清醒。这就像错过一些非常重要的东西。

一个全栈开发人员的能力在变化的情况下应付不足。在接下来的两年内，没有Al技能，全栈就不是合格的全栈。

是时候采取行动了。于是我做了我现在迫切需要学习的行动——作为一个开发者去提高我的技术，改变我作为产品人员的心态以及我作为企业家的哲学观念，从而实现数据导向。

正如著名的风险资本家Spiros Margaris 和Al和芬科公司的思想领袖对我说的那样，如果一个创业公司只依赖尖端的人工智能和机器学习算法来竞争—这是不够的。Al在未来不是一个竞争优势而是一个基本要求。你有听说谁将自己使用电当作竞争优势的吗？

### 建立我的第一个神经网络

推荐去学习吴恩达在Coursera上的[课程](https://www.coursera.org/learn/machine-learning)。这是一个很适合入门的课程，但是我发现很难长时间保持清醒。不是说这个课程不好，而是我真的很喜欢在课堂上保持专注。我的学习模式一直都是这样的，所谓我想实现自己的神经网络。

我没有直接跳到神经网络，因为我认为有更好的方法去学。我试着去熟悉领域中的所有单词，这样我就可以说这种语言了。

第一个任务不是去学，而是去熟悉。

我来自纯JavaScript和Node.js的背景，我不想更换我当前的栈。因此，我搜索了一个简单的叫做nn的神经网络模块，并且用它实现了一个带有虚拟输入的与门。在[这个教程](https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1)的启发下，我选择了解决这样的问题：对于任意输入的三个X,Y,Z，输出都应该为X和Y。

```javascript
var nn = require('nn')
var opts = {
    layers: [ 4 ],
    iterations: 300000,
    errorThresh: 0.0000005,
    activation: 'logistic',
    learningRate: 0.4,
    momentum: 0.5,
    log: 100   
}
var net = nn(opts)
net.train([
    { input: [ 0,0,1 ], output: [ 0 ] },
    { input: [ 0,1,1 ], output: [ 0 ] },            
    { input: [ 1,0,1 ], output: [ 0 ] },
    { input: [ 0,1,0 ], output: [ 0 ] },
    { input: [ 1,0,0 ], output: [ 0 ] },
    { input: [ 1,1,1 ], output: [ 1 ] },
    { input: [ 0,0,0 ], output: [ 0 ] }
    ])
// send it a new input to see its trained output
var output = net.send([ 1,1,0]) 
console.log(output); //0.9971279763719718
```

多么幸福！

在我个人看来，这一步建立了我的自信心。当输出闪烁为0.9971时，我意识到网络学会了如何执行与操作，并且自己忽略了额外的输入。

这就是机器学习的要点。你给电脑程序一组数据，它以此调整自己的内部参数，不断的将从原始数据中观察到的误差减小，然后获得了处理新数据的能力。

这种方式，我到后面才知道，叫做[梯度下降](https://en.wikipedia.org/wiki/Gradient_descent)。

### 预充我的关于人工智能的知识

当我写出了我的第一个人工智能程序后我充满自信，我想知道我做为一个开发者还能运用机器学习做些什么。

- 我解决了几个监督学习的问题比如[分类](https://en.wikipedia.org/wiki/Logistic_regression)和[回归问题](https://en.wikipedia.org/wiki/Linear_regression)
- 我使用了[多变量线性回归](https://www.hackerearth.com/practice/machine-learning/linear-regression/multivariate-linear-regression-1/tutorial/)模型，通过有限的数据去预测哪个队伍会赢得IPL比赛。（预测结果不尽人意，但还是很酷）
- 我通过运行[谷歌机器学习云](https://cloud.google.com/products/machine-learning/)中的demo去了解如今的Al都能做哪些事。
- 我偶然发现了 AI Playbook,由一个受人尊敬的风险组织**Andreessen-Horowitz**提供的资源。它真的是开发者和企业家十分方便的资源。
- 我开始在Youtube上看Siraj Rawal的 很好的[节目](https://www.youtube.com/channel/UCWN3xxRkmTPmbKwht9FuE5A)，节目以深度学习和机器学习为中心。
- 我读了这篇令人敬畏的黑客午后文章，它讲述了硅谷的展示者是如何制作Not Hotdog应用程序(热狗识别应用程序)的。这是与我们生活很相近的一个机器学习例子。
- 我读了特斯拉的Ai指导Andrej Karpathy的文章，虽然我一点也不懂，这一点让我很头痛。但是，当我尝试了一些之后，我发现文章中的概念确实很有意义。
- 当我有了一点勇气后，我开始实践一些深度学习的详细教程（复制和粘贴），并尝试在我自己的电脑上进行训练和跑代码。很多时候，因为我的电脑没有GPU，训练时就得耗费大量的时间。

逐步的，我将我的工具从Javascript换成了Python,并且在我的windows系统上安装了Tensorflow。

整个过程都围绕着被动的吸收内容在我的脑海里形成结构，这样在我以后遇到问题的时候我可以很好的解决它。

正如 Steve Jobs所说，你只能连接向后看的点。

#### 登上聊天机器人的车

作为一名电影《Her》的死粉，我想要创建一个聊天机器人。我开始了挑战，打算在两个小时内运用Tensorflow创建一个聊天机器人。我在几天前的文章中介绍了这段旅程以及相关的商业需求。

幸运的是，这篇文章十分的有感染力，它在[TechlnAsia](https://www.techinasia.com/talk/built-chatbot-2-hours)、[CodeMentor](https://www.codementor.io/shivalgupta1/i-built-a-chatbot-in-2-hours-and-this-is-what-i-learned-be677twav)、[KDNuggets](http://www.kdnuggets.com/2017/09/chatbot-2-hours-what-i-learned.html)上都有报道。这对我而言是一个十分重要的时刻，因为我谷歌开始了写科技博客。但我认为这篇文章是我AI学习旅程中里程碑式的时刻之一。

这让我在Twitter和Linkedln上认识了很多朋友，我可以和他们深入的讨论人工智能的未来，还可以在我困惑的地方给我一些解答。我受到了一些项目的邀请，最棒的是，年轻的开发人员和AI的初学者开始问我如何开始学习AI。

这就是我写这篇文章的原因，我希望更多的人从我的学习旅程中能有帮助并开始他们的学习之旅。

所有的旅程最难的就是开始。



学习AI对我来说并不容易。我最开始是写Javascript的，我几乎一夜跳转到了Python,学习如何用它编程。当我的模型在我的电脑上不能跑，或者跑了经过几个小时的训练后，它的正确率仍然十分的低，我会十分生气，这感觉就一个队伍赢了板球比赛。学习AI不像学习一个Web框架。

这种技巧需要你这段在计算的内部发生了什么，这样你才能从输出的结果中找到你的代码和数据的重点。

AI不仅仅是一个课题，它是一个可以用于任何简单的回归问题的杀手的术语，它有一天会杀了我们。就像你刚进入的其他学科一样，你可能像要挑选一些你擅长的东西，比如计算机视觉和自然语言处理，或者上帝不允许的统治世界。

在和Atlantis 首都著名的AI、Fintech和密码学行业领袖Gaurav Sharma的谈话中，他和我说到：在人工智能时代，“变得聪明”将有着完全不同的意思。我们需要人们进行更高层次的创造，对工作有更强烈的热情。

你要对电脑如何突然学会如何自己做事情着迷。耐心和好奇事你必须坚持的两个原则。

这是一次长远的旅行。非常累、非常令人生气、非常耗费时间。但是这个旅行的好的部分是，就像其他的旅行一样，这都是一步一步的。

