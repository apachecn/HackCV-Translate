# AlphaGo Zero: 从零开始学习

原文链接：[AlphaGo Zero: Learning from scratch](https://deepmind.com/blog/alphago-zero-learning-scratch/?from=hackcv&hmsr=hackcv.com&utm_medium=hackcv.com&utm_source=hackcv.com)

人工智能研究在语音识别和图片分类到基因学和药物发现等不同的领域取得了巨大的进步。在许多场景中，都有大量利用人们专业之和数据的专业的系统

然而，对于一些问题来说，利用人们的知识也许代价太大，太过不可靠，或者根本不可用。因此，人工智能长期的目标就是跨过这一步——创造在没有人为输入的情况下可以在绝大多数充满挑战性的领域里展现出超人一般的表现的算法。在[《自然杂志》](https://www.nature.com/)上发表的最新[论文](http://nature.com/articles/doi:10.1038/nature24270)中，我们展示了向这个目标迈出的重要一步。

#  从零开始

![img](https://storage.googleapis.com/deepmind-live-cms/images/AlphaGoZero-Illustration-WideScreen.width-320_oOByzmR.jpg)

这篇论文介绍了最新一代的AlphaGo产品AlphaGo Zero，第一个在古中国下棋游戏中打败了世界冠军的电脑程序。AlphaGo Zero甚至可以说是能力最强的并且按理说是历史上最强的下棋选手。

以前版本的AlpGo最初与成千上万的人类业余爱好者和专业游戏者训练来学习如何下棋。AlphaGoZero跳过了这一步骤并通过和自己下棋来学习，从最开始的胡乱下棋开始。通过这个方法，它很快的就超越了人类下棋的水平，并且以100比0的战绩打败了[之前发布](http://www.nature.com/nature/journal/v529/n7587/full/nature16961.html?foxtrotcallback=true)的打败了下棋冠军的AlpGo版本。

![Training time graphic](https://storage.googleapis.com/deepmind-live-cms/documents/TrainingTime-Graph-171019-r01.gif)

之所以能够这么做是因为它使用了[强化学习](https://en.wikipedia.org/wiki/Reinforcement_learning)的一种新的形式——AlpGoZero变成了他自己的老师。这个系统由一个完全不知道下棋游戏的神经网络系统开始。然后，通过将这个神经网络系统和强大的搜索算法相结合，他就会他自己和自己下棋。随着它不断地和自己下棋，神经网络不断被调整和升级来预测下一步动作，并最终成为游戏的赢家。

这个被更新过的神经网络之后又和搜索算法结合来创造新的更加强大版本的AlpGo Zero，并且这个过程还会再次开始。在每次迭代中这个系统的表现都会有微小的提升，并且自己玩游戏的质量也会有所增加，导致越来越多的准确的神经网络和甚至更加强大版本的AlphaGoZero。

这个技术比之前版本的AlpGo更加强大因为它不再局限于人们现有的知识。取而代之的是他能够从零开始向世界上最强的下棋选手:AlpGo他自己学习。

它还有其他一些有别于之前版本的显著的区别：

- AlpGoZero仅仅将棋盘上的黑棋白棋作为输入，而之前版本的AlpGo则是包括了少量的手工设计的特性
- 他使用一个神经网络而不是两个，之前版本的AlpGo使用的是policy network来选择下一步做什么然后使用value network来预测游戏赢家的每一个位置。这两个神经网络在AlpGoZero中被结合了，使得它能够更加有效的去训练和评估。
- AlphaGo Zero不使用“初赛”——其他围棋程序使用快速、随机的游戏来预测哪位棋手将在目前的棋盘位置上获胜。取而代之的它依赖于他自己高质量的神经网络来评估位置

所有这些差异都有助于提高系统的性能，使其更加通用。但是这正是算法的改变才使得系统更加的强力和高效。

![img](https://storage.googleapis.com/deepmind-live-cms/images/AlphaGo%2520Efficiency.width-400_cHoMue6.png)

AlphaGo的效率越来越高得益于硬件的进步和算法的优化。

经过三天的自我游戏训练，AlphaGo Zero以100比0的比分击败了之前发布的[AlphaGo](https://research.googleblog.com/2016/01/alphago master -ancient-game-of-go.html)。经过40天的自我训练，AlphaGo Zero变得更加强大，超过了被称为“Master”的AlphaGo版本，后者击败了世界上最好的选手和世界第一的棋手[柯洁](https://deepmind.com/research/alphago/alphago-china/)。

![img](https://storage.googleapis.com/deepmind-live-cms/images/Elo%2520Ratings.width-400_ahXVKga.png)

Elo评定-围棋等竞技游戏中玩家相对技能水平的的衡量方法-展示处理AlphaGo如何在它的发展中变得强大的。

AlphaGo在数以百万计的和自己比赛的过程中，系统逐渐地从零开始学习围棋，在短短几天内积累了数千年的人类知识。AlphaGo也发现了新的知识，自创了非传统的策略和创造性的新行为。与李世石(Lee Sedol)和柯洁(Ke Jie)的比赛中使用的新技术相呼应，并超越了后者。

![AlphaGo Zero knowledge timeline](https://storage.googleapis.com/deepmind-live-cms/documents/Knowledge%2520Timeline.gif)

这些创造性的时刻给了我们信心——人工智能将是人类创造力的倍增器，帮助我们完成我们自己的[任务](https://deepmind.com/about/)以解决人类正在面临的一些重要挑战



#  发现新的知识

![img](https://storage.googleapis.com/deepmind-live-cms/images/AlphaGoZero-Illustration-Square.width-320_RDH0108.jpg)

虽然现在还为时尚早，但AlphaGo Zero是迈向这一目标的关键一步。如果类似的技术可以应用于其他结构性问题，如蛋白质折叠、降低能源消耗或寻找革命性的新材料，那么由此产生的突破有可能对社会产生积极影响。

------

阅读 [这篇文章](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ)

阅读这篇文章相关的 [Nature News and Views article](https://www.nature.com/articles/550336a.epdf?shared_access_token=QbXlOw9nSIP_MS1moc_M0tRgN0jAjWel9jnR3ZoTv0PvinEKRXS2Dk736vL8i-Uo2-6AN8KRxOlLhDGorUgFzEgC3fwrX95r3LQ7u2FBwQ5axjmpMSZrWg4i6D7_g5rV5ze0zLhgo4jufsSKL-UZmw%3D%3D)

下载 [AlphaGo Zero games](http://www.alphago-games.com/)

阅读 [更多AlpGo的文章](https://deepmind.com/research/alphago/)

**\*这个作品是由David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, Yutian Chen, Timothy Lillicrap, Fan Hui, Laurent Sifre, George van den Driessche, Thore Graepel和Demis Hassabis完成的***