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
- AlphaGo Zero does not use “rollouts” - fast, random games used by other Go programs to predict which player will win from the current board position. Instead, it relies on its high quality neural networks to evaluate positions.

All of these differences help improve the performance of the system and make it more general. But it is the algorithmic change that makes the system much more powerful and efficient.

![img](https://storage.googleapis.com/deepmind-live-cms/images/AlphaGo%2520Efficiency.width-400_cHoMue6.png)

AlphaGo has become progressively more efficient thanks to hardware gains and more recently algorithmic advances

After just three days of self-play training, AlphaGo Zero emphatically defeated the previously [published version of AlphaGo](https://research.googleblog.com/2016/01/alphago-mastering-ancient-game-of-go.html) - which had itself [defeated 18-time world champion Lee Sedol](https://deepmind.com/research/alphago/alphago-korea/) - by 100 games to 0. After 40 days of self training, AlphaGo Zero became even stronger, outperforming the version of AlphaGo known as “Master”, which has defeated the world's best players and [world number one Ke Jie](https://deepmind.com/research/alphago/alphago-china/).

![img](https://storage.googleapis.com/deepmind-live-cms/images/Elo%2520Ratings.width-400_ahXVKga.png)

Elo ratings - a measure of the relative skill levels of players in competitive games such as Go - show how AlphaGo has become progressively stronger during its development

Over the course of millions of AlphaGo vs AlphaGo games, the system progressively learned the game of Go from scratch, accumulating thousands of years of human knowledge during a period of just a few days. AlphaGo Zero also discovered new knowledge, developing unconventional strategies and creative new moves that echoed and surpassed the novel techniques it played in the games against Lee Sedol and Ke Jie.

![AlphaGo Zero knowledge timeline](https://storage.googleapis.com/deepmind-live-cms/documents/Knowledge%2520Timeline.gif)

These moments of creativity give us confidence that AI will be a multiplier for human ingenuity, helping us with [our mission](https://deepmind.com/about/) to solve some of the most important challenges humanity is facing.



#  Discovering new knowledge

![img](https://storage.googleapis.com/deepmind-live-cms/images/AlphaGoZero-Illustration-Square.width-320_RDH0108.jpg)

While it is still early days, AlphaGo Zero constitutes a critical step towards this goal. If similar techniques can be applied to other structured problems, such as protein folding, reducing energy consumption or searching for revolutionary new materials, the resulting breakthroughs have the potential to positively impact society.

------

Read [the paper](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ)

Read the accompanying [Nature News and Views article](https://www.nature.com/articles/550336a.epdf?shared_access_token=QbXlOw9nSIP_MS1moc_M0tRgN0jAjWel9jnR3ZoTv0PvinEKRXS2Dk736vL8i-Uo2-6AN8KRxOlLhDGorUgFzEgC3fwrX95r3LQ7u2FBwQ5axjmpMSZrWg4i6D7_g5rV5ze0zLhgo4jufsSKL-UZmw%3D%3D)

Download [AlphaGo Zero games](http://www.alphago-games.com/)

Read [more about AlphaGo](https://deepmind.com/research/alphago/)

**\*This work was done by David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, Yutian Chen, Timothy Lillicrap, Fan Hui, Laurent Sifre, George van den Driessche, Thore Graepel and Demis Hassabis.***