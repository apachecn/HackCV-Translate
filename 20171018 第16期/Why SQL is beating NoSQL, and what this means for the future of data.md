# 为什么SQL打败NoSQL，这对未来的数据意味着什么

原文链接：[Why SQL is beating NoSQL, and what this means for the future of data](Why SQL is beating NoSQL, and what this means for the future of data)

*经过多年的死亡，SQL今天正在卷土重来。 这是怎么发生的？ 这会对数据社区产生什么影响？*

*(Update: #1 on Hacker News!* [*Read the discussion here.*](https://news.ycombinator.com/item?id=15335717)*)*

*(Update 2:* [*TimescaleDB*](http://www.timescale.com/) *is hiring! Open positions in Engineering, Marketing, and Sales.* [*Interested?*](http://www.timescale.com/careers)*)*



![img](https://cdn-images-1.medium.com/max/2000/1*HMEoq1e2RNxSwiQo_RL6tw.gif)

**SQL唤醒了对抗NoSQL的黑暗势力**

自计算开始以来，我们一直在收集指数级增长的数据，不断从我们的数据存储，处理和分析技术中获取更多信息。在过去十年中，这导致软件开发人员抛弃SQL作为遗留物，无法随着这些不断增长的数据量而扩展，导致NoSQL的兴起：MapReduce和Bigtable，Cassandra，MongoDB等等。

然而今天SQL正在复苏。所有主要的云提供商现在都提供流行的托管关系数据库服务：例如， [Amazon RDS](https://aws.amazon.com/rds/), [Google Cloud SQL](https://cloud.google.com/sql/docs/), [Azure Database for PostgreSQL](https://azure.microsoft.com/en-us/services/postgresql/) （Azure今年刚刚推出）。用亚马逊的话来说，它的PostgreSQL和MySQL兼容的数据库Aurora数据库产品一直是“[AWS历史上发展最快的服务]((http://www.businesswire.com/news/home/20161130006131/en/AWS-Extends-Amazon-Aurora-PostgreSQL-Compatibility))”。 Hadoop和Spark之上的SQL接口继续蓬勃发展。就在上个月，[Kafka发起了SQL支持](https://www.confluent.io/blog/ksql-open-source-streaming-sql-for-apache-kafka/)。你卑微的作者本身就是一个完全包含SQL的新[时间序列数据库](https://github.com/timescale/timescaledb)的开发人员。

在这篇文章中，我们将研究为什么今天的钟摆回到SQL，以及这对数据工程和分析社区的未来意味着什么。

### 第一部分: 新希望

为了理解SQL为什么可以卷土重来，让我们从回顾设计它的原因。

![img](https://cdn-images-1.medium.com/max/1600/0*fAiBMwVRHoAPwLL7.)

**像所有好故事一样，我们的故事始于20世纪70年代**

我们的故事始于20世纪70年代早期的IBM Research，关系数据库诞生于此。 那时，查询语言依赖于复杂的数学逻辑和符号。 两位新创建的博士Donald Chamberlin和Raymond Boyce对关系数据模型印象深刻，但发现查询语言将成为采用的主要瓶颈。 他们开始设计一种新的查询语言（用他们自己的话说）：“[没有正式的数学或计算机编程基础的用户更容易使用](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6359709).”



![img](https://cdn-images-1.medium.com/max/1600/0*Y5w_pCl0K9Fo9AF8.)

**在SQL（a，b）与SQL（c）之前查询语言 (**[**source**](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6359709)**)**

想想看。互联网发展之前，在个人计算机诞生之前，当编程语言C首次从世界上诞生时，两位年轻的计算机科学家意识到了这一点, “[计算机行业的成功大部分取决于除受过训练的计算机专家以外的一类用户。](http://www.almaden.ibm.com/cs/people/chamberlin/sequel-1974.pdf)” 他们想要一种像英语一样容易阅读的查询语言，并且还包括数据库管理和操作。

结果是SQL，于1974年首次在世界上诞生。在接下来的几十年中，SQL将被证明非常受欢迎。 随着System R，Ingres，DB2，Oracle，SQL Server，PostgreSQL，MySQL等关系数据库接管软件行业，SQL成为与数据库交互的优秀语言，并成为纷繁复杂生态系统中的*通用语言* 。

(可悲的是，Raymond Boyce没有机会见证SQL的成功， [他死于脑动脉瘤](https://en.wikipedia.org/wiki/Raymond_F._Boyce) 。在给出一个最早的SQL演示文稿之后的一个月，仅仅26岁，留下了一个妻子和年幼的女儿。)

有一段时间，似乎SQL已经成功完结。 但随后进入了互联网时代。



<iframe data-width="800" data-height="400" width="700" height="350" src="https://blog.timescale.com/media/254441eea2ea320d7081c26599169d9e?postId=348b777b847a" data-media-id="254441eea2ea320d7081c26599169d9e" allowfullscreen="" frameborder="0" style="user-select: text !important; display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 619.987px;"></iframe>

------

### 第二部分: NoSQL反击

虽然Chamberlin和Boyce正在开发SQL，但他们没有意识到的是，加利福尼亚的第二组工程师正在研究另一个新兴的项目，该项目后来会广泛传播并威胁SQL的存在。 该项目是[ARPANET](https://en.wikipedia.org/wiki/ARPANET)，并于1969年10月29日，[它诞生](http://all-that-is-interesting.com/internet-history)。

![img](https://cdn-images-1.medium.com/max/1600/0*L-W7e8jSXtgdWSXu.)

**ARPANET的一些创造者，最终演变成今天的互联网 (**[**source**](http://all-that-is-interesting.com/internet-history)**)**

但是，在另一位工程师出现并发明了SQL之前，SQL实际上很好 [World Wide Web](https://en.wikipedia.org/wiki/World_Wide_Web), in 1989.



![img](https://cdn-images-1.medium.com/max/1600/0*6kZJR84blb_BkDxc.)

**发明网络的物理学家 (**[**source**](https://webfoundation.org/about/vision/history-of-the-web/)**)**

就像杂草一样，互联网和网络蓬勃发展，以无数种方式大规模地扰乱了我们的世界，但对于数据社区来说，它造成了一个特别令人头疼的问题：新数据比以前数据的生成速度更快。

随着互联网的不断发展和壮大，软件界发现当时的关系数据库无法处理这种新的负载。 *部队受到干扰，好像有数百万个数据库喊叫并突然超载。*

然后两个新的互联网巨头取得了突破，并开发了自己的分布式非关系系统，以帮助这一新的数据冲击: **MapReduce** ([published 2004](https://static.googleusercontent.com/media/research.google.com/en//archive/mapreduce-osdi04.pdf)) 与 **Bigtable** ([published 2006](https://static.googleusercontent.com/media/research.google.com/en//archive/bigtable-osdi06.pdf)) 由谷歌开发, 和 **Dynamo** ([published 2007](http://www.allthingsdistributed.com/files/amazon-dynamo-sosp2007.pdf)) 由Amazon开发. 这些开创性的论文导致了更多的非关系型数据库, 包括**Hadoop** (基于MapReduce论文, [2006](https://en.wikipedia.org/wiki/Apache_Hadoop)), **Cassandra** (受Bigtable和Dynamo论文的启发, [2008](https://en.wikipedia.org/wiki/Apache_Cassandra)) 和 **MongoDB** ([2009](https://en.wikipedia.org/wiki/MongoDB)). 因为这些是从头开始编写的新系统，所以它们也避开了SQL，导致了NoSQL运动的兴起。

而且男孩让软件开发者社区吃掉了NoSQL，可以说它比原来的谷歌/亚马逊作者想的要广泛得多。 很容易理解为什么：NoSQL是新的，有光泽的; 它具有一定的规模和力量; 这似乎是工程成功的捷径。 但随后问题开始出现。



![img](https://cdn-images-1.medium.com/max/1600/0*G6Hx2C1l9abkVkxq.)

**经典软件开发人员受NoSQL的诱惑。 不要成为这个人。**

开发人员很快发现，没有SQL实际上是非常有限的。每个NoSQL数据库都提供了自己独特的查询语言，这意味着：学习更多语言（以及向同事传授）;将这些数据库连接到应用程序的难度增加，导致大量脆弱的胶水代码;缺乏第三方生态系统，要求公司开发自己的运营和可视化工具。

这些NoSQL语言虽然是新的，但还没有完全开发出来。例如，关系数据库中已经有多年的工作要为SQL添加必要的功能（例如，JOIN）; NoSQL语言的不成熟意味着在应用程序级别需要更多的复杂性。缺乏JOIN也导致非规范化，导致数据膨胀和僵化。

一些NoSQL数据库添加了自己的“类SQL”查询语言，如Cassandra的CQL。但这往往使问题变得更糟。使用与更常见的*几乎相同的界面实际上创造了更多的精神摩擦：工程师不知道支持什么和不支持什么。



![img](https://cdn-images-1.medium.com/max/1600/0*NxNoLnTnFQ7LkqBj.)

**类SQL的查询语言就像** [**Star Wars Holiday Special**](https://www.youtube.com/watch?v=ZX0x-I06Fpc)**. 不接受任何模仿.** [*(And always avoid the Star Wars Holiday Special.)*](https://xkcd.com/653/)

社区中的一些人早期就看到了NoSQL的问题 (e.g., [DeWitt and Stonebraker in 2008](https://homes.cs.washington.edu/~billhowe/mapreduce_a_major_step_backwards.html)). 随着时间的推移，通过不断积攒的个人经验，越来越多的软件开发人员加入了他们。

[**Time-series data: Why (and how) to use a relational database instead of NoSQL**
*Contrary to the belief of most developers, we show that relational databases can be made to scale for time-series data.*blog.timescale.com](https://blog.timescale.com/time-series-data-why-and-how-to-use-a-relational-database-instead-of-nosql-d0cd6975e87c)

------

### 第三部分：回归SQL



![img](https://cdn-images-1.medium.com/max/1600/1*QsZLtPL0t9bspQ16fpmeLA.gif)

最初被黑暗的一面诱惑，软件界开始看到光明并回到SQL。

首先是在Hadoop（以及后来的Spark）之上的SQL接口，引领业界将“back-cronym”NoSQL改为“Not Only SQL”（是的，不错的尝试）。

然后是NewSQL的兴起：新的可扩展数据库完全包含SQL。 麻省理工学院和布朗研究人员的**H-Store** [(published 2008](http://hstore.cs.brown.edu/papers/hstore-demo.pdf))是最早的横向扩展OLTP数据库之一。 Google再次凭借他们的第一篇**Spanner** 论文引领了地理复制的SQL接口数据库 [(published 2012](https://static.googleusercontent.com/media/research.google.com/en//archive/spanner-osdi2012.pdf)) (其作者包括原始的MapReduce作者), 其次是其他**CockroachDB** ([2014](https://en.wikipedia.org/wiki/Cockroach_Labs))开拓者.

同时,  **PostgreSQL** 社区开始复活, 添加像JSON数据类型这样的关键改进(2012), 和一个新的功能的 [PostgreSQL 10](https://wiki.postgresql.org/wiki/New_in_postgres_10): 更好的本机支持分区和复制，JSON的全文搜索支持等（将在今年晚些时候发布）。 其他公司如**CitusDB** ([2016](https://www.citusdata.com/blog/2016/03/24/citus-unforks-goes-open-source/)) 和 ([**TimescaleDB**](https://github.com/timescale/timescaledb), [released this year](https://blog.timescale.com/when-boring-is-awesome-building-a-scalable-time-series-database-on-postgresql-2900ea453ee2))找到了为专业数据工作负载扩展PostgreSQL的新方法。



![img](https://cdn-images-1.medium.com/max/1600/1*iGyZFQzaXJwP6gPAjqdgwQ.png)

事实上，我们开发[**TimescaleDB**](https://github.com/timescale/timescaledb)的过程非常反映了该行业的发展方向。 [TimescaleDB](http://www.timescale.com/)的早期内部版本使用了我们自己的类似SQL的查询语言“ioQL”。是的，我们也受到黑暗面的诱惑：构建我们自己的查询语言感觉强大。虽然看起来很简单，但我们很快意识到我们还需要做更多的工作：例如，决定语法，构建各种连接器，教育用户等等。我们还发现自己不断地查找正确的查询语法对于我们自己编写的查询语言，我们已经可以在SQL中表达了！

有一天，我们意识到构建我们自己的查询语言毫无意义。关键是要拥抱SQL。这是我们做出的最佳设计决策之一。立刻开启了一个全新的世界。今天，即使我们只是一个5个月大的数据库，我们的用户也可以在生产中使用我们并获得开箱即用的各种精彩内容：可视化工具（Tableau），常见ORM的连接器，各种工具和备份选项，在线等丰富的教程和语法解释等。

[**Eye or the Tiger: Benchmarking Cassandra vs. TimescaleDB for time-series data**
*How a 5 node TimescaleDB cluster outperforms 30 Cassandra nodes, with higher inserts, up to 5800x faster queries, 10%…*blog.timescale.com](https://blog.timescale.com/time-series-data-cassandra-vs-timescaledb-postgresql-7c2cc50a89ce)

------

### 但是不要相信我们的话。看看Google



![img](https://cdn-images-1.medium.com/max/1600/1*CiKNT6_V8VH5hRVoWNcIHA.png)

十多年来，谷歌显然一直处于数据工程和基础设施的前沿。 我们应该密切关注他们正在做的事情。

看一下四个月前发布的谷歌第二篇主要的**Spanner**论文（[Spanner: Becoming a SQL System](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46103.pdf)，2017年5月），你会发现它支持我们的独立发现。

例如，Google开始在Bigtable之上构建，但后来发现缺少SQL会产生问题（强调所有引号内容都在）：

> “虽然这些系统提供了数据库系统的一些优点，但它们缺少应用程序开发人员经常依赖的许多传统数据库功能。 **一个关键的例子是强大的查询语言**，这意味着开发人员必须编写复杂的代码来处理和聚合应用程序中的数据。 **因此，我们决定将Spanner变成一个功能齐全的SQL系统**，查询执行与Spanner的其他架构特性紧密集成（例如强一致性和全局复制）。”

在本文的后面，他们进一步了解了从NoSQL转换到SQL的基本原理：

> Spanner的原始API为单个和交错表的点查找和范围扫描提供了NoSQL方法。 虽然NoSQL方法提供了启动Spanner的简单路径，并且在简单的检索方案中继续有用，但是**在表达更复杂的数据访问模式和将计算推送到数据方面提供了显着的附加价值**。

本文还描述了SQL的采用如何不止于Spanner，而是实际扩展到Google的其他部分，其中多个系统现在共享一种常见的SQL方言：

> **Spanner的SQL引擎共享一种常见的SQL方言，称为“标准SQL”，**与谷歌的其他几个系统，包括内部系统，如F1和Dremel（以及其他系统），以及外部系统，如BigQuery ......

> **对于Google中的用户，这降低了跨系统工作的障碍。**针对Spanner数据库编写SQL的开发人员或数据分析师可以将他们对语言的理解转移到Dremel，而无需担心语法上的细微差别，NULL处理 等

这种方法的成功说明了一切。 对于主要的Google系统，包括AdWords和Google Play，扳手已经是*“真相来源”*，而*“潜在云客户对使用SQL非常感兴趣。”*

考虑到谷歌首先帮助启动了NoSQL运动，今天它正在接受SQL是非常值得注意的。 （引导一些人最近想知道： “[Did Google Send the Big Data Industry on a 10 Year Head Fake?](https://medium.com/@garyorenstein/did-google-send-the-big-data-industry-on-a-10-year-head-fake-9c94d553925a)”.)

------

### 这对数据的未来意味着什么：SQL作为通用接口

在计算机网络中，有一个叫做“[narrow waist](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1.4614&rep=rep1&type=pdf),”的概念用来描述通用接口。

出现这个想法是为了解决一个关键问题：在任何给定的网络设备上，想象一下堆栈，底层有硬件层，顶层有软件层。 可以存在各种网络硬件; 类似地，可以存在各种软件和应用程序。 人们需要一种方法来确保无论硬件如何，软件仍然可以连接到网络; 无论软件如何，网络硬件都知道如何处理网络请求。



![img](https://cdn-images-1.medium.com/max/1600/0*qm2HH4Ob3YnH3C3f.)

**IP作为网络通用接口 (**[**source**](http://slideplayer.com/slide/7597601/)**)**

在网络中，通用接口的作用由[因特网协议（IP）](https://en.wikipedia.org/wiki/Internet_Protocol)起作用，作为为本地设计的低级网络协议之间的连接层。区域网络，以及更高级别的应用程序和传输协议。 （[这是一个很好的解释](https://www.youtube.com/watch?v=uXumm52oBMo)。）和（在一个广泛的过度简化），这个通用接口成为计算机的*通用语言*，使网络互连，设备进行通信，这个“网络中的网络”将成长为今天丰富多彩的互联网。

**我们相信SQL已成为数据分析的通用接口。**

我们生活在一个数据正在成为“世界上最宝贵的资源”的时代（[The Economist，2017年5月](https://www.economist.com/news/leaders/21721656-data-economy-demands-new-approach-antitrust-rules-worlds-most-valuable-resource)）。结果，我们看到寒武纪爆炸的专业数据库（OLAP，时间序列，文档，图形等），数据处理工具（Hadoop，Spark，Flink），数据总线（Kafka，RabbitMQ）等。还有更多需要依赖这种数据基础架构的应用程序，无论是第三方数据可视化工具（Tableau，Grafana，PowerBI，Superset），Web框架（Rails，Django）还是定制的数据驱动应用程序。



![img](https://cdn-images-1.medium.com/max/1600/1*iC7lwedryNOSSYiQc3M7-Q.png)

与网络一样，我们有一个复杂的堆栈，底层有基础设施，顶层有应用程序。通常，我们最终编写了大量的胶水代码来使这个堆栈工作。但胶水代码可能很脆弱：它需要维护和倾向于。

我们需要的是一个接口，允许这个堆栈的各个部分相互通信。理想情况下，业界已经标准化了。能够让我们以最小的摩擦交换各种层的东西。

这就是SQL的强大功能。与IP一样，SQL是一种通用接口。

但SQL实际上远不止IP。因为数据也会被人类分析。对于SQL创建者最初分配给它的目的而言，SQL是可读的。

SQL完美吗？不，但这是我们社区大多数人都知道的语言。虽然已经有工程师在开发更自然的语言界面，但这些系统会连接到什么？ SQL。

所以在堆栈的最顶层还有另一层。那层是我们。

------

### SQL回来了

SQL回来了。 不只是因为编写胶水代码以合并NoSQL工具是令人讨厌的。 这不仅仅是因为重新培训劳动力来学习无数新语言也很困难。 不仅仅因为标准可以是一件好事。

但也因为世界充满了数据。 它环绕着我们，束缚着我们。 起初，我们依靠人类的感官和感觉神经系统来处理它。 现在，我们的软件和硬件系统也变得足够聪明，可以帮助我们。 随着我们收集越来越多的数据以更好地了解我们的世界，我们用于存储，处理，分析和可视化数据的系统的复杂性也将继续增长。



![img](https://cdn-images-1.medium.com/max/1600/0*0NbRxZrtmccWwYJ_.)

**硕士数据科学家尤达**

要么我们生活在一个脆弱的系统和一百万个接口的世界里。 或者我们可以继续接受SQL。 并恢复力量的平衡。

------

*Like this post and interested in learning more?*

*Follow us* [*here*](https://blog.timescale.com/) *on Medium, check out our* [*GitHub*](https://github.com/timescale/timescaledb)*, join our* [*Slack community*](http://slack.timescale.com/)*, and sign up for the community mailing list below. We’re also* [*hiring*](http://www.timescale.com/careers)*!*

*Suggested reading for those who’d like to learn more about the history of databases (aka syllabus for the future TimescaleDB Intro to Databases Class):*

- [A Relational Model of Data for Large Shared Data Banks](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6359709) (IBM Research, 1970)
- [SEQUEL: A Structured English Query Language](http://www.almaden.ibm.com/cs/people/chamberlin/sequel-1974.pdf) (IBM Research, 1974)
- [System R: Relational Approach to Database Management](http://daslab.seas.harvard.edu/reading-group/papers/astrahan-1976.pdf) (IBM Research, 1976)
- [MapReduce: Simplified Data Processing on Large Clusters](https://static.googleusercontent.com/media/research.google.com/en//archive/mapreduce-osdi04.pdf) (Google, 2004)
- [C-Store: A Column-oriented DBMS](http://cs-www.cs.yale.edu/homes/dna/papers/vldb.pdf) (MIT, others, 2005)
- [Bigtable: A Distributed Storage System for Structured Data](https://static.googleusercontent.com/media/research.google.com/en//archive/bigtable-osdi06.pdf) (Google, 2006)
- [Dynamo: Amazon’s Highly Available Key-value Store](http://www.allthingsdistributed.com/files/amazon-dynamo-sosp2007.pdf) (Amazon, 2007)
- [MapReduce: A major step backwards](https://homes.cs.washington.edu/~billhowe/mapreduce_a_major_step_backwards.html) (DeWitt, Stonebreaker, 2008)
- [H-Store: A High-Performance, Distributed Main Memory Transaction Processing System](http://hstore.cs.brown.edu/papers/hstore-demo.pdf) (MIT, Brown, others, 2008)
- [Spark: Cluster Computing with Working Sets](https://cs.stanford.edu/~matei/papers/2010/hotcloud_spark.pdf) (UC Berkeley, 2010)
- [Spanner: Google’s Globally-Distributed Database](https://static.googleusercontent.com/media/research.google.com/en//archive/spanner-osdi2012.pdf) (Google, 2012)
- [Early History of SQL](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6359709) (Chamberlin, 2012)
- [How the Internet was Born](http://all-that-is-interesting.com/internet-history) (Hines, 2015)
- [Spanner: Becoming a SQL System](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46103.pdf) (Google, 2017)