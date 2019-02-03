# 在Reddit上搜索更好的搜索

原文链接：[The Search for Better Search at Reddit](https://redditblog.com/2017/09/07/the-search-for-better-search-at-reddit/?from=hackcv&hmsr=hackcv.com&utm_medium=hackcv.com&utm_source=hackcv.com)



[TECHNOLOGY](https://redditblog.com/topic/technology/)	[Staff](https://redditblog.com/author/blabyrinth/) • [September 7, 2017](https://redditblog.com/2017/09/07/the-search-for-better-search-at-reddit/)

**Chris Slowe, Nick Caldwell, & Luis Bitencourt-Emilio***CTO, VP of Engineering, Director of Engineering*

## **什么是Fuss?**

我们从Reddit的新手工程团队成员那里得到的一个常见问题是“我们什么时候才能修复搜索？”直到今年，答案总是“去询问5楼的搜索团队。”这很有趣，因为

1. 到5楼的电梯按钮没有工作
2. 没有搜索团队

但是时代在进步，这是一个改革。我们很高兴地宣布，我们正在Reddit推出一个新的搜索引擎。实际上，在过去的几周里它已经启动了50％的流量，并且已经提供了近5亿次查询。现在我们对我们的系统充满信心，我们将其推向100％的流量。我们希望您享受更快，更可靠的结果！

更重要的是，我们还在Reddit开设了一个专门用于搜索和相关的整个产品部门，由我们的工程总监Luis领导。我们认识到这些技术对Reddit的未来至关重要。我们的平台包含世界上最有趣的内容集合之一，目前索引超过25亿个搜索帖子，并且它每天都在变大。但我们知道这个内容很难找到。改进搜索和相关性将使Reddit能够筛选数百万个帖子，评论和社区，以便直接为您的家庭Feed提供定制的精彩内容流。

那就是未来。就目前而言，我们认为沿着记忆之路旅行会很有趣。

<iframe height="326px" width="100%" scrolling="no" frameborder="0" src="https://www.redditmedia.com/r/announcements/comments/59k22p/hey_its_reddits_totally_politically_neutral_ceo/d992fwq/?embed=true&amp;context=1&amp;depth=2&amp;showedits=true&amp;created=2018-10-17T03:25:43.723226+00:00&amp;uuid=548c946c-d1bc-11e8-b1de-0e16448f7cb2&amp;showmore=false" style="user-select: text !important; box-sizing: border-box; word-break: keep-all; font-family: &quot;Open Sans&quot; !important; border: 0px; max-width: 800px; height: 326px; width: 720px; display: block; min-width: 220px; margin: 10px 0px; box-shadow: rgba(0, 0, 0, 0.05) 0px 0px 5px 0.5px;"></iframe>

## **Reddit搜索简洁的历史**

不用说，搜索不是一个容易解决的挑战。在Reddit上搜索的时候，我们就像坐过山车一样，但现在我们已经是第六次搜索了，我们对大规模搜索的困难并不陌生。下面是关于12年历史的粗略概述，以及一些来自我们团队的精选引语，我们通过迭代来将我们的infra扩展到Reddit的需求:

- 2005 – Steve Huffman ([u/spez](https://www.reddit.com/user/spez)), 创始人之一，现任首席执行官, 开启了 postgres 7.4’s contrib/[tsearch2](http://www.sai.msu.su/~megera/postgres/gist/tsearch/V2/). 

  当有人说 “哦，我们可以用 Postgres来完成!” ， “对我来说听起来不错!?” “我们当时也非常喜欢TRIGGERs（”不，这很酷。数据库完成所有工作，并且保证准确无误“是我们毫无疑问的说法）。它工作得很好，但它不是很可调，我们很快发现我们正在以少数（约2％）搜索流量阻塞大多数Postgres查询：

  - “我们修复了搜索结果排序中的错误。” —[Steve](https://redditblog.com/2006/02/27/if-you-want-something-done-right-do-it-yourself/)
  - “我们今天早上更新了搜索系统，以帮助缓解一些负载问题。” —[Steve](https://redditblog.com/2006/07/25/searching/)
  - “Jeremy正致力于搜索！这不是一个复杂的修复（排序很糟糕）。” —[Steve](https://redditblog.com/2007/04/28/updates/)

- 2007 – Chris Slowe ([u/KeyserSosa](https://www.reddit.com/user/KeyserSosa)), 创始工程师（现在是CTO），与PyLucene一起重新实施。

   

  这实际上是在10年前的2007年7月实现的。它由一个Python进程组成，该进程被设置为TCP上的线程RPC服务器。在初始版本中，我们实际上支持搜索帖子标题和评论，并且Lucene索引文件可以舒适地存储在一个盒子上。这也是在我们搬到AWS之前，当时我们已经认真考虑过使用Google Search Appliance，这对我们的单机架来说是一个很好的补充。这个版本很灵活，但我们没有以一种易于扩展的方式进行设置：

  - “搜索的效果变得更好这标记着用户可以更好的进行控制。” —[Steve](https://redditblog.com/2007/07/26/new-reddit-on-the-horizon/)
  - “搜索效果更好，但是不是我们喜欢的地方。” —[Steve](https://redditblog.com/2007/08/21/its-slow-its-unstable-its-beta/)
  - “统计数据和搜索暂时被禁用，但只要我们能够修复它们就会回来。” —[Steve](https://redditblog.com/2007/10/16/reddit-status-update/)
    - “我们希望包含升级后的搜索，与上一版本不同，它实际上非常有用，可以帮助您找到所需内容。不幸的是，我们确定的版本并没有很好地加载测试。” —[Steve](https://redditblog.com/2007/10/18/reddit-status-update-part-ii/)
  - “我快速修复了搜索，我希望有所帮助，直到我们有机会真正解决它。” —[Steve](https://redditblog.com/2007/06/08/a-note-on-search-and-what-were-working-on/)

- 2008 – David King ([u/ketralnis](https://www.reddit.com/user/ketralnis)), 第三名员工，现在是搜索工程师，实施Solr。 

  实际上，他实现了一个自制的pysolr，它能够以XML格式将更新文档发送给Solr，并以这样的方式包装响应，以便模拟我们现有的Query模型，足以将其放入任何类型或列表中。它实际上很甜蜜。初始版本不支持评论，但后来确实如此。

  - “[David]一直在修复Erlang的搜索和黑客攻击项目。” —[Alexis Ohanian](https://redditblog.com/2008/04/17/welcome-david/)
  - “我完全取代了reddit搜索功能。” —[David King](https://redditblog.com/2008/04/21/new-search-2/)

- 2010 – David将Solr替换为第三方搜索提供商IndexTank。

  当你喜欢某些东西时，将其外包......从来没有人说过。随着网站持续增长，我们首先在一个月内与一个四人工程团队一起破解了十亿次网页浏览，我们将所有努力投入503缓解，继续添加Postgres读取，添加更多缓存，开始利用Cassandra的早期版本（之后很快就发生了一次令人难忘的停电），并且通常无视搜索的糟糕程度。我们有一个勇敢的决定，永远使用第三方搜索提供商，比我们为保持Solr运行所付出的更少，所以我们签了！

  - “我们昨天推出了一个新的搜索引擎。冷静。没关系。我知道。你以前受伤了。” —[David King](https://redditblog.com/2010/07/21/new-search/)

- 2012 – Keith Mitchell (u/kemitche) 在LinkedIn关闭IndexTank后实施CloudSearch。

  很明显，这个永远过于短暂，IndexTank在公司被收购之前为我们提供了很好的帮助。当我们发现他们正在关闭时，我们不得不离开IndexTank并快速过渡到AWS CloudSearch。继续我们长期以来的传统“让新人照顾它”，这项任务落到了Keith身上，在接下来的几年里，我们将CloudSearch扩展到了爆炸状态：

  - “今天，我们从旧的Amazon CloudSearch域迁移到新的Amazon CloudSearch域。旧的搜索域存在严重的性能问题：大约33％的查询需要5秒才能完成，并且会导致搜索错误页面。” —[u/bsimpson](https://www.reddit.com/r/changelog/comments/694o34/reddit_search_performance_improvements/)

- TODAY – Lucidworks Fusion!

  这一次，我们希望确保搜索符合三个标准：它需要快速，需要与Reddit的增长很好地扩展，最重要的是，它需要具有相关性。最终，这促使我们与Lucidworks的搜索专家合作，利用Fusion及其由多个Solr提交者组成的团队的独特搜索专业知识。下面，我们将更详细地解释我们如何进行此操作。

  - “As [/u/bitofsalt](https://www.reddit.com/u/bitofsalt) [几个月前我们提到过](https://www.reddit.com/r/funny/comments/65ryr3/and_now_a_look_at_the_machine_that_powers_reddits/dgd22mi/), 我们一直在努力改进搜索。我们甚至可能领先于 [spez’s 的10年计划](https://www.reddit.com/r/announcements/comments/59k22p/hey_its_reddits_totally_politically_neutral_ceo/d992fwq/?context=1).” —[u/starfishjenga](https://www.reddit.com/r/changelog/comments/6pi0kk/improving_search/)

## **感受更多**

今年早些时候，对Reddit的搜索变得非常糟糕。简单的查询只能在一半的时间内成功。想要使用两个关键字进行搜索？离开这里！

![img](https://redditupvoted.files.wordpress.com/2017/09/screen-shot-2017-09-07-at-11-43-15-am.png?w=720&h=505)

图1：我们的CloudSearch集群负载过重时的示例错误页面。

在查看了几个选项后，我们与 [Lucidworks](https://lucidworks.com/) 合作，重振Reddit的搜索系统。 Lucidworks是Fusion的创建者，Fusion是一个基于Solr的搜索堆栈，支持巨大的文档规模和高查询吞吐量。

## **第一件事：以Reddit量表摄取**

迁移到新搜索系统的最大挑战是我们的索引管道需要更新。第一次尝试很艰难。为了速度，我们匆忙将它放在我们由 [Jenkins](https://jenkins.io/)和[Azkaban](https://azkaban.github.io/) 组成的遗留ETL系统上，编排了许多Hive查询。正如您在下图中所看到的，将来自多个来源的数据汇总到一个有索引的规范视图中进行索引，证明比最初预期的更复杂。

![img](https://redditupvoted.files.wordpress.com/2017/09/screen-shot-2017-09-07-at-11-44-37-am.png?w=720&h=433)

图2：我们新的搜索提取管道的第一次迭代，现在被替换为显着简化的版本。

我们的第二次尝试既简单又产生了明显更好的结果。我们设法将整个管道修剪为仅仅四个更简单和更准确的Hive查询，这使得索引的帖子增加了33％。另一个重大改进是，我们不仅索引新的帖子创作，而且还实时更新其相关信号，因为投票，评论和其他信号全天都在流动。

## **使其相关**

如果它们不相关，搜索结果并不意味着什么。对于我们的初始部署，主要目标是避免降低返回结果的整体相关性。

为了监控这一点，我们测量了搜索结果页面上的点击次数，并比较了在新旧搜索系统中点击的结果的排名。一个完美的搜索引擎会在返回的最高结果上产生100％的点击次数，这是另一种表示您希望在顶部获得最相关结果的方式。由于我们知道完美的搜索引擎不是一个可实现的目标，我们使用 [平均互惠等级](https://en.wikipedia.org/wiki/Mean_reciprocal_rank)和 [折扣累积增益](https://en.wikipedia.org/wiki/Discounted_cumulative_gain)等措施来比较我们的结果质量。

虽然它在我们的实验中还处于早期阶段，但迄今为止的数据指向了我们的旧堆栈与新堆栈之间非常可比的相关性测量，而Fusion具有轻微的优势。这个有希望的部分是我们还没有进行太多的相关调整 - 这是我们的新系统实际支持的东西。个性化，机器学习模型以及查询意图和重写等进步现在已经成为现实。

![img](https://redditupvoted.files.wordpress.com/2017/09/screen-shot-2017-09-07-at-11-46-10-am.png?w=720&h=276)

图3：Fusion和CloudSearch堆栈之间搜索结果点击位置的比较。

## **首次展示**

在我们克服数据提取挑战和监控相关性时，我们继续将使用率提高到越来越多的redditors。这个早期小组的 [反馈](https://www.reddit.com/r/changelog/comments/6pi0kk/improving_search/)非常宝贵，我们非常感谢社区帮助我们解决漏洞和不太常见的用例。我们在新筹码上只有1％的用户开始工作，处理报告的问题并改进了摄取管道，因为我们在GA之前将推出百分比提高到5,10,25和最终50％的流量。在这段时间里，我们将所有搜索查询作为黑暗流量发送到我们的新搜索群集，以确保随着我们增加推出百分比，它可以全面扩展。

![img](https://redditupvoted.files.wordpress.com/2017/09/screen-shot-2017-09-07-at-11-47-24-am.png?w=720&h=376)

图4：黄色的CloudSearch错误和绿色的Fusion。

我们很自豪地说Reddit Search比以往更好！所有Reddit内容的完整重新索引现在在大约5个小时内完成（从大约11个小时开始），我们不断将实时更新流式传输到索引。错误率下降了两个数量级，99％的搜索结果在500毫秒内完成。运行搜索所需的机器数量从今年早些时候的约200台减少到30台左右，因此我们甚至设法节省了一些成本。

![img](https://redditupvoted.files.wordpress.com/2017/09/screen-shot-2017-09-07-at-11-48-12-am.png?w=720&h=308)

图5：Reddit新搜索堆栈概述。

更快，更可靠，更相关，更低成本！当然这应该是我们最后一次需要更改搜索堆栈！

## **展望未来**

严肃地说，我们认为你会喜欢这个更新。我们希望新的搜索堆栈将成为改进的基础，以便更容易地发现Reddit上的所有优秀内容。更重要的是：我们没有完成。修复搜索只是一系列新功能的第一步，这些功能将使Reddit更加个性化并与您的兴趣相关。 Reddit最终拥有一个Search＆Relevance团队，我们正在疯狂招聘。如果您对在数亿人使用的搜索和相关性平台上使用世界上最有趣的数据集之一感到兴奋，那么请查看我们的工作列表：

**Head of Search:** [https://boards.greenhouse.io/reddit/jobs/723000#.Wa3yONOGOEI
](https://boards.greenhouse.io/reddit/jobs/723000#.Wa3yONOGOEI)**Head of Relevance:** [https://boards.greenhouse.io/reddit/jobs/611466#.WbC_ltOGOEI
](https://boards.greenhouse.io/reddit/jobs/611466#.WbC_ltOGOEI)**Head of Discovery:** [https://boards.greenhouse.io/reddit/jobs/764831#.WbC_2NOGOEI
](https://boards.greenhouse.io/reddit/jobs/764831#.WbC_2NOGOEI)**Search Engineers:** <https://boards.greenhouse.io/reddit/jobs/612128#.Wa3yQtOGOEI>

最后，感谢Lucidworks团队提供了一个惊人的合作伙伴关系，并帮助我们在Reddit上寻找更好的搜索。