# 每个人的数据分析! Part 1

原文链接：[Intro To Data Analysis For Everyone! Part 1](https://towardsdatascience.com/intro-to-data-analysis-for-everyone-part-1-ff252c3a38b5?from=hackcv&hmsr=hackcv.com)

数据分析是任何数据科学家日常工作的一部分([以及数据篡改和清理](https://www.thoughtworks.com/insights/blog/let-data-scientists-be-data-mungers))。这对现代劳动力中其他大部分人来说也是非常重要的。可以是系统分析师、业务所有者、财务团队和项目经理。

然而，大多数本科课程并没有[或至少没有](https://www.coursera.org/browse/datscience/datanalysis?)数据分析，而是有数学和统计学的课程，还有大量涉及数据结构和算法的计算机编程课程。

然而，这些都没有关注如何查看来自数据库、csvs或现代数据世界中存在的数十个其他数据源的数据集。

可能偶尔会有需要分析数据的项目。有些人可能很幸运地收到了一组项目，迫使他们第一次从数据库中分析数据。然而，大多数学生都在他们的第一份工作中试图自己解决这个问题。

对于不打算成为程序员的学生来说， [理解数据库和SQL是一项非常有价值的技能](http://www.skilledup.com/articles/learn-sql-it-most-in-demand-skill-in-single-day)，这样可以让他们理解那些已经被数据库团队分析后的数据。

管理人员不再能接受他们的团队看不懂数据，或不知道如何进行数据分析!因此，即使是营销专业的学生也需要知道如何使用和设计数据分析!

数据分析是抽象的。它不是数学(虽然涉及数学)，也不是英语或会计。要真正理解优秀分析师会遇到的陷阱，就需要有实际的方法。然而，很遗憾的是大多数学生在进入第一份工作时，还不需要处理模糊的参数和庞大的数据集，许多学生甚至没有听说过数据仓库，然后这正是帮助管理者做出关键决策的大部分数据所在之处。

在现代商业世界中，数据分析并不局限于数据科学家。对于分析师、系统工程师、金融团队、公关、人力资源、营销等等来说，这也是很重要的技能。

因此，我们的团队想提供一个指南，帮助新学生和那些有兴趣学习更多数据科学和分析的人。

### 良好数据科学和分析的基础

本系列的第一部分将介绍良好分析所需的重要软技能。 [数据分析不仅仅是数学、SQL和脚本](https://www.theseattledataguy.com/statistics-data-scientist-review/)。它还包括保持组织有序，能够清晰地向管理者阐明已经发现的发现。这是[成功的数据科学和分析团队所描绘](https://www.theseattledataguy.com/top-30-tips-data-science-team-succeeds/)的众多特征之一。我们认为首先指出这些是很重要的，因为它为我们接下来的几个部分奠定了基础。

在本节之后，我们将讨论分析过程、技术，并通过数据集、SQL和python笔记给出示例。

**沟通**

[术语数据讲故事者已经与数据科学家联系在一起](https://www.ted.com/talks/hans_rosling_shows_the_best_stats_you_ve_ever_seen)，但对于使用数据的人来说，擅长传达他们的发现也很重要！

这种技能子集符合通信的一般技能。数据科学家可以访问来自不同部门的多个数据源。这使他们有责任并且需要能够清楚地解释他们在多个领域向高管和中小企业发现的内容。他们采用复杂的数学和技术概念，创建清晰简洁的信息，管理人员可以采取行动。不只是躲在他们的行话背后，而是将他们复杂的想法转化为商业话语。分析师和数据科学家都必须能够获取数字并返回明确规定的投资回报率和可行的决策。

这意味着不仅要记笔记，还要创造扎实的工作簿。它还意味着为其他团队创建可靠的报告和遍历。

这是如何做到的？（这可能是一个帖子本身），但这里有一些快速提示，可以更好地在报告或演示文稿中传达你的想法。

   1. 标记每个图形，轴，数据点等
   2. 在笔记本中创建自然的数据和笔记流
   3. 确保突出您的主要发现！不要藏起来，把你的结论展示出来！使用大量数据证明您的观点时，这说起来容易做起来难。
   4. 想象一下，你实际上在讲故事或写一篇有关数据的文章
   5. 不要让观众觉得枯燥，保持甜美和简洁
   6. 避免繁重的数学术语！如果你不能用简单的英语解释你的计算，你就没有完全理解。
   7. 让同行审核您的报告和演示文稿，以确保最大程度的清晰度

**我们最喜欢的数据故事之一！**



<iframe data-width="640" data-height="480" width="640" height="480" data-src="/media/72cf832079445b8dbf1e634afe63bd30?postId=ff252c3a38b5" data-media-id="72cf832079445b8dbf1e634afe63bd30" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Fi.ytimg.com%2Fvi%2FhVimVzgtD6w%2Fhqdefault.jpg&amp;key=a19fcc184b9711e1b4764040d3dc5c07" class="progressiveMedia-iframe js-progressiveMedia-iframe" allowfullscreen="" frameborder="0" src="https://towardsdatascience.com/media/72cf832079445b8dbf1e634afe63bd30?postId=ff252c3a38b5" style="user-select: text !important; display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 525px;"></iframe>

**善于倾听**

数据科学家和分析师并不总是与企业主和管理人员在同一个团队中提出问题。这使得分析师非常重视聆听实际被问到的内容。

在大公司工作，试图寻找其他团队的痛点和问题并帮助他们度过难关是很有价值的！这意味着要有同理心。这项技能的一部分需要劳动力的经验，而这项技能的其他部分只需要了解其他人。

为什么他们真的要求进行分析?你如何使分析尽可能清晰准确?

与企业主沟通不畅很容易发生。因此 [认真倾听和倾听言外之意是一项很棒的技能](https://www.forbes.com/sites/glennllopis/2013/05/20/6-effective-ways-listening-can-make-you-a-better-leader/#3fafb2421756)。



![img](https://cdn-images-1.medium.com/max/1000/0*x4gXpuM1k7rgyHi9.)

**关注背景**

除了关注细节。数据分析师和数据科学家还需要关注他们分析的数据背后的背景。这意味着理解请求项目的其他部门的需求，以及实际理解他们分析的数据背后的过程。

数据通常表示业务的流程。这可能是一个用户与电子商务网站交互，一个病人在医院，一个项目获得批准，软件被购买和开发等等。

这意味着，数据分析师需要理解这些业务规则和逻辑!否则，他们就无法进行良好的分析，他们会做出错误的假设，并且常常会创建脏的、重复的数据。

这都是因为他们不理解使用的场景。上下文允许以数据为中心的团队更清楚地做出假设。他们不需要在假设阶段花太多时间去检验所有可能的理论。相反，他们可以利用上下文来帮助加速分析的过程。

数据周围的元数据(例如上下文)对于数据科学家来说就像黄金。它并不总是在那里，但当它在的时候。它使我们的工作更容易!

**记录能力**

[无论是用excel还是Jupyter笔记本](http://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what_is_jupyter.html).。对于数据分析师来说，了解如何跟踪他们的工作是很重要的!

分析需要大量的假设和问题，如果没有记录下来，就会失去思路。

第二天回来时，很容易忘记分析了什么，不同的查询和指标是如何以及为什么被提取的，等等。因此，以一种勤奋的方式记录下每一件事情是很重要的。这个技巧是不能留给第二天的，因为总是会有信息丢失!

创建一个清晰的记录方式使每个人都更容易参与。我们在之前的交流中提到过。然而,一次。

标签，创造自然流通的笔记，避免商业术语可以帮助每个人参与，包括当初的记录人!当记录者都不理解自己的笔记，这将是相当尴尬的一件事。

记笔记很重要！

**创造性和抽象思维**

创造力和 [抽象思维 ](http://www.projectlearnet.org/tutorials/concrete_vs_abstract_thinking.html)有助于数据科学家更好地假设他们在最初探索阶段看到的可能模式和特征。将逻辑思维与最小的数据点结合起来，数据科学家可以得出几种可能的解决方案。然而，这需要跳出框框进行思考。

分析是有纪律的研究和创造性思维的结合。如果分析师受到确认偏差或过程的限制，那么他们可能无法得出正确的结论。

另一方面，如果他们过于疯狂地思考，没有使用基本的推论和归纳来驱动他们的搜索。在浏览各种数据集时，他们可能会花上数周时间试图回答一个简单的问题，而没有任何明确的目标。

**Engineering Mindset**

分析师需要能够将大问题和数据集分解成更小的部分。有时候，一个单独的团队提出的2-3个问题无法用2-3个答案来回答。

相反，2-3个问题本身可能需要被分解成小问题，这些问题可以被数据分析和支持。

只有这样，分析师才能回去回答更大的问题。特别是对于大而复杂的数据集。[能够清楚地将分析分解成适当的部分](http://www.thwink.org/sustain/articles/000_AnalyticalApproach/index.htm).变得越来越重要。

**注意细节**

分析需要注意细节。仅仅因为一个分析师或数据科学家可能是一个大局观的人。这并不意味着他们不负责找出围绕项目的所有有价值的细节。

公司，甚至是小公司都有很多角落和缝隙。流程上有流程，但不理解这些流程及其细节会影响可执行的分析级别。

特别是在编写复杂的查询和编程脚本时。很容易不正确地连接表或过滤错误的东西。因此，总是进行两次和三次检查工作是非常关键的(而且，如果涉及脚本，同行评审也应该如此!)



![img](https://cdn-images-1.medium.com/max/1000/0*R-Rff55mXIQkP7mJ.)

**好奇心**

分析需要的好奇心。当我们分解这个过程时，我们会讲到这个。然而，分析过程中的一个步骤是列出所有您认为对分析有价值的问题。这需要一个好奇的心去关心答案。

为什么数据是这样，为什么我们看到模式，我们能用什么来找到答案，谁知道呢?

这些只是一些模糊的问题，可以帮助我们开始向正确的方向进行分析。 [需要有那种动力和欲望去知道为什么!](http://www.ibmbigdatahub.com/podcast/curious-data-scientist)

**宽容失败**

数据科学与科学领域有许多相似之处。从这个意义上说，可能有99个失败的假设导致1个成功的解决方案。一些数据驱动型的公司只希望他们的机器学习工程师和数据科学家每年创造新的算法，或者每年半的相关性。这取决于任务的大小和所需的实现类型(例如流程实现、技术、策略等)。在所有这些工作中都有失败后的失败，有未回答的问题后的问题和分析师不得不继续。

关键是要得到答案，或者清楚地说明为什么你不能回答这个问题。然而，它不能仅仅因为最初的几次尝试失败而放弃。

分析可以成为时间的黑洞。一个接一个的问题可能是不正确的。这就是为什么半结构化过程很重要。它可以指导分析师，但不会阻止他们。

### **数据科学和分析软技能**

这些技能分析人员和数据科学家需要的不仅仅是编程和统计分析。相反，这些技巧的重点在于确保所发现的洞见是易于转移的。这允许其他团队成员和经理也从分析中获益!

分析师需要做的不仅仅是得出结论。他们需要能够创造出易于复制和传播的工作。

**为什么?**

它不仅节省时间!

更重要的是，这有助于领导信任分析师的结论。否则，分析师可能是对的，但如果他或她听起来不自信，如果他们记错了笔记，甚至漏掉了一个数据点。这会立即导致领导层之间的不信任!

不幸的是，这是真的!当仅仅一个数据点不正确或沟通不畅时，分析师的工作就会立即受到质疑。我们经常建议数据团队检查他们的报告和演示文稿，检查漏洞。在这种情况下，有一个善于质疑每个角度的团队成员是很好的!

你的团队可以提前回答高管可能提出的问题越多。高管们更有可能在项目的下一阶段签字!



![img](https://cdn-images-1.medium.com/max/1000/0*J7W2YgdjexxKsr4X.)

**数据分析的过程**

在下一部分中，我们将介绍分析数据的过程。我们将建立基本的笔记和描述简单的过程，这将帮助新的和有经验的数据科学家和分析师确保他们有效地跟踪他们的工作。

### [Part 2 每个人的数据分析](https://medium.com/@SeattleDataGuy/data-analysis-for-everyone-part-2-cf1c79441940)

**其他关于数据科学和策略的资源**

[How To Apply Data Science To Real World Problems](https://www.theseattledataguy.com/data-science-case-studies/)

[Amazon Using Data To Win The Grocery Store Game](https://www.theseattledataguy.com/amazon-taking-lunch-data-driven-strategies/)

[30 Tips To Ensure Your Data Science Team Succeeds](https://www.theseattledataguy.com/top-30-tips-data-science-team-succeeds/)

[A Brilliant Explanation of A Decision Tree](http://www.acheronanalytics.com/acheron-blog/brilliant-explanation-of-a-decision-tree-algorithms)