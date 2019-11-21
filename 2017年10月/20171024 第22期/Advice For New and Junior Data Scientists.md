# 给初学者和初级数据科学家的建议

原文链接：[Advice For New and Junior Data Scientists](https://medium.com/@rchang/advice-for-new-and-junior-data-scientists-2ab02396cf5b?from=hackcv&hmsr=hackcv.com&utm_medium=hackcv.com&utm_source=hackcv.com)

## 我会在几年前告诉自己的



![img](https://cdn-images-1.medium.com/max/2000/1*n1_oY6S0tqy4QaWVXaAmxQ.png)

Image credit: Alice Truong

### 动机

两年前，我分享了我在数据科学领域内的行业[经验](https://medium.com/@rchang/my-two-year-journey-as-a-data-scientist-at-twitter-f0c13298aee6)。最初写这本书的目的是为了对我自己进行自我反思，以庆祝我在Twitter上的两年推特活动，但我最后将其发表在Medium上，因为我认为它对许多有抱负的数据科学家来说非常有用。

快进到2017年，我在Airbnb工作了不到两年，最近晋升为一名高级数据科学家。这是一个行业头衔，用来表示某些人技术上已经达到了一定的高度。当我回顾迄今为止的旅程并想象接下来会发生什么时，我再次写下了一些我希望在职业生涯的早些时候就知道的课程。

如果我之前文章的目标读者是有抱负的数据科学家和该领域的新手，那么本文适用于已经在该领域但刚刚起步的人。我的目标不仅是希望通过这篇文章来提醒自己我学到的重要知识，而且还可以激励其他人踏上DS生涯！

### 你在谁的关键路径上？

郭飞（Philip Guo）是一位杰出的学术和多才多艺的博客作者，他回顾了自己在学生，实习生和研究员期间与各种导师互动的经验。 在他的污点文章“ [你的关键路径在哪里？](http://www.pgbovine.net/critical-path.htm)”中，他进行了以下观察：

> 如果我处于导师的**关键道路**[为了职业发展或成就]，那么他们会尽力提供我成功的所需。 相反，如果我不在导师的关键路线上，那么我只能自供自给。 […]如果你在某人的关键道路，那么你会迫使他们将你的成功与他们的成功联系起来，这将激励他们尽最大努力帮助你。



![img](https://cdn-images-1.medium.com/max/600/1*ev5_ddW1FE367USX2d46zA.png)

Image credit: The Icefields Parkway // Daniel Han

这种工作动态非常直观，我希望我可以在职业生涯的早期将其内部化，这时可以选择项目，选择团队，甚至可以选择要为哪些导师或公司工作。

例如，在Twitter上，我一直想了解有关机器学习的更多信息，尽管我的团队非常依赖数据驱动，但仍非常需要数据科学家专注于实验设计和产品分析。 虽然我尽了最大的努力，但我发现很难将这种理想的愿望与团队的关键项目相结合。

结果，当我到Airbnb时，我做出了一个明智的决定，即专注于加入ML对于其成功至关重要的项目/团队。 我与我的经理一起确定了一些有前景的机会，其中之一是对Airbnb上房源的终身价值（LTV）进行建模。

这个项目不仅对我们业务的成功至关重要，而且对我的职业发展也至关重要。我了解到很多[大规模构建机器学习模型](https://medium.com/airbnb-engineering/using-machine-learning-to-predict-value-of-homes-on-airbnb-9272d3d4739d)的工作流程，除了在解决具体业务问题的背景下进行学习之外，没有更好的学习方法。

毫无疑问，我很幸运地找到了一个与我的志向以及我想在其中培养技能的项目。我相信，在导师的关键路径上挑选项目的框架可以使我们随着时间的流逝，将自己的理想与正确的项目相匹配，从而使我们越来越“幸运”。

**我学到的原则**：*我们所有人都有我们想发展的技能和我们追求的知识兴趣。评估我们的理想与我们所处环境的关键路径的契合度非常重要。找到关键路径与你的关键路径最相符的项目，团队和公司。*

### 选择正确的工具解决问题

在加入Airbnb之前，我一生的大部分时间都在使用R和[dplyr](https://github.com/tidyverse/dplyr)。 在开始LTV项目之后，我很快意识到可交付成果不是一段分析代码，而是一条生产机器学习流水线。 鉴于使用Python在[Airflow](https://medium.com/the-astronomer-journey/airflow-and-the-future-of-data-engineering-aqa-266f68d956a9)中构建复杂的流水线要容易得多， 我面临两难的境地— —我应该从R切换到Python吗？

![img](https://cdn-images-1.medium.com/max/800/1*dZaHst97QwHkKtWd0HlZ4Q.png)

Image source: quickmeme.com (besides R or Python, Excel is also a serious contender 👊)

事实证明，这是数据科学家中一个非常普遍的问题，因为许多人都在选择哪种语言上纠结。对我来说，一旦承诺一个或另一个，显然会有转换成本。我通过分析利弊来了解折衷，但是我对它的思考越多，我越容易陷入两难之地。 （这里是一个有趣的[talk](https://blog.dominodatalab.com/video-huge-debate-r-vs-python-data-science/)（展示了这个概念）。最终，我在[Reddit](https://www.reddit.com/r/Python/comments/2tkkxd/considering_putting_my_efforts_into_python/)上阅读以下回复后，摆脱了这种纠结：

>不用考虑学习哪种编程语言，而是考虑哪种语言可以为你提供适合你问题的正确的领域特定语言（DSL）。

工具的适当性始终取决于上下文和具体问题。这与我是否应该学习Python无关，而与Python是否适合这项工作无关。为了详细说明这一点，下面是一些示例：

- 如果您的目标是最新应用，最先进的统计方法，则R可能是更好的选择。 为什么？ 因为R是由统计学家为统计学家建立的。 如今，学者们不仅在论文中发表研究成果，还在R包中发表研究成果。 每周，[CRAN](https://cran.r-project.org/mirrors.html)上都会提供许多有趣的新R软件包，例如[这个](https://github.com/susanathey/ causalTree)。
- 另一方面，Python是构建生产数据管道的理想选择，因为它是一种通用的编程语言。 例如，可以使用[Python UDF](http://www.florianwilhelm.info/2016/10/python_udf_in_hive/)轻松包装[scikit-learn](http://scikit-learn.org/)模型 在Hive中进行分布式计算，使用复杂的逻辑编排Airflow DAG，或编写Flask Web应用程序以在浏览器中展示模型的输出。

对于我的特定项目，我需要构建一个生产机器学习管道，如果我使用Python进行工作，我的生活会容易得多。 最终，我卷起袖子迎接了这个新挑战！

**我学到的原则**: *不问自己只使用一种技术或编程语言，而是问自己，什么是最能帮助你解决问题的工具或技术？ 专注于解决问题，使用工具就变成很自然地*。

### 建立学习项目

即使我以前从未使用过Python从事数据科学工作，但我还是以[不同身份](https://medium.com/@rchang/learning-how-to-build-a-web-application-c5499bd15c8f)使用了该语言。 但是，我从来没有真正正确地学习过Python基础知识。 结果，当代码组织成 [classes](https://jeffknupp.com/blog/2014/06/18/improve-your-python-python-classes-and-object-oriented-programming/)时，我感到恐惧，我一直想知道 *__init__.py*  的用途

To really learn the fundamentals properly this time, I took inspiration from Anders Ericsson’s research on [**Deliberate Practice**](https://www.amazon.com/Peak-Secrets-New-Science-Expertise/dp/1531864880):

> **Deliberate Practice** is activities designed, typically by a teacher, for the sole purpose of effectively improving specific aspects of an individual’s performance.

Given that I was my own teacher, insights from Dr. Ericsson were very helpful. For example, I kicked off my “learning project” by curating a set of materials that were most relevant for doing ML in Python. This process took me a few weeks until I settled on a personalized [curriculum](https://github.com/robert8138/python-deliberate-practice). I stress tested this curriculum by asking experienced Pythonistas to review my plan. All of this pre-work was meant to ensure I would be on the right learning path.



![img](https://cdn-images-1.medium.com/max/800/1*JgVutu1PA5x-kh2WjLDJJA.png)

Here is a glimpse of my personalized curriculum

Once I had a clearly defined curriculum, I used the following strategies to deliberately practice on the job:

- **Practice Repeatedly**: I forced myself to carry out mundane, non mission-critical analyses in Python instead of in R. This dragged down my productivity initially, but it forced me to get familiar with the basic API of [pandas](http://pandas.pydata.org/), without the burden of needing to meet an urgent deadline.
- **Create Feedback Loop**: I found opportunities to review other people’s code and fix small bugs when appropriate. For example, I tried to understand how our internal Python libraries were designed before using them. When writing my own code, I also tried to refactor it several times and make it more readable for everyone.
- **Learn By Chunking and Recalling:** By the end of each week, I wrote down my [weekly progress](https://github.com/robert8138/python-deliberate-practice/blob/master/Planning.md), which included the important resources I studied in that week, concepts I learned, and any major takeaways during that week. By recalling the materials I learned, I was able to internalize the concepts better.

Slowly and gradually, I got better each week. It certainly wasn’t easy though: there were times when I had to look up basic syntax in both R and Python because I was switching back and forth between the two languages. That said, I kept in mind that this is a long term investment, and dividends will be paid as I dived into the ML project.

P**rinciple I learned**: *As supported by many* [*field experiments*](https://qz.com/978273/a-stanford-professors-15-minute-study-hack-improves-test-grades-by-a-third-of-a-grade/), *before diving into a project, planning ahead helps you to practice more deliberately. Repeating, chunking, recalling, and getting feedback are among the most useful activities to reinforce learning.*

### Partnering With Experienced Data Scientists

One of the key ingredients of **deliberate practice** is to receive timely and actionable feedback. No great athletes, musicians, or mathematicians are able to achieve greatness without coaching or targeted feedback.

One common trait I have observed from people who have a strong [growth mindset](https://www.ted.com/talks/carol_dweck_the_power_of_believing_that_you_can_improve) is that they are generally not ashamed of acknowledging what they don’t know and they constantly ask for feedback.

Looking back at my own academic and professional career so far, many times in the past I self-censored my questions because I did not want to appear incapable. However, over time I realized that this attitude was rather detrimental — in the long run, most instances of self-censorship are missed opportunities for learning rather than shame.



![img](https://cdn-images-1.medium.com/max/800/1*lgG5Z6FUEdZOVRZ8d1O2WQ.png)

Image source: edutopia — It’s important to have a growth mindset!

Before this project, I had very little experience putting machine learning models into [production](https://www.slideshare.net/SharathRao6/lessons-from-integrating-machine-learning-models-into-data-products). Of the many decisions that I made for the project, one of the best decisions was to declare early and shamelessly to my collaborators that I know very little about ML infrastructure, but that I wanted to learn. I promised them, however, as I got more knowledgeable, I would make myself useful for the team.

This turned out to be a pretty good strategy, because people generally love to share their knowledge, especially when they know their mentorship will benefit themselves eventually. Below are a few examples that I would not have learned so quickly without the guidance of my partners:

- [**Scikit-Learn Pipelines**](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)**:** My collaborator suggested to me that I can make my code more modular by adopting Sklearn’s pipeline construct. Essentially, pipelines define a series of data transformation that are consistent across training and scoring. This tool made my code cleaner, more reusable, and more easily compatible with production models.
- **Model Diagnostics**: Given that our prediction problem involves time, my collaborator taught me that typical cross validation will not work, as we could run into the risk of predicting the past using future data. Instead, a better method would be to use [time series cross validation](https://robjhyndman.com/hyndsight/tscv/). I also learned different diagnostic techniques such as [lift chart](https://www.analyticsvidhya.com/blog/2016/02/7-important-model-evaluation-error-metrics/) and various other evaluation metrics such as [SMAPE](https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error).
- **Machine Learning Infrastructure**: With the help from ML infra engineers, I learned about managing package dependency via virtualenvs, how to serialize models using [pickling](https://docs.python.org/3/library/pickle.html), and how to make the model available at scoring time using [Python UDFs](http://www.florianwilhelm.info/2016/10/python_udf_in_hive/). All these are data engineering skills that I didn’t know before.

As I learned more new concepts, not only was I able to apply them for my own project, I was able to drive engaging discussions with the machine learning infrastructure team so they can build better ML tools for data scientists. This creates a virtuous cycle because the knowledge that was shared with me made me a better partner and collaborator.

P**rinciple I learned**: *In the long run, most instances of self-censorship are missed opportunities for learning rather than shame. Declare early and shamelessly your desire to learn, and make yourself useful as you become better.*

### Teaching And Evangelizing

As I got closer to putting my model into production, I noticed that a lot of the skills that I picked up could be very valuable for other data scientists on our team. Having been a graduate student instructor for years, I always knew I had a passion for teaching, and I always learned more about the subject when I became the teacher. Richard Feynman, the late Nobel Laureate in Physics and a [phenomenal teacher](https://www.youtube.com/watch?v=0KmimDq4cSU), spoke about his view on teaching:

> Richard Feynman was once asked by a Caltech faculty member to explain why spin one-half particles obey Fermi Dirac statistics. Rising to the challenge, he said, “I’ll prepare a freshman lecture on it.” But a few days later he told the faculty member, “You know, I couldn’t do it. I couldn’t reduce it to the freshman level. That means we really don’t understand it.”

This was really inspiring — if you can’t reduce the subject to its core and make it accessible for others, that means you don’t really understand it. Knowing that teaching these skills can improve my understanding, I seek opportunities to carefully document my model implementations, give learning lunches, and encourage others to try out the tools. This was a win-win because evangelization raises awareness, which in tern helps to drive tool adoption across the team.

As of late September, I have started collaborating with our internal [Data University](https://medium.com/airbnb-engineering/how-airbnb-democratizes-data-science-with-data-university-3eccc71e073a) team to prepare a series of classes on our internal ML tools. I am not exactly sure where this will go, but I am very excited about driving more ML education at Airbnb.

Finally, I would end this section with a tweet from [Hadley Wickham](https://twitter.com/hadleywickham):



<iframe data-width="500" data-height="185" width="500" height="185" data-src="/media/d2a4a2b22832b73a7c1aa1d7da9a4eb1?postId=2ab02396cf5b" data-media-id="d2a4a2b22832b73a7c1aa1d7da9a4eb1" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Fpbs.twimg.com%2Fprofile_images%2F677589103710306304%2Fm56O6Wgf_400x400.jpg&amp;key=a19fcc184b9711e1b4764040d3dc5c07" class="progressiveMedia-iframe js-progressiveMedia-iframe" allowfullscreen="" frameborder="0" src="https://medium.com/media/d2a4a2b22832b73a7c1aa1d7da9a4eb1?postId=2ab02396cf5b" style="user-select: text !important; display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 259px;"></iframe>
<https://twitter.com/hadleywickham/status/890107458219368448>

P**rinciple I learned:** *Teaching is the best way to test your understanding of the subject and the best way to improve your skills. When you learn something valuable, share it with others. You don’t always have to create new software, explaining how existing tools work can also be super valuable.*

### At Step K, Think About Your Step K+1

From focusing on my own deliverables, to partnering with the ML infrastructure team, to finally teaching and enabling other data scientists to learn more about ML tools, I am really happy that the scope of my original project was much larger than it was a few months ago. Yet, admittedly, I never anticipated this in the first place.

As I reflected on the evolution of this project, one thing that was different from my previous projects was that I always had a slight dissatisfaction with the current state of things, and I always wanted to make it a little bit better. The most eloquent way to characterize this is from [Claude Shannon’s essay](https://medium.com/the-mission/a-genius-explains-how-to-be-creative-claude-shannons-long-lost-1952-speech-fbbcb2ebe07f):



![img](https://cdn-images-1.medium.com/max/600/1*EmbwmvVXC1bV7Jv7ZpOPZw.png)

Image source: Book cover from “A Mind at Play: How Claude Shannon Invented the Information Page” by Jimmy Soni, Rob Goodman

> “There’s the idea of **dissatisfaction**. By this I don’t mean a pessimistic dissatisfaction of the world — we don’t like the way things are — I mean a constructive dissatisfaction. The idea could be expressed in the words, This is OK, but I think things could be done better. I think there is a neater way to do this. I think things could be improved a little. In other words, there is continually a slight irritation when things don’t look quite right; and I think that dissatisfaction in present days is a key driving force in good scientists.”

By no means I am a qualified scientist (even though that is somehow in my job title), but I do think the characterization of slight dissatisfaction is quite telling for whether you will be able to extend the impact of your project. Throughout my project, whenever I am at step K, I naturally would start thinking about what to do for step K+1 and beyond:

- From “*I don’t know how to build a production model, let me figure out how”*to “*I think the tools can be improved, here are my pain points, suggestion and feedback for how to make the tools better”,* I reframed myself from a customer to a partner with ML infrastructure team.
- From “*let me learn the tools so I can be good at it”* to “*let’s make these tools more accessible for all the other Data Scientists interested in ML”,* I reframed myself from a partner to an evangelizer.

I think this mindset is extremely helpful — use your good taste and slight dissatisfaction to fuel your progress with persistence. That said, I do think that this dissatisfaction cannot be manufactured, and can only come from working on a problem you care about, which brings to my last point.

P**rinciple I learned:** *Pay attention to your inner dissatisfaction when working on a project. These are clues to how you can improve and scale your project to the next level.*

### Parting Thoughts: You And Your Work

Recently, I came across a lecture from [Richard Hamming](https://en.wikipedia.org/wiki/Richard_Hamming), who is an American Mathematician well known for many of his scientific contributions, including Hamming code and Hamming distance. The lecture was titled [You And Your Research](https://www.youtube.com/watch?v=a1zDuOPkMSw), where Dr. Hamming said it can very well be renamed as **“You And Your Career”**.



<iframe data-width="640" data-height="480" width="640" height="480" data-src="/media/70791d910c2b5748671fad34eec86b8c?postId=2ab02396cf5b" data-media-id="70791d910c2b5748671fad34eec86b8c" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Fi.ytimg.com%2Fvi%2Fa1zDuOPkMSw%2Fhqdefault.jpg&amp;key=4fce0568f2ce49e8b54624ef71a8a5bd" class="progressiveMedia-iframe js-progressiveMedia-iframe" allowfullscreen="" frameborder="0" src="https://medium.com/media/70791d910c2b5748671fad34eec86b8c?postId=2ab02396cf5b" style="user-select: text !important; display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 525px;"></iframe>
<https://www.youtube.com/watch?v=a1zDuOPkMSw>

As he shared his stories, a few important points stood out for me.

> If what you are doing is not important, not likely to be important, **why are you doing it?** You must work on important problems. I spent Friday afternoon for years thinking about the important problems in my field [that’s 10% of my working time].

> Let me warn you about important problems, importance is not the consequence, some problems are not important because you haven’t gotten an attack. The importance of problem, to a great extent, depends on if you got a way of attacking the problem.

> This whole course, I am trying to teach you something about **style** and **taste**, so you’ll be able to have some hunch on when the problem is right, what problem is right, how to go about it. The right problem at the right time at the right way counts, and nothing else counts. Nothing.

When Dr. Hamming speaks about importance, he means problems that are important **to you.** For him, it was scientific problems, and for many of us, it might be something different. He also talked about the importance of having a plan of attack. If you don’t have a plan, the problem does not matter, however big the consequences. Lastly, he mentioned doing it with your own unique style and taste.

His bar for doing great work is extremely high, but it’s one worth pursuing. When you find your important problem, you will naturally try to make it better and make it more impactful; you will find ways to teach other about its significance; you will spend time to learn from other great people and build your craft.

**What’s a problem that is important to you that is on your critical path?**

------

*I would like to thank* [*Jason Goodman*](https://medium.com/@jasonkgoodman) *and Tim Kwan for reviewing my post and giving me feedback*