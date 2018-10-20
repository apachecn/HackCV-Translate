# Advice For New and Junior Data Scientists

原文链接：[Advice For New and Junior Data Scientists](https://medium.com/@rchang/advice-for-new-and-junior-data-scientists-2ab02396cf5b?from=hackcv&hmsr=hackcv.com&utm_medium=hackcv.com&utm_source=hackcv.com)

## What I Would Have Told Myself a Few Years ago



![img](https://cdn-images-1.medium.com/max/2000/1*n1_oY6S0tqy4QaWVXaAmxQ.png)

Image credit: Alice Truong

### Motivation

Two years ago, I shared my [experience](https://medium.com/@rchang/my-two-year-journey-as-a-data-scientist-at-twitter-f0c13298aee6) on doing data science in the industry. The writing was originally meant to be a private reflection for myself to celebrate my two year twitterversary at Twitter, but I instead published it on Medium because I believe it could be very useful for many aspiring data scientists.

Fast forward to 2017, I have been working at Airbnb for a little bit less than two years and have recently become a senior data scientist — an industry title used to signal that one has acquired a certain level of technical expertise. As I reflect on my journey so far and imagine what’s next to come, I once again wrote down a few lessons that I wish I had known in the earlier days of my career.

If the intended audience of my previous post was for aspiring data scientists and people who are completely new to the field, then this article is for people who are already in the field but are just starting out. My goal is to not only use this post as a reminder to myself about the important things that I have learned, but also to inspire others as they embark onto their DS careers!

### Whose Critical Path Are You On?

Philip Guo, an outstanding academic and prolific blogger, reflected on his experience interacting with various mentors throughout his years as a student, intern, and researcher. In his blot post “[Whose Critical Path Are You On?](http://www.pgbovine.net/critical-path.htm)”, he made the following observation:

> If I was on my mentor’s **critical path** [for career advancement or fulfillment], then they would fight hard to make sure I got the help that I needed to succeed. Conversely, if I wasn’t on my mentor’s critical path, then I was usually left to fend for myself. […] If you get on someone’s critical path, then you force them to tie your success to theirs, which will motivate them to lift you up as hard as they can.



![img](https://cdn-images-1.medium.com/max/600/1*ev5_ddW1FE367USX2d46zA.png)

Image credit: The Icefields Parkway // Daniel Han

This work dynamic is pretty intuitive, and I wish I had internalized it earlier in my career when choosing projects, selecting teams, or even evaluating which mentors or companies to work for.

As an example, while at Twitter, I had always wanted to learn more about machine learning, but my team, despite being very data driven, largely needed data scientists to focus on experiment design and product analytics. Despite my best efforts, I often found it difficult to marry this intellectual desire with the critical projects of my team.

As a result, when I arrived at Airbnb, I made a conscious decision to focus on joining a project/team where ML is critical to its success. I worked with my manager to identify a few promising opportunities, one of which is to model the lifetime value (LTV) of listings on Airbnb.

This project was not only critical to the success of our business, but also to the development of my career. I learned so much about the workflow of [building machine learning model at scale](https://medium.com/airbnb-engineering/using-machine-learning-to-predict-value-of-homes-on-airbnb-9272d3d4739d), and there was no better way to learn other than learning in the context of solving a concrete business problem.

Undoubtedly, I was very lucky to find a project that aligned with my aspirations and where I wanted to build my skills. I believe the framework of picking projects on our mentors’ critical paths can make us increasingly “lucky” over time on matching our aspirations with the right projects at work.

P**rinciple I learned**: *We all have skills that we would like to develop and intellectual interests that we would love to pursue. It’s important to evaluate how well our aspirations align with the critical path of the environment we are in. Find projects, teams, and companies whose critical path best aligned with yours.*

### Picking the Right Tools For The Problem

Before Airbnb, I had been coding in R and [dplyr](https://github.com/tidyverse/dplyr) for most of my professional life. After starting on the LTV project, I soon realized the deliverable was not a piece of analysis code, but rather a production machine learning pipeline. Given that it is much easier to build complex pipelines in [Airflow](https://medium.com/the-astronomer-journey/airflow-and-the-future-of-data-engineering-a-q-a-266f68d956a9) using Python, I was faced with a dilemma — should I switch from R to Python?



![img](https://cdn-images-1.medium.com/max/800/1*dZaHst97QwHkKtWd0HlZ4Q.png)

Image source: quickmeme.com (besides R or Python, Excel is also a serious contender 👊)

This turns out to be a very common question among data scientists, since many struggled to decide which language to choose. For me, there is clearly a switching cost once committed to one or the other. I went through the pros and cons to understand the tradeoffs, but the more I thought about it, the more I fell into the trap of decision paralysis. (Here is an entertaining [talk](https://blog.dominodatalab.com/video-huge-debate-r-vs-python-data-science/) that demonstrates this concept). Eventually, I escaped from this paralysis after reading this response on [Reddit](https://www.reddit.com/r/Python/comments/2tkkxd/considering_putting_my_efforts_into_python/):

> Instead of thinking about which programming language to learn, think about which language offers you the right set of Domain Specific Languages (DSL) that fit your problems.

The appropriateness of a tool is always context dependent and problem specific. It’s not about whether I should learn Python, it’s whether Python is the right tool for the job. To elaborate more on this point, here are a few examples:

- If your goal is to apply the most current, cutting-edge statistical methods, R is likely to be the better choice. Why? Because R is built by statisticians and for statisticians. Nowadays, academics publish their research not only in papers but also in R packages. Each week, there are many interesting new R packages made available on [CRAN](https://cran.r-project.org/mirrors.html), like this [one](https://github.com/susanathey/causalTree).
- On the other hand, Python is great for building production data pipelines, since it is a general-purpose programming language. For example, one can easily wrap a [scikit-learn](http://scikit-learn.org/) model using [Python UDF](http://www.florianwilhelm.info/2016/10/python_udf_in_hive/) to do distributed scoring in Hive, orchestrate Airflow DAGs with complex logic, or write a Flask web app to showcase the output of the model in a browser.

For my particular project, I needed to build a production machine learning pipeline, and my life would be a lot easier if I did it in Python. Eventually, I rolled up my sleeves and embraced this new challenge!

P**rinciple I learned**: *Instead of fixating on a single technique or programming language, ask yourself, what is the best set of tools or techniques that will help you to solve your problem? Focus on problem solving, and the tools will come naturally.*

### Building A Learning Project

Even though I have not used Python to do Data Science work before, I did play with the language in a [different capacity](https://medium.com/@rchang/learning-how-to-build-a-web-application-c5499bd15c8f). However, I never really learned Python fundamentals properly. As a result, I got scared when code was organized into [classes](https://jeffknupp.com/blog/2014/06/18/improve-your-python-python-classes-and-object-oriented-programming/), and I always wondered what *__init__.py* was [used](https://stackoverflow.com/questions/448271/what-is-init-py-for) for*.*

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