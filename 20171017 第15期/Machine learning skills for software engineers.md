# Machine learning skills for software engineers

原文链接：[Machine learning skills for software engineers](https://www.infoworld.com/article/3223688/machine-learning/machine-learning-skills-for-software-engineers.html?from=hackcv&hmsr=hackcv.com&utm_medium=hackcv.com&utm_source=hackcv.com)

### You don’t need to be a data scientist to do machine learning, but you do need data skills. Start with these

![Machine learning skills for software engineers](https://images.techhive.com/images/article/2017/04/1_proven-skills-development-100719359-large.jpg)

Thinkstock

*Ted Dunning is chief applications architect at MapR Technologies.*

A long time ago in the mid 1950’s, Robert Heinlein wrote a story called “A Door into Summer” in which a competent mechanical engineer hooked up some “Thorsen tubes” for pattern matching memory and some “side circuits to add judgment” and spawned an entire industry of intelligent robots. To make the story more plausible, it was set well into the future, in 1970. These robots could have a task like dishwashing demonstrated to them and then replicate it flawlessly.

I don’t think I have to tell you, but it didn’t turn out that way. It may have seemed plausible in 1956, but by 1969 it was clear it wouldn’t happen in 1970. And then a bit later it was clear that it wouldn’t happen in 1980, either, nor in 1990 or 2000. Every 10 years, the ability for a normal engineer to build an artificially intelligent machine seemed to retreat at least as fast as time passed. As technology improved, the enormous difficulty of the problem became clear as layer after layer of difficulties were found.

**[ Review: TensorFlow, Spark MLlib, Scikit-learn, MXNet, Microsoft Cognitive Toolkit, and Caffe machine learning and deep learning frameworks. | Roundup: 13 frameworks for mastering machine learning. | Cut to the key news and issues in cutting-edge enterprise technology with the InfoWorld Daily newsletter. ]**

It wasn’t that machine learning wasn’t solving important problems; it was. For example, by the mid-90’s essentially all credit card transactions were being scanned for fraud using neural networks. By the late 90’s Google was analyzing the web for advanced signals to aid in search. But your day to day software engineer didn’t have a chance of building such a system unless they went back to school for a Ph.D. and found a gaggle of like-minded friends who would do the same thing. Machine learning was hard, and each new domain required breaking a significant amount of new ground. Even the best researchers couldn’t crack hard problems like image recognition in the real world.

I am happy to say that this situation has changed dramatically. I don’t think that any of us is about to found a Heinlein-style, auto-magical, all-robotic engineering company in the near future, but it is now possible for a software engineer without any particularly advanced training to make systems that do really amazing stuff. The surprising part is not that computers could do these things. (It has been known since 1956 that this would be possible any day now!) What is surprising is how far we’ve come in the last decade. What would have made really good Ph.D. research 10 years ago is now a cool project for a weekend.

## Machine learning is getting easier (or at least more accessible)

In our forthcoming book “Machine Learning Logistics”(coming in late September 2017 from O’Reilly), Ellen Friedman and I describe a system known as TensorChicken that our friend and software engineer, Ian Downard, has built as a fun home project. The problem to be solved was that blue jays were getting into our friend’s chicken coop and pecking the eggs. He wanted to build a computer vision system that could recognize a blue jay so that some kind of action could be taken to stop the pecking.

After seeing a deep learning presentation by Google engineers from the TensorFlow team, Ian got cracking and built just such a system. He was able to do this by starting with a partial model known as [Inception-v3](https://www.tensorflow.org/tutorials/image_recognition) and training it to the task of blue jay spotting with a few thousand new images taken by a webcam in his chicken coop. The result could be deployed on a Raspberry Pi, but plausibly fast response time requires something a bit beefier, such as an Intel Core i7 processor.



And Ian isn’t alone. There are all sorts of people, many of them *not* trained as data scientists, building cool bots to do all kinds of things. And an increasing number of developers are beginning to work on a variety of different, serious machine learning projects as they recognize that machine learning and even deep learning have become more accessible. Developers are beginning to fill roles as data engineers in a “data ops” style of work, where data-focused skills (data engineering, architect, data scientist) are combined with a devops approach to build things such as machine learning systems. 

It’s impressive that a computer can fairly easily be trained to spot a blue jay, using an image recognition model. In many cases, ordinary folks can sit down and just do this and a whole lot more besides. All you need is a few pointers to useful techniques, and a bit of a reset in your frame of mind, particularly if you’re mainly used to doing software development.



Building models is different from building ordinary software in that it is data-driven instead of design-driven. You have to look at the system from an empirical point of view and rely a bit more than you might like on experimental proofs of function rather than careful implementation of a good design accompanied with unit and integration tests. Also keep in mind that in problem domains where machine learning has become easy, it can be stupidly easy. Right next door, however, are problems that are still very hard and that do require more sophisticated data science skills, including more math. So prototype your solution. Test it. Don’t bet the farm (or the hen house) until you know your problem is in the easy category, or at least in the not-quite-bleeding-edge category. Don’t even bet the farm after it seems to work for the first time. Be suspicious of good looking results just like any good data scientist.

## Essential data skills for machine learning beginners

The rest of this article describes some of the skills and tactics that developers need in order to use machine learning effectively.

### Let the data speak

In good software engineering, you can often reason out a design, write your software, and validate the correctness of your solution directly and independently. In some cases, you can even mathematically prove that your software is correct. The real world does intrude a bit, especially when humans are involved, but if you have good specifications, you can implement a correct solution.

With machine learning, you generally don’t have a tight specification. You have data that represents the past experience with a system, and you have to build a system that will work in the future. To tell if your system is really working, you have to measure performance in realistic situations. Switching to this data-driven, specification-poor style of development can be hard, but it is a critical step if you want to build systems with machine learning inside. 



### Learn to spot the better model

Comparing two numbers is easy. Assuming they are both valid values (not NaN’s), you check which is bigger, and you are done. When it comes to the accuracy of a machine learning model, however, it isn’t so simple. You have lots of outcomes for the models you are comparing, and there isn’t usually a clean-cut answer. Pretty much the most basic skill in building machine learning systems is the ability to look at the history of decisions that two models have made and determine which model is better for your situation. This judgment requires basic techniques to think about values that have an entire cloud of values rather than a single value. It also typically requires that you be able to visualize data well. Histograms and scatter plots and lots of related techniques will be required.

### Be suspicious of your conclusions

Along with the ability to determine which variant of a system is doing a better job, it is really important to be suspicious of your conclusions. Are your results a statistical fluke that will go the other way with more data? Has the world changed since your evaluation, thus changing which system is better? Building a system with machine learning inside means that you have to keep an eye on the system to make sure that it is still doing what you thought it was doing to start with. This suspicious nature is required when dealing with fuzzy comparisons in a changing world.

### Build many models to throw away

It is a well-worn maxim in software development that you will need to build one version of your system just to throw away. The idea is that until you actually build a working system, you won’t really understand the problem well enough to build that system well. So you build one version in order to learn and then use that learning to design and build the real system.

With machine learning, the situation is the same, but more so. Instead of building just one disposable system, you must be prepared to build dozens or hundreds of variants. Some of these variants might use different learning technologies or even just different settings for the learning engine. Other variants might be completely different restatements of the problem or the data that you use to train the models. For instance, you might determine that there is a surrogate signal that you could use to train the models even if that signal isn’t really what you want to predict. That might give you 10 times more data to train with. Or you might be able to restate the problem in a way that makes it simpler to solve.

The world may well change. This is particularly true, for instance, when you are building models to try to catch fraud. Even after you build a successful system, you will need to change in the future. The fraudsters will spot your countermeasures, and they will change their behavior. You will have to respond with new countermeasures.

So for successful machine learning, plan to build a bunch of models to throw away. Don’t expect to find a golden model that is the answer forever.

### Don’t be afraid to change the game

The first question that you try to solve with machine learning is usually not quite the right one. Often it is dramatically the wrong one. The result of asking the wrong question can be a model that is nearly impossible to train, or training data that is impossible to collect. Or it may be a situation where a model that finds the best answer still has little value.

Recasting the problem can sometimes give you a situation where a very simple model to build gives very high value. I had a problem once that was supposed to do with recommendation of sale items. It was really hard to get even trivial gains, even with some pretty heavy techniques. As it turned out, the high value problem was to determine *when* good items went on sale. Once you knew *when*, the problem of *which* products to recommend became trivial because there were many good products to recommend. At the wrong times, there was nothing worth recommending anyway. Changing the question made the problem vastly easier.

### Start small

It is extremely valuable to be able to deploy your original system to just a few cases or to just a single sub-problem. This allows you to focus your effort and gain expertise in your problem domain and gain support in your company as you build models.

### Start big

Make sure that you get enough training data. In fact, if you can, make sure that you get 10 times more than you think you need.

### Domain knowledge still matters

In machine learning, figuring out how a model can make a decision or a prediction is one thing. Figuring out what really are the important questions is much more important. As such, if you already have a lot of domain knowledge, you are much more likely to ask the appropriate questions and to be able to incorporate machine learning into a viable product. Domain knowledge is critical to figuring out where a sense of judgment needs to be added and where it might plausibly be added.

### Coding skills still matter

There are a number of tools out there that purport to let you build machine learning models using nothing but drag-and-drop tooling. The fact is, most of the work in building a machine learning system has nothing to do with machine learning or models and has everything to do with gathering training data and building a system to use the output of the models. This makes good coding skills extremely valuable. There is a different flavor to code that is written to manipulate data, but that isn’t hard to pick up. So the basic skills of a developer turn out to be useful skills in many varieties of machine learning.

Many tools and new techniques are becoming available that allow practically any software engineer to build systems that use machine learning to do some amazing things. Basic software engineering skills are highly valuable in building these systems, but you need to augment them with a bit of data focus. The best way to pick up these new skills is to start today in building something fun.

\---

*Ted Dunning is chief applications architect at MapR Technologies and a board member for the Apache Software Foundation. He is a PMC member and committer of the Apache Mahout, Apache Zookeeper, and Apache Drill projects and a mentor for various incubator projects. He was chief architect behind the MusicMatch (now Yahoo Music) and Veoh recommendation systems and built fraud detection systems for ID Analytics (LifeLock). He has a Ph.D. in computing science from University of Sheffield and 24 issued patents to date. He has co-authored a number of books on big data topics including several published by O’Reilly related to machine learning. Find him on Twitter as @ted_dunning.*

**New Tech Forum provides a venue to explore and discuss emerging enterprise technology in unprecedented depth and breadth. The selection is subjective, based on our pick of the technologies we believe to be important and of greatest interest to InfoWorld readers. InfoWorld does not accept marketing collateral for publication and reserves the right to edit all contributed content. Send all inquiries to newtechforum@infoworld.com.**