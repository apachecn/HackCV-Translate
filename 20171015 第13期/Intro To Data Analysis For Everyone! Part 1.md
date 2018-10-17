# Intro To Data Analysis For Everyone! Part 1

原文链接：[Intro To Data Analysis For Everyone! Part 1](https://towardsdatascience.com/intro-to-data-analysis-for-everyone-part-1-ff252c3a38b5?from=hackcv&hmsr=hackcv.com)

Data analysis is part of any data scientists daily work ([along with data munging and cleansing](https://www.thoughtworks.com/insights/blog/let-data-scientists-be-data-mungers)). It is also very important for a good portion of everyone else in the modern workforce. This could be a systems analysts, business owners, financial teams, and project managers.

However, most undergrad courses do not ([or at the very least did not](https://www.coursera.org/browse/data-science/data-analysis?languages=en)) teach the basics of data analysis in any of their courses. There were math courses, and statistics, as well as plenty of computer programming courses that involved data structures and algorithms.

Yet, none of these focused on how to look at data sets from databases, csvs or the dozens of other data sources that exist in the modern data world.

There might be the occasional project that requires analyzing data. Some individuals might have been lucky enough to receive a set of projects that forced them to analyze data for the first time out of a database. However, most students are left to attempt to figure it out themselves during their first job!

For students not planning to be programmers, u[nderstanding databases and SQL is a super-valuable skill](http://www.skilledup.com/articles/learn-sql-it-most-in-demand-skill-in-single-day). It allows them access to data that was once held hostage by database teams.

Managers are no longer ok with their teams not having access to data! Thus, even a marketing major needs to know how to work with and devise analysis from data!

Data analysis is abstract. It is not math(although math is involved), it is not english or accounting. It requires a hands on approach in order to truly understand the pitfalls good analysts will run into. Yet, most students have not had to deal with vague parameters, and large data sets by the time they get into their first job, which is shame! Many students haven’t even heard of a data warehouse, and this is where most of the data that helps managers make critical decisions reside.

In the modern business world, data analysis is not limited to data scientists. It is also key for analysts, systems engineers, financial teams, PR, HR, marketing, and so on.

Thus, our team wanted to give a guide to helping both new students and those interested in learning more about data science and analysis.

### The Foundation Of Good Data Science And Analytics

This first part in this series will cover the important soft skills required for good analysis. [Data analysis is not only math, SQL and scripting](https://www.theseattledataguy.com/statistics-data-scientist-review/). It is also about staying organized and being able to clearly articulate to managers the discoveries that have been unearthed. This is one of many traits that [successful teams in data science and analytics portray](https://www.theseattledataguy.com/top-30-tips-data-science-team-succeeds/). We believe it is important to point these out first because it lays the groundwork for our next few parts.

After this section, we will discuss analysis processes, techniques and give examples with data sets, SQL and python notebooks.

**Communication**

[The term data storyteller has become correlated with data scientist](https://www.ted.com/talks/hans_rosling_shows_the_best_stats_you_ve_ever_seen), but it is also important for anyone who uses data to be good at communicating their findings!

This skill-subset fits in the general skill of communication. Data scientists have access to multiple data sources from various departments. This gives them the responsibility and need to be able to clearly explain what they are discovering to executives and SMEs in multiple fields. They take complex mathematical and technological concepts and create clear and concise messages that executives can act upon. Not just hiding behind their jargon, but actually transcribing their complex ideas into business speak. Analysts and data scientists alike must be able to take numbers and return clearly stated ROIs and actionable decisions.

This means not only taking good notes and creating solid work books. It also means creating solid reports and walk throughs for other teams.

How do you do that?(this could be a post in itself), but here are some quick tips to better communicate your ideas in a report or presentation.

1. Label every figure, axis, data point, etc
2. Create a natural flow of data and notes in a note book
3. Make sure to highlight your key findings! Don’t bury the lead, sell your big conclusion! This is easier said than done when you have lots of data to prove your point.
4. Imagine you are actually telling a story or writing an essay with data
5. Don’t bore your audience to death, keep it sweet and succinct
6. Avoid heavy math jargon! If you can’t explain your calculations in plain English, you don’t understand them
7. Peer review your reports and presentations to ensure for maximum clarity

**One Of Our Favorite Examples Of Data Story Telling!**



<iframe data-width="640" data-height="480" width="640" height="480" data-src="/media/72cf832079445b8dbf1e634afe63bd30?postId=ff252c3a38b5" data-media-id="72cf832079445b8dbf1e634afe63bd30" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Fi.ytimg.com%2Fvi%2FhVimVzgtD6w%2Fhqdefault.jpg&amp;key=a19fcc184b9711e1b4764040d3dc5c07" class="progressiveMedia-iframe js-progressiveMedia-iframe" allowfullscreen="" frameborder="0" src="https://towardsdatascience.com/media/72cf832079445b8dbf1e634afe63bd30?postId=ff252c3a38b5" style="user-select: text !important; display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 525px;"></iframe>

**Empathetic Listening**

Data scientists and analysts aren’t always on the same team as the business owners, and managers that come to them with questions. This makes it very important for analysts to listen diligently to what is actually being asked of them.

Working in large corporations, there is a lot of value in trying to seek out other teams pain points and problems and help them through it! This means having empathy. Part of this skill requires experience in the workforce and other parts of this skill simply require understanding other human beings.

Why are they really asking for the analysis and how can you make it as clear and accurate for them as possible?

Miscommunication with the business owners can happen quiet easily. Thus, the combination of [listening diligently as well as listening for what is not being said is a great asset.](https://www.forbes.com/sites/glennllopis/2013/05/20/6-effective-ways-listening-can-make-you-a-better-leader/#3fafb2421756)



![img](https://cdn-images-1.medium.com/max/1000/0*x4gXpuM1k7rgyHi9.)

**Context Focused**

Besides being focused on details. Data analysts and data scientists also need to focus on what context is behind the data they are analyzing. This means understanding the needs of the other departments who have requested the project as well as actually understanding the processes behind the data they are analyzing.

Data typically represents the processes of a business. This could be a user interacting with a ecommerce site, a patient in a hospital, a project getting approved, software being purchased and invoiced and so on.

All of these get represented in thousands of data warehouses and databases across the world and all of them are often stored just slightly differently with different business rules.

That means, data analysts need to understand those business rules and logic! Otherwise, they can’t perform good analysis, they will make bad assumptions and they will often create dirty and duplicate data.

All because they did not understand context. Context allows data focused teams to make assumptions more clearly. They are not forced to spend too much time in the hypothesis phase where they are testing every possible theory. Instead, they can utilize context to help speed up the process of their analysis.

The metadata (e.g. context) around data, is like gold to a data scientists. It isn’t always there, but when it is. It makes our jobs much easier!

**Note Taking Prowess**

[Whether using excel or Jupyter notebook](http://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what_is_jupyter.html). It is important for a data analyst to understand how to track their work!

Analysis requires a lot of assumptions and questions, and single track thinking that can be lost if not noted down.

It is easy to come back the next day and forget what was analyzed, how and why different queries and metrics were pulled, etc. Thus, it is important to note everything down in a diligent manner. This skill is not to be left to the next day, because there will always be loss of information!

Creating a clear note taking style makes it easier for everyone involved. We brought this up earlier in communication. However, again.

Labeling, creating a natural flow of notes, and avoiding business jargon can help everyone involved. Even the original note taker! It is pretty embarrassing when even the original note taker does not understand their notes!

Note taking saves lives!

**Creative and Abstract Thinking**

Creativity and [abstract thinking ](http://www.projectlearnet.org/tutorials/concrete_vs_abstract_thinking.html)helps data scientists better hypothesize possible patterns and features they are seeing in their initial exploration phases. Combining logical thinking with minimal data points, data scientists can lead themselves to several possible solutions. However, this requires thinking outside of the box.

Analysis is a combination of disciplined research and creative thinking. If an analysts is too limited by confirmation bias or process, then they might not reach the correct conclusions.

If, on the other hand, they are too wildly thinking, and not using basic deduction and induction to drive their search. They could spend weeks trying to answer a simple question as they wander through various data sets without any real clear cut goal.

**Engineering Mindset**

Analysts need to be able to take large problems and data sets and break them down into smaller pieces. Sometimes, the 2–3 questions asked by a separate team can’t be answered by 2–3 answers.

Instead, the 2–3 questions themselves might need to be broken down into small bite size questions that can be analyzed and supported by data.

Only then, can the analyst go back and answer the larger questions. Especially with large and complex data sets. It is becoming more and more important to be[ able to clearly breakdown analysis into its proper pieces](http://www.thwink.org/sustain/articles/000_AnalyticalApproach/index.htm).

**Attention To Details**

Analysis requires attention to details. Just because an analyst or data scientist might be a big picture person. This does not mean they are not responsible for figuring out all the valuable details that surround a project.

Companies, even small ones have lots of nooks and crannies. There are processes on processes and not understanding those processes and their details affects the level of analysis that can be done.

Especially when writing complex queries and programming scripts. It is very easy to incorrectly join a table or filter the wrong thing. Thus, it is key to always double and triple check work(also, if scripts are involved, peer reviews should be too!).



![img](https://cdn-images-1.medium.com/max/1000/0*R-Rff55mXIQkP7mJ.)

**Curiosity**

Analysis requires curiosity. We will get into this when we break down the process. However, a step in the analysis process is listing out all the questions you believe are valuable to the analysis. This requires a curious mind that cares to know the answer.

Why is the data the way it is, why are we seeing patterns, what can we use to find the answer, and Who would know?

These are just some vague questions that can help start pointing analysis in the right direction. [There needs to be that drive and desire to know why!](http://www.ibmbigdatahub.com/podcast/curious-data-scientist)

**Tolerance of Failure**

Data science has a lot of similarities to the science field. In the sense that there might be 99 failed hypotheses that lead to 1 successful solution. Some data driven companies only expect their machine learning engineers and data scientists to create new algorithms, or correlations every year to year and a half. This depends on the size of the task and the type of implementation required (e.g. process implementation, technical, policy, etc). In all of this work there is failure after failure, there is unanswered question after unanswered question and analysts have to continue.

The point is to get the answer, or clearly state why you can’t answer the question. However, it can’t just be giving up because the first few attempts failed.

Analysis can be a black hole for time. Question after question can be incorrect. That is why it is important to have a semi-structured process. One that guides analysts but doesn’t keep them back.

### **Data Science and Analytics Soft Skills**

These skills analysts and data scientists need aren’t all about programming and statistical analysis. Instead, these skills are about focusing on making sure the the insights that are discovered are easily transferable. This allows other team members and managers to also gain from the analysis done!

Analysts need to be able to do more than just come to a conclusion. They need to be able to create work that is easily reproducible and communicable.

**Why?**

It not only saves time!

It more importantly helps leadership trust the analyst’s conclusion. Otherwise, the analysts might be correct, but if he or she sounds unconfident, if they have bad notes, or are even missing one data point. It can instantly lead to distrust among leadership!

Sadly this is very true! Analysts work can instantly come into question when even just one data point is incorrect or communicated poorly. We often recommended that data teams do a walk through of their reports and presentations just to check for holes. Having a team member that is good at questioning every angle is great in these situations!

The more your team can pre-answer questions executives may have. The more likely the executives will sign off on the next leg of the project!



![img](https://cdn-images-1.medium.com/max/1000/0*J7W2YgdjexxKsr4X.)

**The Process of Data Analysis**

In the next portion we will lay out a process for analyzing data. We will be setting up basic notebooks and describing simple processes that will help new and experienced data scientists and analysts make sure they are tracking their work effectively.

### [Part 2 Of Data Analysis For Everyone](https://medium.com/@SeattleDataGuy/data-analysis-for-everyone-part-2-cf1c79441940)

**Other Resources About Data Science And Strategy**

[How To Apply Data Science To Real World Problems](https://www.theseattledataguy.com/data-science-case-studies/)

[Amazon Using Data To Win The Grocery Store Game](https://www.theseattledataguy.com/amazon-taking-lunch-data-driven-strategies/)

[30 Tips To Ensure Your Data Science Team Succeeds](https://www.theseattledataguy.com/top-30-tips-data-science-team-succeeds/)

[A Brilliant Explanation of A Decision Tree](http://www.acheronanalytics.com/acheron-blog/brilliant-explanation-of-a-decision-tree-algorithms)