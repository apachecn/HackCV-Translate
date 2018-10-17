# Why SQL is beating NoSQL, and what this means for the future of data

原文链接：[Why SQL is beating NoSQL, and what this means for the future of data](Why SQL is beating NoSQL, and what this means for the future of data)

*After years of being left for dead, SQL today is making a comeback. How come? And what effect will this have on the data community?*

*(Update: #1 on Hacker News!* [*Read the discussion here.*](https://news.ycombinator.com/item?id=15335717)*)*

*(Update 2:* [*TimescaleDB*](http://www.timescale.com/) *is hiring! Open positions in Engineering, Marketing, and Sales.* [*Interested?*](http://www.timescale.com/careers)*)*



![img](https://cdn-images-1.medium.com/max/2000/1*HMEoq1e2RNxSwiQo_RL6tw.gif)

**SQL awakens to fight the dark forces of NoSQL**

Since the dawn of computing, we have been collecting exponentially growing amounts of data, constantly asking more from our data storage, processing, and analysis technology. In the past decade, this caused software developers to cast aside SQL as a relic that couldn’t scale with these growing data volumes, leading to the rise of NoSQL: MapReduce and Bigtable, Cassandra, MongoDB, and more.

Yet today SQL is resurging. All of the major cloud providers now offer popular managed relational database services: e.g., [Amazon RDS](https://aws.amazon.com/rds/), [Google Cloud SQL](https://cloud.google.com/sql/docs/), [Azure Database for PostgreSQL](https://azure.microsoft.com/en-us/services/postgresql/) (Azure launched just this year). In Amazon’s own words, its PostgreSQL- and MySQL-compatible database Aurora database product has been the “[fastest growing service in the history of AWS](http://www.businesswire.com/news/home/20161130006131/en/AWS-Extends-Amazon-Aurora-PostgreSQL-Compatibility)”. SQL interfaces on top of Hadoop and Spark continue to thrive. And just last month, [Kafka launched SQL support](https://www.confluent.io/blog/ksql-open-source-streaming-sql-for-apache-kafka/). Your humble authors themselves are developers of a new [time-series database](https://github.com/timescale/timescaledb) that fully embraces SQL.

In this post we examine why the pendulum today is swinging back to SQL, and what this means for the future of the data engineering and analysis community.

------

### Part 1: A New Hope

To understand why SQL is making a comeback, let’s start with why it was designed in the first place.



![img](https://cdn-images-1.medium.com/max/1600/0*fAiBMwVRHoAPwLL7.)

**Like all good stories, ours starts in the 1970s**

Our story starts at IBM Research in the early 1970s, where the relational database was born. At that time, query languages relied on complex mathematical logic and notation. Two newly minted PhDs, Donald Chamberlin and Raymond Boyce, were impressed by the relational data model but saw that the query language would be a major bottleneck to adoption. They set out to design a new query language that would be (in their own words): “[more accessible to users without formal training in mathematics or computer programming](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6359709).”



![img](https://cdn-images-1.medium.com/max/1600/0*Y5w_pCl0K9Fo9AF8.)

**Query languages before SQL ( a, b ) vs SQL ( c ) (**[**source**](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6359709)**)**

Think about this. Way before the Internet, before the Personal Computer, when the programming language C was first being introduced to the world, two young computer scientists realized that, “[much of the success of the computer industry depends on developing a class of users other than trained computer specialists.](http://www.almaden.ibm.com/cs/people/chamberlin/sequel-1974.pdf)” They wanted a query language that was as easy to read as English, and that would also encompass database administration and manipulation.

The result was SQL, first introduced to the world in 1974. Over the next few decades, SQL would prove to be immensely popular. As relational databases like System R, Ingres, DB2, Oracle, SQL Server, PostgreSQL, MySQL (and more) took over the software industry, SQL became established as the preeminent language for interacting with a database, and became the *lingua franca* for an increasingly crowded and competitive ecosystem.

(Sadly, Raymond Boyce never had a chance to witness SQL’s success. [He died of a brain aneurysm](https://en.wikipedia.org/wiki/Raymond_F._Boyce) 1 month after giving one of the earliest SQL presentations, just 26 years of age, leaving behind a wife and young daughter.)

For a while, it seemed like SQL had successfully fulfilled its mission. But then the Internet happened.



<iframe data-width="800" data-height="400" width="700" height="350" src="https://blog.timescale.com/media/254441eea2ea320d7081c26599169d9e?postId=348b777b847a" data-media-id="254441eea2ea320d7081c26599169d9e" allowfullscreen="" frameborder="0" style="user-select: text !important; display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 619.987px;"></iframe>

------

### Part 2: NoSQL Strikes Back

While Chamberlin and Boyce were developing SQL, what they didn’t realize is that a second group of engineers in California were working on another budding project that would later widely proliferate and threaten SQL’s existence. That project was [ARPANET](https://en.wikipedia.org/wiki/ARPANET), and on October 29, 1969, [it was born](http://all-that-is-interesting.com/internet-history).



![img](https://cdn-images-1.medium.com/max/1600/0*L-W7e8jSXtgdWSXu.)

**Some of the creators of ARPANET, which eventually evolved into today’s Internet (**[**source**](http://all-that-is-interesting.com/internet-history)**)**

But SQL was actually fine until another engineer showed up and invented the [World Wide Web](https://en.wikipedia.org/wiki/World_Wide_Web), in 1989.



![img](https://cdn-images-1.medium.com/max/1600/0*6kZJR84blb_BkDxc.)

**The physicist who invented the Web (**[**source**](https://webfoundation.org/about/vision/history-of-the-web/)**)**

Like a weed, the Internet and Web flourished, massively disrupting our world in countless ways, but for the data community it created one particular headache: new sources generating data at much higher volumes and velocities than before.

As the Internet continued to grow and grow, the software community found that the relational databases of that time couldn’t handle this new load. *There was a disturbance in the force, as if a million databases cried out and were suddenly overloaded.*

Then two new Internet giants made breakthroughs, and developed their own distributed non-relational systems to help with this new onslaught of data: **MapReduce** ([published 2004](https://static.googleusercontent.com/media/research.google.com/en//archive/mapreduce-osdi04.pdf)) and **Bigtable** ([published 2006](https://static.googleusercontent.com/media/research.google.com/en//archive/bigtable-osdi06.pdf)) by Google, and **Dynamo** ([published 2007](http://www.allthingsdistributed.com/files/amazon-dynamo-sosp2007.pdf)) by Amazon. These seminal papers led to even more non-relational databases, including **Hadoop** (based on the MapReduce paper, [2006](https://en.wikipedia.org/wiki/Apache_Hadoop)), **Cassandra** (heavily inspired by both the Bigtable and Dynamo papers, [2008](https://en.wikipedia.org/wiki/Apache_Cassandra)) and **MongoDB** ([2009](https://en.wikipedia.org/wiki/MongoDB)). Because these were new systems largely written from scratch, they also eschewed SQL, leading to the rise of the NoSQL movement.

And boy did the software developer community eat up NoSQL, embracing it arguably much more broadly than the original Google/Amazon authors intended. It’s easy to understand why: NoSQL was new and shiny; it promised scale and power; it seemed like the fast path to engineering success. But then the problems started appearing.



![img](https://cdn-images-1.medium.com/max/1600/0*G6Hx2C1l9abkVkxq.)

**Classic software developer tempted by NoSQL. Don’t be this guy.**

Developers soon found that not having SQL was actually quite limiting. Each NoSQL database offered its own unique query language, which meant: more languages to learn (and to teach to your coworkers); increased difficulty in connecting these databases to applications, leading to tons of brittle glue code; a lack of a third party ecosystem, requiring companies to develop their own operational and visualization tools.

These NoSQL languages, being new, were also not fully developed. For example, there had been years of work in relational databases to add necessary features to SQL (e.g., JOINs); the immaturity of NoSQL languages meant more complexity was needed at the application level. The lack of JOINs also led to denormalization, which led to data bloat and rigidity.

Some NoSQL databases added their own “SQL-like” query languages, like Cassandra’s CQL. But this often made the problem worse. Using an interface that is *almost* identical to something more common actually created more mental friction: engineers didn’t know what was supported and what wasn’t.



![img](https://cdn-images-1.medium.com/max/1600/0*NxNoLnTnFQ7LkqBj.)

**SQL-like query languages are like the** [**Star Wars Holiday Special**](https://www.youtube.com/watch?v=ZX0x-I06Fpc)**. Accept no imitations.** [*(And always avoid the Star Wars Holiday Special.)*](https://xkcd.com/653/)

Some in the community saw the problems with NoSQL early on (e.g., [DeWitt and Stonebraker in 2008](https://homes.cs.washington.edu/~billhowe/mapreduce_a_major_step_backwards.html)). Over time, through hard-earned scars of personal experience, more and more software developers joined them.

[**Time-series data: Why (and how) to use a relational database instead of NoSQL**
*Contrary to the belief of most developers, we show that relational databases can be made to scale for time-series data.*blog.timescale.com](https://blog.timescale.com/time-series-data-why-and-how-to-use-a-relational-database-instead-of-nosql-d0cd6975e87c)

------

### Part 3: Return of the SQL



![img](https://cdn-images-1.medium.com/max/1600/1*QsZLtPL0t9bspQ16fpmeLA.gif)

Initially seduced by the dark side, the software community began to see the light and come back to SQL.

First came the SQL interfaces on top of Hadoop (and later, Spark), leading the industry to “back-cronym” NoSQL to “Not Only SQL” (yeah, nice try).

Then came the rise of NewSQL: new scalable databases that fully embraced SQL. **H-Store** [(published 2008](http://hstore.cs.brown.edu/papers/hstore-demo.pdf)) from MIT and Brown researchers was one of the first scale-out OLTP databases. Google again led the way for a geo-replicated SQL-interfaced database with their first **Spanner** paper [(published 2012](https://static.googleusercontent.com/media/research.google.com/en//archive/spanner-osdi2012.pdf)) (whose authors include the original MapReduce authors), followed by other pioneers like **CockroachDB** ([2014](https://en.wikipedia.org/wiki/Cockroach_Labs)).

At the same time, the **PostgreSQL** community began to revive, adding critical improvements like a JSON datatype (2012), and a potpourri of new features in [PostgreSQL 10](https://wiki.postgresql.org/wiki/New_in_postgres_10): better native support for partitioning and replication, full text search support for JSON, and more (release slated for later this year). Other companies like **CitusDB** ([2016](https://www.citusdata.com/blog/2016/03/24/citus-unforks-goes-open-source/)) and yours truly ([**TimescaleDB**](https://github.com/timescale/timescaledb), [released this year](https://blog.timescale.com/when-boring-is-awesome-building-a-scalable-time-series-database-on-postgresql-2900ea453ee2)) found new ways to scale PostgreSQL for specialized data workloads.



![img](https://cdn-images-1.medium.com/max/1600/1*iGyZFQzaXJwP6gPAjqdgwQ.png)

In fact, our journey developing [**TimescaleDB**](https://github.com/timescale/timescaledb) closely mirrors the path the industry has taken. Early internal versions of [TimescaleDB](http://www.timescale.com/) featured our own SQL-like query language called “ioQL.” Yes, we too were tempted by the dark side: building our own query language felt powerful. But while it seemed like the easy path, we soon realized that we’d have to do a lot more work: e.g., deciding syntax, building various connectors, educating users, etc. We also found ourselves constantly looking up the proper syntax to queries that we could already express in SQL, for a query language we had written ourselves!

One day we realized that building our own query language made no sense. That the key was to embrace SQL. And that was one of the best design decisions we have made. Immediately a whole new world opened up. Today, even though we are just a 5 month old database, our users can use us in production and get all kinds of wonderful things out of the box: visualization tools (Tableau), connectors to common ORMs, a variety of tooling and backup options, an abundance of tutorials and syntax explanations online, etc.

[**Eye or the Tiger: Benchmarking Cassandra vs. TimescaleDB for time-series data**
*How a 5 node TimescaleDB cluster outperforms 30 Cassandra nodes, with higher inserts, up to 5800x faster queries, 10%…*blog.timescale.com](https://blog.timescale.com/time-series-data-cassandra-vs-timescaledb-postgresql-7c2cc50a89ce)

------

### But don’t take our word for it. Take Google’s.



![img](https://cdn-images-1.medium.com/max/1600/1*CiKNT6_V8VH5hRVoWNcIHA.png)

Google has clearly been on the leading edge of data engineering and infrastructure for over a decade now. It behooves us to pay close attention to what they are doing.

Take a look at Google’s second major **Spanner** paper, released just four months ago ([Spanner: Becoming a SQL System](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46103.pdf), May 2017), and you’ll find that it bolsters our independent findings.

For example, Google began building on top of Bigtable, but then found that the lack of SQL created problems (emphasis in all quotes below ours):

> “While these systems provided some of the benefits of a database system, they lacked many traditional database features that application developers often rely on. **A key example is a robust query language**, meaning that developers had to write complex code to process and aggregate the data in their applications. **As a result, we decided to turn Spanner into a full featured SQL system**, with query execution tightly integrated with the other architectural features of Spanner (such as strong consistency and global replication).”

Later in the paper they further capture the rationale for their transition from NoSQL to SQL:

> The original API of Spanner provided NoSQL methods for point lookups and range scans of individual and interleaved tables. While NoSQL methods provided a simple path to launching Spanner, and continue to be useful in simple retrieval scenarios, **SQL has provided significant additional value in expressing more complex data access patterns and pushing computation to the data**.

The paper also describes how the adoption of SQL doesn’t stop at Spanner, but actually extends across the rest of Google, where multiple systems today share a common SQL dialect:

> **Spanner’s SQL engine shares a common SQL dialect, called “Standard SQL”,** with several other systems at Google including internal systems such as F1 and Dremel (among others), and external systems such as BigQuery…

> **For users within Google, this lowers the barrier of working across the systems.** A developer or data analyst who writes SQL against a Spanner database can transfer their understanding of the language to Dremel without concern over subtle differences in syntax, NULL handling, etc.

The success of this approach speaks for itself. Spanner is already the *“source of truth”* for major Google systems, including AdWords and Google Play, while *“Potential Cloud customers are overwhelmingly interested in using SQL.”*

Considering that Google helped initiate the NoSQL movement in the first place, it is quite remarkable that it is embracing SQL today. (Leading some to recently wonder: “[Did Google Send the Big Data Industry on a 10 Year Head Fake?](https://medium.com/@garyorenstein/did-google-send-the-big-data-industry-on-a-10-year-head-fake-9c94d553925a)”.)

------

### What this means for the future of data: SQL as the universal interface

In computer networking, there is a concept called the “[narrow waist](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1.4614&rep=rep1&type=pdf),” describing a universal interface.

This idea emerged to solve a key problem: On any given networked device, imagine a stack, with layers of hardware at the bottom and layers of software on top. There can exist a variety of networking hardware; similarly there can exist a variety of software and applications. One needs a way to ensure that no matter the hardware, the software can still connect to the network; and no matter the software, that the networking hardware knows how to handle the network requests.



![img](https://cdn-images-1.medium.com/max/1600/0*qm2HH4Ob3YnH3C3f.)

**IP as the Networking Universal Interface (**[**source**](http://slideplayer.com/slide/7597601/)**)**

In networking, the role of the universal interface is played by [Internet Protocol (IP)](https://en.wikipedia.org/wiki/Internet_Protocol), acting as a connecting layer between lower-level networking protocols designed for local-area network, and higher-level application and transport protocols. ([Here’s one nice explanation](https://www.youtube.com/watch?v=uXumm52oBMo).) And (in a broad oversimplification), this universal interface became the *lingua franca* for computers, enabling networks to interconnect, devices to communicate, and this “network of networks” to grow into today’s rich and varied Internet.

**We believe that SQL has become the universal interface for data analysis.**

We live in an era where data is becoming “the world’s most valuable resource” ([The Economist, May 2017](https://www.economist.com/news/leaders/21721656-data-economy-demands-new-approach-antitrust-rules-worlds-most-valuable-resource)). As a result, we have seen a Cambrian explosion of specialized databases (OLAP, time-series, document, graph, etc.), data processing tools (Hadoop, Spark, Flink), data buses (Kafka, RabbitMQ), etc. We also have more applications that need to rely on this data infrastructure, whether third-party data visualization tools (Tableau, Grafana, PowerBI, Superset), web frameworks (Rails, Django) or custom-built data-driven applications.



![img](https://cdn-images-1.medium.com/max/1600/1*iC7lwedryNOSSYiQc3M7-Q.png)

Like networking we have a complex stack, with infrastructure on the bottom and applications on top. Typically, we end up writing a lot of glue code to make this stack work. But glue code can be brittle: it needs to be maintained and tended to.

What we need is an interface that allows pieces of this stack to communicate with one another. Ideally something already standardized in the industry. Something that would allow us to swap in/out various layers with minimal friction.

That is the power of SQL. Like IP, SQL is a universal interface.

But SQL is in fact much more than IP. Because data also gets analyzed by humans. And true to the purpose that SQL’s creators initially assigned to it, SQL is readable.

Is SQL perfect? No, but it is the language that most of us in the community know. And while there are already engineers out there working on a more natural language oriented interface, what will those systems then connect to? SQL.

So there is another layer at the very top of the stack. And that layer is us.

------

### SQL is Back

SQL is back. Not just because writing glue code to kludge together NoSQL tools is annoying. Not just because retraining workforces to learn a myriad of new languages is hard. Not just because standards can be a good thing.

But also because the world is filled with data. It surrounds us, binds us. At first, we relied on our human senses and sensory nervous systems to process it. Now our software and hardware systems are also getting smart enough to help us. And as we collect more and more data to make better sense of our world, the complexity of our systems to store, process, analyze, and visualize that data will only continue to grow as well.



![img](https://cdn-images-1.medium.com/max/1600/0*0NbRxZrtmccWwYJ_.)

**Master Data Scientist Yoda**

Either we can live in a world of brittle systems and a million interfaces. Or we can continue to embrace SQL. And restore balance to the force.

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