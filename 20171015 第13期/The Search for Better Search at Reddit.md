# The Search for Better Search at Reddit

原文链接：[The Search for Better Search at Reddit](https://redditblog.com/2017/09/07/the-search-for-better-search-at-reddit/?from=hackcv&hmsr=hackcv.com&utm_medium=hackcv.com&utm_source=hackcv.com)

Because, certainly, we’ve solved it this time

[TECHNOLOGY](https://redditblog.com/topic/technology/)	[Staff](https://redditblog.com/author/blabyrinth/) • [September 7, 2017](https://redditblog.com/2017/09/07/the-search-for-better-search-at-reddit/)

**Chris Slowe, Nick Caldwell, & Luis Bitencourt-Emilio***CTO, VP of Engineering, Director of Engineering*

## **What’s the Fuss?**

A common question we get from newbie engineering team members here at Reddit is “When are we going to fix search?” Until this year, the answer was always “Go ask the search team on the 5th floor.” Which was great fun because a) the elevator button to the 5th floor didn’t work and b) there was no search team.

But the times, they are a-changin’. We’re happy to announce that we’re launching a new search engine at Reddit. Actually, it’s been launched to 50% of traffic for the past couple weeks and has already served up nearly half a billion queries. Now that we’re confident in our system, we’re pushing it to 100% of traffic. We hope you enjoy faster and more reliable results!

More importantly, we’ve also started an entire product unit dedicated to search and relevance here at Reddit, led by our Director of Engineering Luis. We recognize that these technologies are critical to Reddit’s future. Our platform contains one of the world’s most interesting collections of content, currently indexing over a quarter billion posts for search, and it gets bigger every day. But we know this content is hard to find. Improving search and relevance will allow Reddit to sift through millions of posts, comments, and communities to create a custom-fit stream of great content straight to your home feed.

That’s the future. For now, we thought it’d be fun to take a trip down memory lane.

<iframe height="326px" width="100%" scrolling="no" frameborder="0" src="https://www.redditmedia.com/r/announcements/comments/59k22p/hey_its_reddits_totally_politically_neutral_ceo/d992fwq/?embed=true&amp;context=1&amp;depth=2&amp;showedits=true&amp;created=2018-10-17T03:25:43.723226+00:00&amp;uuid=548c946c-d1bc-11e8-b1de-0e16448f7cb2&amp;showmore=false" style="user-select: text !important; box-sizing: border-box; word-break: keep-all; font-family: &quot;Open Sans&quot; !important; border: 0px; max-width: 800px; height: 326px; width: 720px; display: block; min-width: 220px; margin: 10px 0px; box-shadow: rgba(0, 0, 0, 0.05) 0px 0px 5px 0.5px;"></iframe>

## **A Brief History of Reddit Search**

Needless to say, search is not an easy challenge to solve. We’ve been on a bit of a roller coaster when it comes to search at Reddit, but now that we’re on our sixth search stack, we’re no strangers to the struggles of doing search at scale. Below is a rough outline of the 12-year history, along with a few select quotes from the team as we’ve iterated to scale our infra to Reddit’s needs:

- 2005 – Steve Huffman ([u/spez](https://www.reddit.com/user/spez)), co-founder and now CEO, turns on postgres 7.4’s contrib/[tsearch2](http://www.sai.msu.su/~megera/postgres/gist/tsearch/V2/). 

  This was a simpler time, when the statement “Oh, we can just have Postgres do it!” was greeted with “Sounds good to me!  What 

  can’t

   Postgres do!?” We also really liked 

  TRIGGER

  s back then (“No, it’s cool. The database does all the work and it’s guaranteed to be accurate” is something we no doubt said). It worked well, but it wasn’t very tunable, and we quickly discovered we were bogging down the majority of Postgres queries with a small minority (~2%) of search traffic:

  - “We fixed a bug in the search results ordering.” —[Steve](https://redditblog.com/2006/02/27/if-you-want-something-done-right-do-it-yourself/)
  - “We updated the search system this morning to help alleviate some load problems.” —[Steve](https://redditblog.com/2006/07/25/searching/)
  - “Jeremy is working on search! It’s not a complicated fix (basically, the sorting is whacky).” —[Steve](https://redditblog.com/2007/04/28/updates/)

- 2007 – Chris Slowe ([u/KeyserSosa](https://www.reddit.com/user/KeyserSosa)), founding engineer (and now CTO), re-implements with PyLucene.

   

  This was actually implemented just over 10 years ago in July 2007. It consisted of a single Python process which was set up as a threaded RPC server over TCP. In the initial version, we had actually supported searching for both post titles and comments, and the Lucene index files were comfortably stored on a single box. This was also before we 

  moved to AWS

  , and at the time we had seriously considered getting a 

  Google Search Appliance

  , which would have made a nice addition to our 

  single

   rack. This version was flexible, but we didn’t set it up in a way to make it easily scalable:

  - “Search works much better, tagging and user-controlled subreddits are right around the corner” —[Steve](https://redditblog.com/2007/07/26/new-reddit-on-the-horizon/)
  - “Search is better, but not quite where we’d like it.” —[Steve](https://redditblog.com/2007/08/21/its-slow-its-unstable-its-beta/)
  - “Stats and search are temporarily disabled, but will be coming back as soon as we can get them repaired.” —[Steve](https://redditblog.com/2007/10/16/reddit-status-update/)
  - “We were hoping to include an upgraded search, which, unlike the last version, was actually useful and helped you find what you were looking for. Unfortunately, the version we settled on didn’t quite load test as nicely” —[Steve](https://redditblog.com/2007/10/18/reddit-status-update-part-ii/)
  - “I made a quick fix to search that I hope helps until we get a chance to really fix it.” —[Steve](https://redditblog.com/2007/06/08/a-note-on-search-and-what-were-working-on/)

- 2008 – David King ([u/ketralnis](https://www.reddit.com/user/ketralnis)), third employee and now search engineer, implements Solr. 

  In fact, he implemented a home-built pysolr, which was capable of shipping update documents to Solr in XML and wrapping the response in such a way as to emulate our existing 

  Query

   models enough to drop it into any sort or listing. It was actually pretty sweet. The initial version didn’t support comments, but that did come later.

  - “[David]’s been fixing search and hacking mystery projects in Erlang.” —[Alexis Ohanian](https://redditblog.com/2008/04/17/welcome-david/)
  - “I’ve totally replaced the reddit search function.” —[David King](https://redditblog.com/2008/04/21/new-search-2/)

- 2010 – David replaces Solr with IndexTank, a third-party search provider.

   

  When you love something, outsource it… said no one ever. As the site continued to grow and we first cracked a billion 

  pageviews 

  in a month with an engineering team of four, we put all of our effort into 503 mitigation, continuing to add Postgres read slaves, adding more cache, starting to take advantage of a 

  very early version of Cassandra

   (which was followed shortly thereafter by a memorable 24-hour, thundering-herd-related outage), and generally ignoring how bad search was getting. We had an intrepid startup approach us and offer to take search off of our hands 

  forever

   for less than we were paying to keep Solr running, so we signed on!

  - “We launched a new search engine yesterday. Calm down. It’s okay. I know. You’ve been hurt before.” —[David King](https://redditblog.com/2010/07/21/new-search/)

- 2012 – Keith Mitchell (u/kemitche) implements CloudSearch after LinkedIn shut down IndexTank. 

  Clearly, it was one of the 

  shorter

   forevers, but IndexTank served us well until the company was acquired. When we found out they were shutting down, we had to ween off of IndexTank and make a quick transition to AWS CloudSearch. Continuing our long-standing tradition of ‘Let the new guy take care of it,’ that task fell to Keith, and over the next several years we scaled and stretched CloudSearch to bursting:

  - “Today we moved from the old Amazon CloudSearch domain to a new Amazon CloudSearch domain. The old search domain had significant performance issues: roughly 33% of queries took over 5 seconds to complete and would result in the search error page.” —[u/bsimpson](https://www.reddit.com/r/changelog/comments/694o34/reddit_search_performance_improvements/)

- TODAY – Lucidworks Fusion!

   

  This time around, we wanted to ensure that search would meet three criteria: it needed to be fast, it needed to scale well with Reddit’s growth, and most importantly, it needed to be relevant. Ultimately, this led us to partner with the search experts at Lucidworks, leveraging Fusion and their unique search expertise from a team comprised of multiple Solr committers. Below, we’ll explain how we went about this in more detail.

  - “As [/u/bitofsalt](https://www.reddit.com/u/bitofsalt) [mentioned a few months ago](https://www.reddit.com/r/funny/comments/65ryr3/and_now_a_look_at_the_machine_that_powers_reddits/dgd22mi/), we’ve been working on some improvements to search. We may even be ahead of [spez’s 10 year plan](https://www.reddit.com/r/announcements/comments/59k22p/hey_its_reddits_totally_politically_neutral_ceo/d992fwq/?context=1).” —[u/starfishjenga](https://www.reddit.com/r/changelog/comments/6pi0kk/improving_search/)

## **Once More with Feeling**

Earlier this year, search on Reddit had become truly abysmal. Simple queries could be expected to succeed only half of the time. Want to search with two keywords? Get out of here!

![img](https://redditupvoted.files.wordpress.com/2017/09/screen-shot-2017-09-07-at-11-43-15-am.png?w=720&h=505)

Fig. 1: Example error page when our CloudSearch cluster is under heavy load.

After looking at several options, we partnered with with [Lucidworks](https://lucidworks.com/) to revitalize Reddit’s search system. Lucidworks is the creator of Fusion, a Solr-based search stack that supports huge document scale and high query throughput.

## **First Things First: Ingesting at Reddit Scale**

The biggest challenge in moving to a new search system was that our indexing pipeline needed to be updated. The first attempt was a bit of a beast. In the interest of speed, we hastily put it together on our legacy ETL system comprised of [Jenkins](https://jenkins.io/) and [Azkaban](https://azkaban.github.io/) orchestrating numerous Hive queries. As you can see in the diagram below, pulling together data from several sources into one cohesive canonical view to be indexed proved to be more complex than originally expected.

![img](https://redditupvoted.files.wordpress.com/2017/09/screen-shot-2017-09-07-at-11-44-37-am.png?w=720&h=433)

Fig. 2: First iteration at our new search ingestion pipeline, now replaced with a significantly simplified version.

Our second attempt was both simpler and produced significantly better results. We managed to trim the entire pipeline to just four simpler and more accurate Hive queries, which led to a 33% increase in posts indexed. Another great improvement is that we not only index new post creations but also update their relevance signals in real time as votes, comments, and other signals flow in throughout the day.

## **Make it Relevant**

Search results don’t mean much if they’re not relevant. For our initial rollout the primary goal was to avoid degrading the overall relevance of results returned.

To monitor this, we measured clicks on the search results page and compared the rank of results being clicked across old and new search systems. A perfect search engine would yield 100% of clicks on the top result being returned, which is another way of saying you want the most relevant result at the top. Since we know a perfect search engine isn’t an achievable goal, we use measures like [Mean Reciprocal Rank](https://en.wikipedia.org/wiki/Mean_reciprocal_rank) and [Discounted Cumulative Gain](https://en.wikipedia.org/wiki/Discounted_cumulative_gain) to compare the quality of our results.

While it’s still early in our experiments, the data so far points towards very comparable relevancy measurements between our old vs. new stacks, with Fusion having a slight edge. The promising part of this is that we haven’t done much relevancy tuning yet — something that our new system actually supports. Advancements like personalization, machine learning models, and query intent and rewriting are now low-hanging fruit.

![img](https://redditupvoted.files.wordpress.com/2017/09/screen-shot-2017-09-07-at-11-46-10-am.png?w=720&h=276)

Fig. 3: Comparison of search result click positions between Fusion and CloudSearch stacks.

## **The Rollout**

As we overcame the data ingestion challenges and monitored relevance, we continued to ramp up usage to more and more redditors. The [feedback](https://www.reddit.com/r/changelog/comments/6pi0kk/improving_search/) from this early group was invaluable, and we owe the community a huge thank-you in helping us surface bugs and less common use cases. We started out with just 1% of users on the new stack, working through issues reported and improving the ingestion pipeline as we increased rollout percentages to 5, 10, 25 and ultimately 50% of traffic prior to GA. Throughout this time, we sent all search queries as dark traffic to our new search cluster to ensure it would be ready for full scale as we increased rollout percentages.

![img](https://redditupvoted.files.wordpress.com/2017/09/screen-shot-2017-09-07-at-11-47-24-am.png?w=720&h=376)

Fig. 4: CloudSearch Errors in yellow and Fusion in green.

We’re proud to say that Reddit Search is better than ever! A full reindex of all Reddit content now completes in about 5 hours (down from around 11 hours), and we’re constantly streaming live updates to the index. The error rate is down by two orders of magnitude with 99% of search results served in under 500ms. The number of machines needed to run search dropped from ~200 earlier this year down to ~30 so we even managed to get some cost savings.

![img](https://redditupvoted.files.wordpress.com/2017/09/screen-shot-2017-09-07-at-11-48-12-am.png?w=720&h=308)

Fig. 5: Overview of Reddit’s new search stack.

Faster, more reliable, more relevant, and lower cost! Certainly this shall be the last time we ever need to change our search stack!

## **The Future**

In all seriousness, we think you’ll love this update. It’s our hope that the new search stack will be a foundation for improvements that make it easier to discover all the great content on Reddit. More importantly: we’re not done. Fixing search is just the first step in a series of new capabilities that will make Reddit feel more personalized and relevant to your interests. Reddit finally has a Search & Relevance team, and we are hiring like crazy. If you’re excited about working with one of the world’s most interesting datasets on a search and relevance platform used by hundreds of millions of people, then check out our job listings:

**Head of Search:** [https://boards.greenhouse.io/reddit/jobs/723000#.Wa3yONOGOEI
](https://boards.greenhouse.io/reddit/jobs/723000#.Wa3yONOGOEI)**Head of Relevance:** [https://boards.greenhouse.io/reddit/jobs/611466#.WbC_ltOGOEI
](https://boards.greenhouse.io/reddit/jobs/611466#.WbC_ltOGOEI)**Head of Discovery:** [https://boards.greenhouse.io/reddit/jobs/764831#.WbC_2NOGOEI
](https://boards.greenhouse.io/reddit/jobs/764831#.WbC_2NOGOEI)**Search Engineers:** <https://boards.greenhouse.io/reddit/jobs/612128#.Wa3yQtOGOEI>

Finally, thanks to the Lucidworks team for an amazing partnership and helping us end the search for better search at Reddit.