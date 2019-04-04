# A genetic algorithm scheduled our developer conference. Here’s what I learnt.

In December, I posted about my plan to use an evolutionary algorithm to create a conference schedule. The event ([DartConf 2018](https://events.dartlang.org/2018/dartconf/)) took place last week, so this is a good time for me to write a postmortem. I’m also releasing the [source code](https://github.com/filiph/conference_darwin) for the algorithm.

If you want a quick primer on genetic algorithms, I think that [the original article](https://medium.com/@filiph/using-a-genetic-algorithm-to-optimize-developer-conference-schedules-27f13d97fa9a) is quite a good explanation-by-example. But you don’t need it to understand this article. The gist of it is that:

* Scheduling is a (NP-)hard optimization problem.

* Most conference organizers will solve this problem on a “good-enough” basis.

* Hypothesis: a genetic algorithm can help.

### End result

I think this was the first time in the history of Earth that a conference of any kind was scheduled by a genetic algorithm. I may be wrong, though, so we never advertised this.

**UPDATE**: [I was wrong](https://www.reddit.com/r/programming/comments/7uh4vg/a_genetic_algorithm_scheduled_our_developer/dtkbsme/). Though that genetic algorithm (and others like that) was used only for constraint solving, while the algorithm I’m describing has wider scope (see below).

The conference went great. Average 4.6 out of 5 stars, with only a single 3-star rating and zero below that. Happy faces and congrats all around. Many people came up to me after the event and complimented the program, which is something you don’t see very often.

![](https://cdn-images-1.medium.com/max/1600/1*1ZRGUp8KUhyNAGWwch6w2Q.jpeg)

Best of all, the main AV person told one of my colleagues that this was the best conference he’s ever seen — and this guy is at a conference basically every week.

![](https://cdn-images-1.medium.com/max/1200/1*xZ6dVc7ht85CsVugmjqVbQ.png)

Of course, I can’t really take credit for all that — I was only responsible for the program & schedule, and a good conference is so much more than that. I’d say about ¾ of an attendee’s perception is directly influenced by non-program aspects. (Even things like the weather have tremendous effect on how an attendee perceives an event. DartConf took place in Santa Monica: clearly an unfair advantage.) In other words, most of the credit goes to Linda, the main organizer, the speakers, the AV people, the attendees, and generally people who are not me.

That said, the sequence of talks worked really well. Despite the diverse set of topics (Dart, Flutter, AngularDart), experience levels (beginners, intermediates, experts, half-gods) and one or two sub-par talks, there was only one significant dip in attention. (I take this seriously and actually watch attendees as they listen to talks. When people start looking down from the stage and onto their screens, it’s really hard to reverse.)

### The Joy of Forgetting

As I wrote in my first article on this topic, there’s a huge amount of constraints and considerations and best practices that a good conference program must adhere to. No organizer, however genius, can ever hope to hold them all in their mind at once. Here’s a sampling of what it feels like to build a schedule for a conference manually (this follows first-hand experience from a past event):

* You start by making the conference work in its time & space constraints.

* You realize that the flow of topics is completely broken down.

* You try to fix that but it’s not 100% possible without tearing down the whole program.

* After 20 minutes of pointless swapping of session slots, you give up on trying to make it any better and move on to something else.

* Next day you realize there’s a 3-session block of 301 deepdives that will bore the beginners to death (among other issues).

* You spend another 30 minutes swapping.

* You look at the schedule again and realize you’ve broken one of the original space constraints.

* You spend another 20 minutes fixing that problem while making sure you don’t break the other constraints.

* Next day, you look at the program again and realize a 201 session is scheduled before its corresponding 101 talk.

* Also, someone tells you they completely changed the topic of their talk so now it doesn’t fit where it’s scheduled.

Welcome to hell.

This is where my algorithm comes in. All the considerations and constraints are—quite literally—codified. My mind can forget all of them and how they relate to each other. I just modify the inputs and re-run the algorithm and go grab a cup of coffee.

### Tightening constraints

Constraints of a conference schedule tighten over time.

1. As the day of the event approaches, some things that were fluid before get set in stone. For example, at some point you’ll need to tell the catering team when the lunch will be, and you can’t change that a day before the conference.

2. People generally don’t like things to change constantly. For example, while you technically can change the lunch timing every week for 3 months straight, one consequence of that is that the catering team will hate you (and you don’t want that).

![](https://cdn-images-1.medium.com/max/1600/1*YR6i5F1-Jg3ys8Gkcd5Q4g.jpeg)

To point #1: You’ll have to be ready to add new constraints, some of them complex. I was okay with that, because it was my code, not some external tool, and so the sky was the limit. What I’m saying is that you shouldn’t expect your algorithm to be built once and then reused without modification until the day of the conference. In practice, I added [custom evaluator functions](https://github.com/filiph/conference_darwin/blob/3b3602dcaa329d6f28506b99994550bb074787b2/lib/src/evaluator.dart#L273) that you can pass to each run to customize the scoring.

To point #2: There’s a solution that I didn’t get to implement. Basically, one of the objective functions of the scheduling algorithm could be “how different is this from the currently published schedule.” You could slowly increase the weight of this parameter as the day of the conference approaches, so that in the last week before the event, the algorithm would only make changes if it was really worth it. Without this, nothing stops the algorithm from completely reshuffling the conference schedule each time it’s run.

### The many objectives of a schedule

At first, I downplayed the multi-objective character of the problem. I thought I could flatten them all into a single “attendee happiness” fitness function.

In the original version of the algorithm, every evaluated schedule could be measured by a single floating point number.

* Inconvenient things (like when the lunch is too late) would worsen this score.

* Broken constraints (like when there is no lunch scheduled at all) would strongly worsen this score.

![](https://cdn-images-1.medium.com/max/1600/1*suZ45tSEst1GVPsRMxJo7A.jpeg)

This worked reasonably well, but it was more and more obvious that I really need a multi-objective algorithm. With a single scalar, you soon get into a tuning game: is constraint A three times or five times more important than best practice B? You never know. All you can do is to test things, and that’s time-intensive.

Even after much tuning, some of your generated schedules will be “min-maxed”. They will over-optimize in one dimension just because it’s weighted slightly higher than it should be. Remember, a genetic algorithm doesn’t care. It doesn’t have any context. It just does its thing, and its thing is to improve the fitness of the population.

What you really want are high-quality, well-rounded schedules.

Well-rounded, you say? Enter multi-objective genetic algorithms. You can split the different constraints and best practices into categories, and then ask the algorithm to provide pareto fronts.



This results in a higher overall quality of schedules, and it’s much easier to assign scores. The current algorithm has 7 categories.



### Incomplete info

DartConf had a lightning talks session. From the perspective of my algorithm, this was just a single, 90-minute block of content. The lightning talks themselves were organized aside, by my colleague Wm.

A couple of weeks before the conference, it came to my attention that one of the lightning talks is about animations using the redux paradigm, while a long-form session about redux was scheduled for the day after the lightning talks session. At that point it was too costly to reschedule.

In the end, it was fine, but it underlines the fact that the algorithm can’t work with things it doesn’t know.

For future, I think info about sub-content like the aforementioned lightning talks must somehow find its way into the inputs of the algorithm. In theory, there’s nothing stopping me from scheduling even the 5-minute lightning talks using the algorithm but that seems like overkill—we already know they’ll be scheduled together, and the sequence doesn’t really matter with this kind of content.

### The code

[Here’s the code.](https://github.com/filiph/conference_darwin) It’s open source and quite hackable. The README file should bring you up to speed.

![](https://cdn-images-1.medium.com/max/1600/1*QALFivSFrV9FPhDX3KeRRQ.png)

For something more general than scheduling conferences, you can have a look at the underlying [darwin package](https://pub.dartlang.org/packages/darwin) which has been around since 2014. You can get very creative with darwin, as it supports things like fitness sharing and multi-objective optimization, yet maintains a very general approach.

### The conclusion

If you’re in charge of a program of a sizeable conference, and you’re at least a little bit of a perfectionist, trust me: you want this (or something similar). The few hours you’ll need to tweak the inputs is nothing compared to the hours spent scheduling and rescheduling (and rescheduling).

![](https://cdn-images-1.medium.com/max/1600/1*Vdj6MTpdJYwlr-B7js9mHA.jpeg)

There is no way your brain will be able to contain and weigh all the pros and cons of even a single schedule, let alone thousands of its variants. If you’re like me, the thought of trying to keep all the information in your head while on a tight schedule is extremely stressful. It’s great to commit it to an external memory / code.

Don’t get me wrong: you can still have an excellent conference even with a “manual” schedule. Many things are more important than the sequence of talks, and many things about the sequence are easy enough to contain in one’s head. But as you explore this problem more, you realize just how much there is to optimise, and once you have this knowledge, it’s hard to unlearn it.

I think DartConf is a good indication that this attention to detail can nudge a great conference to pretty epic status.

![](https://cdn-images-1.medium.com/max/1600/1*NVVpz3WKraTqrahpOuyPIg.gif)

Thanks for reading this far. If you haven’t already, I really recommend reading [the previous article](https://medium.com/@filiph/using-a-genetic-algorithm-to-optimize-developer-conference-schedules-27f13d97fa9a) explaining more about the algorithm itself.

