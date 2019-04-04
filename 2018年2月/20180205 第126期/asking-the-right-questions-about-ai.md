# Asking the Right Questions About AI

In the past few years, we’ve been deluged with discussions of how artificial intelligence (AI) will either save or destroy the world. Self-driving cars will keep us alive; social media bubbles will destroy democracy; robot toasters will rob us of our ability to heat bread.

It’s probably pretty clear to you that some of this is nonsense, and that some of this is real. But if you aren’t deeply immersed in the field, it can be hard to guess which is which. And while there are endless primers on the Internet for people who want to learn to program AI, there aren’t many explanations of the ideas, and the social and ethical challenges they imply, for people who aren’t and don’t want to be software engineers or statisticians.

And if we want to be able to have real discussions about this as a society, we need to fix that. So today, we’re going to talk about the realities of AI: what it can and can’t actually do, what it might be able to do in the future, and what some of the social, cultural, and ethical challenges it poses are. I won’t cover every possible challenge; some of them, like filter bubbles and disinformation, are so big that they need entire articles of their own. But I want to give you enough examples of the real problems that we face that you’ll be situated to start to ask hard questions on your own.

I’ll give you one spoiler to start with: most of the hardest challenges aren’t technological at all. The biggest challenges of AI often start when writing it makes us have to be very explicit about our goals, in a way that almost nothing else does — and sometimes, we don’t want to be that honest with ourselves.


-

#### Artificial Intelligence and Machine Learning

As I write this, I’m going to use the terms “artificial intelligence” (AI) and “machine learning” (ML) more or less interchangeably. There’s a stupid reason these terms mean almost the same thing: it’s that “artificial intelligence” has historically been defined as “whatever computers can’t do yet.” For years, people argued that it would take true artificial intelligence to play chess, or simulate conversations, or recognize images; every time one of those things actually happened, the goalposts got moved. The phrase “artificial intelligence” was just too frightening: it cut too close, perhaps, to the way we define ourselves, and what makes us different as humans. So at some point, professionals started using the term “machine learning” to avoid the entire conversation, and it stuck. But it never really stuck, and if I only talked about “machine learning” I’d sound strangely mechanical — because even professionals talk about AI all the time.

So let’s start by talking about what machine learning, or artificial intelligence, is. In the strictest sense, machine learning is part of the field of “predictive statistics:” it’s all about building systems which can take information about things which happened in the past, and make out of those some kind of model of the world around them which they can then use to predict what might happen under other circumstances. This can be as simple as “when I turn the wheel left, the car tends to turn left, too,” or as complicated as trying to understand a person’s entire life and tastes.

You can use this picture to understand what every AI does:

![](https://cdn-images-1.medium.com/max/1600/1*C1nHSAeA5wmHUsBFPlMPww.png)

There’s a system with some sensors that can perceive the world — these can be anything from video cameras and LIDAR to a web crawler looking at documents. There’s some other system which can act on the world, by doing anything from driving a car to showing ads to sorting fish. Sometimes, this system is a machine, and sometimes it’s a person, who has to make decisions based on something hopelessly complex or too large to think about at once — like the entire contents of the Internet.

To connect the two, you need a box that takes the perceptions of the world, and comes out with advice about what what is likely to happen if you take various courses of action. That box in the center is called a “model,” as in “a model of how the world works,” and that box is the AI part.

The diagram above has some extra words in it, which are ones you may hear when professionals discuss AI. “Features” are simply some distillation of the raw perceptions, the parts of those perceptions which the designers of the model thought would be useful to include. In some AI systems, the features are just the raw perceptions — for example, the color seen by every pixel of a camera. Such a huge number of features is good for the AI in that it doesn’t impose any preconceptions of what is and isn’t important, but makes it harder to build the AI itself; it’s only in the past decade or so that it’s become possible to build computers big enough to handle that.

“Predictions” are what comes out the other end: when you present the model with some set of features, it will generally give you a bunch of possible outcomes, and its best understanding of the likelihood of each. If you want an AI to make a decision, you then apply some rule to that — for example, “pick the one most likely to succeed,” or “pick the one least likely to cause a catastrophic failure.” That final rule, of weighing possible costs and benefits, is no less important to the system than the model itself.

Now, you could imagine a very simple “model” that gives rules that are just fine for many uses: for example, the mechanical regulator valves on old steam engines were a kind of simple “model” which read the pressure in on one end, and if that pressure pushed a lever beyond some set point, it would open a valve. It was a simple rule: if the pressure is above the set point, open the valve; otherwise, close it.

The reason this valve is so simple is that it only needs to consider one input, and make one decision. If it had to decide something more complicated that depended on thousands or millions of inputs — like how to control a car (that depends on all of your vision, hearing, and more), or which web page might give the best answer to your question about wombat farming (that depends on whether you’re casually interested or a professional marsupial wrangler, and on interpreting if the site was written by an enthusiast or is just trying to sell you cheap generic wombat Viagra) — you would find not one simple comparison, but millions, even tens of millions, needed to decide.

> AI’s don’t get bored or distracted: a model can keep making decisions over different pieces of data, millions or billions of times in a row, and not get any worse (or better) at it.

What makes AI models special is that they are designed for this. Inside any AI model are a bunch of rules to combine features, each of which depends on one of hundreds, thousands, or even millions of individual knobs, telling it how much to weigh the significance of each feature under different circumstances. For example, in one kind of AI model called a “decision tree,” the model looks like a giant tree of yes/no questions. If the AI’s job were to tell tuna from salmon, the very first question may be “is the left half of the picture darker than the right half?,” and by the end of it it would look like “given the answers to the past 374 questions, is the average color of pixels in this square more orange or red?” The “knobs” here are the order in which questions are asked, and what the boundaries between a yes and a no for each of them are.

Here’s the magic: It would be impossible to find the right combination of settings which would reliably tell a tuna from a salmon. There are just too many of them. So to start out with, AI’s run in “training mode.” The AI is shown one example after another, each time adjusting its knobs so that it gets better at guessing what will come next, correcting itself after each mistake. The more examples it sees, and the more different examples it sees, the better it gets at telling the crucial from the incidental. And once it has been trained, the values of the knobs are fixed, and the model can be put to use, connected to real actuators.

The advantage that ML models have over humans doing the same task isn’t speed; an ML model typically takes a few milliseconds to make a decision, which is roughly what a human takes as well. (You do this all the time while driving a car) Their real advantage is that they don’t get bored or distracted: an ML model can keep making decisions over different pieces of data, millions or billions of times in a row, and not get any worse (or better) at it. That means you can apply them to problems that humans are very bad at — like ranking billions of web pages for a single search, or driving a car.

(Humans are terrible at driving cars: that’s why [35,000 people were killed](https://en.wikipedia.org/wiki/List_of_motor_vehicle_deaths_in_U.S._by_year) by them in the US alone in 2015. The huge majority of these crashes were due to distraction or driver error — things that people normally do just fine, but failed to do just once at a critical moment. Driving requires tremendous awareness and the ability to react within a small fraction of a second, something which if you think about it is kind of amazing we can do at all. But worse, it requires the ability to consistently do that for hours on end, something which it turns out we actually can’t do.)

When someone is talking about using AI in a project, they mean breaking the project down into the components drawn above, and then building the right model. That process starts by gathering training examples, which is often the hardest part of the task; then choosing the basic shape of the model (which is what things like “neural networks,” “decision trees,” and so on are; these are basic kinds of model which are good for different problems) and running the training; and then, most importantly, figuring out what’s broken and adjusting it.

For example, look at the following six pictures, and figure out the key difference between the first three and the second three:

If you guessed “the first three have carpet in them,” you’re right!

You would also be right, of course, if you had guessed that the first three were pictures of grey cats, and the second three were pictures of white cats. But if you had used these images to train your Grey Cat Detector, you might get excellent performance when the model tries to rate your training pictures, and terrible performance in the real world, because what the model actually learned was “grey cats are cat-shaped things which sit on carpets.”

This is called “overfitting:” when your model has learned idiosyncrasies of the training data, rather than what you actually cared about. Avoiding this is what people who build ML systems spend most of their time worrying about.


-

#### What AI is good at and bad at

So now that we’ve talked about what AI (or ML) is, let’s talk about where it’s actually useful or useless.

Problems where both the goals, and the means to achieve those goals, are well-understood don’t even require AI. For example, if your goal is “tighten all the nuts on this car wheel to 100 foot-pounds,” all you need is a mechanism that can tighten and measure torque, and stops tightening when the torque reaches 100. This is called a “torque wrench,” and if someone offers you an artificially intelligent torque wrench the correct first question to ask them is why would I want that. These are the steam relief valves of AI; all you need is a simple mechanism.

AI shines in problems where the goals are understood, but the means aren’t. This is easiest to do when:

* The number of possible external stimuli is limited, so the model has a chance to learn about them, and

* The number of things you have to control is limited, so you don’t need to look at an overwhelming range of options, and

* The number of stimuli or decisions is still so big that you can’t just write down the rule; and separately, that

* It’s easy to connect one of your actions to an observable consequence in the outside world, so you can easily figure out what did and didn’t work.

These things are harder than they seem. For example, pick up an object sitting next to you right now — I’ll do it with an empty soda can. Now do that again slowly, and watch what your arm did.

My arm rotated at the elbow quickly to move my hand from horizontal on the keyboard to vertical, a few inches from the can, then quickly stopped. Then it moved forward, while I opened my fingers just a bit larger than the can, more slowly than the first motion but still somewhat rapidly, until I saw that my thumb was on the opposite side of the can from my other fingers — despite the fact that my other fingers were obscured from my sight by the can. Then my fingers closed until they met resistance, and stopped almost immediately. And as my arm started to raise up, this time from the shoulder (keeping the elbow fixed), their grip tightened infinitesimally, until it was securely holding, but not deforming, the can.

The fact that we can walk without falling on our faces in confusion is a lot more amazing than it seems. Next time you walk across the room, pay attention to the exact path you take, each time you bend or move your body or put your foot down anywhere except directly in front of you. “Motion planning,” as this problem is called in robotics, is really hard.

This is one of the tasks which is so hard that our brains have a double-digit percentage of their mass dedicated to nothing else. That makes them seem far easier to us than they actually are. Other tasks in this category are face recognition (a lot of our brain is dedicated not to general vision, but specifically to recognizing faces), understanding words, identifying 3D objects, and moving without running into things. We don’t think of these as hard because they’re so intuitive to us — but they’re intuitive because we have evolved specialized organs to do nothing but be really good at those.

For this narrow set of tasks, computers do very poorly, not because they do worse at them than they do at similar tasks, but because we’re intuitively so good at them that our baseline for what constitutes “acceptable performance” is very high. If we didn’t have a huge chunk of our brain doing nothing but recognizing faces, people would look about as different to us as armadillos do — which is just what happens to computers.

(Conversely, the way humans are wired makes other tasks artificially easy for computers to get “right enough.” For example, human brains are wired to assume, in case of doubt, that something which acts more-or-less alive is actually animate. This means that having convincing dialogues with humans doesn’t require understanding language in general; so long as you can keep the conversation on a more-or-less focused topic, humans will autocorrect around anything unclear. This is why voice assistants are possible. The most famous example of this is [ELIZA](https://en.wikipedia.org/wiki/ELIZA), a 1964 “AI” which mimicked a Rogerian psychotherapist. It would understand just enough of your sentences to ask you to tell it more about various things, and if it got confused, it would fall back on safe questions like “Tell me about your mother.” While it was half meant as a joke, people did report feeling better after talking to it. If you have access to a Google Assistant-powered device, you can tell it “OK Google; talk to Eliza” and see for yourself.)

To understand the last of the problems described above — a case where it’s hard to connect your immediate actions to a consequence — think about learning to play a video game. Some action-consequences are pretty obvious: you zigged when you should have zagged, ran into a wall, game over. But as you get better at a game, you start to realize “crap, I missed that one boost, I’m going to be totally screwed five minutes from now,” and can attribute that decision to a much later consequence. You had to spend a lot of time understanding the mechanics of the game before that connection became understandable to you; AI’s have the same problem.

We’ve talked about cases where the goals and means are understood, and cases where the goals but not the means are understood. There’s a third category, where AI can’t help at all: problems where the goal itself isn’t well understood. After all, if you can’t give the AI a bunch of examples of what is and isn’t a good solution look like, what’s it going to learn from?

We’ll talk about these problems a lot more in a moment, because problems which actually are like this but which we think aren’t are often where the thorniest ethical issues come up. What’s really happening a lot of the time is that either we don’t know what “success” really means (in which case, how do you know if you’ve succeeded?), or worse, we do know — but don’t really want to admit it to ourselves. And the first rule of programming computers is that they’re no good at self-deception: if you want them to do something, you have to actually explain to them what you want.

Before we go into ethics, here’s another way to divide up what AI is good and bad at.

The easiest problem is **clear goals in a predictable environment.**That’s anything from a very simple environment (one lug nut, where you don’t even need AI) to a more complicated, but predictable one (a camera looking at an assembly line, where it knows a car will show up soon and it has to spot the wheels). We’ve been good at automating this for several years.

A harder problem is **clear goals in an unpredictable environment**. Driving a car is a good example of this: the goals (get from point A to point B safely and at a reasonable speed) are straightforward to describe, but the environment can contain arbitrarily many surprises. AI has only developed to the point where these problems can really be attacked in the past few years, which is why we’re now attacking problems like self-driving cars or self-flying airplanes.

Another kind of hard problem is **indirect goals in a predictable environment.**These are problems where the environment makes sense, but the relationship between your actions and these goals is very distant — like playing games. This is another field where we’ve made tremendous progress in the recent past, with AI’s able to do previously-unimaginable things like winning at Go.

Winning at board games isn’t very useful in its own right, but it opens up the path to **indirect goals in an unpredictable environment,**like planning your financial portfolio. This is a harder problem, and we haven’t yet made major inroads on it, but I would expect us to get good at these over the next decade.

And finally you have the hardest case, of **undefined goals**. These can’t be solved by AI at all; you can’t train the system if you can’t tell it what you want it to do. Writing a novel might be an example of this, since there isn’t a clear answer to what makes something a “good novel.” On the other hand, there are specific parts of that problem where goals could be defined — for example, “write a novel which will sell well if marketed as horror.”

Whether this is a good or bad use of AI is left to the reader’s wisdom.


-

#### Ethics and the Real World

So now we can start to look at the meat of our question: what do real-world hard questions look like, ones where AI working or failing could make major differences in people’s lives? And what kinds of questions keep coming up?

I could easily fill a bookshelf with discussions of this; there’s no way to look at every interesting problem in this field, or even at most of them. But I’ll give you six examples which I’ve found have helped me think about a lot of other problems, in turn — not in that they gave me the right answers, but in that they helped me ask the right questions.

#### 1. The Passenger and the Pedestrian

A self-driving car is crossing a narrow bridge, when a child suddenly darts out in front of it. It’s too late to stop; all the car can do is go forward, striking the child, or swerve, sending itself and its passenger into the rushing river below. What should it do?

I’m starting with this problem because it’s been discussed a lot in public in the past few years, and the discussion has often been remarkably intelligent, and shows off the kinds of question we really need to ask.

First of all, there’s a big caveat to this entire question: this problem matters very little in practice, because the whole point of self-driving cars is that they don’t get into this situation in the first place. Children rarely appear out of nowhere; mostly when that happens, either the driver was going too fast for their own reflexes to handle a child jumping out from behind an obstruction they could see, or the driver was distracted and for some reason didn’t notice the chid until too late. These are both exactly the sorts of things that an automatic driver has no problem with: looking at all the signals around at once, for hours on end, without getting bored or distracted. A situation like this one would become vanishingly rare, and that’s where the lives saved come from.

But “almost never” isn’t the same thing as “never,” and we have to accept that sometimes this will happen. When it does, what should the car do? Should it prioritize the life of its passengers, or of pedestrians?

This isn’t a technology question: it’s a policy question, and in the form above, it’s been boiled down to its simple core. We could agree on either answer (or any combination) as a society, and we can program the cars to do that. If we don’t like the answer, we can change it.

There’s one big way in which this is different from the world we inhabit today. If you ask people what they would do in this situation, they’ll give a wide variety of answers, and caveat them with all sorts of “it depends”es. The fact is that we don’t want to have to make this decision, and we certainly don’t want to publicly admit if our decision is to protect ourselves over the child. When people actually are in such situations, their responses end up all over the map.

Culturally, we have an answer for this: in the heat of the moment, in that split-second between when you see oncoming disaster and when it happens, we recognize that we can’t make rational decisions. We will end up both holding the driver accountable for their decision, and recognizing it as inevitable, no matter what they decide. (Although we might hold them much more accountable for decisions they made before that final split-second, like speeding or driving drunk.)

With a self-driving car, we don’t have that option; the programming literally has a space in it where it’s asking us now, years before the accident happens: “When this happens, what should I do? How should I weight the risk to the passenger against the risk to the pedestrian?”

And it will do what we tell it to. The task of programming a computer requires brutal honesty about what we want it to decide. When these decisions affect society as a whole, as they do in this case, that means that as a society, we are faced with similarly hard choices.

#### 2. Polite fictions

Machine-learned models have a very nasty habit: they will learn what the data shows them, and then tell you what they’ve learned. They obstinately refuse to learn “the world as we wish it were,” or “the world as we like to claim it is,” unless we explicitly explain to them what that is — even if we like to pretend that we’re doing no such thing.

In mid-2016, high school student Kabir Alli tried doing Google image searches for “three white teenagers” and “three black teenagers.” The results were even worse than you’d expect.

![](https://cdn-images-1.medium.com/max/1600/1*_f8L6Et6QQIloKwnqjUTpg.jpeg)

“Three white teenagers” turned up stock photography of attractive, athletic teens; “three black teenagers” turned up mug shots, from news stories about three black teenagers being arrested. (Nowadays, either search mostly turns up news stories about this event)

What happened here wasn’t a bias in Google’s algorithms: it was a bias in the underlying data. This particular bias was a combination of “invisible whiteness” and media bias in reporting: if three white teenagers are arrested for a crime, not only are news media much less likely to show their mug shots, but they’re less likely to refer to them as “white teenagers.” In fact, nearly the only time groups of teenagers were explicitly labeled as being “white” was in stock photography catalogues. But if three black teenagers are arrested, you can count on that phrase showing up a lot in the press coverage.

Many people were shocked by these results, because they seemed so at odds with our national idea of being a “post-racial” society. (Remember that this was in mid-2016) But the underlying data was very clear: when people said “three black teenagers” in media with high-quality images, they were almost always talking about them as criminals, and when they talked about “three white teenagers,” they were almost always advertising stock photography.

The fact is that these biases do exist in our society, and they’re reflected in nearly any piece of data you look at. In the United States, it’s a good bet that if your data doesn’t show a racial skew of some sort, you’ve done something wrong. If you try to manually “ignore race” by not letting race be an input to your model, it comes in through the back door: for example, someone’s zip code and income predict their race with great precision. An ML model which sees those but not race, and which is asked to predict something which actually is tied to race in our society, will quickly figure that out as its “best rule.”

AI models hold a mirror up to us; they don’t understand when we really don’t want honesty. They will only tell us polite fictions if we tell them how to lie to us ahead of time.

This kind of honesty can force you to be very explicit. A good recent example was in a technical paper about “[word debiasing](https://arxiv.org/abs/1607.06520).” This was about a very popular ML model called word2vec which learned various relationships between the meanings of English words — for example, that “king is to man, as queen is to woman.” The authors of this paper found that it contained quite a few examples of social bias: for example, it would also say that “computer programmer is to man, as homemaker is to woman.” The paper is about a technique they came up with for eliminating that bias.

What isn’t obvious to the casual reader of this paper — including many of the people who wrote news articles about it — is that there’s no automatic way to eliminate bias. Their procedure was quite reasonable: first, they analyzed the word2vec model to find pairs of words which were sharply split along the he/she axis. Next, they asked a bunch of humans to identify which of those pairs represented meaningful splits (e.g., “boy is to man as girl is to woman”) and which represented social biases. Finally, they applied a mathematical technique to subtract off the biases from the model as a whole, leaving behind an improved model.

This is all good work, but it’s important to recognize that the key step in this — of identifying which male/female splits should be removed — was a human decision, not an automatic process. It required people to literally articulate which splits they thought were natural and which ones weren’t. Moreover, there’s a reason the original model derived those splits; it came from analysis of millions of written texts from all over the world. The original word2vec model accurately captured people’s biases; the cleaned model accurately captured the raters’ preference about which of these biases should be removed.

The risk which this highlights is the “[naturalistic fallacy](https://en.wikipedia.org/wiki/Naturalistic_fallacy),” what happens when we confuse what is with what ought to be. The original model is appropriate if we want to use it to study people’s perceptions and behavior; the modified model is appropriate if we want to use it to generate new behavior and communicate some intent to others. It would be wrong to say that the modified model more accurately reflects what the world is; it would be just as wrong to say that because the world is some way, it also ought to be that way. After all, the purpose of any model — AI or mental — is to make decisions. Decisions and actions are entirely about what we wish the world to be like; if they weren’t, we would never do anything at all.

#### 3. The Gorilla Incident

In July of 2015, when I was technical leader for Google’s social efforts (including photos), I received an urgent message from a colleague at Google: our photo indexing system had [publicly described a picture of a Black man and his friend as “gorillas,”](https://www.forbes.com/sites/mzhang/2015/07/01/google-photos-tags-two-african-americans-as-gorillas-through-facial-recognition-software/#653c0745713d) and he was — with good reason — furious.

My immediate response, after swearing loudly, was to page the team and [publicly respond](https://twitter.com/yonatanzunger/status/615355996114804737) that this was not something we considered to be okay. The team sprung into action and disabled the offending characterization, as well as several other potentially risky ones, until they could solve the underlying issue.

Many people suspected that this issue was the same one as the one that caused [HP’s face-tracking webcams to not work on Black people](https://gizmodo.com/5431190/hp-face-tracking-webcams-dont-recognize-black-people) six years earlier: that the training data for “faces” had been composed exclusively of white people. This was the first thing we suspected as well, but it we quickly crossed it off the list: the training data included a wide range of people of all races and colors.

What actually happened was the intersection of three subtle problems.

The first problem was that face recognition is hard. Different people look so vividly different to us precisely because a tremendous fraction of our brain matter is dedicated to nothing but recognizing people’s faces; we’ve spent millions of years evolving tools for nothing else. But if you compare how different two different faces are in to how different, say, two different chairs are, you’ll see that faces are tremendously more similar than you would guess — even across species.

In fact, we discovered that this bug was far from isolated: the system was also prone to misidentifying white faces as dogs and seals.

And this goes to the second problem, which is the real heart of the matter: ML systems are very smart in their domain, but know nothing at all about the broader world, unless they were taught it. And when trying to think about all the ways in which different pictures could be identified as different objects — this AI isn’t just about faces— nobody thought to explain to it the long history of Black people being dehumanized by being compared to apes. That context is what made this error so serious and harmful, while misidentifying someone’s toddler as a seal would just be funny.

There’s no simple answer to this question. When dealing with problems involving humans, the cost of errors is typically tied in with tremendously subtle cultural issues. It’s not so much that it’s hard to explain them as that it’s hard to think of them in advance: quickly, list for me the top cultural sensitivities that might show up around pictures of arms!

This problem doesn’t just manifest in AI: it also manifests when people are asked to make value judgments across cultures. One particular challenge for this is when detecting harassment and abuse online. Such questions are almost entirely handled by humans, rather than AI’s, because it’s extremely difficult to set down rules that even humans can use to judge these things. I spent a year and a half developing such rules at Google, and consider it to be one of the greatest intellectual challenges I’ve ever faced. To give a very simple example: people often say “well, an obvious rule is that if you say n****r, that’s bad.” I challenge you to apply that rule to the different meanings of the word in (1) nearly any of Jay-Z’s songs, (2) Langston Hughes’ poem “[Christ in Alabama](http://www.english.illinois.edu/maps/poets/g_l/hughes/christ.htm),” (3) [that routine](https://www.youtube.com/watch?v=f3PJF0YE-x4) by Chris Rock, (4) that same routine if he had performed it in front of a white audience, (5) and that same routine if Ted Nugent had performed it, verbatim, to one of his audiences, and come up with a coherent explanation of what’s going on. It’s possible; it’s far from simple. And those are just five examples involving published, edited, creative works, not even normal conversation.

Even with teams of people coming up with rules, and humans, not AI’s, enforcing them, cultural barriers are a huge problem. A reviewer in India won’t necessarily have the cultural context around the meaning of a racial slur in America, nor would one in America have cultural context for one in India. But the number of cultures around the world is huge: how do you express these ideas in a way that anyone can learn them?

The lesson is this: often the most dangerous risks in a system come, not from problems within the system, but from unexpected ways that the system can interact with the broader world. We don’t yet have a good way to manage this.

(The third problem in the Gorilla Incident — for those of you who are interested — is a problem of racism in photography. Since the first days of commercial film, the standards for color and image calibration have included things like “[Shirley Cards](http://www.npr.org/2014/11/13/363517842/for-decades-kodak-s-shirley-cards-set-photography-s-skin-tone-standard),” pictures of standardized models. These models were exclusively white until the 1970’s — when [furniture manufacturers complained](https://www.youtube.com/watch?v=d16LNHIEJzs) that film couldn’t accurately capture the brown tones of dark wood! Even though modern color calibration standards are more diverse, our standards for what constitute “good images” still overwhelmingly favor white faces rather than black ones. As a result, amateur pictures of white people with cell phone cameras turn out reasonably well, but amateur pictures of black people — especially dark-skinned people — often come out underexposed. Faces are reduced to vague blobs of brown with eyes and sometimes a mouth, which unsurprisingly are hard for image recognition algorithms to make much sense of. Photography director Ava Berkofsky recently gave an excellent interview on [how to light and photograph Black faces well](https://mic.com/articles/184244/keeping-insecure-lit-hbo-cinematographer-ava-berkofsky-on-properly-lighting-black-faces#.aK60yvtLn).)

#### 4. Unfortunately, the AI will do what you tell it

“The computer has it in for me / I wish that they would sell it. / It never does just what I want / but only what I tell it.” — Anonymous

One important use of AI is to help humans make better decisions: not to directly operate some actuator, but to tell a person what it recommends, and so better-equip them to make a good choice. This is most valuable when the choices have high stakes, but the factors which really affect long-term outcomes aren’t immediately obvious to the humans in the field. In fact, absent clearly useful information, humans may easily act on their unconscious biases, rather than on real data. That’s why many courts started to use automated “risk assessments” as part of their sentencing guidelines.

Modern risk assessments are ML models, tasked with predicting the likelihood of a person committing another crime in the future. Trained on the full corpus of an area’s court history, it can form a surprisingly good picture of who is and isn’t a risk.

If you’ve been reading carefully so far, you may have spotted a few ways this could go horribly, terribly, wrong. And that’s exactly what happened across the country, as revealed by a 2016 [ProPublica exposé](https://www.forbes.com/sites/daniellecitron/2016/07/13/unfairness-of-risk-scores-in-criminal-sentencing/#52fd26854ad2).

The designers of the COMPAS system, the one used by Broward County, Florida, followed best practices. They made sure their training data hadn’t been artificially biased by group, for example making sure there was equal training data about people of all races. They took care to ensure that race was not one of the input features that their model had access to. There was only one problem: their model didn’t predict what they thought it was predicting.

The question that a sentencing risk assessment model ought to be asking is something like, “what is the probability that this person will commit a serious crime in the future, as a function of the sentence you give them now?” That would take into account both the person and the effect of the sentence itself on their future life: will it imprison them forever? Release them with no chance to get a straight job?

> It was trained to answer, “who is more likely to be convicted,” and then asked “who is more likely to commit a crime,” without anyone paying attention to the fact that these are two entirely different questions.

But we don’t have a magic light that goes off every time someone commits a crime, and we certainly don’t have training examples where the same person was given two different sentences at once and turned out two different ways. So the COMPAS model was trained on a proxy for the real, unobtainable data: given the information we know about a person at the time of sentencing, what is the probability that this person will be convicted of a crime? Or phrased as a comparison between two people, “Which of these two people is most likely to be convicted of a crime in the future?”

If you know anything at all about the politics of the United States, you can answer that question immediately: “The Black one!” Black people are tremendously more likely to be [stopped, arrested, convicted, and given long sentences](http://www.slate.com/articles/news_and_politics/crime/2015/08/racial_disparities_in_the_criminal_justice_system_eight_charts_illustrating.html) for identical crimes than white people, so an ML model which looked at the data and, ignoring absolutely everything else, always predicted that a Black defendant is more likely to be convicted of another crime in the future, would in fact be predicting quite accurately.

But what the model was being trained for wasn’t what the model was being used for. It was trained to answer, “who is more likely to be convicted,” and then asked “who is more likely to commit a crime,” without anyone paying attention to the fact that these are two entirely different questions.

(COMPAS’ not using race as an explicit input made no difference: housing is very segregated in much of the US, very much so in Broward County, and so knowing somebody’s address is as good as knowing their race.)

There are obviously many problems at play here. One is that the courts took the AI model far too seriously, using it as a direct factor in sentencing decisions, skipping human judgment, with far more confidence than any model should warrant. (A good rule of thumb, also recently [encoded into EU law](https://gdpr-info.eu/art-22-gdpr/), is that decisions with serious consequences of people should be sanity-checked by a human — and that there should be a human override mechanism available.) Another problem, of course, is the underlying systemic racism which this exposed: the fact that Black people are more likely to be arrested and convicted of the same crimes.

But there’s an issue specific to ML here, and it’s one that bears attention: there is often a difference between the quantity you want to measure, and the one you can measure. When these differ, your ML model will become good at predicting the quantity you measured, not the quantity for which it was meant to be a proxy. You need to very carefully reason about the ways in which these are similar and differ before trusting your model.

#### 5. Man is a rationalizing animal

There is a new buzzword afoot in the discussion of machine learning: the “right to explanation.” The idea is that, if ML is being used to make decisions of any significance at all, people have a right to understand how those decisions were made.

Intuitively, this seems obvious and valuable — yet when this is mentioned around ML professionals, their faces turn colors and they try to explain that what’s requested is physically impossible. Why is this?

First, we should understand why it’s hard to do this; second, and more importantly, we should understand why we expect it to be easy to do, and why this expectation is wrong. And third, we can look at what we can actually do.

Earlier, I described an ML model as containing between hundreds and millions of dials. This doesn’t do justice to the complexity of real models. For example, modern [ML-based language translation systems](https://research.googleblog.com/2016/09/a-neural-network-for-machine.html) take as their input one letter at a time. That means that the model has to express conditions about the state of its understanding of a text after reading however many letters, and how each successive next letter might affect its interpretation of meaning. (And it works; with some language pairs like English and Spanish, it performs as well as humans!)

For any situation the model encounters, the only “explanation” it has of what it’s doing is “well, the following thousand variables were in these states, and then I saw the letter ‘c,’ and I know that this should change the probability of the user talking about a dog according to the following polynomial…”

This isn’t just incomprehensible to you: it’s also incomprehensible to ML researchers. Debugging ML systems is one of the hardest problems in the field, since examining the individual state of the variables at any given time tells you approximately as much about the model as measuring a human’s neural potentials will tell you about what they had for dinner.

And yet — this is coming to the second part — we always feel that we can explain our own decisions, and it’s this kind of explanation that people (especially regulators) keep expecting. “I set the interest rate for this mortgage at 7.25% because of their median FICO score,” they expect it to say, “had their FICO score from Experian been 35 points higher, the rate would have dropped to 7.15%.” Or perhaps, “I recommended we hire this person because of the clarity with which they explained machine learning during our interview.”

But there’s a dark secret which everyone in cognitive or behavioral psychology knows: All of these explanations are nonsense. Our decisions about whether we like someone or not are set within the first few seconds of conversation, and can be influenced by something as seemingly random as [whether they were holding a hot or cold drink before shaking your hand](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2737341/). Unconscious biases pervade our thinking, and [can be measured](https://implicit.harvard.edu/implicit/education.html), even though we aren’t aware of them. Cognitive biases are one of the largest (and IMO most interesting) branches of psychology research today.

What people are good at, it turns out, isn’t explaining how they made decisions: it’s coming up with a reasonable-sounding explanation for their decision [after the fact](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3150852/). Sometimes this is perfectly innocent: for example, we identify some fact which was salient for us in the decision-making process (“I liked the color of the car”) and focus on that, while ignoring things which may have been important to us but were invisible. (“My stepfather had a hatchback. I hated him.”) It can also have deeper motivations: to resolve cognitive dissonance by explaining how we did or didn’t want something anyway (“the grapes were probably sour, anyway”), or to avoid thinking too closely about something we may not want to admit. (“The first candidate sounded just like I did when I graduated. That woman was good, but she felt different… she wouldn’t fit as well working with me.”)

If we expect ML systems to provide actual explanations for their decisions, we will have as much trouble as if we asked humans to explain the actual basis for their own decisions: they don’t know any more than we do.

But when we ask for explanations, what we’re really often interested in is which facts were both salient (in that changing them would have changed the outcome materially) and mutable (in that changes to them are worth discussing). For example, “you were shown this job posting; had you lived ten miles west, you would have seen this one instead” may be interesting in some context, but “you were shown this job posting; had you been an emu, you would instead have been shown a container of mulga seeds” is not.

This information is particularly useful when it’s also provided as an axis for providing feedback to ML systems: for example, by showing people a few salient and mutable items, they may offer corrections to those items, and provide updated data.

Mathematical techniques for producing this kind of explanation are in active development, but you should be aware that there are nontrivial challenges in them. For example, most of these techniques are based on building a second “explanatory” ML model which is less accurate, only useful for inputs which are small variations on some given input (your own), more comprehensible, but based on entirely different principles than the “main” ML model being described. (This is because only a few kinds of ML model, like decision trees, are at all comprehensible by people, while the models most useful in many real applications, like neural nets, decidedly are not.) This means that if you try to give the system feedback saying “no, change this variable!” in terms of the explanatory model, there may be no obvious way to translate that into inputs for the main model at all. Yet if you give people an explanation tool, they’ll also demand the right to change it in the same language — reasonably, but not feasibly.

Humans deal with this by having an extremely general intelligence in their brains, which can handle all sorts of concepts. You can tell it that it should be careful with its image recognition when it touches on racial history, because the same system can understand both of those concepts. We are not yet anywhere close to being able to do that in AI’s.

#### 6. AI is, ultimately, a tool

It’s hard to discuss AI ethics without bringing up everybody’s favorite example: artificially intelligent killer drones. These aircraft fly high in the sky, guided only by a computer which helps them achieve their mission of killing enemy insurgents while preserving civilian life… except when they decide that the mission calls for some “collateral damage,” as the euphemism goes.

People are rightly terrified of such devices, and would be even more terrified if they heard more of the stories of people who [already live](https://www.theguardian.com/world/2017/mar/30/yemen-drone-strikes-trump-escalate) under the perpetual threat of death coming suddenly out of a clear sky.

AI is part of this conversation, but it’s less central to it than we think. Large drones differ from manned aircraft in that their pilots can be thousands of miles away, out of harm’s way. Improvements in autopilot AI’s mean that a single drone operator could soon fly not one aircraft, but a small flight of them. Ultimately, large fleets of drones could be entirely self-piloting 99% of the time, calling in a human only when they needed to make an important decision. This would open up the possibility of much larger fleets of drones, or drone air forces at much lower cost — democratizing the power to bomb people from the sky.

In another version of this story, humans might be taken entirely out of the “kill chain” — the decision process about whether to fire a weapon. (Most Western armies have made quite clear that they have no intention of doing any such thing, because it would be obviously stupid. But an army in extremis may easily do so, if nothing else for the terror it could create — unknown numbers of aircraft flying around, killing at will — and we may expect far more armies to have drones in the future.) Now we might ask, who is morally responsible for a killing decided on entirely by a robot?

The question is both simpler and more complicated than we at first imagine. If someone hits another person over the head with a rock, we blame the person, not the rock. If they throw a spear, even though the spear is “under its own power” for some period of flight, we would never think of blaming it. Even if they construct a complex deathtrap, Indiana Jones-style, the volitional act is the human’s. This question only becomes ambiguous to the extent that the intermediate actor can decide on their own.

The simplicity comes because this question is far from new. Much of the point of military discipline is to create a fighting force which does not try to think too autonomously during battle. In countries whose militaries are descended from European systems, the role of enlisted and noncommissioned officers is to execute on plans; the role of commissioned officers is to decide on which plans to execute. Thus, in theory, the decision responsibility is entirely on the shoulders of the officers, and the clear demarcation of areas of responsibility between officers based on rank, area of command, and so on, determines who is ultimately responsible for any given order.

While in practice, this is often considerably more fuzzy, the principles are ones we’ve understood for millennia, and AI’s add nothing new to the picture. Even at their greatest decision-making capability and autonomy, they would still fit into this discussion — and we’re decades away from them actually having enough autonomy for the conversation to even start to approach the levels we have long had established for these discussions around people.

Perhaps this is the last important lesson of the ethics of AI: many of the problems we face with AI are simply the problems we have faced in the past, brought to the fore by some change in technology. It’s often valuable to look for similar problems in our existing world, to help us understand how we might approach seemingly new ones.


-

#### Where do we go from here?

There are many other problems that we could discuss — many of which are very urgent for us as a society right now. But I hope that the examples and explanations above have given you some context for understanding the kinds of ways in which things can go right and wrong, and where many of the ethical risks in AI systems come from.

These are rarely new problems; rather, the formal process of explaining our desires to a computer — the ultimate case of someone with no cultural context or ability to infer what we don’t say — forces us to be explicit in ways we generally aren’t used to. Whether this involves making a life-or-death decision years ahead of time, rather than delaying it until the heat of the moment, or whether it involves taking a long, hard look at the way our society actually is, and being very explicit about which parts of that we want to keep and which parts we want to change, AI pushes us outside of our comfort zone of polite fictions and into a world where we have to discuss things very explicitly.

Every one of these problems existed long before AI; AI just made us talk about them in a new way. That might not be easy, but the honesty it forces on us may be the most valuable gift our new technology can give us.

