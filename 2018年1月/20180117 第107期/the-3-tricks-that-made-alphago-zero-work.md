# The 3 Tricks That Made AlphaGo Zero Work

There were [many advances](http://www.wildml.com/2017/12/ai-and-deep-learning-in-2017-a-year-in-review/) in Deep Learning and AI in 2017, but few generated as much publicity and interest as DeepMind’s [AlphaGo Zero](https://deepmind.com/blog/alphago-zero-learning-scratch/). This program was truly a shocking breakthrough: not only did it beat the prior version of AlphaGo — the program that beat 17 time world champion Lee Sedol just a year and a half earlier — 100–0, it was trained without any data from real human games. [Xavier Amatrain](https://xamat.github.io) called it [“more [significant] than anything…in the last 5 years”](https://www.quora.com/What-is-the-significance-of-AlphaGo-Zero-in-AI-research/answer/Xavier-Amatriain) in Machine Learning.

So how did DeepMind do it? In this essay, I’ll try to give an intuitive idea of the techniques AlphaGo Zero used, what made them work, and what the implications for future AI research are. Let’s start with the general approach that both AlphaGo and AlphaGo Zero took to playing Go.

### DeepMind’s General Approach

Both AlphaGo and AlphaGo Zero evaluated the Go board and chose moves using a combination of two methods:

1. Performing “**lookahead**” search: looking ahead several moves by simulating games, and thus seeing which current move is most likely to lead to a “good” position in the future.

2. Evaluating positions based on an “**intuition**”, of whether a position is “good” or “bad” — that is, likely to lead to a win or a loss.

AlphaGo and AlphaGo Zero both worked by cleverly combining these two methods. Let’s look at each one in turn:

#### Go-Playing Method #1: “Lookahead”

Go is a sufficiently complex game that computers can’t simply search all possible moves using a brute force approach to find the best one (indeed, [they can’t even come close](https://en.wikipedia.org/wiki/Go_and_mathematics)).

![](https://cdn-images-1.medium.com/max/1600/1*qpzAxoUR9POLYl__zJhU5g.png)

The best Go programs prior to AlphaGo overcame this by using “[Monte Carlo Tree Search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)” or MCTS. At a high level, this method involves initially exploring many possible moves on the board, and then focusing this exploration over time as certain moves are found to be more likely to lead to wins than others.

Both AlphaGo and AlphaGo Zero use a relatively straightforward version of MCTS for their “lookahead”, simply using many of the best practices listed in the [Monte Carlo Tree Search Wikipedia page](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) to properly manage the tradeoff between exploring new sequences of move or more deeply explore already-explored sequences (for more, see the details in the “Search” section under “Methods” in [the original AlphaGo Paper published in Nature](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf)).

Though, MCTS had been the core of all successful Go programs prior to AlphaGo, it was DeepMind’s clever combination of this technique with a neural network-based “intuition” that allowed it to surpass human performance.

#### Go-Playing Method #2: “Intuition”

DeepMind’s major innovation with AlphaGo was to use deep neural networks to understand the state of the game, and then use this understanding to intelligently guide the search of the MCTS. More specifically: they trained networks that could look at

1. The current board position

2. Which player was playing,

3. The sequence of recent moves (necessary to rule out certain moves as illegal)

Given this information, the neural networks could recommend:

1. Which move should be played

2. Whether the current player was likely to win or not.

How did DeepMind train neural networks to do this? Here, AlphaGo and AlphaGo Zero used very different approaches; we’ll start first with AlphaGo’s:

#### AlphaGo’s “Intuition”: Policy Network and Value Network

AlphaGo had two separately trained neural networks.

![](https://cdn-images-1.medium.com/max/1600/1*rnmOlqtr_bTSpP6i7GuJ4w.png)

1. The first neural network (initialized randomly) was trained to mimic the play of human experts by being shown 30 million moves from a large database of real games. Solving this was a difficult but straightforward pattern recognition problem of the kind [at which deep neural networks excel](https://towardsdatascience.com/the-5-deep-learning-breakthroughs-you-should-know-about-df27674ccdf2); and indeed, once trained, this network did indeed learn to recommend moves similar to those that it observed human experts playing in their games.

2. DeepMind didn’t just want AlphaGo to mimic human players: they also wanted it to win. To learn to play moves more likely to lead to winning instead of losing, the network — that had been trained to play like a human expert — played games against itself. Moves were then randomly sampled from these “self-play” games; if a given move happened in a game in which the current player ended up winning, the network was trained to be more likely to play moves like that in the future, and vice versa.

DeepMind then combined these two neural networks with MCTS — that is, the program’s “intuition” with its brute force “lookahead” search— in a very clever way: it used the network that had been trained to predict moves to guide which branches of the game tree to search and used the network that had been trained to predict whether a position was “winning” to evaluate the positions it encountered during its search. This allowed AlphaGo to intelligently search upcoming moves and ultimately allowed it to beat Lee Sedol.

![](https://cdn-images-1.medium.com/max/1600/1*s2kMOSdl2AaUwo5QVjpTHA.png)

AlphaGo Zero, however, took this to a whole new level.

### The three tricks that made AlphaGo Zero work

At a high level, AlphaGo Zero works the same way as AlphaGo: specifically, it plays Go by using MCTS-based lookahead search, intelligently guided by a neural network.

However, AlphaGo Zero’s neural network — its “intuition” — was trained completely differently from that of AlphaGo:

#### Trick #1: How to Train Your AlphaGo Zero, Part 1

Let’s say you have a neural network that is attempting to “understand” the game of Go: that is, for every board position, it is using a deep neural network to generate evaluations of what the best moves are. What DeepMind realized is that no matter how intelligent this neural network is — whether it is completely clueless or a Go master — its evaluations can always be made better by MCTS.

Fundamentally, MCTS performs the kind of lookahead search that we would imagine a human master would perform if given enough time: it intelligently guesses which variations— sequences of future moves — are most promising, simulates those variations, evaluates how good they actually are, and updates its assessments of its current best moves accordingly.

An illustration of this is below. Suppose we have a neural network that is reading the board and determining that a given move results in a game being even, with an evaluation of 0.0. Then, the network intelligently looks ahead a few moves and finds a sequence of moves that can be forced from the current position that ends up resulting in an evaluation of 0.5. It can then update its evaluation of the current board position to reflect that it leads to a more favorable position down the road.

![](https://cdn-images-1.medium.com/max/1600/1*hBzorPuADtitET2SZaLN2A.png)

This lookahead search, therefore, can always give us improved data on how good the various moves in the current position that the neural network is evaluating are. This is true whether our neural network is playing at an amateur level or an expert level: we can always generate improve evaluations for it by looking ahead and seeing which of its current options actually lead to better positions.

#### Trick #1 (continued): How to Train Your AlphaGo Zero, Part 2

In addition, just as in AlphaGo, we would also want our neural network to learn which moves are likely to lead to wins. So, also as before, our agent—using its MCTS-improved evaluations and the current state of its neural network — could play games against itself, winning some and losing others.

![](https://cdn-images-1.medium.com/max/1600/1*DB99saQWkvVwPleKaWj-1A.png)

This data, generated purely via lookahead and self-play, is what DeepMind used to train AlphaGo Zero. More specifically:

1. The neural network was trained to play moves that reflected the improved evaluations from performing the “lookahead” search.

2. The neural network was adjusted so that it was more likely to play moves similar to those that led to wins and less likely to play moves similar to those that led to losses during the self-play games.

Much was made of the fact that no games between humans were used to train AlphaGo Zero, and this first “trick” was the reason why: for a given state of a Go agent, it can always be made smarter by performing MCTS-based lookahead and using the results of that lookahead to improve the agent. This is how AlphaGo Zero was able to continuously improve, from when it was an amateur all the way up to when it better than the best human players.

The second trick was a novel neural network structure that I’ll call the “Two Headed Monster”.

#### Trick #2: The Two Headed Monster

AlphaGo Zero’s was its neural network architecture, a “two-headed” architecture. Its first 20 layers or so were layer “blocks” of a type often seen in modern neural net architecures. These layers were followed by **two “heads”**: one head that took the output of the first 20 layers and produced probabilities of the Go agent making certain moves, and another that took the output of the first 20 layers and outputted a probability of the current player winning.

![](https://cdn-images-1.medium.com/max/1600/1*96DnPFNDD8YyN-GK737bBQ.png)

This is quite unusual. In [almost all applications, ](https://towardsdatascience.com/the-5-deep-learning-breakthroughs-you-should-know-about-df27674ccdf2)neural networks output a single, fixed output — such as the probability of an image containing a dog, or a vector containing the probabilities of an image containing one of 10 types of objects. How can a net learn if it is receiving two sets of signals: one on how good its evaluations of the board are, and another how good the specific moves it is selecting are?

The answer is simple: remember that neural networks are fundamentally just mathematical functions with a bunch of parameters that determine the predictions that they make; we “teach” them by repeatedly showing them “correct answers” and having them update their parameters so the answers they produce more closely match these correct answers.

So, when we use the two headed neural net to make a prediction using Head #1, we simply update the parameters that led to making that prediction, namely the parameters in the “Body” and in “Head #1”. Similarly, when we make a prediction using Head #2, we update the parameters in the “Body” and in “Head #2”.

![](https://cdn-images-1.medium.com/max/1600/1*kxeOANM2_4aqGXwGAXsusQ.png)

![](https://cdn-images-1.medium.com/max/1600/1*Bqt3g_0RlcAJNn6KEICNkQ.png)

This is how DeepMind trained its single, “two-headed” neural network that it used to guide MCTS during its search, just as AlphaGo did with two separate neural networks. This trick accounted for half of AlphaGo Zero’s increase in playing strength over AlphaGo.

(this trick is known more technically as Multi-Task Learning with Hard Parameter Sharing. [Sebastian Ruder has a great overview here](http://ruder.io/multi-task/index.html#introduction)).

The other half of the increase in playing strength simply came from bringing the neural network architecture up-to-date with the latest advances in the field:

#### Trick #3: “Residual” Nets

AlphaGo Zero used a more “cutting edge” neural network architecture than AlphaGo. Specifically, they used a “residual” neural network architecture instead of a purely “convolutional” architecture. Residual nets were [pioneered by Microsoft Research in late 2015](https://arxiv.org/pdf/1512.03385.pdf), right around the time work on the first version of AlphaGo would have wrapped up, so it both understandable that DeepMind did not use them in the original AlphaGo program.

![](https://cdn-images-1.medium.com/max/1600/1*aJCekYFA3jG0NDBmBEYYPA.png)

Interestingly, as the chart below shows, each of these two neural network-related tricks — switching from convolutional to residual architecture and using the “Two Headed Monster” neural network architecture instead of separate neural networks — would have resulted in about half of the increase in playing strength as was achieved when both were combined.

![](https://cdn-images-1.medium.com/max/1600/1*3Yl6HAo-3YVwdbKpSVZY_A.png)

#### Summary of Tricks

These three tricks are what enabled AlphaGo Zero to achieve its incredible performance that blew away even Alpha Go:

1. Using the evaluations provided by Monte Carlo Tree Search— “intelligent lookahead” — to continually improve the neural network’s evaluation of board positions, instead of using human games.

2. Using one neural network — the “Two Headed Monster” that simultaneously learns both which moves “intelligent lookahead” would recommend and which moves are likely to lead to victory — instead of two separate neural networks.

3. Using a more cutting edge neural network architecture — a “residual” architecture rather than a “convolutional” architecture.

#### One comment

It is worth noting that AlphaGo did not use any classical or even “cutting edge” reinforcement learning concepts — no Deep Q Learning, Asynchronous Actor-Critic Agents, or anything else we typically associate with reinforcement learning. It simply used simulations to generate training data for its neural nets to then learn from in a supervised fashion. [Denny Britz](https://twitter.com/dennybritz) sums this idea up well in this Tweet from just after when the AlphaGo Zero paper was released:



#### The Numbers: Training AlphaGo Zero, Step-by-Step

Here’s a “step-by-step” timeline of how AlphaGo Zero was trained:

1. Initialize neural network.

2. Play self-play games, using **1,600** MCTS simulations per move (which takes about 0.4 seconds).

![](https://cdn-images-1.medium.com/max/1600/1*IRbL8abD_fN4tafR6cfMlg.png)

3. As these self-play games are happening, sample **2,048** positions from the most recent **500,000** games, along with whether the game was won or lost. For each move, record both A) the results of the MCTS evaluations of those positions — how “good” the various moves in these positions were based on lookahead — and B) whether the current player won or lost the game.

4. Train the neural network, using both A) the move evaluations produced by the MCTS lookahead search and B) whether the current player won or lost.

5. Finally, every 1,000 iterations of steps 3–4, evaluate the current neural network against the previous best version; if it wins at least 55% of the games, begin using it to generate self-play games instead of the prior version.

Repeat steps 3–4 **700,000** times, while the self-play games are continuously being played — after three days, you’ll have yourself an AlphaGo Zero!

### Implications for the rest of AI

There are many implications of DeepMind’s incredible achievement for the future of AI research. Here are a couple of key ones:

First, the fact that self-play data generated from simulations was “good enough” to be able to train the network suggests that **simulated self-play data can train agents to surpass human performance in extremely complex tasks, even starting completely from scratch** — data generated from human experts may not be needed.

Second, **the “Two Headed Monster” trick seems to significantly help agents learn to perform several related tasks in many domains**, since it seems to prevent the agents from overfitting their behavior to any individual task. DeepMind seems to really like this trick, and has used it and more advanced versions of it to build agents that can learn multiple tasks in [several](https://arxiv.org/pdf/1707.04175.pdf) [different](https://arxiv.org/pdf/1701.08734.pdf) [domains](https://arxiv.org/pdf/1708.07860.pdf).

![](https://cdn-images-1.medium.com/max/1600/1*ZFqEYHfcP-8mLAYi2wz8gw.png)

Many projects in robotics, especially the burgeoning field of using simulations to teach robotic agents to use their limbs to accomplish tasks, are using these two tricks to great effect. [Pieter Abbeel’s recent NIPS keynote](https://www.youtube.com/watch?v=TyOooJC_bLY) highlights many impressive new results that use these tricks along with many bleeding edge reinforcement learning techniques. Indeed, locomotion seems like a perfect use case for the “Two Headed Monster” trick in particular: for example, robotic agents could be simultaneously trained to hit a baseball using a bat and to throw a punch to hit a moving target, since the two tasks require learning some common skills (e.g. balance, torso rotation).

![](https://cdn-images-1.medium.com/max/1600/1*la16q_VbyN_l3iejXSJ3og.jpeg)

DeepMind’s AlphaGo Zero was one of the most intriguing advancements in AI and Deep Learning in 2017. I can’t wait to see what 2018 brings!

