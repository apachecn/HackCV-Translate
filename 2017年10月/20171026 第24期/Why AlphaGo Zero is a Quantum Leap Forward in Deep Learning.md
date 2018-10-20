# Why AlphaGo Zero is a Quantum Leap Forward in Deep Learning

原文链接：[Why AlphaGo Zero is a Quantum Leap Forward in Deep Learning](https://medium.com/intuitionmachine/the-strange-loop-in-alphago-zeros-self-play-6e3274fcdd9f?from=hackcv&hmsr=hackcv.com&utm_medium=hackcv.com&utm_source=hackcv.com)

![img](https://cdn-images-1.medium.com/max/1000/1*2OfOOA-dvERoxgbzkCp-CA.jpeg)

Credit: War Games (1983) <http://www.imdb.com/title/tt0086567/>

> Self-play is Automated Knowledge Creation

The 1983 movie “War Games” has a memorable climax where the supercomputer known as WOPR (War Operation Plan Response) is asked to train on itself to discover the concept of an un-winnable game. The character played by Mathew Broderick asks “Is there any way that it can play itself?”



<iframe data-width="640" data-height="480" width="640" height="480" data-src="/media/672b1a5a3423932477f1f4a0b64571ca?postId=6e3274fcdd9f" data-media-id="672b1a5a3423932477f1f4a0b64571ca" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Fi.ytimg.com%2Fvi%2FNHWjlCaIrQo%2Fhqdefault.jpg&amp;key=a19fcc184b9711e1b4764040d3dc5c07" class="progressiveMedia-iframe js-progressiveMedia-iframe" allowfullscreen="" frameborder="0" src="https://medium.com/media/672b1a5a3423932477f1f4a0b64571ca?postId=6e3274fcdd9f" style="user-select: text !important; display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 525px;"></iframe>

34 years later, DeepMind has shown how this is exactly done in real life! The solution is the same, set the number of players to zero (i.e. zero humans).

There is plenty to digest about this latest breakthrough in Deep Learning technology. DeepMind authors use the term “self-play reinforcement learning”. As I remarked in the piece about “[Tribes of AI](https://medium.com/intuitionmachine/the-many-tribes-problem-of-artificial-intelligence-ai-1300faba5b60)”, DeepMind is particularly fond of their Reinforcement Learning (RL) approach. DeepMind has taken the use of Deep Learning layers in combination with more classical RL approaches to an art form.

AlphaGo Zero is the latest incarnation of its Go-playing automation. One would think that it would be hard to top the AlphaGo version that bested the human world champion in Go. AlphaGo Zero however not only beats the previous system, but does it in a manner that validates a revolutionary approach. To be more specific, this is what AlphaGo has been able to accomplish:

1. Beat the previous version of AlphaGo (Final score: 100–0).
2. Learn to perform this task from scratch, without learning from previous human knowledge (i.e. recorded game play).
3. World champion level Go playing in just 3 days of training.
4. Do so with an order of magnitude less neural networks ( 4 TPUs vs 48 TPUs).
5. Do this with less training data (3.9 million games vs 30 millions games).

Each of the above bullet points is a newsworthy headline. The combination of each bullet point and what it reveals is completely overwhelming. This is my honest attempt to make sense of all of this.

The first bullet point for many will seem unremarkable. Perhaps it’s because incremental improvements in technology have always been the norm. Perhaps one algorithm besting another algorithm 100 straight times intuitively doesn’t have the same appeal of one human besting another human 100 straight times. Algorithms don’t have the kind of inconsistency that we find in humans.

One would expect though that the game of Go would have a large enough search space that there would be a chance of a less capable algorithm to be lucky enough to beat a better own. Could it be that AlphaGo Zero has learned new alien moves that its competitors are unable to reason about the same search space and thus having an insurmountable disadvantage. This apparently seems to be the case and is sort of alluded to by the fact that AlphaGo Zero requires less compute resources to best its competitors. Clearly, it’s doing a lot less work, but perhaps it is just working off a much richer language of Go strategy. Less work is what biological creatures aspire to do. Language compression is a means to arrive at less cognitive work.

The second bullet point challenges our current paradigm of supervised only machine learning. The original AlphaGo was bootstrapped using previously recorded tournament gameplay. This was then followed with self-play to improve its two internal neural networks (i.e. policy and value networks). In contrast, AlphaGo Zero started from scratch with just the rules of Go programmed. It also required a single network rather than two. It is indeed surprising that it was able to bootstrap itself and then eventually learning more advanced human strategies as well as previously unknown strategies. Furthermore, the order in what strategies it learned first were sometimes unexpected. It is as if the system had learned a new internal language of how to play Go. It is also interesting to speculate as to the effect of a single integrated neural network versus two disjoint neural networks. Perhaps there are certain strategies that a disjoint network cannot learn.

Humans learn languages through metaphors and stories. The human strategies discovered in Go are referred to with names so as to be recognizable by a player. It could be possible that the human language of Go is inefficient in that it is unable to express more complex compound concepts. What AlphaGo Zero seems to be able to do is perform its moves in a way that satisfies multiple objectives at the same time. So humans and perhaps earlier versions of AlphaGo were constrained to a relatively linear way of thinking, while AlphaGo Zero was not encumbered with an inefficient language of strategy. It is also interesting that one may consider this a system that actually doesn’t use the implicit bias that may reside in a language. David Silver, of DeepMind, has an even more bold claim:

> It’s more powerful than previous approaches because by not using human data, or human expertise in any fashion, we’ve removed the constraints of human knowledge and it is able to create knowledge itself.

The [Atlantic reports](https://www.theatlantic.com/technology/archive/2017/10/alphago-zero-the-ai-that-taught-itself-go/543450/) about some interesting observation of the game play of this new system:

> Expert players are also noticing AlphaGo’s idiosyncrasies. Lockhart and others mention that it almost fights various battles simultaneously, adopting an approach that might seem a bit madcap to human players, who’d probably spend more energy focusing on smaller areas of the board at a time.

The learned language is devoid of any historical baggage that it may have accumulated over the centuries of Go study.

The third bullet point says that training time is also surprisingly less than its previous incarnation. It is as if AlphaGo Zero learns how to improve its own learning.



![img](https://cdn-images-1.medium.com/max/1000/1*MlwJE5cDvc3WL76v_ViERw.gif)

Source: <https://deepmind.com/blog/alphago-zero-learning-scratch/>

It took only 3 days to get to a level that beats the best human player. Furthermore, it just keeps getting better even after it surpasses the best previous AlphaGo implementation. How is it capable of improving its learning continuously? This ability to incrementally learn and improve the same neural network is something we’ve seen in another architecture known as FeedbackNet. In the commonplace SGD based learning, the same network is fed data across multiple epochs.

Here however, each training set is entirely new and increasingly more challenging. It is also analogous to curriculum learning, however the curriculum is intrinsic in the algorithm. The training set is self generated and the calculation of the objective function is derived from the result of MCTS. The network learns by comparing itself not from external training data but from synthetic data that is generated from a previous version of the neural network.

The fourth bullet point, the paper reports that it took only 4 Google TPUs ( [180 teraops each](https://medium.com/intuitionmachine/googles-ai-processor-is-inspired-by-the-heart-d0f01b72defe) ) as compared to 48 TPUs for previous systems. Even surprisingly, the Nature paper notes that this ran on a single system and did not use distributed computing. So anyone with four Volta based Nvidia GPUs has the horse power to replicate these results. Performing a task with 1/10th the amount of compute resources should be a hint to anyone that something fundamentally different is happening over here. I have yet to analyze this in detail, but perhaps the explanation is due to just a simpler architecture.

Finally, the last bullet point where it appears that AGZ advanced its capabilities using less training data. It appears that the synthetic data generated by self-play has more ‘teachable moments’ than data that’s derived from human play. Usually, the way to improve a network is to generate more synthetic data. The usual practice is to augment data by doing all sorts of data manipulations (ex. cropping, translations, etc), however in AGZ’s case, the automation seemed to be able to select richer training data.

Almost every new Deep Learning paper that is published (or found in Arxiv) tends to show at best a small percentage improvement over previous architectures. Almost every time, the newer implementation also requires more resources to achieve higher prediction accuracies. What AlphaGo has shown is unheard of, that is, it requires an order of magnitude less resources and a less complex design, while unequivocally besting all previous algorithms.

Many long time practitioners of reinforcement learning applied to games have commented that the actual design isn’t even novel and has been formulated decades ago. Yet, the efficacy of this approach has finally been experimentally validated by the DeepMind team. In Deep Learning like in sports, you can’t win on paper, you actually have to play the game to see who wins. In short, no matter a simple an idea may be, you just never know how well it will work unless the experiments are actually run.

There is nothing new about the [policy iteration algorithm](https://arxiv.org/abs/1405.2878) or the architecture of the neural network. Policy iteration is a old algorithm that learns improving policies, by alternating between policy estimation and policy improvement . That is, between estimating the value function of the current policy and using the current value function to find a better policy.

The single neural network that it uses is a pedestrian convolution network:

> The overall network depth, in the 20- or 40-block network, is 39 or 79 parameterized layers, respectively, for the residual tower, plus an additional 2 layers for the policy head and 3 layers for the value head.

Like the previous incarnations of AlphaGo, Monte Carlo Tree Search (MCTS) is used to select the next move. AlphaGo Zero takes advantage of the calculations of the tree search as a way to evaluate and train the neural network. So basically, MCTS employing a previously trained neural network, performs a search for winning moves. The policy evaluation estimates the value function from many sampled trajectories. The results of this search is then used to drive the learning of the neural network. So after every game, a new and potentially improved network is selected for the next self-play game. DeepMind calls this “Self-play reinforcement learning”:

> A novel reinforcement learning algorithm. MCTS search is executed, guided by the neural network fθ. The MCTS search outputs probabilities π of playing each move. These search probabilities usually select much stronger moves than the raw move probabilities p of the neural network fθ(s); MCTS may therefore be viewed as a powerful policy improvement operator.

> Self-play with search — using the improved MCTS-based policy to select each move, then using the game winner z as a sample of the value — may be viewed as a powerful policy evaluation operator.

With each iteration of self-play, the system learns to become a stronger player. I find it odd that the exploitive search mechanism is able to creatively discover new strategies while simultaneous using less training data. It is as if self-play is feeding back into itself and learning to learn better.

This self-play reminds me of an earlier writing about “[The Strange Loop in Deep Learning](https://medium.com/intuitionmachine/the-strange-loop-in-deep-learning-38aa7caf6d7d).” I wrote about many recent advances in Deep Learning such as Ladder networks and Generative Adversarial Networks (GANs) that exploited a loop based method to improve recognition and generation. It seems that when you have this kind of mechanism, that is able to perform assessments of its final outputs, that the fidelity is much higher with less training data. In the case of AlphaGo Zero, there’s is no training data to speak of. The training data is generated through self-play. A GAN for example, collaboratively improves its generation by having two networks (discriminator and generator) work with each other. AlphaGo Zero, in contrast pits the capabilities of a network trained in a previous game against that of the current network. In both cases, you have two networks that feed of each other in training.

An important question that should be in everyone’s mind is: “How general is AlphaGo Zero’s algorithm?” DeepMind has publicly stated that they will be [applying this technology to drug discovery](https://www.bloomberg.com/news/articles/2017-10-18/deepmind-s-superpowerful-ai-sets-its-sights-on-drug-discovery?lipi=urn%3Ali%3Apage%3Ad_flagship3_feed%3BDl0KiForSoS7oqq5GWTDCw%3D%3D). Earlier I wrote about how to assess the appropriateness of Deep Learning technologies (see: [Reality Checklist](https://medium.com/intuitionmachine/where-is-deep-learning-applicable-understand-the-known-unknowns-f4272e3136ec)). In that assessment, there are six uncertainties in any domain that needs to be addressed: execution uncertainty, observational uncertainty, duration uncertainty, action uncertainty, evaluation uncertainty and training uncertainty.

In the AlphaGo Zero, the training uncertainty, seems to have been addressed. AlphaGo Zero learns the best strategies by just playing against itself. That is, it is able to “imagine” situations and then discover through self-improvement the best strategies. It can do this efficiently because all the other uncertainties are known. That is, there is no indeterminism in the results of a sequence of actions. There is complete information. The effects of actions are predictable. There is a way to measure success. In short, the behavior of the game of Go is predictable, real world systems however are not.

In many real world contexts however, we can still build accurate simulations or virtual worlds. Certainly the policy iteration methods found here may seem to be applicable to these virtual worlds. Reinforcement learning has been applied to virtual worlds (i.e. video games and strategy games). DeepMind has not yet reported experiments of using policy iteration in Atari games. Most games of course don’t need this sophisticated look ahead that requires MCTS, however there are some games like Montezuma’s Revenge that do. DeepMind’s Atari game experiments were like AlphaGo Zero, in that there was no need for human data to teach a machine.

The difference between AlphaGo Zero and the video game playing machines is that the decision making at every state in the game is much more sophisticated. In fact there is an entire spectrum of decision making required for different games. Is MCTS the most sophisticated algorithm that we will ever need?

There is also a question on strategies that require remembering one’s previous move. AlphaGo Zero appears to only care about the current board state and does not have a bias on what it moved previously. A human sometimes may determine its own action based on its previous move. It is a way of telegraphing actions to an opponent, but it usually is more like a head fake. Perhaps that’s a strategy that only works on humans and not machines! In short, a machine cannot see motion if it was never trained to recognize its value.

This lack of memory affecting strategy may in fact be advantageous. Humans when playing a strategy game will stick to a specific strategy until an unexpected event disrupts that strategy. So long as an opponent’s moves are as expected, there is no need to change a strategy. However, as we’ve seen in the most advanced Poker playing automation, there is a distinct advantage of always calculating strategy from scratch with every move. This approach avoids telegraphing any plans and therefore a good strategy. However, misdirection is a strategy that is effective against humans but not machines that are not trained to be distracted by them. (Editors Note: Apparently previous board states are used as input to the network, so appears this lack of memory observation is incorrect).

Finally, there is a question about the applicability of a turn based game to the real world. Interactions in the real world are more dynamic and continuous, furthermore the time of interaction is unbounded. Go games have a limited number of moves. Perhaps, it doesn’t matter, after all, all interactions require two parties that act and react and predicting the future will always be boxed in time.

If I were to pinpoint the one pragmatic Deep Learning discovery in AlphaGo Zero then it would be the fact that Policy Iteration works surprisingly well using Deep Learning networks. We’ve have hints in previous research that incremental learning was a capability that existed. However, DeepMind has shown unequivocally that incremental learning indeed works effectively well.

AlphaGo Zero appears also to have evolutionary aspects. That is, you select the best version of the newly latest trained network and you discard the previous one. There is indeed something going on here that is eluding a good explanation. The self-play is intrinsically competitive and the MCTS mechanism is an exploratory search mechanism. Without exploration, the system will eventually not be able to beat itself in play. To be effective, the system should be inclined to seek out novel strategies to avoid any stalemate. Like nature’s own evolutionary process that abhors a vacuum, AGZ seems to discover unexplored areas and somehow take advantage of these finds.

One perspective to think about these systems as well as the human mind is in terms of the language that we use. Language is something that you layer more and more complex concepts on top of each other. In the case of AlphaGo Zero, it learned a new language that doesn’t have legacy baggage and it learned one that is so advanced that it is incomprehensible. Not necessarily mutually exclusive. As humans, we understand the world with concepts that originate from our embodiment with our world. That is we have evolved to understand visual-spatial, sequence, rhythm and motion. All our understanding is derived from these basic primitives. However, a machine may possibly discover a concept that may simply not be decomposable to these basic primitives.

Such irony, when DeepMind trained an AI without human bias, humans discovered they didn’t understand It! This in another dimension of incomprehensibility. The concept of “incomprehensibility in the large” in that there is just too much information. Perhaps there is this other concept, that is “incomprehensibility in the small”. That there are primitive concepts that we simply are incapable of understanding. Let this one percolate in your mind for a while. For indeed it is one that is fundamentally shocking and a majority will overlook what DeepMind may have actually uncovered!.


  