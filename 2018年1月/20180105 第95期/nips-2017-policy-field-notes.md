# NIPS 2017: Policy Field Notes

By [Tim Hwang](https://medium.com/@timhwang) and [Jack Clark](https://medium.com/@jackclarksf)

![](https://cdn-images-1.medium.com/max/1600/1*b2GmbgbtGuiIHvegPEax1Q.jpeg)

NIPS, as per usual, was insane. But, we survived.

Following on our [policy field notes from NIPS 2016](https://medium.com/@timhwang/policy-field-notes-nips-update-b77e346265a6), we’ve decided to jot down some thoughts inspired by this year’s conference, highlighting a few recent technical developments and impressions on their likely impact on the public policy discussion around machine learning.

We hope recaps like these might help navigate through the neverending waves of research crashing through the field, and bridge the research work with broader discussions happening around the social impact of the technology.

—

[Can Decentralized Algorithms Outperform Centralized Algorithms? A Case Study for Decentralized Parallel Stochastic Gradient Descent](https://arxiv.org/abs/1705.09056)****— Lian, Zhang, et al.

Though often missed in popular coverage of machine learning, the architecture on which training is done has important implications for the social impact of the technology. The dominant model, which requires data to be centralized and then trained on, means that effective, optimal use of machine learning rests somewhat in tension with the demands to preserve privacy (though research is ongoing that works to mitigate the extent of this trade-off).

One alternative architecture takes a more distributed approach, lowering the data that needs to pass through a centralized point. However, these have been disfavored as being less efficient. Focusing on parallel stochastic gradient descent, the authors find a set of conditions under which decentralized setups actually outperform centralized architectures. That’s a neat finding even if what’s contemplated is that all the servers are running a decentralized algorithm within a single data center.

One big question on the tension between privacy and machine learning performance is the economics of the situation: do companies investing and deploying these systems have incentives to preserve both? Making sure that happens will require that the performance of centralized and decentralized algorithms here are at least at parity, and for the latter to surpass the former in the best case. This paper marks a contribution in that direction.

—

[A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)****— Lundberg, Lee

Interest in the research questions around model interpretability continues to grow. This year’s Interpretable ML workshop had huge attendance (and resulted in a [ton of awesome papers you should check out](http://interpretable.ml/)).

One major debate that continues to play out is the ultimate goal of research into interpretability, and what significance it practically has, if any at all. As Zach Lipton observes, claims that the “[doctor just won’t accept that](https://arxiv.org/abs/1711.08037)” remains an open and under-investigated question, and [a position taken in an evening debate](https://twitter.com/davegershgorn/status/938992170836430849?lang=en) on the topic asserted that the usefulness of interpretability is actually quite limited.

Of course, a core challenge here is that whether or not something is interpretable depends very much on the beholder in question. So, investigating it seriously would take the field towards research methods and experimentation that look much more like what happens in, say, human-computer interaction, rather than pure computer science. How the field evolves to adapt or reject this opening to go across disciplines will be an interesting thing to keep an eye on.

This paper represents an approach straddling in some sense this debate. The authors propose a common approach to interpretability that links a number of earlier approaches — LIME, DeepLIFT, Shapely Value Estimation, and so on — and evaluates “interpretability” based on a user study that evaluates machine interpretation against human understandings of the model. It’s unclear, though, if that really is the kind of interpretability success metric that should hold sway, particularly given research showing the extent to which interpretability can be manipulated (see, e.g., [The (Un)reliability of Saliency Methods](https://arxiv.org/abs/1711.00867); [Interpretability of Neural Networks is Fragile](https://arxiv.org/abs/1710.10547)). Whether a consensus among technical experts emerges is an open question, but if it does, it might prove influential as [governments around the world consider GDPR style options](https://arxiv.org/abs/1711.01134) which would demand explanation from ML systems under certain circumstances.

—

[Test-of-Time Award Presentation (Ali Rahimi)](https://www.youtube.com/watch?v=Qi1Yry33TQE)**and**[The Trouble with Bias (Kate Crawford)](https://www.youtube.com/watch?v=fMym_BKWQzk)

Admittedly, it’s a bit silly to do a recap of papers based on the schedule of the research conferences. At this point, most papers appear on arxiv and elsewhere prior to the events themselves, and it’s a common complaint at NIPS and ICML that everything has already been seen before.

In that light, it’s cool to see that this year featured a pair of keynotes — one from Ali Rahimi and another from Kate Crawford — that became core to the conversation happening in and around NIPS. Both are, in our minds, important calls to action for the research community: Rahimi pushing back against the slide of ML towards being “alchemy” and Crawford pushing the envelope on the field’s approach towards fairness and highlighting the politics of the work itself.

Both talks are asking and urging the field to reflect and decide what it wants to become. For Rahimi that involves the field taking a deeper account for the questions of why, for Crawford it involves taking into account the complex social context classification takes place in.

Rahimi and Crawford’s agendas would place the machine learning research community in a broader game, accepting more responsibility in its already ongoing role in shaping the integration of the technology into society. We’ll have to see if the broader community takes up that charge as the hype around the field continues to swirl, or if this becomes delegated by default to others outside the field.

—

[One-Shot Visual Imitation Learning via Meta-Learning (Finn, Yu, et al.)](https://arxiv.org/abs/1709.04905)**and**[Self-Supervised Visual Planning with Temporal Skip Connections (Ebert et al.)](https://arxiv.org/abs/1710.05268)

One of the recurring topics at NIPS was getting AI systems to learn how to learn fast. That’s inspired by one of the most remarkable traits of intelligence — the capacity for people to rapidly reach proficiency in a new task. One area where this research is particularly vibrant is at the intersection of few/one-shot learning, meta-learning, and robotics. That’s because many researchers think we need to make it dramatically easier for our AI systems to acquire new skills so that they can be trained to accomplish useful things in the real world. This type of research is also predicated on the assumption that we won’t be able to simulate everything and — despite techniques like randomization — it could be difficult for us to adapt to some unseen aspect of reality that was poorly modeled in our simulator, if sufficiently novel.

One-Shot Visual Imitation Learning via Meta-Learning, a research paper from UC-Berkeley, gives us an early look at what a future world might look like where robots can learn more rapidly: the method works from pure pixel inputs (though can also take in other datapoints, like the state space and action space, etc), and requires the robots to be trained over a distribution of tasks — such as pushing a disparate set of different objects — before being tested on an entirely new task (such as pushing a new object in a new direction). The technique generalizes well and is validated on a real-world robot exposes to real-world video footage. The results show promise but also indicate this technology is still a ways away from being productized: the success rate of this approach on one-shot object placement is about ~70% (90% if you give it access to state and actions.) The researchers have also conducted parallel, related research on developing robots that are better able to predicts traits about their world to carry out actions, and [demoed this work at NIPS](http://rail.eecs.berkeley.edu/nips_demo.html).

As it matures, this sort of research opens the way to deploying powerful, data-efficient learning algorithms on real, physical robots. This poses a number of policy challenges around what happens when malicious actors are able to physically access these robots — could a bad actor be able to retrain powerful industrial machines to carry out somewhat different behaviors? Could robots be trivially repurposed for dangerous ends? Today, systems are somewhat limited by the breadth of simulated tasks the robot has been trained on and been able to run meta-learning policies over. In the future, this distribution of tasks will be far wider, so a real question faced by robot deployers will be what subset of tasks to train a meta-learning policy on, as they’ll be trading off robot flexibility for the scope of actions it could carry out if accessed, rooted, and repurposed.

—

“**The Turtle and The Tank”** — [3D Adversarial Examples](https://machine-learning-and-security.github.io/slides/Andrew_Synthesizing_Robust_Adversarial_Examples.pdf) @ [Machine Learning and Computer Security Workshop](https://machine-learning-and-security.github.io/)

A recurring theme of NIPS 2017 was the AI community grappling with the real world impact of its increasingly smart, applicable contributions. In the Machine Learning and Computer Security Workshop, a multitude of researchers grappled with some of the security issues brought about by widescale AI deployment.

Anish Athalye and his collaborators presented their work on robust adversarial examples: creating images that fool computer-based classification systems which are robust to transforms. The most striking demonstration by Athalye and his colleagues is of a three-dimensional turtle which has been covered with a specific visual pattern to cause it to be misclassified as a ‘rifle’ as opposed to a 3D turtle.

This raises a range of potential real-world issues: does this mean it is possible to, for instance, repaint a large vehicle such as a tank so that, when viewed from overhead via drone and/or satellite-based sensors, it is misclassified as a normal car, or perhaps a tree? “Yeah, I think it should be possible to use the same method to target satellite images,” Athalye told us via email. “You could use a bunch of satellite imagery and simulate your adversarial painting on top of it, and find a painting pattern that’s simultaneously adversarial no matter what the translation/zoom level.” Since countries field multiple satellites viewing the world at a variety of distributions we also wondered if it’d be possible to make a 3D adversarial object — like our example tank — and make it be adversarial across all resolutions, guaranteeing mis-labeling at arbitrary zooms. “It’s possible to make something that’s adversarial across a distribution of resolutions. You don’t do it by training at a discrete distribution of resolutions like {1, 5, 10, 20, 100}, you’d train over the continuous distribution (e.g. uniform distribution over [1,100]),” he says.

The satellite example points to a larger issue. Future policymakers will need to prepare for an environment full of synthetic data generated by AI systems, and with some of that data potentially containing various traps designed to hijack classification or analysis algorithms and cause them to make different predictions. That could have a couple of interesting effects: for one, it’ll likely increase interest in having good software tools for guaranteeing the ‘data’ and/or ‘inference’ supply chain so it’s easier to identify places where an attacker might insert themselves.

—

[“Found in Translation”: Predicting Outcomes of Complex Organic Chemistry Reactions using Neural Sequence-to-Sequence Models](https://arxiv.org/abs/1711.04810) by Schwaller, Gaudin et al (Best paper award @ the [Machine Learning for Molecules and Materials Workshop](http://www.quantum-machine.org/workshops/nips2017/))

In the years since deep learning started to take over the AI sector we’ve seen generic tools, like convolutional neural networks and recurrent nets, and approaches like sequence-to-sequence domain translation techniques, propagate into other fields. This means that incremental advances in mainstay research areas like speech recognition and computer vision have a tendency to enable domain-specific advances in other sectors. One of the most interesting places to observe this phenomenon is in the field of chemistry, which is seeing basic exploratory methods for drug design and analysis be transformed by deep learning techniques.

A new paper from IBM researchers shows how to apply some AI techniques originally developed by Google (and subsequently used in Google Translate and the Google Assistant) to the task of predicting organic chemistry reactions — specifically, looking at chemical reactions catalogued in the SMILES dataset and trying to predict how the reactants and reagents interact to create a certain target molecule. The scientific goal here is “to solve the forward-reaction prediction problem, where the starting materials are known and the interest is in generating the products”, as they write in the paper. To try to do this, they designed a network that can ingest chemical recipes written in the SMILEs format, use sequence-to-sequence techniques to perform a multi-stage translation from the original string into a tokenized string, and map the source input string to a target string, in this case the products of the reactions.

The results are encouraging, with the method’s approach leading to an 80.3% top-1 accuracy, compared to 74% for previous state of the art. The paper does have a few limitations, for example 1.3% of its top-1 predictions are grammatically erroneous and therefore invalid in SMILES format. There also likely needs to be a way to create larger datasets and there is some evidence of overfitting (On Jin’s USPTO dataset, the training plateaued because an accuracy of 99.9% was reached and the network had memorized almost the entire training set. Even on Lowe’s noisier dataset, a training accuracy of 94.5% was observed.)

In the future the scientists hope that “with this type of model, chemists can codify and perhaps one day fully automate the art of organic synthesis.” While that may be a ways off the gestation of the paper is interesting — it’s almost entirely derived from another very different-seeming domain (consumer-grade AI-infused internet services from Google) yet uses the same fundamental machinery to make headway on a different problem that relates to scientific exploration. This highlights how traditional notions of ‘dual use’ technology struggle when analyzing AI: these techniques are so generic that they veer into the category of ‘omniuse’ — regulating them might be as hard as regulating screwdrivers and hammers and plastics. It also suggests there is the possibility that smart funding of basic scientific research in AI-adjacent disciplines could unlock new paths of experimentation and development in other parts of the economy.

—

[Unsupervised Image-to-Image Translation Networks](https://arxiv.org/abs/1703.00848) — Liu, Bruel, Kautz

The researchers use a combination of GANs and VAEs to produce a set of high-quality translations from one image domain to another. Of particular note is their demo (images available in the paper) which take a selection of street scenes and converts them from winter to summer conditions, rainy to clear conditions, day to night, and so on.

From a policy perspective, this continues a broader pair of trends that we’re seeing in the space: image generation quality continues to improve — it seems likely that these techniques will over time make it harder to discern real from generated media. Perhaps more importantly, the barrier to entry continues to fall — whether in the specific datasets required, the computational power, or the specific expertise necessary. That means that it’ll be more and more cost effective for a broader range of actors to leverage these techniques going forwards, for good or for ill.

—

And that’s it! As always, we’re keeping an eye out for new policy-relevant papers as we get into 2018. Give us a shout on Twitter ([Jack Clark](https://medium.com/@jackclarksf) / [Tim Hwang](https://medium.com/@timhwang)) or comment here if you see anything good.

