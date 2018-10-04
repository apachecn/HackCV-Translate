# 大脑vs深度学习的第一部分：计算复杂度（为什么奇点离我们还很远）

原文链接：[The Brain vs Deep Learning Part I: Computational Complexity — Or Why the Singularity Is Nowhere Near](http://timdettmers.com/2015/07/27/brain-vs-deep-learning-singularity/?from=hackcv&hmsr=hackcv.com&utm_medium=hackcv.com&utm_source=hackcv.com)

在这篇博客中，我将深入研究大脑，并解释其基本信息处理机制，并将其与深度学习进行比较。我通过一步一步地沿着大脑电化学和生物信息处理管道进行操作，并将其直接与卷积网络的架构相关联。因此，我们将看到神经元和卷积网络是非常相似的信息处理机器。在进行比较时，我还将讨论这些过程的计算复杂性，从而推导出对大脑总体计算能力的估计。我将使用这些估计，以及来自高性能计算的知识来表明，本世纪不太可能出现技术奇点。

这篇博客很复杂，因为它涉及多个主题，以便将它们统一为一个连贯的思想框架。我尽量使这篇文章具有可读性，但可能并没有在所有地方都成功。因此，如果你发现自己读到一个不明确的段落中，那么我可能会在接下来的几段中将其与另一门学科相结合，重新解释地更清楚。

首先，我将简要介绍技术奇点的预测和与之相关的主题。 然后我将开始整合大脑和深度学习之间的思想。 最后，我将讨论高性能计算以及这一切与预测技术奇点的关系。

将大脑信息处理步骤与深度学习相比较的部分是独立的，对技术奇点的预测不感兴趣的读者可以会跳过这一部分。

## 第一部分: 评估技术奇点的当前预测

最近有很多关于人工智能最早将在2030年达到超人类智能的头条预测新闻，这可能预示着人类灭绝的开始，或者至少对日常生活产生很大的影响。那么，这个预测是如何做出的？

### 有助于预测奇点的因素

Ray Kurzweil做了很多非常准确的[预测](https://en.wikipedia.org/wiki/Predictions_made_by_Ray_Kurzweil#2029)，他实现这些预测的方法对于计算设备来说非常简单:观察计算能力、效率和大小的指数增长，然后进行推断。通过这种方式，你可以很容易预测出适合掌上小型计算机的出现，只要有一点创造力，就可以想象有一天会出现平板电脑和智能手机。趋势已经出现了，你只需要想象之后掌上计算机可以用来做些什么。

雷·库兹韦尔(Ray Kurzweil)同样预言了强人工智能的出现，这种人工智能与人类一样聪明，甚至更聪明。在这个预测中，他还使用了计算能力指数增长的数据，并将其与大脑计算能力的估计进行了比较。

他还承认该软件与硬件一样重要，并且强人工智能软件的开发需要更长的时间，因为这种软件只有在快速计算机系统可用时才能开发。 这可以在深度学习领域感受到，由于计算机速度慢，上世纪90年代的坚实想法是不可行的。 一旦使用图形处理单元（GPU），这些计算限制很快就会被消除，并且可以快速取得进展。

然而，Kurzweil还强调，一旦达到硬件水平，第一个“简单”的强人工智能系统将很快地被开发。 他将类似大脑的计算能力的出现设定为2020年，强人工智能(第一种类似人类的智能或更好的智能)的出现时间设定为2030年。为什么这些数字？ 随着2019年计算能力的持续增长，我们将达到相当于人脑的计算能力 - 我们真的会吗？

这个估计基于两件事：（1）对大脑复杂性的估计，（2）对计算能力增长的估计。 正如我们将要看到的，这两种估计都不是最新的神经科学和高性能计算的技术和知识。

我们对神经科学的了解每年都会翻倍。用这个翻倍的时间，在2005年我们只掌握了我们今天所掌握的神经科学知识的0.098%。这个数字有点偏差，因为2005年的倍增时间约是2年，而现在还不到一年，但总体来说还是低于1%。

事实上，Ray Kurzweil根据他对2005年神经科学的预测，从未更新过它们。 基于1％的神经科学知识对大脑计算能力的估计似乎并不正确。 以下是过去两年里的几项重大发现，这些发现将大脑的计算能力提高了许多个数量级：

- 研究表明，大脑连接本身可以以有意义的方式处理信息和改变神经元的行为，而不是被动的连接，例如： 大脑连接可以帮助你在日常生活中看到物体。仅这一事实就将大脑的计算复杂度提高了几个数量级
- 神经元不被触发却在学习：在神经元和大脑连接处有更多电峰值：蛋白质，这些小生物机器让你的身体里的所有东西工作，结合局部电势进行大量的信息处理——不需要激活神经元
- 神经元动态改变其基因组以产生正确的蛋白质来进行日常信息处理任务。 大脑：“哦，你在看博客。等一下，我上调这个阅读基因来帮助你更好地理解博客的内容。”(这有点夸张，但也不算太离谱)

在我们研究大脑的复杂性之前，我们先来看看大脑模拟。大脑模拟通常被用来预测人类智智力。如果我们能模拟人脑，那么不久我们能够开发出类似人类的智慧，对吧？所以下一段看看这个推理。大脑模拟真的能提供可靠的证据来预测人工智能的出现吗?

### 大脑模拟的问题

大脑模拟模拟了神经元发出的电信号以及神经元之间连接的大小。大脑模拟从随机信号开始，整个系统依靠控制大脑中信息处理步骤的规则来稳定。在运行这些规则一段时间后，可以形成稳定的信号，与大脑的信号进行比较。如果模拟的信号与大脑的记录相似，这增加了我们对所选规则有些类似于大脑使用规则的信心。因此，我们可以验证大脑中大规模的信息处理规则。然而，大脑模拟的一个大问题是，这几乎是我们所能做的。

我们无法理解这些信号的含义或它们可能具有的功能。除了模糊的“我们的规则产生相似的活动”之外，我们无法用这个大脑模型来检验任何有意义的假设。缺乏精确的假设来做出准确的预测(“如果活动是这样的，那么电路检测到的是苹果而不是橘子”)是对欧洲大脑模拟项目[最大的批评之一](http://www.nature.com/news/neuroscience-where-is-the-brain-in-the-human-brain-project-1.15803)。大脑项目被许多神经科学家认为是[无用的](http://www.neurofuture.eu/) ，甚至是危险的，因为它会把钱浪费在有用的神经科学项目上，这些项目实际上为神经信息处理提供了线索。

另一个问题是这些大脑模拟依赖于过时、不完整并且这些模型在神经信息处理中忽略了许多生物部分。这主要是因为大脑中的电子信息处理更容易理解。 另一个更方便的原因是，当前模型已经能够重现所需的输出模式（毕竟这是主要目标），因此不需要更新这些模型以使其更像大脑。

总而言之，大脑模拟的问题是：

- 不可能测试具体的科学假设（将其与大型强子对撞机项目(large hadron collider project，简称lhc)的假设进行比较）
- 不能模拟真实的大脑处理(没有触发连接，没有生物相互作用)
- 没有深入了解大脑处理的功能（未评估模拟活动的意义）

最后一点是反对大脑处理对强AI评估有用的最重要的论据。如果我们能开发一个视觉系统的大脑模拟，这将在MNIST和ImageNet数据集上表现得更好，这将有助于估计大脑AI的进展。但是如果没有这些，或者没有任何类似的可观察功能，大脑模拟对于AI来说仍然是无用的。

根据这个说法，大脑模拟对于测试大脑中信息处理的假设一般规则仍然是有价值的——我们没有更好的办法—— 但它们对于理解大脑中的信息处理意味着什么是毫无用处的。 这就为AI的发展提供了不可靠的证据。 任何依靠大脑模拟作为预测未来强AI的证据都应该以极大的怀疑态度来看待。

### 估计大脑的计算复杂性

正如引言中所提到的，对大脑复杂性的估计已有十年之久，许多新发现使这一旧的估计过时了。 我从来没有看到过最新的估计，所以在这里我得出了自己的估计。 在此过程中，我将主要关注电化学信息处理并忽略神经元内的生物相互作用，因为它们太复杂了（这篇博客已经很长了）。 因此，这里得出的估计可以被认为是复杂性的下限 - 应该总是假设大脑比这更复杂。

在构建这种复杂模型的过程中，我还将模型中的每一步与其深度学习等价的东西联系起来。 这将使你更好地理解深度学习与人脑的紧密联系，人脑与深度学习相比有多快

### 定义模型的参考编号

我们知道一些事实和估计可以帮助我们开始构建模型：

- 大脑使用的学习算法与深度学习非常不同，但神经元的结构类似于卷积网络
- 成人大脑有860亿个神经元，大约10万亿个突触，大约3000亿个树突（树状结构上有突触）
- 儿童的大脑有1000多亿个神经元，突触和树突分别超过15万亿和1500亿
- 胎儿的大脑有超过一万亿的神经元;错位的神经元很快就会死亡(这也是为什么成年人的神经元比儿童少的原因)
- 小脑、大脑的超级计算机包含大约¾的神经元(这个比例在大多数哺乳动物物种中是一致的)
- 大脑是“智力”的主要驱动力,大约占所有神经元的¼
- 小脑中的一个普通神经元大约有25000个突触
- 大脑中的一个普通神经元大约有5000-15000个突触

[![小脑动画](https://i2.wp.com/timdettmers.com/wp-content/uploads/2015/07/cerebellum_animation_small.gif?resize=150%2C150)](https://i2.wp.com/timdettmers.com/wp-content/uploads/2015/07/cerebellum_animation_small.gif)

小脑的位置，其中包含大约3/4的所有神经元和连接。 图像来源: [1](https://commons.wikimedia.org/wiki/File:Cerebellum_animation_small.gif)

[![大脑动画](https://i0.wp.com/timdettmers.com/wp-content/uploads/2015/07/cerebrum_animation_small.gif?zoom=1.25&resize=150%2C150)](https://i0.wp.com/timdettmers.com/wp-content/uploads/2015/07/cerebrum_animation_small.gif)

大脑的位置，也被称为“皮质”。 更确切地说，皮质是大脑的外层，其包含大脑的大多数神经元。 图片来源：[1](https://commons.wikimedia.org/wiki/File:Cerebrum_animation_small.gif)

神经元的数量是已知的; 突触和树突的数量只有在一定的范围内才知道，我在这里保守估计一下。

每个神经元的平均突触在神经元之间差异很大，这里粗略计算了平均值。众所周知，小脑中的大多数突触是在浦肯野神经元的树突和两种不同类型的神经元之间形成的，这两种神经元与浦肯野的突触形成“攀爬”或“交叉平行”的连接。已知浦肯野细胞每个约有10万个突触。 因为这些细胞在小脑中具有迄今为止观察到的最大的重量，所以如果观察这些神经元及其产生的相互作用，我们就能最好地估计大脑的复杂性。

[![神经元类型](https://i0.wp.com/timdettmers.com/wp-content/uploads/2015/07/neuron_types.gif?zoom=1.25&resize=500%2C302)](https://i0.wp.com/timdettmers.com/wp-content/uploads/2015/07/neuron_types.gif)有数百种不同类型的神经元; 这里有一些比较常见的神经元。 感谢[Robert Stufflebeam](http://www.uno.edu/cola/philosophy/faculty/stufflebeam.aspx) 对于这个图像([来源](http://www.mind.ilstu.edu/curriculum/neurons_intro/neurons_intro.php)).

区分大脑区域的复杂性和功能重要性是很重要的。虽然几乎所有的计算都是由小脑完成，但几乎所有重要的功能都是由大脑(或皮层)完成。大脑皮层使用小脑做出预测、校正和下结论，但是皮层积累了这些见解并对它们起作用。

对于大脑来说，我们知道神经元具有的突触数量几乎从来没有超过50000个，并且与小脑不同的是，大多数神经元的突触数量都在5000-15000之间。

### 我们如何使用这些数字?

估计大脑计算复杂性的一种常用方法是假设大脑中的所有信息处理都可以由神经元发出脉冲（动作电位）和每个神经元突触大小（主要是受体数量）的组合来表示。 因此，可以将神经元数量及其突触的估计值相乘，并将所有数据相加。 然后将其乘以平均神经元的发射速度，即每秒约200个动作电位。 这个模型是Ray Kurzweil用来创建他的估计的模型。 虽然几十年前这种模型表现还可以，但从目前的观点来看，它并不适合对大脑进行建模，因为它遗漏了许多重要的神经信息处理，而这些信息处理远不止是激活神经元那么简单。

A model which approximates the behavior of neurons more accurately is the extended linear-nonlinear-Poisson cascade model (LNP). The extended LNP model is [currently viewed as an accurate model of how neurons process information](http://www.sciencedirect.com/science/article/pii/S0959438814000130). However, the extended LNP model still leaves out some fine details, which are deemed unimportant to model large scale brain function. Indeed adding these fine details to the model will add almost no additional computational complexity, but makes the model more complex to understand — thus including these details in simulations would violate the scientific method which seeks to find the simplest models for a given theory. However, this extended model is actually very similar to deep learning and thus I will include these details here.

There are other good models that are also suitable for this. The primary reason why I chose the LNP model is that it is very close to deep learning. This makes this model perfect to compare the architecture of a neuron to the architecture of a convolutional net. I will do this in the next section and at the same time I will derive an estimate for the complexity of the brain.

## Part II: The brain vs. deep learning — a comparative analysis

Now I will explain step by step how the brain processes information. I will mention the steps of information processing which are well understood and which are supported by reliable evidence. On top of these steps, there are many intermediary steps at the biological level (proteins and genes) which are still poorly understood but known to be very important for information processing. I will not go into depth into these biological processes but provide a short outline, which might help the knowledge hungry readers to delve into these depths themselves. We now begin this journey from the neurotransmitters released from a firing neuron and walk along all its processes until we reach the point where the next neuron releases its neurotransmitters, so that we return to where we started.

The next section introduces a couple of new terms which are necessary to follow the rest of the blog post, so read it carefully if you are not familiar with basic neurobiology.

[![neuron_anatomy](https://i2.wp.com/timdettmers.com/wp-content/uploads/2015/07/neuron_anatomy1.jpg?zoom=1.25&resize=680%2C390)](https://i2.wp.com/timdettmers.com/wp-content/uploads/2015/07/neuron_anatomy1.jpg)Image sources: [1](https://commons.wikimedia.org/wiki/File:Neuron_Hand-tuned.svg),[2](https://commons.wikimedia.org/wiki/File:SynapseSchematic_lines.svg),[3](http://faculty.ivytech.edu/~shopper6/ANPweb/gallery/Week_011-2.html),[4](http://faculty.ivytech.edu/~shopper6/ANPweb/gallery/Week_011-2.html)

Neurons use the axon — a tube like structure— to transmit their electric signals over long stretches in the brain. When a neuron fires, it fires an action potential — an electrical signal— down its axon which branches into a tree of small endings, called axon terminals. On the ending of each of these axon terminals sit some proteins which convert this electrical message back into a chemical one: Small balls — called synaptic vesicles — filled with a couple of neurotransmitters each are released into an area outside of the neuron, called synaptic cleft. This area separates the axon terminal from the beginning of the next neuron (a synapse) and allows the neurotransmitter to move freely to pursue different tasks.

The synapses are most commonly located at a structure which looks very much like the roots of a tree or plant; this is the dendritic tree composed of dendrites which branch into larger arms (this represents the connections between neurons in a neural network), which finally reach the core of the cell, which is called soma. These dendrites hold almost all synapses which connect one neuron to the next and thus form the principal connections. A synapse may hold hundreds of receptors to which neurotransmitter can bind themselves.

You can imagine this compound of axon terminal and synapses at a dendrite as the (dense) input layer (of an image if you will) into a convolutional net. Each neuron may have less than 5 dendrites or as many as a few hundred thousand. Later we will see that the function of the dendritic tree is similar to the combination of a convolutional layer followed by max-pooling in a convolutional network.

Going back to the biological process, the synaptic vesicles merge with the surface of the axon terminal and turn themselves inside-out spilling their neurotransmitters into the synaptic cleft. There the neurotransmitters drift in a vibrating motion due to the temperature in the environment, until they (1) find a fitting lock (receptor protein) which fits their key (the neurotransmitter), (2) the neurotransmitters encounter a protein which disintegrates them, or (3) the neurotransmitters encounter a protein which pulls them back into the axon (reuptake) where they are reused. Antidepressants mostly work by (3) preventing, or (4) enhancing the reuptake of the neurotransmitter serotonin; (3) preventing reuptake will yield changes in information processing after some days or weeks, while (4) enhancing reuptake leads to changes within seconds or minutes. So neurotransmitter reuptake mechanisms are integral for minute to minute information processing. Reuptake is ignored in the LNP model.

However, the combination of the amount of neurotransmitters released, the number of synapses for a given neurotransmitter, and how many neurotransmitters actually make it into a fitting protein on the synapse can be thought of as the weight parameter in a densely (fully) connected layer of a neural network, or in other words, the total input to a neuron is the sum of all axon-terminal-neurotransmitter-synapse interactions. Mathematically, we can model this as the dot product between two matrices (A dot B; [amount of neurotransmitters of all inputs] dot [amount of fitting proteins on all synapses]).

After a neurotransmitter has locked onto a fitting protein on a synapse, it can do a lot of different things: Most commonly, neurotransmitters will just (1) open up channels, to let charged particles flow (through diffusion) into the dendrites, but it can also cause a rarer effect with huge consequences: The neurotransmitter (2) binds to a G-protein which then produces a protein signaling cascade which, (2a) activates (upregulates) a gene which is then used to produce a new protein which is integrated into either the surface of the neuron, its dendrites, and/or its synapses; which (2b) alerts existing proteins to do a certain function at a specific site (create or remove more synapses, unblock some entrances, attach new proteins to the surface of the synapse). This is ignored in the NLP model.

Once the channels are open, negatively or positively charged particles enter into the dendritic spine. A dendritic spine is a small mushroom-like structure on to which the synapse is attached. These dendritic spines can store electric potential and have their own dynamics of information processing. This is ignored in the NLP model.

[![dendritic_spine](https://i0.wp.com/timdettmers.com/wp-content/uploads/2015/07/dendritic_spine.jpg?zoom=1.25&resize=471%2C335)](https://i0.wp.com/timdettmers.com/wp-content/uploads/2015/07/dendritic_spine.jpg)Dendritic spines have their own internals information processing dynamics which is largely determined by its shape and size. Image source: [1](https://en.wikipedia.org/wiki/File:Spline_types_3D.png),[2](https://en.wikipedia.org/wiki/File:Dendritic_spines.jpg)

The charge of the particles that may enter the dendritic spine are either negatively or positively charged — some neurotransmitters only open channels for negative particles, others only for positive ones. There are also channels which let positively charged particles leave the neuron, thus increasing the negativity of the electric potential (a neuron “fires” if it becomes too positive). The size and shape of the mushroom-like dendritic spine corresponds to its behavior. This is ignored in the NLP model.

Once particles entered the spine, there are many things they can affect. Most commonly, they will (1) just travel along the dendrites to the cell body in the neuron and then, if the cell gets too positively charged (depolarization) they induce an action potential (the neuron “fires”). But other actions are also common:  The charged particles accumulate in the dendritic spine directly and (2) open up voltage-gated channels which may polarize the cell further (this is an example of the dendritic spine information processing mentioned above). Another very important process are (3) dendritic spikes.

### Dendritic spikes

Dendritic spikes are a phenomenon which has been known to exist for some years, but only in 2013 the techniques were advanced enough to collect the data to show that these spikes were important for information processing. To measure dendritic spikes, you have to attach some very tiny clamps onto dendrites with the help of a computer which moves the clamp with great precision. To have some sort of idea where your clamp is, you need a special microscope to observe the clamp as you progress onto a dendrite. Even then you mostly attach the clamp in a rather blind matter because at such tiny scale every movement made is a rather giant leap. Only a few teams in the world have the equipment and skill to attach such clamps onto dendrites.

However, the direct data gathered by those few teams was enough to establish dendritic spikes as important information processing events. Due to the introduction of dendritic spikes into computational models of neurons, the complexity of a single neuron has become very similar to a convolutional net with two convolutional layers. As we see later the LNP model also uses non-linearities very similar to a rectified linear function, and also makes use of a spike generator which is very similar to dropout – so a neuron is very much like an entire convolutional net. But more about that later and back to dendritic spikes and what exactly they are.

Dendritic spikes occur when a critical level of depolarization is reached in a dendrite. The depolarization discharges as an electric potential along the walls of the dendrite and may trigger voltage-gated channels along its way through the dendritic tree and eventually, if strong enough, the electric potential reaches the core of the neuron where it may trigger a true action potential. If the dendritic spike fails to trigger an action potential, the opened voltage-gated channels in neighboring dendrites may do exactly that a split second later. Due to channels opened from the dendritic spike more charged particles enter the neuron, which then may either trigger (common) or stifle (rare) a full action potential at the neurons cell body (soma).

[![dendritic_spikes](https://i0.wp.com/timdettmers.com/wp-content/uploads/2015/07/dendritic_spikes.png?zoom=1.25&resize=677%2C263)](https://i0.wp.com/timdettmers.com/wp-content/uploads/2015/07/dendritic_spikes.png)A shows a computer model of a neuron that does not model dendritic spikes; B models simple dynamics of dendritic spikes; C models more complex dynamics of dendritic spikes which takes into account the one dimensional diffusion of particles (which is similar to a convolution operation). Take note that these images are only snapshots in a particular moment of time. A big thanks to [Berd Kuhn](https://groups.oist.jp/onu). Image copyright © 2014 Anwar, Roome, Nedelescu, Chen, Kuhn and De Schutter as published in *Frontiers in Cellular Neuroscience (Anwar et al. 2014)*

This process is very similar to max-pooling, where a single large activation “overwrites” other neighboring values. However, after a dendritic spike, neighboring values are not overwritten like during max-pooling used in deep learning, but the opening of voltage-gated channels greatly amplifies the signals in all neighboring branches within the dendritic tree. Thus a dendritic spike may heighten the electrochemical levels in neighboring dendrites to a level which is more similar to the maximum input — this effect is close to max-pooling.

Indeed it was shown that dendritic spikes in the visual system serve the same purpose as max pooling in convolutional nets for object recognition: In deep learning, max-pooling is used to achieve (limited) rotation, translation, and scale invariance (meaning that our algorithm can detect an object in an image where the object is rotated, moved, or shrunk/enlarged by a few pixels). One can think of this process as setting all surrounding pixels to the same large activation and make each activation share the weight to the next layer (in software the values are discarded for computational efficiency — this is mathematically equivalent). Similarly, it was shown that dendritic spikes in the visual system are sensitive to the orientation of an object. So dendritic spikes do not only have computational similarity, but also similarities in function.

The analogy does not end here. During neural back-propagation — that is when the action potential travels from the cell body back into the dendritic tree — the signal cannot backpropagate into the dendritic branch where the dendritic spike originated because these are “deactivated” due to the recent electrical activity. Thus a clear learning signal is sent to inactivated branches. At first this may seem like the exact opposite from the backpropagation used for max-pooling, where everything but the max-pooling activation is backpropagated. However, the absence of a backpropagation signal in a dendrite is a rare event and represents a learning signal on its own. Thus, dendrites which produce dendritic spikes have special learning signals just like activated units in max-pooling.

To better understand what dendritic spikes are and what they look like, I very much want to encourage you to watch [this video](http://www.hhmi.org/research/how-do-neurons-compute-output-their-inputs) (for which I do not have the copyright). The video shows how two dendritic spikes lead to an action potential.

This combination of dendritic spikes and action potentials and the structure of the dendritic tree has been found to be critical for learning and memory in the hippocampus, the main brain region responsible for forming new memories and writing them to our “hard drive” at night.

Dendritic spikes are one of the main drivers of computational complexity which have been left out from past models of the complexity of the brain. Also, these new findings show that neural back-propagation does not have to be neuron-to-neuron in order to learn complex functions; a single neuron already implements a convolutional net and thus has enough computational complexity to model complex phenomena. As such, there is little need for learning rules that span multiple neurons — a single neuron can produce the same outputs we create with our convolutional nets today.

But these findings about dendritic spikes are not the only advance made in our understanding of the information processing steps during this stage of the neural information processing pathway. Genetic manipulation and targeted protein synthesis are sources that increase computational complexity by orders of magnitude, and only recently we made advances which reveal the true extend of biological information processing.

### Protein signaling cascades

As I said in the introduction of this part, I will not cover the parts of biological information processing extensively, but I want to give you enough information so that you can start learning more from here.

One thing one has to understand is that a cell looks much different from how it is displayed in text books. Cells crawl with proteins: There are about 10 billion proteins in any given human cell and these proteins are not idle: They combine with other proteins, work on a task, or jitter around to find new tasks to work on.

All the functions described above are the work of proteins. For example the key-and-lock mechanism and the channels that play the gatekeeper for the charged particles that leave and enter the neuron are all proteins. The proteins I mean in this paragraph are not these common proteins, but proteins with special biological functions.

As an example the abundant neurotransmitter glutamate may bind to a NDMA receptor which then opens up its channels for many different kinds of charged particles and after being opened, the channel only closes when the neuron fires. The strength of synapses is highly dependent on this process, where the synapse is adjusted according to the location of the NDMA receptor and the timing of signals which are backpropagated to the synapses. We know this process is critical to learning in the brain, but it is only a small piece in a large puzzle.

The charged particles which may enter the neuron may additionally induce protein signaling cascades own their own. For example the cascade below shows how an activated NMDA receptor (green) lets charged calcium CA2+ inside which triggers a cascade which eventually leads to AMPAR receptors (violet) being trafficked and installed on the synapse.

[![RegulationOfAMPARTrafficking](https://i0.wp.com/timdettmers.com/wp-content/uploads/2015/07/regulationofampartrafficking.jpg?zoom=1.25&resize=680%2C531)](https://i0.wp.com/timdettmers.com/wp-content/uploads/2015/07/regulationofampartrafficking.jpg)Image source: [1](https://commons.wikimedia.org/wiki/File:RegulationOfAMPARTrafficking.jpg)

It was shown again and again that these special proteins have a great influence on the information processing in neurons, but it is difficult to pick out a specific type of protein from this seemingly chaotic soup of 10 billion proteins and study its precise function. Findings are often complex with a chain of reactions involving many different proteins until a desired end-product or end-function is reached. Often the start and end functions are known but not the exact path which led from one to the other. Sophisticated technology helped greatly to study proteins in detail, and as technology gets better and better we will further our understanding of biological information processing in neurons.

### Genetic manipulation

The complexity of biological information processing does not end with protein signaling cascades, the 10 billion proteins are not a random soup of workers that do their tasks, but these workers are designed in specific quantities to serve specific functions that are relevant at the moment. All this is controlled by a tight feedback loop involving helper proteins, DNA, and messenger RNA (mRNA).

If we use programming metaphors to describe this whole process, then the DNA represents the whole github website with all its public packages, and messenger RNA is a big library which features many other smaller libraries with different functions (something like the C++ boost library).

It all begins with a programming problem you want to solve (a biological problem is detected). You use google and stackoverflow to find recommendations for libraries which you can use to solve the problem and soon you find a post that suggests that you use library X to solve problem Y (problem Y is detected on a local level in a cell with known solution of protein X; the protein that detected this defect then cascades into a chain of protein signals which leads to the upregulation of the gene G which can produce protein X; here upregulation is a “Hey! Produce more of this, please!” signal to the nucleus of the cell where the DNA lies). You download the library and compile it (the gene G is copied (transcribed) as a short string of mRNA from the very long string of DNA). You then do configure the install (the mRNA leaves the core) with the respective configuration (the mRNA is translated into a protein, the protein may be adjusted by other proteins after this), and install the library in a global “/lib” directory (the protein folds itself into its correct form after which it is fully functional). After you have installed the library, you import the needed part of the library to your program (the folded protein travels (randomly) to the site where it is needed) and you use certain functions of this library to solve your problem (the protein does some kind of work to solve the problem).

Additional to this, neurons may also dynamically alter their genome, that is they can dynamically change their github repository to add or remove libraries.

To understand this process further, you may want to watch the following video, which shows how HIV produces its proteins and how the virus can change the host DNA to suit its needs. The process described in this video animation is very similar to what is going on in neurons. To make it more similar to the process in neurons, imagine that HIV is a neurotransmitter and that everything contained in the HIV cell is in the neuron in the first place. What you have then is an accurate representation of how neurons make use of theirs genes and proteins:



You may ask, isn’t it so that every cell in your body has (almost) the same DNA in order to be able to replicate itself? Generally, this is true for most cells, but not true for most neurons. Neurons will typically have a genome that is different from the original genome that you were assigned to at birth. Neurons may have additional or fewer chromosomes and have sequences of information removed or added from certain chromosomes.

It was shown, that this behavior is important for information processing and if gone awry, this may contribute to brain disorders like depression or Alzheimer’s disease. Recently it was also shown, that neurons change their genome on a daily basis to improve information processing demands.

So when you sit at your desk for five days, and then on the weekend decide to go on a hike, it makes good sense that the brain adapts its neurons for this new task, because entirely different information processing is needed after this change of environment.

Equally, in an evolutionary sense, it would be beneficial to have different “modes” for hunting/gathering and social activity within the village — and it seems that this function might be for something like this. In general, the biological information processing apparatus is extremely efficient in responding to slower information processing demands that range from minutes to hours.

With respect to deep learning, an equivalent function would be to alter the function of a trained convolutional net in significant but rule-based ways; for example to apply a transformation to all parameters when changing from one to another task (recognition of street numbers -> transform parameters -> recognition of pedestrians).

Nothing of this biological information processing is modeled by the LNP model.

Looking back at all this, it seems rather strange that so many researchers think they that they can replicate the brain’s behavior by concentrating on the electrochemical properties and inter-neuron interactions only. Imagine that every unit in a convolutional network has its own github, from which it *learns* to dynamically download, compile and use the best libraries to solve a certain task. From all this you can see that a single neuron is probably more complex than an entire convolutional net, but we continue from here in our focus on electrochemical processes and see where it leads us.

### Back to the LNP model

After all this above, there is only one more relevant step in information processing for our model. Once a critical level of depolarization is reached, a neuron will most often fire, but not always. There are mechanisms that prevent a neuron from firing. For example shortly after a neuron fired, its electric potential is too positive to produce a fully-fledged action potential, and thus it cannot fire again. This blockage may be present even when a sufficient electric potential is reached, because this blockade is a biological function and not a physical switch.

In the LNP model, this blockage of an action potential is modeled as an inhomogeneous Poisson process which has a Poisson distribution. A Poisson process with a Poisson distribution as a model means that the neuron has a very high probability to fire the first or second time it reached its threshold potential, but it may also be (with a exponentially decreasing probability) that a neuron may not fire for many more times.

[![Poisson](https://i0.wp.com/timdettmers.com/wp-content/uploads/2015/07/poisson.png?zoom=1.25&resize=652%2C347)](https://i0.wp.com/timdettmers.com/wp-content/uploads/2015/07/poisson.png)A Poisson(0.5) distribution with a randomly drawn sample. Here 0,1,2,3 represents the waiting time until the neuron fires, thus 0 means it fires without delay, while 2 means it will not fire for two cycles even if it could fire physically.

There are exceptions to this rule, where neurons disable this mechanism and fire continuously at the rates which are governed by the physics alone — but these are special events which I will ignore at this point. Generally, this whole process is very similar to dropout used in deep learning which uses a uniform distribution instead of a Poisson distribution; thus this process can be viewed as some kind of regularization method that the brain uses instead of dropout.

In the next step, if the neuron fires, it releases an action potential. The action potential has very little difference in its amplitude, meaning the electric potential generated by the neuron almost always has the same magnitude, and thus is a reliable signal. As this signal travels down the axon it gets weaker and weaker. When it flows into the branches of the axon terminal, its final strength will be dependent on the shape and length of these branches; so each axon terminal will receive a different amount of electrical potential. This spatial information, together with the temporal information due to the spiking pattern of action potentials, is then translated into electrochemical information (it was shown that they are translated into spikes of neurotransmitters themselves that last about 2ms). To adjust the output signal, the axon terminal can move, grow or shrink (spatial), or it may alter its protein makeup which is responsible for releasing the synaptic vesicles (temporal).

Now we are back at the beginning: Neurotransmitters are released from the axon terminal (which can be modeled as a dense matrix multiplication) and the steps repeat themselves.

### Learning and memory in the brain

Now that we went through the whole process back to back, let us put all this into context to see how the brain uses all this in concert.

Most neurons repeat the process of receive-inputs-and-fire about 50 to 1000 times per second; the firing frequency is highly dependent on the type of neuron and if a neuron is actively processesing tasks. Even if a neuron does not process a task it will fire continuously in a random fashion.  Once some meaningful information is processed, this random firing activity makes way for a highly synchronized activity between neighboring neurons in a brain region. This synchronized activity is poorly understood, but is thought to be integral to understanding information processing in the brain and how it learns.

Currently, it is not precisely known how the brain learns. We do know that it adjusts synapses with some sort of reinforcement learning algorithm in order to learn new memories, but the precise details are unclear and the weak and contradicting evidence indicates that we are missing some important pieces of the puzzle. We got the big picture right, but we cannot figure out the brain’s learning algorithm without the fine detail which we are still lacking.

Concerning memories, we know that some memories are directly stored in the hippocampus, the main learning region of the brain (if you lose your hippocampus in each brain hemisphere, you cannot form new memories). However, most long-term memories are created and integrated with other memories during your REM sleep phase, when so called sleep spindles unwind the information of your hippocampus to all other brain areas. Long-term memories are generally all local: Your visual memories are stored in the visual system; your memories for your tongue (taste, texture) are stored in the brain region responsible for your tongue, etcetera.

It is also known, that the hippocampus acts as a memory buffer. Once it is full, you need to sleep to empty its contents to the rest of your brain (through sleep spindles during REM sleep); this might be why babies sleep so much and so irregularly —once their learning buffer is full, they sleep to quickly clear their buffer in order to learn more after they wake. You can still learn when this memory buffer is full, but retention is much worse and new memories might wrangle with other memories in the buffer for space and displace them —so really get your needed amount of sleep. Sleeping less and irregularly is unproductive, especially for students who need to learn.

[![Hippocampus_small](https://i0.wp.com/timdettmers.com/wp-content/uploads/2015/07/hippocampus_small.gif?zoom=1.25&resize=200%2C200)](https://i0.wp.com/timdettmers.com/wp-content/uploads/2015/07/hippocampus_small.gif)The hippocampus in each hemisphere is shown in red. Image source: [1](https://commons.wikimedia.org/wiki/File:Hippocampus_small.gif)

Because memories are integrated with other memories during your “write buffer to hard-drive” stage, sleep is also very important for creativity. The next time you recall a certain memory after you slept, it might be altered with some new information that your brain thought to be fitting to attach to that memory.

I think we all had this: We wake up with some crazy new idea, only to see that it was quite nonsensical in the first place — so our brain is not perfect either and makes mistakes. But other times it just works: One time I tortured myself with a math problem for 7 hours non-stop, only to go to bed disappointed with only about a quarter of the whole problem solved. After I woke, I immediately had two new ideas how to solve the problem: The first did not work; but second made things very easy and I could sketch a solution to the math problem within 15 minutes — an ode to sleep!

Now why do I talk about memories when this blog post is about computation? The thing is that memory creation — or in other words — a method to store computed results for a long time, is critical for any intelligence. In brain simulations, one is satisfied if the synapse and activations occur in the same distribution as they do in the real brain, but one does not care if these synapses or activations correspond to anything meaningful — like memories or “distributed representations” needed for functions such as object recognition. This is a great flaw. Brain simulations have no memories.

In brain simulation, the diffusion of electrochemical particles is modeled by differential equations. These differential equations are complex, but can be modeled with simple techniques like Euler’s method to approximate these complex differential equations. The result has poor accuracy (meaning high error) but the algorithm is very computationally efficient and the accuracy is sufficient to reproduce the activities of real neurons along with their size and distribution of synapses. The great disadvantage is that we generally cannot learn parameters from a method like this — we cannot create meaningful memories.

However, as I have shown in [my blog post about convolution](https://timdettmers.wordpress.com/2015/03/26/convolution-deep-learning/), we can also model diffusion by applying convolution — a very computationally complex operation. The advantage about convolution is that we can use methods like maximum-likelihood estimation with backpropagation to learn parameters which lead to meaningful representations which are akin to memories (just like we do in convolutional nets). This is exactly akin to the LNP model with its convolution operation.

So besides its great similarity to deep learning models, the LNP model is also justified in that it is actually possible to learn parameters which yield meaningful memories (where with memories I mean here distributed representations like those we find in deep learning algorithms).

This then also justifies the next point where I estimate the brain’s complexity by using convolution instead of Euler’s method on differential equations.

Another point to take away from for our model is, that we currently have no complexity assigned for the creation of memories (we only modeled the forward pass, not the backward pass with backpropagation). As such, we underestimate the complexity of the brain, but because we do not know how the brain learns, we cannot make any accurate estimates for the computational complexity of learning. With that said and kept in the back of our mind, let us move on to bringing the whole model together for a lower bound of computational complexity.

### Bringing it all together for a mathematical estimation of complexity

[![brain_complexity](https://i2.wp.com/timdettmers.com/wp-content/uploads/2015/07/brain_complexity.png?zoom=1.25&resize=680%2C242)](https://i2.wp.com/timdettmers.com/wp-content/uploads/2015/07/brain_complexity.png)

The next part is a bit tricky: We need to estimate the numbers for N, M, n and m and these differ widely among neurons.

We know that 50 of the 86 billion neurons in the brain are cerebellar granule neurons, so these neurons and their connection will be quite important in our estimation.

Cerebellar granule neurons are very tiny neurons with about 4 dendrites. Their main input is from the cortex. They integrate these signals and then send them along a T-shaped axon which feeds into the dendrites of Purkinje neurons.

Purkinje neurons are by far the most complex neurons, but there are only about 100 million of them. They may have more than a 100000 synapses each and about 1000 dendrites. Multiple Purkinje neurons bundle their outputs in about a dozen deep nuclei (a bunch of densely packed neurons) which then send signals back to the cortex.

This process is very crucial for non-verbal intelligence, abstract thinking and abstract creativity (creativity: Name as many words beginning with the letter A; abstract creativity: What if gravity bends space-time (general relativity)? What if these birds belonged to the same species when they came to this island (evolution)?). It was thought a few decades ago that the cerebellum only computes outputs for movement; for example while Einstein’s cerebrum was handled and studied carefully, his cerebellum was basically just cut off and put away, because it was regarded as a “primitive” brain part.

But since then it was shown that the cerebellum forms 1:1 connections with most brain regions of the cortex. Indeed, changes in the front part of the cerebellum during the ages 23 to 25 may change your non-verbal IQ by up to 30 points, and changes of 10-15 IQ points are common. This is very useful in most instances, whereas we lose neurons which perform a function which we do not need in everyday lives (calculus, or the foreign language which you learned but never used).

So it is crucial to get the estimation of the cerebellum right not only because it contains most neurons, but also because it is important for intelligence and information processing in general.

### Estimation of cerebellar filter dimensions

Now if we look at a single dendrite, it branches off into a few branches and thus has a tree like structure. Along its total length it is usually packed with synapses. Dendritic spikes can originate in any branch of a dendrite (spatial dimension). When we take 3 branches per dendrite, and 4 dendrites in total we have a convolutional filter of size 3 and 4 for cerebellar granule neurons. Since linear convolution over two dimensions is the same as convolution over one dimension followed by convolution over the other dimension, we can also model this as a single 3×4 convolution operation. Also note that this is mathematically identical to a model that describes the diffusion of particles originating from different sources (feature map) which diffuse according to a rule in their neighborhood (kernel) — this is exactly what happens at a physical level. More on this view in [my blog post about convolution](https://timdettmers.wordpress.com/2015/03/26/convolution-deep-learning/).

Here I have chosen to represent the spatial domain with a single dimension. It was shown that the shape of the dendritic tree is also important in the resulting information processing and thus we would need two dimensions for the spatial domain. However, data is lacking to represent this mathematically in a meaningful way and thus I proceed with the simplification to one spatial dimension.

The temporal dimension is also important here: Charged particles may linger for a while until they are pumped out of the neuron. It is difficult to estimate a meaningful time frame, because the brain uses continuous time while our deep learning algorithms only know discrete time steps.

No single estimate makes sense from a biological perspective, but from a psychological perspective we know that the brain can take up unconscious information that is presented in an image in about 20 milliseconds (this involves only some fast, special parts of the brain). For conscious recognition of an object we need more time — at least 65 milliseconds, and on average about 80-200 milliseconds for reliable conscious recognition. This involves all the usual parts that are active for object recognition.

From these estimates, one can think about this process as “building up the information of the seen image over time within a neuron”. However, a neuron can only process information if it can differentiate meaningful information from random information (remember, neurons fire randomly if they do not actively process information). Once a certain level of “meaningful information” is present, the neuron actively reacts to that information. So in a certain sense information processing can be thought of as an epidemic of useful information that spreads across the brain: Information can only spread to one neuron, if the neighboring neuron is already infected with this information. Thinking in this way, such an epidemic of information infects all neurons in the brain within 80-200 milliseconds.

As such we can say that, while the object lacks details in the first 20 milliseconds, there is full detail at about 80-200 milliseconds. If we translate this into discrete images at the rate of 30 frames per second (normal video playback) —or in other words time steps — then 20 milliseconds would be 0.6 time steps, and 80-200 milliseconds 2.4-6 time steps. This means, that all the visual information that a neuron needs for its processing will be present in the neuron within 2.4 to 6 frames.

To make calculations easier, I here now choose a fixed time dimension of 5 time steps for neural processes. This means for the dendrites we have spatio-temporal convolutional filters of size 3x4x5 for cerebellar granule neurons. For Purkinje neurons a similar estimate would be filters of a size of about 10x1000x5. The non-linearity then reduces these inputs to a single number for each dendrite. This number represents an instantaneous firing rate, that is, the number represents how often the neuron fires in the respective interval of time, for example at 5 Hz, 100 Hz, 0 Hz etcetera. If the potential is too negative, no spike will result (0 HZ); if the potential is positive enough, then the magnitude of the spike is often proportional to the magnitude of the electric potential —but not always.

It was shown that dendritic summation of this firing rate can be linear (the sum), sub-linear (less than the sum), supra-linear (more than the sum) or bistable (less than the sum, or more than the sum, depending on the respective input); these behaviors of summation often differ from neuron to neuron. It is known that Purkinje neurons use linear summation, and thus their summation to form a spike rate is very similar to the rectified linear function max(0,x) which is commonly used in deep learning. Non-linear sums can be thought of different activation functions. It is important to add, that the activation function is determined by the type of the neuron.

The filters in the soma (or cell body) can be thought of as an additional temporal convolutional filter with a size of 1 in the spatial domain. So this is a filter that reduces the input to a single dimension with a time dimension of 5, that is, a 1x1x5 convolutional filter (this will be the same for all neurons).

Again, the non-linearity then reduces this to an instantaneous firing rate, which then is dropped out by a Poisson process, which is then fed into a weight-matrix.

At this point I want to again emphasize, that it is not correct to view the output of a neuron as binary; the information conveyed by a firing neuron is more like an if-then-else branch: “if(fire == True and dropout == False){ release_ neurotransmitters(); }else{ sleep(0.02); }”

The neurotransmitters are the true output of a neuron, but this is often confused. The source of this confusion is that it is very difficult to study neurotransmitter release and its dynamics with a synapse, while it is ridiculously easy to study action potentials. Most models of neurons thus model the output as action potentials because we have a lot of reliable data here; we do not have such data for neurotransmitter interactions at a real-time level. This is why action potentials are often confused as the true outputs of neurons when they are not.

When a neuron fires, this impulse can be thought of as being converted to a discrete number at the axon terminal (number of vesicles which are released) and is multiplied by another discrete number which represents the amount of receptors on the synapse (this whole process corresponds to a dense or fully connected weight in convolutional nets). In the next step of information processing, charged particles flow into the neuron and build up a real-valued electric potential. This has also some similarities to batch-normalization, because values are normalized into the range [0,threshold] (neuron: relative to the initial potential of the neuron; convolutional net: relative to the mean of activations in batch-normalization). When we look at this whole process, we can model it as a matrix multiplication between two real-valued matrices (doing a scaled normalization before or after this is mathematically equivalent, because matrix multiplication is a linear operation).

Therefore we can think of axon-terminal-synapse interactions between neurons as a matrix multiplication between two real-valued matrices.

### Estimation of cerebellar input/output dimensions

Cerebellar granule neurons typically receive inputs from about four axons (most often connections from the cortex). Each axon forms about 3-4 synapses with the dendritic claw of the granule neuron (a dendrite ending shaped as if you would hold a tennis ball in your hand) so there are a total of about 15 inputs via synapses to the granule neurons. The granule neuron itself ends in a T shaped axon which crosses directly through the dendrites of Purkinje neurons with which it forms about 100 synapses.

Purkinje neurons receive inputs from about 100000 connections made with granule neurons and they themselves make about 1000 connections in the deep nuclei. There are estimates which are much higher and no accurate number for the number of synapses exists as far as I know. The number of 100000 synapses might be a slight overestimate (but 75000 would be too conservative), but I use it anyways to make the math simpler.

All these dimensions are taken times the time dimension as discussed above, so that the input for granule neurons for example has a dimensionality of 15×5.

So with this we can finally calculate the complexity of a cerebellar granule neuron together with the Purkinje neurons.

[![brain_computational_estimate](https://i2.wp.com/timdettmers.com/wp-content/uploads/2015/07/brain_computational_estimate.png?zoom=1.25&resize=680%2C432)](https://i2.wp.com/timdettmers.com/wp-content/uploads/2015/07/brain_computational_estimate.png)

So my estimate would be 1.075×10^21 FLOPS for the brain, the fastest computer on earth as of July 2013 has 0.58×10^15 FLOPS for practical application (more about this below).

### Part III: Limitations and criticism

While I discussed how the brain is similar to deep learning, I did not discuss how the brain is different. One great disparity is that the dropout in the brain works with respect to all inputs, while dropout in a convolutional network works with respect to each single unit. What the brain is doing makes little sense in deep learning right now; however, if you think about combining millions of convolutional nets with each other, it makes good sense to do as the brain does. The dropout of the brain certainly would work well to decouple the activity of neurons from each other, because no neuron can depend on information from a single other neuron (because it might be dropped out), so that it is forced to take into account all the neurons it is connected with, thus eliminating biased computation (which is basically regularization).

Another limitation of the model is that it is a lower bound. This estimate does not take into account:

- Backpropagation, i.e. signals that travel from the soma to the dendrites; the action potential is reflected within the axon and travels backwards (these two things may almost double the complexity)
- Axon terminal information processing
- Multi-neurotransmitter vesicles (can be thought of multiple output channels or filters, just as an image has multiple colors)
- Geometrical shape of the dendritic tree
- Dendritic spine information processing
- Non-axodendritic synapses (axon-axon and axon-soma connections)
- Electrical synapses
- Neurotransmitter induced protein activation and signaling
- Neurotransmitter induced gene regulation
- Voltage induced (dendritic spikes and backpropagating signals) gene regulation
- Voltage induced protein activation and signaling
- Glia cells (besides having an extremely abnormal brain (about one in a billion), Einstein also had abnormally high levels of glia cells)

All these things have been shown to be important for information processing in the brain. I did not include them in my estimate because this would have made everything:

- Too complex: What I have discussed so far is extremely simple if you compare that to the vastness and complexity of biological information processing
- Too special: Non-axodendritic synapses can have unique information processing algorithms completely different from everything listed here, e.g. direct electrical communication between a neighboring bundle of neurons
- And/or evidence is lacking to create a reliable mathematical model: Neural backpropagation, geometry of the dendritic trees, and dendritic spines

Remember that these estimates are for the whole brain. Local brain regions might have higher computational processing speed than this average when they are actively processing stimuli. Also remember that the cerebellum makes up almost all computational processing. Other brain regions integrate the knowledge of the cerebellum, but the cerebellum acts as a transformation and abstraction module for almost all information in the brain (except vision and hearing).

### But wait, but we can do all this with much less computational power! We already have super-human performance in computer vision!

I would not say that we have super-human performance in computer vision. What we have is a system that beats human at naming things in images that are taken out of context of the real world (what happens before we see something in the real world shapes our perception dramatically). We almost always can recognize things in our environment, but we most often just do not know (or care about) the name of what we see.

Humans do not have the visual system to label things. Try to make a list of 1000 common physical objects in the real world —not an easy task.

To not recognize an object for us humans would mean that we see an object but cannot make sense of it. If you forgot the name of an old classmate, it does not mean you did not recognize her; it just means you forgot her name. Now imagine you get off a train stop and you know a good friend is waiting for you somewhere at the stop. You see somebody 300 meters away waving their hands who is looking in your direction — is it your friend? You do not know; you cannot recognize if it is her. That’s the difference between mere labels and object recognition.

Now if you cannot recognize something in a 30×30 pixel image, but the computer can, this also does not necessarily mean that the computer has super-human object recognition performance. First and foremost this means that your visual system does not work well for pixeled information. Our eyes are just not used to that.

Now take a look outside a window and try to label all the things you see. It will be very easy for most things, but for other things you do not know the correct labels! For example, I do not know the name for a few plants that I see when I look out of my window. However, we are fully aware what it is what we see and can name many details of the object. For example, alone by assessing their appearance, I know a lot about how much water and sunshine the unknown plants need, how fast they grow, in which way they grow, if they are old or young specimens; I know how they feel like if I touch them — or more generally — I know how these plants grow biologically and how they produce energy, and so on. I can do all this without knowing its name. Current deep learning systems cannot do this and will not do this for quite some time. Human-level performance in computer vision is far away indeed! We just reached the very first step (object recognition) and now the task is to make computer vision smart, rather than making it just good at labeling things.

Evolutionarily speaking, the main functions of our visual system have little to do with naming things that we see: Hunt and avoid being hunted, to orient ourselves in nature during foraging and make sure we pick the right berries and extract roots efficiently— these are all important functions, but probably one of the most important functions of our vision is the social function within a group or relationship.

If you Skype with someone it is quite a different communication when they have their camera enabled compared to if they have not. It is also very different to communicate with someone whose image is on a static 2D surface compared to communicating in person. Vision is critical for communication.

Our deep learning cannot do any of this efficiently.

### Making sense of a world without labels

One striking case which also demonstrates the power of vision for true understanding of the environment without any labels is the case of [Genie](https://en.wikipedia.org/wiki/Genie_(feral_child)). Genie was strapped into place and left alone in a room at the age of 20 months. She was found with severe malnutrition 12 years later. She had almost no social interaction during this time and thus did not acquire any form of verbal language.

Once she got in contact with other human beings she was taught English as a language (and later also sign language), but she never really mastered it. Instead she quickly mastered non-verbal language and was truly exceptional at that.

To strangers she almost exclusively communicated with non-verbal language. There are instances where these strangers would stop in their place, leave everything behind, walk up to her and hand her a toy or another item — that item was always something that was known to be something liked and desired.

In one instance a woman got out of her car at a stoplight at an intersection, emptied her purse and handed it to Genie. The woman and Genie did not exchange a word; they understood each other completely non-verbally.

So what Genie did, was to pick up cues with her visual system and translated the emotional and cognitive state of that woman into non-verbal cues and actions, which she would then use to change the mental state of the woman. In turn that the woman would then desire to give the purse to Genie (which Genie probably could not even see).

Clearly, Genie was very exceptional at non-verbal communication — but what would happen if you pitched her against a deep learning object recognition system? The deep learning system would be much better than Genie on any data set you would pick. Do you think it would be fair to say that the convolutional net is better at object recognition than Genie is? I do not think so.

This shows how primitive and naïve our approach to computer vision is. Object recognition is a part of human vision, but it is not what makes it exceptional.

### Can we do with less computational power?

“We do not need as much computational power as the brain has, because our algorithms are (will be) better than that of the brain.”

I hope you can see after the descriptions in this blog post that this statement is rather arrogant.

We do not know how the brain really learns. We do not understand information processing in the brain in detail. And yet we dare to say we can do better?

Even if we did know how the brain works in all its details, it would still be rather naïve to think we could create general intelligence with much less. The brain developed during many hundreds of millions of years through evolution. Evolutionary, it is the most malleable organ there is: The human cortex shrunk by about 10% during the last 20000 years, and the human brain adapted rapidly to the many ways we use verbal language — a very recent development in evolutionary terms.

It was also shown that the number of neurons in each animal’s brain is almost exactly the amount which it can sustain through feeding (we probably killed off the majority of all mammoths by about 20000 years ago). We humans have such large brains because we invented fire and cooking with which we could predigest food which made it possible to sustain more neurons. Without cooking, the intake of calories would not be high enough to sustain our brains and we would helplessly starve (at least a few thousand years ago; now you could survive on a raw vegan diet easily — just walk into a supermarket and buy a lot of calorie-dense foods). With this fact, it is very likely that brains are optimized exhaustively to create the best information processing which is possible for the typical calorie intake of the respective species — the function which is most expensive in an animal will be most ruthlessly optimized to enhance survival and procreation. This is also very much in line with all the complexity of the brain; every little function is optimized thoroughly and only as technology advances we can understand step by step what this complexity is made for.

There are many hundreds of different types of neurons in the brain, each with their designated function. Indeed, neuroscientists often can differentiate different brain regions and their function by looking at the changing architecture and neuron types in a brain region. Although we do not understand the details of how the circuits perform information processing, we can see that each of these unique circuits is designed carefully to perform a certain kind of function. These circuits are often replicated in evolutionary distinct species which share a common ancestor that branched off into these different species hundreds of millions of years ago, showing that such structures are evolutionarily optimal for the tasks they are processing.

The equivalent in deep learning would be, if we had 10000 different architectures of convolutional nets (with its own set of activation functions and more) which we combine meticulously to improve the overall function of our algorithm ― do you really think we can build something which can produce as complex information processing, but which follows a simple general architecture?

It is rather naïve to think that we can out-wit this fantastically complex organ when we are not even able to understand its learning algorithms.

On top of this, the statement that we will develop better algorithms than the brain uses is unfalsifiable. We can only prove it when we achieve it, we cannot disprove it. Thus it is a rather nonsensical statement that has little practical value. Theories are usually useful even when there is not enough evidence to show that they are correct.

The standard model of physics is an extremely useful theory used by physicists and engineers around the world in their daily life to develop the high tech products we enjoy; and yet this theory is not complete, it was amended just a few days ago when a new particle was proven to exist in the LHC experiment.

Imagine if there were another model, but you would only be able to use it when we have proven the existence of *all particles*. This model would then be rather useless. When it makes no predictions at all about the behavior in the world, we would be unable to manufacture and develop electronics with this theory. Similarly, the statement that we can develop more efficient algorithms than the brain does not help; it rather makes it more difficult to make further progress. The brain should really be our main point of orientation.

Another argument, which would be typical for Yann LeCun (he made a similar argument during a panel) would be: Arguably, airplanes are much better at flying than birds are; yet, if you describe the flight of birds it is extremely complex and every detail counts, while the flight of airplanes is described simply by the fluid flow around an airfoil. Why is it wrong to expect this simplicity from deep learning when compared to the brain?

I think this argument has some truth in it, but essentially, it asks the wrong question. I think it is clear that we need not to replicate everything in detail in order to achieve artificial intelligence, but the real question is: Where do we draw the line? If you get to know that neurons can be modeled in ways that closely resemble convolutional nets, would you go so far and say, that this model is too complex and we need to make it simpler?

## Part IV: Predicting the growth of practical computational power

There is one dominant measure of performance in high-performance computing (HPC) and this measure is floating point operations per second (FLOPS) on the High Performance LINPACK (HPL) benchmark – which measures how many computations a system can do in a second when doing distributed dense matrix operations on hundreds or thousands of computers. There exists the TOP 500 list of supercomputers, which is a historical list based on this benchmark which is the main reference point for the performance of a new supercomputer system.

But a big but comes with the LINPACK benchmark. [It does not reflect the performance in real, practical applications](http://www.sandia.gov/~maherou/docs/HPCG-Benchmark.pdf) which run on modern supercomputers on a daily basis, and thus, the fastest computers on the TOP 500 list are not necessarily the fastest computers for practical applications.

Everybody in the high performance computing community knows this, but it is so entrenched in the business routine in this area, that when you design a new supercomputer system, you basically have to show that your system will be able to get a good spot on the TOP 500 in order to get funding for that supercomputer.

Sometimes such systems are practically unusable, like the Tianhe-2 supercomputer which still holds the top spot on the LINPACK benchmark after more than three years. The potential of this supercomputer goes largely unused because it is too expensive to run (electricity) and the custom hardware (custom network, Intel Xeon Phi) requires new software, which would need years of development to reach the levels of sophistication of standard HPC software. The Tianhe-2 runs only at roughly one third of its capacity, or in other words, it practically stands idle for nearly 2 out of 3 minutes. The predecessor of the Tianhe-2, the Tianhe-1, fastest computer in the world in 2010 (according to LINPACK), has not been used since 2013 due to bureaucracy reasons.

While outside of China, other supercomputers of similar design fare better, they typically do not perform so well in practical applications. This is so, because the used accelerators like graphic processing units (GPUs) or Intel Xeon Phis can deliver high FLOPS in such a setup, but they are severely limited by network bandwidth bottlenecks.

To correct the growing uselessness of the LINPACK benchmark a new measure of performance was developed: The high performance conjugate gradient benchmark (HPCG). This benchmark performs conjugate gradient, which requires more communication than LINPACK and as such comes much closer to performance numbers for real applications. I will use this benchmark to create my estimates for a singularity.

[![top500_2](https://i0.wp.com/timdettmers.com/wp-content/uploads/2015/07/top500_2.jpg?zoom=1.25&resize=680%2C544)](https://i0.wp.com/timdettmers.com/wp-content/uploads/2015/07/top500_2.jpg)The TOP500 for the last decade and some data for the HPCG (data collection only began recently). The dashed lines indicate a forecast. The main drivers of computational growth are also shown: Multicore CPU, GPU, and in 2016-2017 3D memory, and some new unknown technology in 2020. Will this growth be sustainable?

However, this benchmark still dramatically overestimates the computing power that can be reached for artificial intelligence applications when we assume that these applications are based on deep learning.

Deep learning is currently the most promising technique for reaching artificial intelligence. It is certain that deep learning — as it is now — will not be enough, but one can say for sure that something similar to deep learning will be involved in reaching strong AI.

Deep learning, unlike other applications has an unusually high demand for network bandwidth. It is so high that for some supercomputer designs which are in the TOP 500 a deep learning application would run slower than on your desktop computer. Why is this so? Because parallel deep learning involves massive parameter synchronization which requires extensive network bandwidth: If your network bandwidth is too slow, then at some point deep learning gets slower and slower the more computers you add to your system. As such, very large systems which are usually quite fast may be extremely slow for deep learning.

The problem with all this is that the development of new network interconnects which enable high bandwidth is difficult and advances are made much more slowly than the advances of computing modules, like CPUs, GPUs and other accelerators. Just recently, Mellanox reached a milestone where they could manufacture switches and InfiniBand cards which operate at 100Gbits per second. This development is still rather experimental, and it is difficult to manufacture fiber-optic cables which can operate at this speed. As such, no supercomputer implements this new development as of yet. But with this milestone reached, there will not be another milestone for many quite a while. The doubling time for network interconnect bandwidth is about 3 years.

Similarly, there is a memory problem. While the speed of theoretical processing power of CPUs and GPUs keeps increasing, the memory bandwidth of RAM is almost static. This is a great problem, because now we are at a point where it costs more time to move the data to the compute circuits than to actually make computations with it.

With new developments such as 3D memory one can be sure that further increases in memory bandwidth will be achieved, but we have nothing after that to increase the performance further. We need new ideas and new technology. Memory will not scale itself by getting smaller and smaller.

However, currently the biggest hurdle of them all is power consumption. The Tianhe-2 uses 24 megawatts of power, which totals to $65k-$100k in electricity cost per day, or about $23 million per year. The power consumed by the Tianhe-2 would be sufficient to power about 6000 homes in Germany or 2000 homes in the US (A/C usage).

[![hpc_constraints](https://i0.wp.com/timdettmers.com/wp-content/uploads/2015/07/hpc_constraints.png?zoom=1.25&resize=680%2C430)](https://i0.wp.com/timdettmers.com/wp-content/uploads/2015/07/hpc_constraints.png)An overview about how the performance constraints changed from old to new supercomputers. Adapted from [Horst Simon](http://www2.lbl.gov/Publications/Deputy-Director/bio.html)‘s [presentation](http://www.researchgate.net/profile/Horst_Simon/publication/261879110_Why_we_need_Exascale_and_why_we_won't_get_there_by_2020/links/0c960535dbade00bbc000000.pdf)

### Physical limitations

Furthermore, there are physical problems around the corner. Soon, our circuits will be so small that electrons will start to show quantum effects. One such quantum effect is quantum tunneling. In quantum tunneling an electron sits in two neighboring circuits at once, and decides randomly to which of these two locations it will go next.

If this would happen at a larger scale, it would be like charging your phone right next to your TV, and the electrons decide they want to go to your cell phone cable rather than to your TV; so they jump over to the phone cable cutting off the power to your TV. Quantum tunneling will become relevant in 2016-2017 and has to be taken into account from there on. New materials and “insulated” circuits are required to make everything work from here on.

With new materials, we need new production techniques which will be very costly because all computer chips relied on the same, old but reliable production process. We need research and development to make our known processes working with these new materials and this will not only cost money but also cost time. This will also fuel a continuing trend where the cost for producing computer chips increases exponentially (and growth may slow due to costs). Currently, the tally is at $9bn for such a semiconductor fabrication plant (fab) increasing at a relatively stable rate of about 7-10% higher costs per year for the past decades.

After this, we are at the plain physical limits. A transistor will be composed of not much more than a handful of atoms. We cannot go smaller than this, and this level of manufacturing will require extensive efforts in order to get such devices working properly. This will start to happen around 2025 and the growth may slow from here due to physical limitations.

### Recent trends in the growth of computational power

So to summarize the previous section: (1) LINPACK performance does not reflect practical performance because it does not test memory and network bandwidth constraints; (2) memory and network bandwidth are now more important than computational power, however (3) advances in memory and network bandwidth will be sporadic and cannot compete with the growth in computational power; (4) electrical costs are a severe limitation (try to justify a dedicated power plant for a supercomputer if citizen face sporadic power outages), and also (5) computational power will be limited by physical boundaries in the next couple of years.

It may not come to a surprise then that the growth in computational power has been slowing down in recent years; this is mainly due to power efficiencies which will only be improved gradually, but the other factors also take its toll, like network interconnects which cannot keep up with accelerators like GPUs.

If one takes the current estimate of practical FLOPS of the fastest supercomputer, the Tianhe-2 with 0.58 petaflops on HPCG, then it would take 21 doubling periods until the lower bound of the brain’s computational power is reached. If one uses Moore’s Law, we would reach that by 2037; if we take the growth of the last 60 years, which is about 1.8 years per doubling period, we will reach this in the year 2053. If we take a lower estimate of 3 years for the doubling period due to the problems listed above we will reach this in 2078. While for normal supercomputing applications memory bandwidth is the bottleneck for practical applications as of now, this may soon change to networking bandwidth, which doubles about every 3 years. So the 2078 estimate might be quite accurate.

[![growth](https://i2.wp.com/timdettmers.com/wp-content/uploads/2015/07/growth.jpg?zoom=1.25&resize=680%2C483)](https://i2.wp.com/timdettmers.com/wp-content/uploads/2015/07/growth.jpg)Growth in computing performance with respect to the HPCG benchmark. Both computing performance and factory costs are assumed to keep growing steadily at an exponential rate with doubling period of 18 or 36 months, respectively.

Now remember that, (1) the HPCG benchmark has much higher performance than typical deep learning applications which rely much more on network and memory bandwidth, and (2) that my estimate for the computational complexity of the brain is a lower bound. One can see that an estimate beyond 2100 might be not too far off. To sustain such a long and merciless increase in computation performance will require that we develop and implement many new ideas while operating at the border of physical limitations as soon as by 2020. Will this be possible?

Where there’s a will, there’s a way — the real question is: Are we prepared to pay the costs?

# Conclusion

Here I have discussed the information processing steps of the brain and their complexity and compared them to those of deep learning algorithms. I focused on a discussion of basic electrochemical information processing and neglected biological information processing.

I used an extended linear-nonlinear-Poisson cascade model as groundwork and related it to convolutional architectures.

With this model, I could show that a single neuron has an information processing architecture which is very similar to current convolutional nets, featuring convolutional stages with rectified non-linearities which activities are then regularized by a dropout-like method. I also established a connection between max-pooling and voltage-gated channels which are opened by dendritic spikes. Similarities to batch-normalization exist.

This straightforward similarity gives strong reason to believe that deep learning is really on the right path. It also indicates that ideas borrowed from neurobiological processes are useful for deep learning (the problem was that progress in deep learning architectures often preceded knowledge in neurobiological processes).

My model shows that it can be estimated that the brain operates at least 10x^21 operations per second. With current rates of growth in computational power we could achieve supercomputers with brain-like capabilities by the year 2037, but estimates after the year 2080 seem more realistic when all evidence is taken into account. This estimate only holds true if we succed to stomp limitations like physical barriers (for example quantum-tunneling), capital costs for semiconductor fabrication plants, and growing electrical costs. At the same time we constantly need to innovate to solve memory bandwidth and network bandwidth problems which are or will be the bottlenecks in supercomputing. With these considerations taken into account, it is practically rather unlikely that we will achieve human-like processing capabilities anytime soon.

## Closing remarks

My philosophy of this blog post was to present all information on a single web-page rather than scatter information around. I think this design helps to create a more sturdy fabric of knowledge, which, with its interwoven strains of different fields, helps to create a more thorough picture of the main ideas involved.  However, it has been quite difficult to organize all this information into a coherent picture and some points might be more confusing than enlightening. Please leave a comment below to let me know if the structure and content need improvement, so that I can adjust my next blog post accordingly.

I would also love general feedback for this blog post.

Also make sure to share this blog post with your fellow deep learning colleagues. People with raw computer science backgrounds often harbor misconceptions about the brain, its parts and how it works. I think this blog post could be a suitable remedy for that.

## The next blog post

The second post in this series on neuroscience and psychology will focus on the most important brain regions and their function and connectivity. The last and third part in the series will focus on psychological processes, such as memory and learning, and what we can learn from that with respect to deep learning.

#### **Acknowledgments**

I would like to thank Alexander Tonn for his useful advice and for proofreading this blog post.

#### **Important references and sources**

**Neuroscience**

Brunel, N., Hakim, V., & Richardson, M. J. (2014). Single neuron dynamics and computation. *Current opinion in neurobiology*, *25*, 149-155.

Chadderton, P., Margrie, T. W., & Häusser, M. (2004). Integration of quanta in cerebellar granule cells during sensory processing. *Nature*, *428*(6985), 856-860.

De Gennaro, L., & Ferrara, M. (2003). Sleep spindles: an overview. *Sleep medicine reviews*, *7*(5), 423-440.

Ji, D., & Wilson, M. A. (2007). Coordinated memory replay in the visual cortex and hippocampus during sleep. *Nature neuroscience*, *10*(1), 100-107.

Liaw, J. S., & Berger, T. W. (1999). Dynamic synapse: Harnessing the computing power of synaptic dynamics. *Neurocomputing*, *26*, 199-206.

Ramsden, S., Richardson, F. M., Josse, G., Thomas, M. S., Ellis, C., Shakeshaft, C., … & Price, C. J. (2011). Verbal and non-verbal intelligence changes in the teenage brain. *Nature*, *479*(7371), 113-116.

Smith, S. L., Smith, I. T., Branco, T., & Häusser, M. (2013). Dendritic spikes enhance stimulus selectivity in cortical neurons in vivo. *Nature*, *503*(7474), 115-120.

[Stoodley, C. J., & Schmahmann, J. D. (2009). Functional topography in the human cerebellum: a meta-analysis of neuroimaging studies. *Neuroimage*,*44*(2), 489-501.](http://pnns.org/pdf/STOODLEY%20and%20SCHMAHMANN%20Cerebellum%20Meta-analysis%20functional%20topography%20NeuroImage%202008.pdf)

**High performance computing**

Dongarra, J., & Heroux, M. A. (2013). Toward a new metric for ranking high performance computing systems. *Sandia Report, SAND2013-4744*, *312*.

[PDF: HPCG Specification](https://software.sandia.gov/hpcg/doc/HPCG-Specification.pdf)

[Interview: Why there will be no exascale computing before 2020](http://www.top500.org/blog/no-exascale-for-you-an-interview-with-berkeley-labs-horst-simon/)

[Slides: Why there will be no exascale computing before 2020](http://www.researchgate.net/profile/Horst_Simon/publication/261879110_Why_we_need_Exascale_and_why_we_won't_get_there_by_2020/links/0c960535dbade00bbc000000.pdf)

[Interview: Challenges of exascale computing](http://www.vrworld.com/2015/03/23/jack-dongarra-on-the-great-exascale-challenge-and-rising-hpc-powers/)

**Image references**

[Anwar, H., Roome, C. J., Nedelescu, H., Chen, W., Kuhn, B., & De Schutter, E. (2014). Dendritic diameters affect the spatial variability of intracellular calcium dynamics in computer models. *Frontiers in cellular neuroscience*, *8*.](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC4107854/)
