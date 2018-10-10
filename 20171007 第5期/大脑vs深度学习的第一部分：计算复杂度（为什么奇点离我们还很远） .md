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

但是，这个扩展模型实际上与深度学习非常相似，因此我将在这里包含这些细节。
扩展的线性-非线性-泊松级联模型(LNP)可以更准确地模拟神经元的行为。扩展的LNP模型目前被看作是[神经元处理信息的精确模型](http://www.sciencedirect.com/science/article/pii/S0959438814000130)。然而，扩展的LNP模型仍然有一些细节问题，这些细节对于模拟大规模脑功能并不重要。实际上，将这些细节加入到模型中几乎不会增加额外的计算复杂度，但会使模型更复杂、难以理解。因此在模拟中包含这些细节会违反为定论找到最简单模型的科学方法。 然而，这个扩展模型实际上与深度学习非常相似，这些细节将包含在本文中。

还有其他好的模型也适用于此。我选择LNP模型的主要原因是它跟深度学习非常类似。我将在下一部分用这个模型比较神经元的结构与卷积网络的结构，同时我将得出对大脑复杂性的估计。

## 第二部分：大脑与深度学习 - 对比分析

现在我将逐步解释大脑是如何处理信息的。我将说明信息处理的步骤，这些步骤是很容易理解的，并且有可靠的证据支持。在这些步骤之上，在生物学层面（蛋白质和基因）有许多中间步骤，这些步骤仍未被充分理解，但已知对信息处理非常重要。我不会深入研究这些生物过程，而是提供一个简短的大纲，这可能会帮助渴望知识的读者自己深入研究。我们现在开始旅程，从一个发射神经元释放的神经递质，沿着它的所有过程走，直到到达下一个神经元释放它的神经递质的位置，这样我们就回到开始的地方。

下一节将介绍几个新术语，这些新术语是博客的其余部分所必需的，所以如果你不熟悉基本的神经生物学，请仔细阅读。

[![neuron_anatomy](https://i2.wp.com/timdettmers.com/wp-content/uploads/2015/07/neuron_anatomy1.jpg?zoom=1.25&resize=680%2C390)](https://i2.wp.com/timdettmers.com/wp-content/uploads/2015/07/neuron_anatomy1.jpg)图片来源: [1](https://commons.wikimedia.org/wiki/File:Neuron_Hand-tuned.svg),[2](https://commons.wikimedia.org/wiki/File:SynapseSchematic_lines.svg),[3](http://faculty.ivytech.edu/~shopper6/ANPweb/gallery/Week_011-2.html),[4](http://faculty.ivytech.edu/~shopper6/ANPweb/gallery/Week_011-2.html)

神经元利用轴突——一种管状结构——在大脑中长时间传输电信号。当一个神经元放电时，它会释放一个动作电位——沿着它的轴突分叉成一个小结尾的树，称为轴突末端。在每个轴突末端的每一个末端都有一些蛋白质将电子信息转化为化学信息:小球体——突触小泡——充满了一些神经递质，每个被释放到神经元外的区域，称为突触间隙。该区域将轴突末端与下一个神经元（突触）的开始分开，并允许神经递质自由移动以完成不同的任务。

突触通常位于一个看起来非常像树或植物的根的结构上，这是由树枝组成的树枝状树，树枝分枝成更大的臂（这代表神经网络中神经元之间的连接），最终到达细胞的核心，称为体细胞。这些树突几乎包含将一个神经元连接到下一个神经元的所有突触，从而形成主要连接。突触可以容纳数百种神经递质可以自身结合的受体。

您可以将这种轴突末端和突触的化合物想象成（卷积）输入层（图片）进入卷积网络。每个神经元可拥有少于5个树突或多达数十万个。之后我们将看到树突树的功能类似于卷积层的组合，随后是卷积网络中的池化层。

回到生物学过程，突触囊泡与轴突末端的表面融合，并从内到外将它们的神经递质溢出到突触间隙中。在那里，神经递质由于环境温度而振动发生漂移，直到它们（1）找到适合其键（神经递质）的合适的锁（受体蛋白），（2）神经递质遇到分解它们的蛋白质，或（3）神经递质遇到一种蛋白质，这种蛋白质将它们拉回轴突（再吸收），在那里它们被重复使用。抗抑郁药主要通过（3）预防或（4）促进神经递质5-羟色胺的再吸收的作用; （3）防止再吸收会在几天或几周后产生信息处理的变化，而（4）促进再吸收会导致在几秒或几分钟内发生变化。因此神经递质再吸收机制对于每分钟的信息处理是不可或缺的。在LNP模型中忽略了重新吸收的过程。

然而，神经递质释放的数量，给定神经递质的突触数量，以及实际上神经递质进入突触上拟合蛋白质的数量可以被认为是（全）连接层中的权重参数，以上都是神经网络中的一部分。换句话说，神经元的总输入是所有轴突 - 末端 - 神经递质 - 突触相互作用的总和。在数学上，我们可以将其等效为两个矩阵的点积（A点乘B; [所有输入的神经递质的数量]点乘[所有突触上拟合蛋白质的量]）。

在神经递质锁定到突触上的拟合蛋白后，它可以做很多不同的事：一般情况下，神经递质（1）打开通道，让带电粒子流入（通过扩散）进入树突，但它也会产生罕见的效果：神经递质（2）结合蛋白，然后产生蛋白质信号级联，（2a）激活（上调）一个基因，然后用于产生一种新的蛋白质。整合到神经元的表面，其树突和/或其突触中;（2b）通知现有蛋白质在特定位点发挥某种功能（产生或移除更多突触，开启一些入口，将新蛋白质附着到突触表面）。这是在NLP模型中被忽略的。

一旦通道打开，带负电或带正电的粒子进入树突棘。树突棘是一种小的蘑菇状结构，突触附着在上面。这些树突棘可以存储电势并具有自己的动态信息处理。这是在NLP模型中被忽略的。

[![dendritic_spine](https://i0.wp.com/timdettmers.com/wp-content/uploads/2015/07/dendritic_spine.jpg?zoom=1.25&resize=471%2C335)](https://i0.wp.com/timdettmers.com/wp-content/uploads/2015/07/dendritic_spine.jpg)树突棘具有其自身的内部信息处理动力学，其主要由其形状和大小决定。 图片来源: [1](https://en.wikipedia.org/wiki/File:Spline_types_3D.png),[2](https://en.wikipedia.org/wiki/File:Dendritic_spines.jpg)

进入树突棘的粒子带负电荷或带正电荷——打开通道的神经递质仅为负粒子，而剩下的为正粒子。还存在使带正电的粒子离开神经元的通道，从而增加电势的负面性（如果神经元变得过于阳性则会被“激发”）。蘑菇状树突棘的大小和形状与其行为相对应。这是在NLP模型中被忽略的。

一旦粒子进入脊柱，他们可以影响许多事。一般情况下，他们将（1）沿树突移动到神经元中的细胞体，然后，如果细胞过度带电（去极化），它们会诱发动作电位（神经元“发射”）。但是其他动作也很常见：带电粒子积聚在树突棘，并且（2）打开电压门控通道，这可以进一步使细胞极化（这是上面提到的树突脊柱信息处理的一个例子）。另一个非常重要的过程是（3）树突状尖峰。

### 树突状尖峰

树突状尖峰是一种已知存在多年的现象，但仅在2013年，这些技术已经足够先进，可以收集数据来显示这些尖峰对于信息处理来说非常重要。要测量树突峰值，必须在计算机的帮助下将一些非常小的夹子（clamps）连接到树突上，该计算机可以非常精确地移动夹具。为了了解夹具的位置，您需要一个特殊的显微镜来观察夹具，即便进入树枝状晶体时，夹子仍然需要长时间固定在一个相当盲目的物体（blind matter）上，因为在如此微小的范围内，世界上只有少数团队拥有将这种夹具连接到树突上的设备和技能。

但是，这些团队收集的直接数据足以将树突状尖峰作为重要的信息处理活动。由于将树突状尖峰引入神经元的计算模型中，单个神经元的复杂性变得非常类似于具有两个卷积层的卷积网络。正如我们后面将看到的，LNP模型也使用十分类似于非线性修正线性函数的功能，并且还使用与dropout非常相似的尖峰发生器（spike generator） - 因此神经元非常像整个卷积网络。但关于这一点的更多内容，需要回到讨论树突状尖峰及其究竟是什么。

当在树突中达到临界水平的去极化时，发生树突状尖峰。去极化放电作为一个电势沿着树突的墙壁去触发电压门控通道，如果电势足够强，那么电势就会达到神经元的核心，触发真正的动作点位。如果树突状尖峰未能触发动作电位，则在一瞬间内，相邻树突打开电压门控通道。由于从树突打开的通道，更多带电粒子进入神经元，然后可以触发（常见）或抑制（罕见）神经元细胞体（体细胞）的完整动作电位。

[![树突状尖峰](https://i0.wp.com/timdettmers.com/wp-content/uploads/2015/07/dendritic_spikes.png?zoom=1.25&resize=677%2C263)](https://i0.wp.com/timdettmers.com/wp-content/uploads/2015/07/dendritic_spikes.png)
A显示了不模拟树突状尖峰的神经元的计算机模型; B模拟树枝状尖峰的简单动态; C模拟树枝状尖峰的更复杂的动力学，其考虑了颗粒的一维扩散（类似于卷积操作）。请注意，这些图像只是特定时刻的快照。非常感谢[Berd Kuhn](https://groups.oist.jp/onu). 图片版权所有© 2014 Anwar, Roome, Nedelescu, Chen, Kuhn和De Schutter发表于*细胞神经科学前沿 （Anwar等人，2014）*

此过程与max-pooling非常相似，将单个大型激活“覆盖”其他相邻值。然而，在树突峰值之后，相邻值不会像在深度学习中使用的最大池化中那样被覆盖，但是电压门控通道的打开极大地放大了树突内所有相邻分支中的信号。因此，树枝状尖峰可以将相邻树突中的电化学水平提高到更接近最大输入的水平 - 该效果接近max-pooling。

实际上，已经证明视觉系统中的树突尖峰与用于物体识别的卷积网络中的max-pooling具有相同的目的：在深度学习中，最大池化用于实现（有限的）旋转，平移和尺度不变性（意味着我们的算法可以检测图像中的目标，其中目标被旋转，移动或缩小/放大几个像素）。可以将此过程视为将所有周围像素设置为相同的激活并使每个激活共享下一层的权重（在软件中，为了计算效率而舍去值 - 这在数学上是等效的）。类似地，已经表明视觉系统中的树突尖峰对物体的方向敏感。因此树突状峰值不仅具有计算相似性，而且还具有与之（方向敏感）相似的功能。

这个类比并没有结束。在神经网络反向传播期间——也就是当动作电位从细胞体传播回到树突时——信号不能反向传播到树突分支的源头，这是因为最近的电活动而被“停用”。因此，清晰（clear）的学习信号被发送到未激活的分支。一开始，这可能看起来与最大池的反向传播完全相反，除了最大池激活之外的所有内容都是反向传播的。然而，树突中没有反向传播信号是罕见的，并且树突本身代表学习信号。因此，产生树突状尖峰的树突具有特殊的学习信号，就像最大池中的激活单元一样。

为了更好地了解树枝状尖峰是什么以及它们看起来像什么，我非常希望你观看[这个视频](http://www.hhmi.org/research/how-do-neurons-compute-output-their-inputs) (我没有版权). 该视频显示了两个树突状尖峰如何导致动作电位。

树突和动作电位的结合以及树突树状结构被发现对海马体的学习和记忆至关重要，海马体是负责形成新记忆并在晚上将它们写入我们的“硬盘”的主要大脑区域。

树突峰值是计算复杂性的主要因素之一，这些因素是从大脑复杂性的过去模型中遗漏下来的。此外，这些新发现表明神经反向传播不一定是神经元到神经元来学习复杂的功能; 单个神经元已经实现了卷积网络，因此具有足够的计算复杂性来模拟复杂现象。因此，几乎不需要跨越多个神经元的学习规则——单个神经元也可以产生我们用卷积网络生成的相同输出。

但是这些发现树突峰值并不是唯一的进步在我们理解信息的处理步骤在这个阶段的神经信息处理途径
但是这些关于树突状尖峰的发现并不是我们理解信息处理步骤的这个阶段所取得的唯一进展。基因操作和蛋白质合成是将计算复杂性提高数量级的来源，直到最近取得了进展，揭示了生物信息处理的真正扩展。

### 蛋白质信号级联

正如我在本部分的介绍中所说，我不会广泛涉及生物信息处理的各个部分，但我想给你足够的信息，以便你可以从这里开始学习到更多。

必须要理解的一点是，细胞与教科书中的显示方式有很大不同。细胞爬行蛋白质：在任何给定的人类细胞中都有大约100亿个蛋白质，这些蛋白质并非空闲：它们与其他蛋白质结合，处理任务或移动以寻找新的任务。

上述所有功能都是蛋白质的。例如，锁定和锁定机制以及为离开和进入神经元的带电粒子起到看门人的通道作用都是蛋白质。我在本部分中所指的蛋白质不是这些常见蛋白质，而是具有特殊生物功能的蛋白质。

作为一个例子，丰富的神经递质谷氨酸可以与NDMA受体结合，然后NDMA受体为许多不同种类的带电粒子打开通道，并且在打开后，在神经元激发时，通道关闭。突触的强度十分依赖于该过程，其中突触根据NDMA受体的位置和反向传播到突触的信号并定时进行调整。我们知道这个过程对于大脑学习至关重要，但它只是大型工程中的一小部分。

可以进入神经元的带电粒子可以另外诱导蛋白质信号级联拥有它们自己的。例如，下面的级联显示了活化的NMDA受体（绿色）如何使带电的钙CA2 +在其内部触发级联，最终导致AMPAR受体（紫色）被搬运并安装在突触上。

[![观察AMPAR搬运](https://i0.wp.com/timdettmers.com/wp-content/uploads/2015/07/regulationofampartrafficking.jpg?zoom=1.25&resize=680%2C531)](https://i0.wp.com/timdettmers.com/wp-content/uploads/2015/07/regulationofampartrafficking.jpg)图片来源: [1](https://commons.wikimedia.org/wiki/File:RegulationOfAMPARTrafficking.jpg)

一次又一次地证明，这些特殊的蛋白质对神经元的信息处理有很大的影响，但很难从这个看似混乱的100亿蛋白质的群体中挑选出特定类型的蛋白质，并研究其准确的功能。研究结果通常很复杂，涉及许多不同蛋白质的反应链，直到达到所需的最终产品或最终功能。开始和结束功能通常是已知的，但不是从一个到另一个的确切路径。先进的技术有助于详细研究蛋白质，随着技术越来越好，我们将进一步了解神经元中的生物信息处理过程。

### 基因操作

生物信息处理的复杂性并不以蛋白质信号级联结束，100亿个蛋白质并不是完成其任务的工人的随机群体，但这些工人具有特定的数量，以满足目前相关的特定功能。所有这些都是由包含辅助蛋白，DNA和信使RNA（mRNA）的紧密反馈环控制的。

如果我们使用编程来比喻描述整个过程，那么DNA代表整个github网站及其所有公共包，而信使RNA是一个大型库，其中包含许多其其它具有不同功能的小型库（类似于C ++ boost库）。




这一切都始于想要解决的编程问题（检测到生物问题）。您可以使用谷歌和stackoverflow来找到可以用解决问题的库的建议，很快你会发现建议你使用库X来解决问题Y（在一个地方发现到了问题Y，根据已有的解决方案找到了蛋白质X细胞，蛋白质检测到这个缺陷然后级联成蛋白质信号的链，这导致该基因将可产生蛋白质X上调;这里的上调是一个“嘿！请生产更多吧！”信号到蛋白质X的细胞核所在的DNA）。你下载该库并进行编译（复制基因G（转录）短串mRNA构造一长串mRNA的DNA）。然后你用相应的配置进行配置安装（mRNA离开核心，mRNA转化为蛋白质，蛋白质可以在此之后通过其他蛋白质调整），并将库安装在全局“/lib”目录中（蛋白质折叠成正确的形状，之后它可以完全发挥作用）。安装完库后，将库中所需的部分导入您的程序（折叠的蛋白质（随机）移动到需要的位置）并使用该库的某些功能来解决您的问题（蛋白质的工作来解决这个问题）。

除此之外，神经元还可以动态地改变它们的基因组，也就是说它们可以动态地改变它们的github仓库来添加或删除库。

为了进一步了解这一过程，您可能需要观看以下视频，其中显示了HIV如何产生蛋白质以及病毒如何改变宿主DNA以满足其需要。此视频动画中描述的过程与神经元中发生的过程非常相似。为了使其与神经元中的过程更加相似，可以想象HIV是一种神经递质，并且HIV细胞中包含的所有物质首先都存在于神经元中。您所拥有的是准确表示神经元如何利用他们的基因和蛋白质：

[https://youtu.be/RO8MP3wMvqg](https://youtu.be/RO8MP3wMvqg)

您可能会问，是不是因为您体内的每个细胞都具有（几乎）相同的DNA以便能够自我复制？一般来说，大多数细胞都是这样，但大多数神经元不是这样。神经元通常具有与您在出生时分配的原始基因组不同的基因组。神经元可以具有额外或更少的染色体，并且从某些染色体中移除或添加信息序列。

结果表明，这种行为对于信息处理非常重要，如果出现问题，这可能会导致抑郁症或阿尔茨海默病等脑部疾病。最近还显示，神经元每天改变其基因组以改善信息处理需求。

因此，当你前五天坐在办公桌，然后在周末决定开始徒步旅行时，大脑会根据这项新任务调整其神经元，这是很有意义的，因为在环境变化后需要完全不同的信息处理。

同样，从进化的角度来看，在村庄内进行狩猎/采集和社交活动有不同的“模式”，这些是有益的——似乎这个功能适合这样的事。通常，生物信息处理设备在响应从几分钟到几小时的较慢信息处理需求方面非常有效。

关于深度学习，一个等效的功能是以重要也是基于规则的方式改变训练有素的卷积网络的功能; 例如，当从一个任务更改为另一个任务时，将变换应用于所有参数（识别街道数量->变换参数->识别行人）。

这种生物信息处理的任何内容都不是由LNP模型建立的。

回顾这一切，似乎很奇怪，许多研究人员认为他们只能通过专注于电化学特性和神经元间相互作用来复制大脑的行为。想象一下，卷积网络中的每个单元都有自己的github，从中*学习*并动态下载，编译和使用最好的库来解决某个任务。从这一切你可以看出，单个神经元可能比整个卷积网络更复杂，但我们继续从这里开始关注电化学过程，看看它在哪里引导我们。

### 回到LNP模型

在介绍完上面的这些之后，我们模型的信息处理只有一个相关的步骤。一旦达到临界水平的去极化，神经元通常会发射。但并非总是如此，也存在着阻止神经元发射的机制。例如，在神经元发射后不久，其电势太强而不能产生完全成熟的动作电位，因此它不能再次发射。即使在达到足够的电势时也可能存在这种阻塞，因为这种阻塞是生物功能而不是物理开关。

在LNP模型中，动作电位的这种阻塞是建模为具有泊松分布的非均匀泊松过程。以泊松分布为模型的泊松过程意味着神经元在第一次或第二次达到其阈值电位时具有非常高的发射概率，但也可能（以指数递减的概率）神经元可能不会发射多次。

[![泊松](https://i0.wp.com/timdettmers.com/wp-content/uploads/2015/07/poisson.png?zoom=1.25&resize=652%2C347)](https://i0.wp.com/timdettmers.com/wp-content/uploads/2015/07/poisson.png)具有随机抽取样本的泊松（0.5）分布。这里0,1,2,3表示神经元发射前的等待时间，因此0表示它会毫无延迟地发射，而2表示即使它可以物理发射也不会发射两个周期。

这个规则有例外，其中神经元禁用这种机制并以仅由物理学控制的速率连续发射——但这些是我将在此时忽略的特殊事件。通常，这整个过程与深度学习中使用的丢弃（dropout）非常相似，它使用均匀分布而不是泊松分布; 因此，这个过程可以看作是大脑使用的某种正则化方法而不是丢弃。

在下一步中，如果神经元发射，它会释放动作电位。动作电位的幅度差别很小，这意味着神经元产生的电位几乎总是具有相同的幅度，因此是可靠的信号。当这个信号沿着轴突传播时，它变得越来越弱。当它流入轴突末端的分支时，其最终强度将取决于这些分支的形状和长度; 因此每个轴突末端将接收不同量的电位。该空间信息与由于动作电位的尖峰模式引起的时间信息一起被转换成电化学信息（显示它们被转化为神经递质自身的峰值，持续约2ms）。调整输出信号,轴突末端可以移动,增加或减少(空间),或者它可能改变其蛋白质组成负责释放突触囊泡(时间)。

现在我们回到开始：神经递质从轴突末端释放（可以建模为稠密矩阵乘法）并且重复此步骤。

### 在大脑中的学习和记忆

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
