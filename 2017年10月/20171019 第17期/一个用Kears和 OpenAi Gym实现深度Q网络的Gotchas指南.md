# 一个用Kears和 OpenAi Gym实现深度Q网络的Gotchas指南

原文链接：[A Tour of Gotchas When Implementing Deep Q Networks with Keras and OpenAi Gym](http://srome.github.io/A-Tour-Of-Gotchas-When-Implementing-Deep-Q-Networks-With-Keras-And-OpenAi-Gym/?from=hackcv&hmsr=hackcv.com&utm_medium=hackcv.com&utm_source=hackcv.com)

从谷歌DeepMind这篇论文开始，关于训练一个模型来玩电子游戏得到了很多人的关注。你，数据科学家/工程师/爱好者，可能不从事强化学习，但可能对教神经网络玩电子游戏感兴趣。谁不是呢?考虑到这一点，这里列出了一些小教程，可以帮助您快速开始自己的实现。

下面的课程是从我自己的[Nature](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html) 论文的[实现](http://www.github.com/srome/ExPyDQN) 中收集的。这些课程针对的是那些从事数据工作的人，但是与典型的监督学习用例相比，强化学习社区中使用的一些非标准方法可能会遇到一些问题。我将讨论神经网络参数的技术细节和所涉及的库。这篇文章首先讲述了关于Nature论文中的基础知识，特别是关于Q学习使用的基本符号。我的实现主要依赖于Keras和Gym，因此很有指导意义，我避免了使用theano/tensorflow中特定的技巧(例如theano的[断开梯度](https://github.com/Theano/theano/blob/52903f8267cff316fc669e207eac4e2ecae952a6/theano/gradient.py#L2002-L2021) )来保持对主要程序逻辑的关注。

## 学习率和损失函数

如果您看过其中的[一些](https://github.com/spragunr/deep_q_rl/blob/master/deep_q_rl/q_network.py) [实现](https://github.com/matthiasplappert/keras-rl/blob/master/rl/agents/dqn.py) ，您会发现通常有一个选项，可以是对一个小批量的损失函数求和，也可以是取平均值。在机器学习领域你听到过的大多数损失函数都是以“平均“开头或者至少在一个小批量中取一个平均值。让我们来谈谈在学习率的背景下，一个总的损失到底意味着什么。

一个典型的梯度下降更新就像这样：

θt+1←θt−λ∇(ℓθt)θt+1←θt−λ∇(ℓθt) 

θtθt  是在tt时刻的权重，λλ 是学习率，ℓθtℓθt 是由θtθt 决定的一个损失函数。我将抑制θtθt依赖损失函数前进。让我们将ℓℓ 定义为一个损失函数（在这里我们假设它求和）且ℓ^ℓ^作为损失函数的意思是小批量(mini batch).对于尺寸为mm的固定小批量，请注意:

ℓ^=1mℓℓ^=1mℓ 

并且在数学上我们有，

∇(ℓ^)=1m∇(ℓ).∇(ℓ^)=1m∇(ℓ). 

这个告诉我们什么？如果你使用学习率λλ训练两个模型，这两种变体将会在mm级别有非常不同的行为。从理论上讲，有了足够小的学习率，你可以考虑到迷你批(mini batch)的大小，并从汇总版本中恢复平均版本的损失行为。然而，这也将导致损失中其他成分系数比如正则化需要调整。坚持使用平均值会使标准系数在许多情况下都能很好地工作。因此，在其他数据科学应用中，使用累加损失函数而不是平均值是很不寻常的，但强化学习中经常出现这种选项。所以，你应该意识到你可能需要调整学习率!也就是说，我的实现使用了平均版本和0.00025的学习率

## 随机性、跳帧和Gym默认值

许多(好的)关于强化学习的博客文章显示了一个模型正在使用“(GameName)-v0”进行训练，所以你可能决定使用ROM，就像你以前看到的那样。到目前为止还不错。然后你看了各种各样的论文，看到了一种叫做“跳帧”的技术，在这种技术中，你把模拟器的神经网络(nn)输出区域堆叠成一个长度宽度为n的图像，然后把它传递给模型，这样你就实现了所有事情都按照计划进行。并不是这样的。取决于你Gym的版本，你可能会遇到麻烦。

在旧版本的Gym，Atari环境随机[重复你的动作2-4步 ](https://github.com/openai/gym/blob/bde0de609d3645e76728b3b8fc2f3bf210187b27/gym/envs/atari/atari_env.py#L69-L71) ，并返回结果帧。就代码而言，这就是在你的跳帧实现中发生的事情。

```
for k in range(frame_skip):
    obs, reward, is_terminal, info = env.step(action) # 2-4 step occur in the emulator with the given action
```

如果你用n=4实现了跳帧，那么你的训练可能是每8-16帧(或者更多!)而不是每4帧。你可以想象它对性能的影响。幸运的是，这已经通过新的rom变得[可调节](https://github.com/openai/gym/blob/master/gym/envs/atari/atari_env.py#L75-L80) ，我们稍后会提到。但是，还有另一个设置需要注意，那就是repeat_action_probability。对于“(Game Name)-v0”rom，这是默认打开的。这是游戏会忽略一个新动作并在每一步重复之前动作的概率。要删除跳帧和重复动作概率，使用“(Game Name)NoFrameskip-v4”rom。可以在[这里](https://github.com/openai/gym/blob/5cb12296274020db9bb6378ce54276b31e7002da/gym/envs/__init__.py#L298-L376) 找到对这些设置的完整理解。

如果我没有指出Gym这么做不是为了破坏你的DQN，那我就太失职了。这样做是有正当理由的，但是当你的神经网络将自己放在无关紧要的位置而不是工作重点时，这种设置会导致无休止的挫败。原因是将随机特性引入到环境中。否则，游戏将是确定性的，你的网络只是简单地记忆一系列步骤，如舞蹈。当使用NoFrameskip ROM时，你必须实现你自己的随机特性来避免网络陷入定式。《自然》杂志的论文(以及许多库)通过“null op max”设置来实现这一点。在每一局游戏的开头(比如乒乓比赛的每一回合)，代理将执行一系列连续的kk空操作(Atari模拟器在Gym中的action=0)，其中kk是一个从[0,null op max]，[0,null op max]均匀采样的整数。在一局的开始可以通过下面的伪代码来实现。

```
obs = env.reset()

# Perform a null operation to make game stochastic
for k in range(np.random.randint(0 , null_op_max, size=1)):
    obs, reward, is_terminal, info = env.step(action)
```

## 梯度裁剪，错误裁剪，奖励裁剪

在《自然》杂志里有许多不同种类的裁剪，并且每一个都很容易被混淆和错误地实现。实际上，如果你认为错误裁剪和梯度裁剪是不同的，那么你已经对此十分困惑了!

### 什么是梯度裁剪

《自然》杂志的这篇论文指出，“删去错误术语”是有帮助的。社区似乎已经拒绝了和“梯度裁剪”一词相似的“错误裁剪”。如果不了解背景，这个词的意思就会模棱两可。在这两种情况下，实际不涉及损失函数、误差或梯度的裁剪。实际上，他们选择的是一个损失函数，它的梯度不会随着误差的大小超过某个区域而增加，因此对于较大的误差限制了梯度更新的大小。特别地，如果损失函数的值大于1，则将该值转换为绝对值。为什么?我们来看看它的导数!

这一项表示平方均误差类函数的损失:

ddx(x−y)2=12(x−y)ddx(x−y)2=12(x−y) 

如果我们考虑损失函数的值或误差,当x−yx−y,我们可以看到一个梯度更新将包含x−yx−y,而另一个没有。没有一个真正好的、朗朗上口的短语来描述上述比“梯度裁剪”更有代表性的数学技巧

标准的方法是使用Huber 损失函数来完成这个任务。该函数的定义如下:

f(x)=12x2 如果 |x|≤δ, δ(|x|−12δ) .

f(x)=12x2 如果|x|≤δ, δ(|x|−12δ) .

实现这一点有一个常见的技巧，像theano和tensorflow这样的符号数学库就可以在不使用switch语句的情况下更容易地求导。下面的叙述和代码展示了这个技巧，并在大多数实现中普遍使用。

实际上需要你编码的函数如下：让q = min(x | |,δ)q = min(δx | |)，然后

g(x)=q22+δ(|x|−q).g(x)=q22+δ(|x|−q). 

当|x|≤δ|x|≤δ 时，

把它代入公式中就可以看出我们得到了

g(x)=12x2g(x)=12x2 

否则当|x|>δ|x|>δ 时，

g(x)=δ22+δ(|x|−δ)=δ|x|−12δ2=δ(|x|−12δ).g(x)=δ22+δ(|x|−δ)=δ|x|−12δ2=δ(|x|−12δ). 

所以,g = fg = f

下面被编码和被绘制出来的函数是一个连续函数，并且有连续导数

```
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

def huber_loss(x, clip_delta=1):
    error = np.abs(x) 
    quadratic_part = np.minimum(error, clip_delta)
    return 0.5 * np.square(quadratic_part) + clip_delta * (error - quadratic_part)

f=np.vectorize(huber_loss)
x = np.linspace(-5,5,100)
plt.plot(x, f(x))
```

[![png](https://camo.githubusercontent.com/7e386e09e50344f2ead0d5125cbc1d3609b0fb5d/687474703a2f2f73726f6d652e6769746875622e696f2f696d616765732f64716e5f696d706c2f6f75747075745f385f312e706e67)](https://camo.githubusercontent.com/7e386e09e50344f2ead0d5125cbc1d3609b0fb5d/687474703a2f2f73726f6d652e6769746875622e696f2f696d616765732f64716e5f696d706c2f6f75747075745f385f312e706e67)

如您所见，该图的斜率(导数)被“裁剪”，其大小永远不会大于1。有趣的一点是它的二阶导数不是连续的。

如果x | |≤δ,0，f′′(x)= 1，否则，如果|x|≤δ, 0 ，f′′(x)=1 (f′′(x)=1 if |x|≤δ, 0 otherwise.f′′(x)=1 if |x|≤δ, 0 otherwise. 原文正确吗？？)

使用二阶方法进行梯度下降可能会导致问题，这就是为什么有些人建议使用伪Huber损失函数，它是Huber损失函数的平滑近似。然而，Huber损失函数对我们的目标来说已经足够了。

### 如果我使用裁剪技巧我会得到什么？

当你把跳帧技术引入方程时，这实际上是一个更加微妙的问题。您是否接受模拟器返回的最后一个结果?但是如果跳过的帧中有一个更好的结果呢?当你对神经网络的实现知之甚少时，如果答案来自基本的Q learning框架时，你会惊讶不已。在Q学习中，一些人使用的是最后一次操作之后的总结果，这也是他们在论文中所做的。这意味着即使在跳过的帧中，您也需要保存观察到的结果，并将它们聚合为给定状态(st,at,rt,st+1)(st,at,rt,st+1)的“结果”。这个累积的结果就是你所获得的。在我的实现中，你可以在TrainingEnvironment类的step函数中看到这一点。伪代码如下:

```
for k in range(frame_skip):
    obs, reward, is_terminal, info = env.step(action) # 1 step in the emulator
    total_reward += reward
```

这个最终的结果是储存在保留区域中的。

## 关于保留区域的思考

“保留区域”是一个存储位置，从训练样本(st,at,rt,st+1)(st,at,rt,st+1)中采样，打破数据点的时间相关性，阻止网络学习发散。最初的论文提到了100万个例子的保留区域(st,at,rt,st+1)(st,at,rt,st+1)。stst和st+1 st+1都是长度宽度为n的图像所以你可以想象内存需求会变得非常大。这种情况对于那些使用Python编程的人来说是很有用的。默认情况下，Gym以数据类型为int8的numpy数组的形式返回图像。如果你对图像进行任何处理，你的图像现在可能是float32类型。所以，当你用int8数据类型储存你的图像时，你一定要确保这是为了节省储存空间，这是十分重要的。对于神经网络来说，任何必要的转换(比如缩放操作)都是在训练之前完成的，而不是在保留区域中存储状态之前。

预分配保留区域所需内存也很有帮助，在我的实现中，我是这样做的:

```
def __init__(self, size, image_size, phi_length, minibatch_size):
    self._memory_state = np.zeros(shape=(size, phi_length, image_size[0], image_size[1]), dtype=np.int8)
    self._memory_future_state = np.zeros(shape=(size, phi_length, image_size[0], image_size[1]), dtype=np.int8)
    self._rewards = np.zeros(shape=(size, 1), dtype=np.float32)
    self._is_terminal = np.zeros(shape=(size, 1), dtype=np.bool)
    self._actions = np.zeros(shape=(size, 1), dtype=np.int8)
```

当然，phi_length=nn是在我们之前的讨论中，模拟器的输出区域数量堆叠在一起形成的状态。

## 闭环调试

有了这么多可变的部件和参数，实现中可能会出现很多错误。当出现错误时，我的建议是将你的起始参数设置为与示例文件相同，以确定一个错误的来源。从最初的NIPS论文到《自然》论文的大部分修改都是为了标准化不同Atari游戏的学习参数和性能。在不同的游戏中会出现很多异常的现象，《自然》杂志上的大多数技术都能防止这个问题。例如，连续求最大用于处理导致对象在某些跳帧设置下消失的屏幕闪烁问题。因此，一旦你更改了一些参数，当你的网络表现不好(或根本不能运行)时，一定要检查如下原因:

​	1.检查你输出的q值是否在批更新之间跳跃，这意味着你的梯度更新是偏大大的。查看你的学习率，从你的梯度发现问题。

​	2.查看你发送给神经网络的数据。跳帧/连续最大值操作是否正确?你的神经网络一段时间之后的状态和先前的状态是否不同？你看到图像的逻辑级数了吗?如果没有，你可能会遇到一些内存引用问题，这可能在尝试在Python中保存内存中的数据时发生。

​	3.在训练目标一致的情况下，验证网络的权重实际上是固定的。在Python中，很容易使固定网络指向内存中的相同位置，但你的固定目标网络内部权重实际上不是固定的!

​	4.如果你发现你的实现不能在特定的游戏上工作，请在更简单的游戏(如Pong)上测试你的代码。你的实现可能很好，但是你没有足够的前沿技术来学习更难的游戏!探索双重DQN，对抗Q网络，优先体验重播。

## 结论

实现DeepMind的论文是强化学习领域迈向现代技术前沿有价值的一步。如果你主要使用一些更传统的监督学习方法，那么本文的许多想法和技巧乍一看可能是陌生的。深度Q网络毕竟也只是神经网络，许多成熟的学习技术也可以应用到传统应用中。对我来说，最有趣的一点是关于正规化的话题。不知你有没有注意到，我们没有使用drop - out、L2L2或L1L1 等传统的方法来稳定神经网络的训练。这是从原始的《自然》的论文开始并一直在后来的论文中被重复和完善的最令人兴奋和新颖的理论方法之一。当您想进一步改进您的实现时，您应该研究技术的下一个迭代方向:[优先体验重放](https://arxiv.org/abs/1511.05952) 、[双重DQN](https://arxiv.org/abs/1509.06461) 和[对抗Q网络](https://arxiv.org/abs/1511.06581) 。目前的标准(主要)来自于一种允许异步学习( [A3C](https://arxiv.org/abs/1602.01783) )的修改版本。

写于2017年7月26日

你也可能喜欢。