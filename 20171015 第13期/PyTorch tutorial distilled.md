# PyTorch教程

原文链接：[PyTorch tutorial distilled](https://towardsdatascience.com/pytorch-tutorial-distilled-95ce8781a89c?from=hackcv&hmsr=hackcv.com)

## 从 TensorFlow 转向了 PyTorch



![img](https://cdn-images-1.medium.com/max/2000/1*aqNgmfyBIStLrf9k7d9cng.jpeg)

在我第一次开始学习PyTorch时候，过了几天我就放弃了，对我来说理解这个框架的核心概念和TensorFlow比起来太难了。这就是为什么我把它放在了我的“知识书架”上，渐渐的遗忘了它。但是不久之后，PyTorch的新版本的发布了，我决定再尝试一次。过了一会，我意识到这个框架简便易行，让我很开心的使用PyTorch来编程。我会尝试清楚地解释它的核心概念，这样你就会有动力，至少现在试一试，而不是几年或更长时间。我们将介绍一些基本原则和一些高级内容，如学习速率调度程序，自定义层等。

#### 学习资料

首先，你应该了解PyTorch， [文档](http://pytorch.org/docs/master/) 和 [教程](http://pytorch.org/tutorials/)是分开存储的。因为更新的太快了，所有他们可能有部分会不一样，所以请查阅 [源代码](http://pytorch.org/tutorials/)，这就非常明确和直截了当。而且，还有一个很棒的[PyTorch论坛](https://discuss.pytorch.org/),在那里你可以提出任何合适的问题，你可以得到一个相对较快的答案。 对于PyTorch用户来说，这个地方似乎比StackOverflow更受欢迎。

#### PyTorch as NumPy

让我们来讨论PyTorch本身，PyTorch的主要构建块是tensors。确实他和[NumPy ones](https://docs.scipy.org/doc/numpy-dev/user/quickstart.html)很相似。 Tensors支持很多和相同的API，因此有时可以使用PyTorch作为NumPy的代替品。你可能想问为什么要这么做，主要的原因是PyTorch的主要目标是使用GPU，这样您就可以将数据预处理或任何其他需要大量计算的内容转移到机器学习中。很容易就可以转换tensors从NumPy转换为PyTorch，反之亦然。我们用代码来举个例子：

<iframe width="700" height="250" data-src="/media/caf8def11adef8f02d682c26b30f288b?postId=95ce8781a89c" data-media-id="caf8def11adef8f02d682c26b30f288b" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Favatars3.githubusercontent.com%2Fu%2F9900548%3Fv%3D4%26s%3D400&amp;key=a19fcc184b9711e1b4764040d3dc5c07" class="progressiveMedia-iframe js-progressiveMedia-iframe" allowfullscreen="" frameborder="0" src="https://towardsdatascience.com/media/caf8def11adef8f02d682c26b30f288b?postId=95ce8781a89c" style="user-select: text !important; display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 666px;"></iframe

#### 从张量到变量

张量是PyTorch一个很棒的部分. 。但我们想要的主要是建立一些神经网络。什么是反向传播？当然, 我们可以手动实现它, 但原因是什么？值得庆幸的是, 存在自动分化。为了支持它, pytorch 为您 [提供了变量](http://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_autograd.html) 。变量是张量的包装。有了它们, 我们就可以建立我们的计算图, 并在以后自动计算梯度。每个变量实例都有两个属性: `. data`, 其中包含初始张量本身, 而 `. gd` 将包含相应张量的渐变。



<iframe width="700" height="250" data-src="/media/214f557a06e55da09ba5bd2f2740b7cb?postId=95ce8781a89c" data-media-id="214f557a06e55da09ba5bd2f2740b7cb" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Favatars3.githubusercontent.com%2Fu%2F9900548%3Fv%3D4%26s%3D400&amp;key=a19fcc184b9711e1b4764040d3dc5c07" class="progressiveMedia-iframe js-progressiveMedia-iframe" allowfullscreen="" frameborder="0" src="https://towardsdatascience.com/media/214f557a06e55da09ba5bd2f2740b7cb?postId=95ce8781a89c" style="user-select: text !important; display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 578.987px;"></iframe>

您可能会注意到, 我们手动计算并应用了渐变。我们有优化器吗？答案是肯定的！



<iframe width="700" height="250" data-src="/media/1b9b56e531553ae43d9af79942cc1462?postId=95ce8781a89c" data-media-id="1b9b56e531553ae43d9af79942cc1462" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Favatars3.githubusercontent.com%2Fu%2F9900548%3Fv%3D4%26s%3D400&amp;key=a19fcc184b9711e1b4764040d3dc5c07" class="progressiveMedia-iframe js-progressiveMedia-iframe" allowfullscreen="" frameborder="0" src="https://towardsdatascience.com/media/1b9b56e531553ae43d9af79942cc1462?postId=95ce8781a89c" style="user-select: text !important; display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 578.987px;"></iframe>

现在我们所有的变量都会自动更新。但是你应该从最后一个片段得到的要点：我们仍然应该在计算新的渐变之前手动归零。这是PyTorch的核心概念之一。有时为什么我们应该这样做可能不是很明显，但另一方面，我们可以完全控制我们的渐变，我们何时以及如何应用它们。

#### 静态与动态计算图的比较

PyTorch和TensorFlow的下一个主要区别是它们对图形表示的方法。 Tensorflow [使用静态图表](https://www.tensorflow.org/programmers_guide/graphs)，这意味着我们一次又一次地执行该图表后定义它。在PyTorch中，每个前向传递定义了一个新的计算图。一开始，这些方法之间的区别并不那么大。但是，当您想要调试代码或定义一些条件语句时，动态图变得非常少。您可以使用自己喜欢的调试器！比较while循环语句的下两个定义 -  TensorFlow中的第一个定义和PyTorch中的第二个定义：



<iframe width="700" height="250" data-src="/media/2e9029abd7bef658d7519be4e8176351?postId=95ce8781a89c" data-media-id="2e9029abd7bef658d7519be4e8176351" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Favatars3.githubusercontent.com%2Fu%2F9900548%3Fv%3D4%26s%3D400&amp;key=a19fcc184b9711e1b4764040d3dc5c07" class="progressiveMedia-iframe js-progressiveMedia-iframe" allowfullscreen="" frameborder="0" src="https://towardsdatascience.com/media/2e9029abd7bef658d7519be4e8176351?postId=95ce8781a89c" style="user-select: text !important; display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 578.987px;"></iframe>



<iframe width="700" height="250" data-src="/media/ca2584bf9ad710e712c2e227ed13f462?postId=95ce8781a89c" data-media-id="ca2584bf9ad710e712c2e227ed13f462" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Favatars3.githubusercontent.com%2Fu%2F9900548%3Fv%3D4%26s%3D400&amp;key=a19fcc184b9711e1b4764040d3dc5c07" class="progressiveMedia-iframe js-progressiveMedia-iframe" allowfullscreen="" frameborder="0" src="https://towardsdatascience.com/media/ca2584bf9ad710e712c2e227ed13f462?postId=95ce8781a89c" style="user-select: text !important; display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 234px;"></iframe>

It seems to me that second solution much easier than first one. And what do you think about it?

#### Models definition

好的，现在我们看到在PyTorch中构建一些if / else / while复杂语句很容易。但是让我们回到通常的模型。该框架提供了与[Keras](https://keras.io/) 非常相似的开箱即用层构造函数：

> `nn`包定义了一组**模块**，大致相当于神经网络层。模块接收输入变量并计算输出变量，但也可以保持内部状态，例如包含可学习参数的变量。 `nn`包还定义了一组在训练神经网络时常用的有用损失函数。



<iframe width="700" height="250" data-src="/media/e26dadacc9034bc873a236617a196a5f?postId=95ce8781a89c" data-media-id="e26dadacc9034bc873a236617a196a5f" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Favatars3.githubusercontent.com%2Fu%2F9900548%3Fv%3D4%26s%3D400&amp;key=a19fcc184b9711e1b4764040d3dc5c07" class="progressiveMedia-iframe js-progressiveMedia-iframe" allowfullscreen="" frameborder="0" src="https://towardsdatascience.com/media/e26dadacc9034bc873a236617a196a5f?postId=95ce8781a89c" style="user-select: text !important; display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 514px;"></iframe>

另外，如果我们想构建更复杂的模型，我们可以子类提供`nn.Module`类。当然，这两种方法可以相互混合。



<iframe width="700" height="250" data-src="/media/76edc5e2f5e6d498bbab08aa2b21062d?postId=95ce8781a89c" data-media-id="76edc5e2f5e6d498bbab08aa2b21062d" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Favatars3.githubusercontent.com%2Fu%2F9900548%3Fv%3D4%26s%3D400&amp;key=a19fcc184b9711e1b4764040d3dc5c07" class="progressiveMedia-iframe js-progressiveMedia-iframe" allowfullscreen="" frameborder="0" src="https://towardsdatascience.com/media/76edc5e2f5e6d498bbab08aa2b21062d?postId=95ce8781a89c" style="user-select: text !important; display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 450px;"></iframe>

在`__init__`方法中，我们应该定义稍后将使用的所有层。在`forward`方法中，我们应该提出我们想要使用已经定义的层的步骤。像往常一样，向后传递将自动计算。

#### 自定义图层

但是如果我们想用非标准的backprop定义一些自定义模型呢？这是一个例子 -  XNOR网络：



![img](https://cdn-images-1.medium.com/max/1000/1*cjzIFgglAP9xGKg8mlRysQ.png)



我不会深入了解详细信息，更多关于您可能在[入门手册](https://arxiv.org/abs/1603.05279)中阅读的此类网络。与我们的问题相关的是，反向传播应仅适用于小于1且大于-1的权重。在PyTorch中，它[可以非常简单地实现](http://pytorch.org/docs/master/notes/extending.html):

<iframe width="700" height="250" data-src="/media/48bf0fc8fecfe815810a138441674709?postId=95ce8781a89c" data-media-id="48bf0fc8fecfe815810a138441674709" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Favatars3.githubusercontent.com%2Fu%2F9900548%3Fv%3D4%26s%3D400&amp;key=a19fcc184b9711e1b4764040d3dc5c07" class="progressiveMedia-iframe js-progressiveMedia-iframe" allowfullscreen="" frameborder="0" src="https://towardsdatascience.com/media/48bf0fc8fecfe815810a138441674709?postId=95ce8781a89c" style="user-select: text !important; display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 730px;"></iframe>

你可能会看到，我们应该只定义两个方法：一个用于前进，一个用于后向传递。如果我们需要从前向传递中访问一些变量，我们可以将它们存储在`ctx`变量中。注意：在以前的API中，前向/后向方法不是静态的，我们将所需的变量存储为`self.save_for_backward(input)`并通过`input,_ = self.saved_tensors`访问。

#### 用CUDA训练模型

如果之前讨论过如何将一个张量传递给CUDA。但是如果我们想要传递整个模型，可以从模型本身调用`.cuda（）`方法，并将每个输入变量包装到`.cuda（）`中就足够了。在所有计算之后，我们应该使用`.cpu（）`方法返回结果。



<iframe width="700" height="250" data-src="/media/a0754eaf7543b84f3a14bfabf1ada845?postId=95ce8781a89c" data-media-id="a0754eaf7543b84f3a14bfabf1ada845" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Favatars3.githubusercontent.com%2Fu%2F9900548%3Fv%3D4%26s%3D400&amp;key=a19fcc184b9711e1b4764040d3dc5c07" class="progressiveMedia-iframe js-progressiveMedia-iframe" allowfullscreen="" frameborder="0" src="https://towardsdatascience.com/media/a0754eaf7543b84f3a14bfabf1ada845?postId=95ce8781a89c" style="user-select: text !important; display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 450px;"></iframe>

此外，PyTorch支持源代码中的直接设备分配：



<iframe width="700" height="250" data-src="/media/a4d012ac9617d0a72a7ddd7b8f02e1bd?postId=95ce8781a89c" data-media-id="a4d012ac9617d0a72a7ddd7b8f02e1bd" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Favatars3.githubusercontent.com%2Fu%2F9900548%3Fv%3D4%26s%3D400&amp;key=a19fcc184b9711e1b4764040d3dc5c07" class="progressiveMedia-iframe js-progressiveMedia-iframe" allowfullscreen="" frameborder="0" src="https://towardsdatascience.com/media/a4d012ac9617d0a72a7ddd7b8f02e1bd?postId=95ce8781a89c" style="user-select: text !important; display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 406px;"></iframe>

因为有时我们想在没有代码修改的情况下在CPU和GPU上运行相同的模型，我建议使用某种包装器：



<iframe width="700" height="250" data-src="/media/e7a51f45014201aef5f5a02c86ed1460?postId=95ce8781a89c" data-media-id="e7a51f45014201aef5f5a02c86ed1460" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Favatars3.githubusercontent.com%2Fu%2F9900548%3Fv%3D4%26s%3D400&amp;key=a19fcc184b9711e1b4764040d3dc5c07" class="progressiveMedia-iframe js-progressiveMedia-iframe" allowfullscreen="" frameborder="0" src="https://towardsdatascience.com/media/e7a51f45014201aef5f5a02c86ed1460?postId=95ce8781a89c" style="user-select: text !important; display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 514px;"></iframe>

#### 权重初始化

在TensorFlow中，权重初始化主要在张量声明期间进行。 PyTorch提供了另一种方法 - 首先应该声明张量，并且在下一步中应该改变该张量的权重。权重可以初始化为对tensor属性的直接访问，作为对`torch.nn.init`包中的一堆方法的调用。这个决定可能不是很简单，但是当你想用相同的初始化初始化某些类型的所有层时它会变得很有用。



<iframe width="700" height="250" data-src="/media/0152c521ff9c9df1ed546564f8dd7431?postId=95ce8781a89c" data-media-id="0152c521ff9c9df1ed546564f8dd7431" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Favatars3.githubusercontent.com%2Fu%2F9900548%3Fv%3D4%26s%3D400&amp;key=a19fcc184b9711e1b4764040d3dc5c07" class="progressiveMedia-iframe js-progressiveMedia-iframe" allowfullscreen="" frameborder="0" src="https://towardsdatascience.com/media/0152c521ff9c9df1ed546564f8dd7431?postId=95ce8781a89c" style="user-select: text !important; display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 752px;"></iframe>

#### 逆向排除子图

有时，当您想要重新训练模型的某些层或为生产模式做好准备时，您可以为某些图层禁用自动编程机制。为此，[PyTorch提供了两个标志](http://pytorch.org/docs/master/notes/autograd.html)：`requires_grad`和`volatile`。第一个将禁用当前图层的渐变，但子节点仍然可以计算一些。第二个将禁用当前层和所有子节点的autograd。



<iframe width="700" height="250" data-src="/media/dfdac4bb40bb2190da57603515b22f73?postId=95ce8781a89c" data-media-id="dfdac4bb40bb2190da57603515b22f73" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Favatars3.githubusercontent.com%2Fu%2F9900548%3Fv%3D4%26s%3D400&amp;key=a19fcc184b9711e1b4764040d3dc5c07" class="progressiveMedia-iframe js-progressiveMedia-iframe" allowfullscreen="" frameborder="0" src="https://towardsdatascience.com/media/dfdac4bb40bb2190da57603515b22f73?postId=95ce8781a89c" style="user-select: text !important; display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 493px;"></iframe>

#### 训练过程

PyTorch中还存在一些其他的花里胡哨。例如，您可以使用[学习率调度程序](http://pytorch.org/docs/master/optim.html#how-to-adjust-learning-rate)来根据某些规则调整学习率。或者您可以使用简单的训练标志来启用或者禁用批次归一化和丢失。如果你想要为CPU和GPU分别更改随机种子，将会很容易实现。

<iframe width="700" height="250" data-src="/media/a0e831526a0d22e2988647ad3f8ea1fc?postId=95ce8781a89c" data-media-id="a0e831526a0d22e2988647ad3f8ea1fc" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Favatars3.githubusercontent.com%2Fu%2F9900548%3Fv%3D4%26s%3D400&amp;key=a19fcc184b9711e1b4764040d3dc5c07" class="progressiveMedia-iframe js-progressiveMedia-iframe" allowfullscreen="" frameborder="0" src="https://towardsdatascience.com/media/a0e831526a0d22e2988647ad3f8ea1fc?postId=95ce8781a89c" style="user-select: text !important; display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 514px;"></iframe>

此外，您可以打印有关模型的信息，或使用几行代码保存/加载它。如果您的模型使用[OrderedDict](https://docs.python.org/3/library/collections.html)或者基于类的模型字符串表示形式初始化的，那么将包含层的名称。



<iframe width="700" height="250" data-src="/media/af9f29a3d4938c4255a98764ccfdd6c2?postId=95ce8781a89c" data-media-id="af9f29a3d4938c4255a98764ccfdd6c2" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Favatars3.githubusercontent.com%2Fu%2F9900548%3Fv%3D4%26s%3D400&amp;key=a19fcc184b9711e1b4764040d3dc5c07" class="progressiveMedia-iframe js-progressiveMedia-iframe" allowfullscreen="" frameborder="0" src="https://towardsdatascience.com/media/af9f29a3d4938c4255a98764ccfdd6c2?postId=95ce8781a89c" style="user-select: text !important; display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 621.987px;"></iframe>根据PyTorch文档保存模型，使用‘state_dict()’方法更为可取(http://pytorch.org/docs/master/notes/serializ.htm)。

根据PyTorch文档保存模型，使用`state_dict()`方法[更为可取](http://pytorch.org/docs/master/notes/serializ.htm)。

#### 记录

记录训练过程是一个非常重要的部分。不幸的是，PyTorch没有tensorboard这样的工具。因此，您可以使用[Python日志模块](https://docs.python.org/3/library/logging.html)中的常规文本日志，或者尝试一些第三方库:

- [一个用于实验的简单记录器](https://github.com/oval-group/logger)
- [TensorBoard与语言无关的界面](https://github.com/torrvision/crayon)
- [在不触及TensorFlow的情况下记录TensorBoard事件](https://github.com/TeamHG-Memex/tensorboard_logger)
- [pytorch使用tensorboard ](https://github.com/lanpa/tensorboard-pytorch)
- [Facebook可视化库智慧](https://github.com/facebookresearch/visdom)

#### 数据处理

您可能还记得[TensorFlow中提出的数据加载器](https://www.tensorflow.org/api_guides/python/reading_data)，甚至尝试实现其中的一些加载器。对我来说，花了大约4个小时或更多的时间来了解所有管道应该如何工作。



![img](https://cdn-images-1.medium.com/max/1000/1*S00VU2HiEjNZ35zlj2kqfw.gif)

图片来源:TensorFlow docs

最初，我想在这里添加一些代码，但我认为这样的gif足以解释所有事情是如何发生的基本思想。

PyTorch的开发者决定不重新发明轮子。他们只是使用多处理。要创建自己的自定义数据加载器，从' torch.utils.data '继承类就足够了。数据集'和改变一些方法:



<iframe width="700" height="250" data-src="/media/7499f54f158531418c5d8fbc27c01f22?postId=95ce8781a89c" data-media-id="7499f54f158531418c5d8fbc27c01f22" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Favatars3.githubusercontent.com%2Fu%2F9900548%3Fv%3D4%26s%3D400&amp;key=a19fcc184b9711e1b4764040d3dc5c07" class="progressiveMedia-iframe js-progressiveMedia-iframe" allowfullscreen="" frameborder="0" src="https://towardsdatascience.com/media/7499f54f158531418c5d8fbc27c01f22?postId=95ce8781a89c" style="user-select: text !important; display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 1118.99px;"></iframe>

你应该知道的两件事。首先 ， 图像尺寸与TensorFlow不同。它们是[batch_size x channels x height x width]。但是，通过预处理步骤`torchvision.transforms.ToTensor()`，可以在没有您交互的情况下进行此转换。[转化包](http://pytorch.org/docs/master/torchvision/transforms.html)中还有很多有用的工具。

第二个重要的事情是你可以在GPU上使用固定内存。为此，您只需要在`cuda()`调用中添加另外的标志`async = True`，并从DataLoader获取带有标志`pin_memory = True`的固定批次。 [更多相关讨论](http://pytorch.org/docs/master/notes/cuda.html#use-pinned-memory-buffers).

#### 最后的体系结构概述

现在你知道了模型，优化器和很多其他的东西。合并它们的正确方法是什么?我建议将你的模型和所有包装在这样的积木上:

![img](https://cdn-images-1.medium.com/max/1000/1*A-cWYNur2lqDEhUF1_gdCw.png)

这里有一些用于阐述的伪代码:

<iframe width="700" height="250" data-src="/media/528d40b05ef5e7aab5007d10cf57018b?postId=95ce8781a89c" data-media-id="528d40b05ef5e7aab5007d10cf57018b" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Favatars3.githubusercontent.com%2Fu%2F9900548%3Fv%3D4%26s%3D400&amp;key=a19fcc184b9711e1b4764040d3dc5c07" class="progressiveMedia-iframe js-progressiveMedia-iframe" allowfullscreen="" frameborder="0" src="https://towardsdatascience.com/media/528d40b05ef5e7aab5007d10cf57018b?postId=95ce8781a89c" style="user-select: text !important; display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 730px;"></iframe>

#### 总结

我希望通过这篇文章，你能理解PyTorch的要点:

- 它可以作为临时代替Numpy
- 这对于原型设计来说非常快
- 调试和使用条件流很容易
- 有很多现成的好工具

PyTorch是一个快速发展的框架，拥有很棒的社区。我认为今天的尝试很棒！