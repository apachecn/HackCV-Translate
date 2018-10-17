# PyTorch与TensorFlow: 一个月总结

原文链接：[PyTorch vs. TensorFlow: 1 month summary](https://towardsdatascience.com/pytorch-vs-tensorflow-1-month-summary-35d138590f9?from=hackcv&hmsr=hackcv.com)

*在使用PyTorch一个月后，比较PyTorch与TensorFlow。*

![img](https://cdn-images-1.medium.com/max/800/1*OFhhTYPS42zjyabSVdBQmQ.jpeg)

我一直是TensorFlow用户，将其更好的用于深度学习工作。但是，当我加入[NVIDIA时](https://medium.com/@NvidiaAI)，我们决定改用PyTorch - 做一个测试一样。以下是我的经历。

### 安装

安装非常简单直接。PyTorch可以通过PIP安装，也可以编译源代码。PyTorch还提供Docker镜像，可用作您自己项目的基本镜像。

PyTorch没有指定的CPU和GPU版本，就像TensorFlow一样。虽然这让安装过程更容易，但如果您想同时支持CPU和GPU使用，它会需要更多代码。

值得一提的是，PyTorch尚未提供正式的Windows发行版。Windows有非官方端口，但没有PyTorch的支持。

### 使用

PyTorch提供了一个非常Pythonic的API。这与TensorFlow非常不同，TensorFlow是定义了所有的Tensors和Graph，然后在会话（session）中运行它。

在我看来，这会增加代码量但是增加了可读性。PyTorch图必须在继承自PyTorch `nn.Module`的类中定义。`forward()`运行Graph时会调用一个函数。使用这种“约定优于配置”方法，Graph的位置始终是已知的，并且在其余代码中并未定义变量。

这种“新”方法需要一些时间来习惯，但我认为如果您之前在深度学习之外运用过Python，那将非常直观。

基于一些评论，与TensorFlow相比，PyTorch在许多模型上也表现出更好的性能

### 文档

文档大部分都是完整的。我没有找不到的函数或模块的定义。与TensorFlow相反，所有函数都有一个页面，PyTorch每个模块都只使用一个页面。如果去Google寻找函数，反而会更加困难。

### 社区

显然，PyTorch的社区并不像TensorFlow那么大。然而，许多人喜欢在业余时间使用PyTorch，即使他们使用TensorFlow进行工作。我认为一旦PyTorch完成公测（Beta），这种情况就会发生变化。

目前，在PyTorch中找到精通它的人仍然有点困难。

但社区还是足够大的，官方论坛中的问题通常可以很快地收到回复，因此很多很棒网络模型的示例实现都使用了PyTorch。

### 工具和助手

即使PyTorch提供了大量的工具，也缺少一些非常有用的工具。缺少最有用的工具之一是TensorFlow的TensorBoard。这使得可视化（vizualization）有点困难。

还有一些非常常见的使用助手丢失。这需要自己编写比TensorFlow更多的代码。

### 结论

PyTorch是TensorFlow的一个很棒的替代品。由于PyTorch仍然处于测试阶段，我希望对可用性，文档和性能进行一些更改和改进。

PyTorch非常pythonic，使用起来很舒服。它有一个很好的社区和文档。它也被认为比TensorFlow快一点。

但是，与TensorFlow相比，社区仍然相当小，并且缺少一些有用的工具，如TensorBoard。