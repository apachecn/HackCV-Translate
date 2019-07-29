# 新的优化改善了CPUS的深度学习框架

原文链接：[NEW OPTIMIZATIONS IMPROVE DEEP LEARNING FRAMEWORKS FOR CPUS](https://www.nextplatform.com/2017/10/13/new-optimizations-improve-deep-learning-frameworks-cpus/?from=hackcv&hmsr=hackcv.com&utm_medium=hackcv.com&utm_source=hackcv.com)

![brainonachip](https://3s81si1s5ygj3mzby34dq6qf-wpengine.netdna-ssl.com/wp-content/uploads/2015/04/brainonachip.jpg)



今天，大多数机器学习是在处理器上完成的。有人会说加速学习必须在GPU上进行，但对于大多数用户而言，这并不是好的建议。最大的原因是现在的英特尔至强SP处理器，以前代号为“Skylake”。

直到最近，用于机器学习的软件通常比GPU更优化。英特尔的一系列努力改变了这一点 - 当与英特尔至强SP系列的白金版本配合使用时，最高性能差距接近2倍，而不是100倍。这可能会使一些人感到震惊，但是当我们理解底层架构时，它已被充分证明并且不令人惊讶。由于性能如此接近，使用GPU加速器是奢侈品而不是必需品 - 当我们真正需要时，'奢侈品'有更好的选择。

毫无疑问，当我们需要机器学习时，“加速器”可以在性能和（或）功耗方面具有优势。我将在本文末尾附上“*如果我们只进行机器学习怎么办？*”。由于我们大多数人不仅需要“机器学习”服务器，我还将关注英特尔至强SP铂金处理器如何成为服务器的最佳选择，包括需要将机器学习作为其工作量的一部分的服务器。

### 抱怨三连 – 基准在哪?

英特尔工程师将告诉您，深度学习的框架已经高度偏向于针对GPU而非CPU进行优化。 因此，英特尔做了一些事情 - 而今天，英特尔通过努力将CPU优化添加到已经针对GPU优化的框架，解决了这些框架中缺乏CPU优化的问题。

结果不言自明。 添加了CPU优化的TensorFlow基准测试，CPU性能提升高达72倍（参见英特尔博客，标题为*TensorFlow Optimizations on Modern Intel Architecture*）。 同样，Caffe基准测试，随着CPU优化的增加，CPU的增益高达82倍（参见英特尔博客标题 [*Benefits of Intel Optimized Caffe in comparison with BVLC Caffe*](https://software.intel.com/en-us/articles/comparison-between-intel-optimized-caffe-and-vanilla-caffe-by-intel-vtune-amplifier))。那只是一个开始。 Torch（torch.ch）的网站宣称“Torch是一个科学计算框架，广泛支持将GPU放在首位的机器学习算法。”英特尔提供了另一个分支，当我们选择使用CPU时，我们可以选择首先使用CPU 。我亲自将这个存储库用于我自己的工作，我知道它有很多帮助。

在本文的后面，我将逐一介绍框架和库，并提供下载的链接，以及迄今为止基准测试结果的详细信息。

当然，最重要的基准是你自己的程序。因此，我建议您在使用提供CPU优化和GPU优化的框架和库时比较结果。感谢英特尔，您现在可以做到这两点。

除非您知道存在针对CPU优化的深度学习框架，工具和库，否则这一点并不明显。实际上，最流行的框架具有针对CPU优化的版本，特别是 - 英特尔至强SP处理器。以下是关键软件的部分版本，用于加速英特尔至强铂金处理器版本的深度学习，足以使GPU的最佳性能优势接近2倍而不是100倍。

### 我们深知和喜爱的深入学习框架

所有这些框架都针对英特尔数学核心函数库（英特尔MKL）和英特尔高级矢量扩展（英特尔AVX）进行了优化。

 -  TensorFlow是由Google创建的领先的深度学习和机器学习框架。在处理器方面，Tensorflow进行了优化以适用于Linux作为[可通过pip安装的wheel](https://software.intel.com/en-us/articles/intel-optimized-tensorflow-wheel-now-available)。英特尔性能测试表明，与没有这些性能优化的基本版TensorFlow相比，CPU的性能提升高达72倍。有关实现此功能的优化工作以及性能数据的更多信息，请参阅[*博客文章标题为现代英特尔架构上的TensorFlow优化*](https://software.intel.com/en-us/articles/tensorflow-optimizations-on-modern-intel-architecture).
 -  Caffe是最受欢迎的图像识别社区应用程序之一。英特尔为优化的分支做出了贡献，该分支致力于在CPU上运行时提高Caffe性能。它可以从<https://github.com/BVLC/caffe/tree/intel>获得。一些性能测试表明，为CPU添加优化产生的效果高达82倍 - 请参阅博客[*与BVLC Caffe *相比，英特尔优化Caffe的优势](https://software.intel.com/en-us/articles/comparison-between-intel-optimized-caffe-and-vanilla-caffe-by-intel-vtune-amplifier)。
 - Torch是深度学习的流行框架。没有应用CPU优化，没有理由在CPU上使用标准Torch。使用*Intel Software for Torch*，专门用于在CPU上运行时提高Torch性能，特别是Intel Xeon Scalable处理器。它可以从<https://github.com/intel/torch>下载。我自己在英特尔处理器上使用它（我使用：install.sh icc off mkl noskip）和英特尔至强Phi处理器（我使用：install.sh icc avx512 mkl noskip）。团队对反馈非常开放，并且已经证明对我提供的问题和反馈有所回应。
 -  Theano是一个开源的Python库，深受机器学习程序员的欢迎，可以帮助定义，优化和评估涉及多维数组的数学表达式。 CPU优化可用于提高CPU设备（尤其是Intel Xeon Scalable处理器和Intel Xeon Phi）的性能，可通过<https://github.com/intel/theano>获得。
 -  Neon是一个基于Python的深度学习框架，旨在实现现代深度神经网络的易用性和可扩展性，并致力于在所有硬件上实现最佳性能。 Neon由Nervana创建，被英特尔收购。在<https://www.intelnervana.com/neon/>上了解有关它的更多信息，包括对所有硬件的优化。

### 深度学习数学库

 -  Python及其库可能是机器学习应用程序最常用的基础。 Python的加速版本在过去几年中得到了广泛采用 - 可直接下载，或通过Conda，或通过yum或apt-get或Docker镜像下载。没有理由不提高Python的性能。我开发的每台机器都安装了Python的这些加速功能。查看<https://software.intel.com/distribution-for-python>，了解使用它需要了解的所有信息。在[Parallel Universe Magazine](https://software.intel.com/intel-parallel-universe-magazine)，第26期，从第33页开始，有一篇名为*克服机器学习的Python性能障碍的好文章*。
 -  BigDL是Apache Spark的分布式深度学习库。使用BigDL，用户可以将他们的深度学习应用程序编写为标准的Apache Spark程序，它可以直接在现有的Apache Spark或Hadoop集群上运行。以Torch为模型，BigDL为深度学习提供全面支持，包括数值计算（通过Tensor）和高级神经网络;此外，用户可以使用BigDL将预先训练的Caffe或Torch模型加载到Spark程序中。据报道，英特尔声称BigDL处理在单节点Xeon处理器上比开箱即用的开源Caffe，Torch或TensorFlow快几个数量级（即与主流GPU相当）。“它可用来自<https://github.com/intel-analytics/BigDL>。在[Parallel Universe Magazine](https://software.intel.com/intel-parallel-universe-magazine)，第28期，从第57页开始，还有一篇关于BigDL的好文章。
 -  MXNet是一个开源的深度学习框架，可从<https://github.com/apache/incubator-mxnet>获得。
 - 英特尔MKL-DNN是一个开源的，性能增强的库，用于加速CPU上的深度学习框架，其中包含[英特尔MKL-DNN概述博客](https://software.intel.com/articles/intel-mkl-dnn-part-1-library-overview-and-installation.)。

除了上面提到的框架和库之外，英特尔数据分析加速库（DAAL）是一个开源的优化算法构建模块库，用于最常与解决大数据问题相关的数据分析阶段。该库设计用于流行的数据平台，包括Hadoop，Spark，R和Matlab。它可以从<https://software.intel.com/intel-daal>获得。在* Parallel Universe Magazine *，第28期，从第26页开始，还有一篇很好的文章，名为*解决英特尔数据分析加速库*的实际机器学习问题。

### 如果我们只做机器学习怎么办？

虽然英特尔至强可扩展处理器可能是我们证明服务器支持各种工作负载的最佳解决方案，但如果我们想要实现跨越并购买“仅限机器学习”的服务器或超级计算机呢？

我最好的建议是“确保你真的知道你需要什么”，并注意事情在这个领域真的发生了变化。我并不是要劝阻任何一个人，但很难猜到我们即将在一年之后所拥有的所有选择。我毫不怀疑现实情况是机器学习的加速器将从GPU转向FPGA，ASIC和产品中的“neural”。在您必须支持各种工作负载的所有这些解决方案中，所选择的CPU仍将是Intel Xeon处理器。

加速器的选择越来越多样化。高核计数CPU（Intel Xeon Phi处理器 - 特别是即将推出的“Knights Mill”版本）和FPGA（Intel Xeon处理器与Intel/Altera FPGA结合）提供高度灵活的选项，具有出色的性价比和功效。基于英特尔至强处理器的系统可以训练或学习AlexNet图像分类系统，速度比使用Nvidia GPU的类似配置系统快2.3倍。 （参见*英特尔内部：更快的机器学习竞赛*）。英特尔表明，与托管GPU解决方案相比，英特尔至强融核处理器的每美元性能提高了9倍，每瓦性能提高了8倍。即将推出更多专为英特尔Nervana设计的AI产品。

成为一名电脑爱好者是一个激动人心的时刻，如果机器学习不好玩，那么机器学习就不算什么了。很高兴看到可用于构建用于机器学习的超快速机器的所有选择。

### 机器学习基础

Xeon SP处理器，特别是Platinum处理器，为机器学习提供了出色的性能，同时为我们提供了比任何其他解决方案更多的功能。 如果我们准备好增加加速度，英特尔至强可扩展处理器仍然是带加速器的多功能系统的核心 - 并且这些加速器的选择正在快速增长。 无论哪种方式，依靠Skylake处理器及其对机器学习的出色支持，我们在一个包装中为我们提供了性能和多功能性的最佳组合。

学到更多：

- [*Inside Intel: The Race for Faster Machine Learning*](https://www.intel.com/content/www/us/en/analytics/machine-learning/the-race-for-faster-machine-learning.html)
- Intel’s official site for information on deep learning frameworks and optimization available to ensure top CPU performance: <https://www.intelnervana.com/framework-optimizations/>
- Video: [How Intel is bringing AI & Machine Learning to the People](https://insidehpc.com/2017/06/intel-bringing-ai-machine-learning-people/) – interview with Pradeep Dubey, Intel Fellow and Director of the Intel’s Parallel Computing Lab.
- Intel Nervana Graph – An open source library for developing frameworks that can efficiently run deep learning computations on a variety of compute platforms – <https://www.intelnervana.com/intel-nervana-graph>
- Intel’s *Deep Learning Insights* website – offering a large number of tutorials on Deep Learning.
- Accelerated Python: <https://software.intel.com/distribution-for-python>; an article titled *Overcome Python Performance Barriers for Machine Learning* in [Parallel Universe Magazine](https://software.intel.com/intel-parallel-universe-magazine), Issue 26, starting on page 33.
- *Solving Real-World Machine Learning Problems with Intel Data Analytics Acceleration Library*, [Parallel Universe Magazine](https://software.intel.com/intel-parallel-universe-magazine), Issue 28, page 26.
- *BigDL: Optimized Deep Learning on Apache Spark*, [Parallel Universe Magazine](https://software.intel.com/intel-parallel-universe-magazine), Issue 28, page 57.
- Caffe – CPU optimized – <https://github.com/BVLC/caffe/tree/intel>; and blog post titled [*Benefits of Intel Optimized Caffe in comparison with BVLC Caffe*.](https://software.intel.com/en-us/articles/comparison-between-intel-optimized-caffe-and-vanilla-caffe-by-intel-vtune-amplifier)
- DAAL – CPU optimized – <https://software.intel.com/intel-daal>; article [Parallel Universe Magazine](https://software.intel.com/intel-parallel-universe-magazine), Issue 28, starting page 26, titled *Solving Real-World Machine Learning Problems with Intel Data Analytics Acceleration Library*.
- Torch – CPU optimized – <https://github.com/intel/torch>
- TensorFlow – CPU optimized –[wheel installable through pip](https://software.intel.com/en-us/articles/intel-optimized-tensorflow-wheel-now-available); and blog post titled [*TensorFlow Optimizations on Modern Intel Architecture*.](https://software.intel.com/en-us/articles/tensorflow-optimizations-on-modern-intel-architecture)
- Theano – CPU optimized – <https://github.com/intel/theano>.
- MKL-DNN – CPU optimized – <https://software.intel.com/en-us/articles/intel-mkl-dnn-part-1-library-overview-and-installation>; blog <https://software.intel.com/articles/intel-mkl-dnn-part-1-library-overview-and-installation>.
- MXNet is an open-source, deep learning framework available from <https://github.com/apache/incubator-mxnet>.
- Neon – all platform optimized – <https://www.intelnervana.com/neon/>

*James Reinders是高性能计算和并行编程的独立顾问。 Reinders最近是英特尔HPC业务的并行编程模型架构师，并且是ASCI Red和Tianhe-2A大规模并行超级计算机设计和实现的关键贡献者。*