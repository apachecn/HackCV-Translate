# TensorFlow在现代英特尔体系结构下的优化

原文链接：[TensorFlow* Optimizations on Modern Intel® Architecture](https://software.intel.com/en-us/articles/tensorflow-optimizations-on-modern-intel-architecture?from=hackcv&hmsr=hackcv.com&utm_medium=hackcv.com&utm_source=hackcv.com)

英特尔：**Elmoustapha Ould-Ahmed-Vall, Mahmoud Abuzaina, Md Faijul Amin,  Jayaram Bobba, Roman S Dubtsov, Evarist M Fomenko, Mukesh Gangadhar,  Niranjan Hasabnis, Jing Huang, Deepthi Karkada, Young Jin Kim, Srihari  Makineni, Dmitri Mishura, Karthik Raman, AG Ramesh, Vivek V Rane,  Michael Riera, Dmitry Sergeev, Vamsi Sripathi, Bhavani Subramanian,  Lakshay Tokas, Antonio C Valles** 

谷歌： **Andy Davis, Toby Boyd, Megan Kacholia, Rasmus Larsen, Rajat Monga, Thiru Palanisamy, Vijay Vasudevan, Yao Zhang** 



TensorFlow是一个领先的深度学习和机器学习框架，这对谷歌和英特尔来说，确保它从英特尔的硬件中获得最大的性能是十分重要的。这篇文章介绍了人工智能社区在基于Intel® Xeon® 和Intel® Xeon Phi™ 处理器的平台上对TensorFlow的优化。这些优化是在去年英特尔AI Day的第一天由英特尔的Diane Bryant 和谷歌的Diane Green 提出的，是英特尔和谷歌工程师亲密合作的成果。

我们叙述了在这个优化过程中遇到的各种性能挑战以及所采用的解决方案。我们还报告了一些常见神经网络模型的性能改进。这些优化可以带来数量级的性能提高。例如,我们在Intel® Xeon Phi™  7250处理器的测试显示可以在训练过程中有70倍的更好表现在验证过程中有最高85倍更好的表现。这些基于 Intel® Xeon® E5 v4 (BDW)处理器 and Intel Xeon Phi  7250 处理器的平台为下一代英特尔的产品奠定了基础。值得一提的是，用户希望看到Intel Xeon可扩展处理器在性能上的提高。

在现代cpu上优化深度学习模型的性能与在高性能计算(HPC)中优化其他性能敏感应用程序时所遇到的挑战并无太大不同:

 1.代码重构需要利用现代向量指令。这意味    着确保所有关键原语(如卷积、矩阵乘法和批标准化)都向最新的SIMD指令(用于Intel Xeon处理器的AVX2和用于Intel Xeon Phi处理器的AVX512)矢量化。

 2.最大的性能要求特别注意有效地使用所有可用的内核。同样，这意味着查看给定层或操作中的并行化以及跨层的并行化。

 3.当执行单元需要数据时，所需的数据必须尽可能地可用。这意味着平衡使用预取、缓存阻塞技术和数据格式，以促进空间和时间局部性。

为了满足这些需求，Intel开发了一些优化的深度学习原语，这些原语可以在不同的深度学习框架中使用，以确保我们能够高效地实现公共构建块。除了矩阵乘法和卷积外，这些构建模块还包括:

* 直接成批的卷积

* Inner product

* 池化：最大、最小、均值

* 标准化:跨信道的局部响应标准化(LRN)，批标准化

* 激活函数：线性修正单元(ReLU)

* 数据处理：多维转换、拆分、合并、求和和比例放缩

  更多细节请参考本文在这些英特尔®深层神经网络优化原语数学内核库(Intel®MKL-DNN)


在TensorFlow中，我们实现了英特尔操作优化版本，以确保这些操作可以在任何可能的情况下利用英特尔MKL-DNN原语。当然，对于在英特尔架构上实现可扩展性能，我们还需要一些其他的优化。特别是，由于性能原因，Intel MKL使用的布局与TensorFlow中的默认布局不同。我们需要确保在两种格式之间转换的开销保持在最小。我们还希望确保数据科学家和其他TensorFlow用户在利用这些优化时不必改变现有的神经网络模型。

[![img](https://camo.githubusercontent.com/67b6aaf6a45f07faefab3fbfbe8e6766521bcecb/68747470733a2f2f736f6674776172652e696e74656c2e636f6d2f73697465732f64656661756c742f66696c65732f6d616e616765642f35352f35642f74656e736f72666c6f772d6f7074696d697a6174696f6e732d696d672d30312e706e67)](https://camo.githubusercontent.com/67b6aaf6a45f07faefab3fbfbe8e6766521bcecb/68747470733a2f2f736f6674776172652e696e74656c2e636f6d2f73697465732f64656661756c742f66696c65732f6d616e616765642f35352f35642f74656e736f72666c6f772d6f7074696d697a6174696f6e732d696d672d30312e706e67)

## 图优化

我们介绍一些图优化：

​	1.在CPU上运行时，用英特尔优化版本替换默认的TensorFlow版本。这确保了用户可以运行他们现有的Python程序，并在不更改其神经网络模型的情况下实现性能提升。

​	2.消除不必要的和开销大的数据布局转换。

​	3.将多个操作融合在一起，以在CPU上实现高效的缓存重用。

​	4.处理中间状态，可以使反向传播操作更快。

这些图优化可以提高性能，而不会给TensorFlow程序员带来额外的负担。数据布局优化是性能优化的关键。通常，对于cpu上的某些张量操作，本地的TensorFlow数据格式并不是最有效的数据布局。在这种情况下，我们将数据布局转换操作从TensorFlow的本机格式插入到内部格式，在CPU上执行操作，并将操作输出转换回TensorFlow格式。然而，这些转换引入了性能开销，应该尽量减少。我们的数据布局优化确定了子图，这些子图可以完全使用Intel MKL优化操作执行，并消除了子图中操作的转换。自动插入的转换节点负责子图边界上的数据布局转换。另一个关键的优化是“融合传递”（fusion pass），它可以自动融合可以高效运行的Intel MKL操作。

## 其他方面的优化

我们还调整了一些TensorFlow中的框架组件，以支持以CPU最高性能允许各种深度学习模型。我们使用TensorFlow中的现有池分配器开发了一个定制的池分配器。我们的自定义池分配器确保TensorFlow和Intel MKL共享相同的内存池(使用Intel MKL imalloc功能)，我们不会过早地将内存返回操作系统，从而避免代价高昂的页面丢失和页面清除。此外，我们仔细地调整了多线程库(TensorFlow使用的pthreads和Intel MKL使用的OpenMP)，使它们能够共存，而不是为了争夺CPU资源而相互竞争。

## 性能测试

正如上文，我们的优化使Intel Xeon和Intel Xeon Phi平台的性能得到显著改进。为了说明性能的提高，我们将在我们最普遍的方法(或BKMs)下测试，并对三个常见的ConvNet基准测试数据进行对比和优化。

​	1.以下参数对于Intel Xeon(变量名：Broadwell)和Intel Xeon Phi(变量名：Knights Landing)处理器的性能非常重要，我们建议针对特定的神经网络模型和平台对它们进行调优。我们已经仔细调整了这些参数，以获得在Intel Xeon和Intel Xeon Phi处理器上的convnet-benchmark的最大性能。

​		i.数据格式:我们建议用户可以为其特定的神经网络模型指定NCHW格式，以获得最大的性能。TensorFlow默认NHWC格式对于CPU来说不是最有效的数据布局，它会导致一些额外的转换开销。

​		ii.Inter-op / intra-op:我们还建议数据科学家和用户试验TensorFlow中的intra-op和intra-op参数，以便为每个模型和CPU平台优化设置。这些设置影响同层以及跨层之间的并行性。

​		iii.批尺寸(batch size):批尺寸是另一个重要的参数，它影响可用的并行性，以利用所有的核心，以及工作集大小和内存性能。

​		iv.OMP_NUM_THREADS:最大的性能要求有效地使用所有可用的核心。这个设置对于Intel Xeon Phi处理器的性能特别重要，因为它控制了超线程级别(1到4)。

​		v.矩阵乘法中的转置:对于某些矩阵大小，转置第二个输入矩阵b可以在Matmul层中提供更好的性能(更好的缓存重用)。下面三个模型中使用的所有Matmul操作都是如此。用户应该对其他矩阵大小使用此设置。

​		vi:KMP_BLOCKTIME:用户应该针对在完成并行区域的执行后，每个线程应该等待多少时间(毫秒)尝试各种设置。

<strong>在Intel®Xeon®处理器的示例设置(代号Broadwell - 2套接字- 22内核)</strong>

[![img](https://camo.githubusercontent.com/7f5fe519021aeaf84974be52feadc041c4b432c2/68747470733a2f2f736f6674776172652e696e74656c2e636f6d2f73697465732f64656661756c742f66696c65732f6d616e616765642f35352f35642f74656e736f72666c6f772d6f7074696d697a6174696f6e732d696d672d30322e706e67)](https://camo.githubusercontent.com/7f5fe519021aeaf84974be52feadc041c4b432c2/68747470733a2f2f736f6674776172652e696e74656c2e636f6d2f73697465732f64656661756c742f66696c65732f6d616e616765642f35352f35642f74656e736f72666c6f772d6f7074696d697a6174696f6e732d696d672d30322e706e67)

<strong>Intel®Xeonφ™处理器的示例设置(代号Knights Landing- 68内核)</strong>

[![img](https://camo.githubusercontent.com/d31983dcf4f45a2a32102cdc76719d02452e0ab6/68747470733a2f2f736f6674776172652e696e74656c2e636f6d2f73697465732f64656661756c742f66696c65732f6d616e616765642f35352f35642f74656e736f72666c6f772d6f7074696d697a6174696f6e732d696d672d30332e706e67)](https://camo.githubusercontent.com/d31983dcf4f45a2a32102cdc76719d02452e0ab6/68747470733a2f2f736f6674776172652e696e74656c2e636f6d2f73697465732f64656661756c742f66696c65732f6d616e616765642f35352f35642f74656e736f72666c6f772d6f7074696d697a6174696f6e732d696d672d30332e706e67)

1.在Intel®Xeon®处理器的运行结果(代号Broadwell - 2套接字- 22内核)

[![img](https://camo.githubusercontent.com/6939fb3814a15680379fe44464ec3664b7453229/68747470733a2f2f736f6674776172652e696e74656c2e636f6d2f73697465732f64656661756c742f66696c65732f6d616e616765642f35352f35642f74656e736f72666c6f772d6f7074696d697a6174696f6e732d696d672d30342e706e67)](https://camo.githubusercontent.com/6939fb3814a15680379fe44464ec3664b7453229/68747470733a2f2f736f6674776172652e696e74656c2e636f6d2f73697465732f64656661756c742f66696c65732f6d616e616765642f35352f35642f74656e736f72666c6f772d6f7074696d697a6174696f6e732d696d672d30342e706e67)

2.在Intel®Xeonφ™处理器上的运行结果(代号Knights Landing- 68内核)

[![img](https://camo.githubusercontent.com/7e2f91683296bb12afb8bcdedefdcf80d6a530b2/68747470733a2f2f736f6674776172652e696e74656c2e636f6d2f73697465732f64656661756c742f66696c65732f6d616e616765642f35352f35642f74656e736f72666c6f772d6f7074696d697a6174696f6e732d696d672d30352e706e67)](https://camo.githubusercontent.com/7e2f91683296bb12afb8bcdedefdcf80d6a530b2/68747470733a2f2f736f6674776172652e696e74656c2e636f6d2f73697465732f64656661756c742f66696c65732f6d616e616765642f35352f35642f74656e736f72666c6f772d6f7074696d697a6174696f6e732d696d672d30352e706e67)

3.训练过程中在不同批尺寸下Intel® Xeon® 处理器 (代号 Broadwell) and Intel® Xeon Phi™ 处理器(代号 Knights Landing) 的表现结果

[![img](https://camo.githubusercontent.com/149e1b63dffd5ae62e7f9a30ad2d81caa0bb844a/68747470733a2f2f736f6674776172652e696e74656c2e636f6d2f73697465732f64656661756c742f66696c65732f6d616e616765642f35352f35642f74656e736f72666c6f772d6f7074696d697a6174696f6e732d696d672d30362e706e67)](https://camo.githubusercontent.com/149e1b63dffd5ae62e7f9a30ad2d81caa0bb844a/68747470733a2f2f736f6674776172652e696e74656c2e636f6d2f73697465732f64656661756c742f66696c65732f6d616e616765642f35352f35642f74656e736f72666c6f772d6f7074696d697a6174696f6e732d696d672d30362e706e67)

[![img](https://camo.githubusercontent.com/82254e49c8c9e1988a9764f72595abc6526499d7/68747470733a2f2f736f6674776172652e696e74656c2e636f6d2f73697465732f64656661756c742f66696c65732f6d616e616765642f35352f35642f74656e736f72666c6f772d6f7074696d697a6174696f6e732d696d672d30372e706e67)](https://camo.githubusercontent.com/82254e49c8c9e1988a9764f72595abc6526499d7/68747470733a2f2f736f6674776172652e696e74656c2e636f6d2f73697465732f64656661756c742f66696c65732f6d616e616765642f35352f35642f74656e736f72666c6f772d6f7074696d697a6174696f6e732d696d672d30372e706e67)

[![img](https://camo.githubusercontent.com/630c1e41069e39d33866d1cc76d54109cfdf2a86/68747470733a2f2f736f6674776172652e696e74656c2e636f6d2f73697465732f64656661756c742f66696c65732f6d616e616765642f35352f35642f74656e736f72666c6f772d6f7074696d697a6174696f6e732d696d672d30382e706e67)](https://camo.githubusercontent.com/630c1e41069e39d33866d1cc76d54109cfdf2a86/68747470733a2f2f736f6674776172652e696e74656c2e636f6d2f73697465732f64656661756c742f66696c65732f6d616e616765642f35352f35642f74656e736f72666c6f772d6f7074696d697a6174696f6e732d696d672d30382e706e67)

## 安装带有CPU优化的TensorFlow

您可以使用pip或conda安装预构建的二进制包， 详细步骤于此链接：[Intel Optimized TensorFlow Wheel Now Available](https://software.intel.com/en-us/articles/intel-optimized-tensorflow-wheel-now-available#) ，或者您可以根据以下方向从源代码构建:

​	1.从TensorFlow源码位置运行"./configure"，如果你选择使用英特尔MKL，它将会在tensorflow/third_party/mkl/mklml 下自动下载最新的英特尔机器学习MKL。

​	2.执行以下命令来创建一个pip包，该包可用于安装优化的TensorFlow。

​		添加指向GCC编译器的环境变量：export PATH=/PATH/gcc/bin:$PATH 

​		将LD_LIBRARY_PATH 改为指向新的GLIBC：export LD_LIBRARY_PATH=/PATH/gcc/lib64:$LD_LIBRARY_PATH

​		为了最好的性能在Intel Xeon and Intel Xeon Phi 处理器上构建包：bazel build --config=mkl --copt=”-DEIGEN_USE_VML” -c opt //tensorflow/tools/pip_package: build_pip_package 

​	3.安装已优化的TensorFlow wheel

​		i.bazel-bin/tensorflow/tools/pip_package/build_pip_package ~/path_to_save_wheel pip install --				upgrade --user ~/path_to_save_wheel /wheel_name.whl 

## 系统配置

[![img](https://camo.githubusercontent.com/6509ddfa2fae9b2a8e699706f1a3a473cead56f8/68747470733a2f2f736f6674776172652e696e74656c2e636f6d2f73697465732f64656661756c742f66696c65732f6d616e616765642f35352f35642f74656e736f72666c6f772d6f7074696d697a6174696f6e732d696d672d30392e706e67)](https://camo.githubusercontent.com/6509ddfa2fae9b2a8e699706f1a3a473cead56f8/68747470733a2f2f736f6674776172652e696e74656c2e636f6d2f73697465732f64656661756c742f66696c65732f6d616e616765642f35352f35642f74656e736f72666c6f772d6f7074696d697a6174696f6e732d696d672d30392e706e67)

## 对于AI来说它意味着什么

优化TensorFlow意味着使用这种广泛应用的框架构建的深度学习应用程序现在可以在英特尔处理器上运行得更快，从而提高灵活性、可访问性和可伸缩性。以英特尔Xeon Phi处理器为例，其设计目的是在核心和节点之间以近乎线性的方式进行扩展，从而大大减少训练机器学习模型的时间。随着我们不断提高英特尔处理器的性能，我们可以处理更大、更有挑战性的人工智能工作任务，这样，TensorFlow可以随着未来性能的提升而扩展。

英特尔和谷歌在优化TensorFlow方面的合作，是让开发人员和数据科学家更容易访问人工智能的持续努力的一部分，也是让人工智能应用程序可以在任何设备上运行——从边缘到云事业的一部分。英特尔相信这是创造下一代解决商业、科学、工程、医学和社会中最紧迫的问题的人工智能算法和模型的关键。

这种合作已经在领先的Intel Xeon和Intel Xeon Phi处理器平台上带来了显著的性能提升。这些改进现在可以通过谷歌的TensorFlow GitHub库轻松获得。我们希望人工智能社区尝试这些优化，并期待基于它们的反馈和贡献。