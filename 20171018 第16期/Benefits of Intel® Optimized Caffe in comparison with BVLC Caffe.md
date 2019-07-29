# 相比BVLC Caffe，英特尔®优化Caffe的优势

原文链接：[Benefits of Intel® Optimized Caffe* in comparison with BVLC Caffe*](https://software.intel.com/en-us/articles/comparison-between-intel-optimized-caffe-and-vanilla-caffe-by-intel-vtune-amplifier?from=hackcv&hmsr=hackcv.com&utm_medium=hackcv.com&utm_source=hackcv.com)

### 总览

本文介绍了Berkeley Vision and Learning Center（BVLC）Caffe 和Caffe 的定制版本，英特尔®优化的Caffe 。 我们将解释英特尔®优化Caffe后在英特尔®架构上通过英特尔®VTune™放大器以及Caffe 本身的性能剖析与如何运用

###  

### BVLC Caffe* 和英特尔®优化版Caffe *简介

[Caffe](http://caffe.berkeleyvision.org/)*是由Berkeley Vision 和Learning Center（[BVLC](http://bvlc.eecs/berkeley.edu/)）开发的一种运用广泛的基于机器视觉的深度学习框架。 它是一个开源框架，目前正在发展。 它允许用户在通过'Makefile.config'构建Caffe *之前修改参数，例如用于BLAS，CPU或GPU关注计算的库，CUDA，OpenCV *，MATLAB和Python *。 您可以轻松更改配置文件中的选项，BVLC可为开发人员提供项目网页上的直接说明。

英特尔®优化版Caffe *是针对英特尔架构的英特尔分布式定制Caffe *版本。英特尔®优化Caffe *通过增加英特尔架构优化功能和多节点分布训练和打分，具有主要Caffe *的所有优点。英特尔®优化版Caffe *可以更有效地利用CPU资源。

要详细了解英特尔®优化Caffe *如何更改以优化自身的英特尔体系结构，请参阅此页：https://software.intel.com/en-us/articles/caffe-optimized-for-intel-architecture-applying-modern-code-techniques

在本文中，我们将首先使用Cifar 10示例分析BVLC Caffe *的性能，然后使用相同的示例来分析英特尔®优化Caffe *的性能。性能评估将通过两种不同的方法进行。

测试的平台：Xeon Phi™7210（1.3Ghz，64核心），96GB RAM，CentOS 7.2

1. Caffe *提供了自己的计时选项，例如：

```bash
./build/tools/caffe time \ 
    --model=examples/cifar10/cifar10_full_sigmoid_train_test_bn.prototxt \
    -iterations 1000
```

2. 英特尔®VTune™放大器：英特尔®VTune™放大器是一款功能强大的分析工具，可提供先进的CPU分析功能和现代分析界面。  <https://software.intel.com/en-us/intel-vtune-amplifier-xe>

 

 

### 如何安装BVLC Caffe*

请参考BVLC Caffe项目网页进行安装：<http://caffe.berkeleyvision.org/installation.html>

如果您的系统上安装了英特尔®MKL，那么最好使用MKL作为BLAS库。

在Makefile.config中，选择BLAS:= mkl并指定MKL地址。 （默认设置为BLAS:= atlas）

在我们的测试中，我们将所有配置保留为默认值，但仅限CPU选项。



### 测试样例

在本文中，我们将使用Caffe *包中包含的“Cifar 10”示例作为默认值。

您可以参考BVLC Caffe项目页面以获取有关此例子的详细信息：<http://caffe.berkeleyvision.org/gathered/examples/cifar10.html>

您可以轻松地运行Cifar 10的训练示例，如下所示：

```bash
cd $CAFFE_ROOT
./data/cifar10/get_cifar10.sh
./examples/cifar10/create_cifar10.sh
./examples/cifar10/train_full_sigmoid_bn.sh
```

首先，我们将尝试使用Caffe自己的基准测试方法来获得其性能结果，如下所示：

```bash
./build/tools/caffe time\
    model=examples/cifar10/cifar10_full_sigmoid_train_test_bn.prototxt \
    -iterations 1000
```

结果，我们得到了逐层前向和后向传播时间。 上面的命令测得每个前向和后向传播批量f图像的时间。 最后，它显示了每层1000次迭代和整个计算的每次迭代的平均执行时间。

![img](https://software.intel.com/sites/default/files/managed/c1/9d/Picture1.png)

该测试在Xeon Phi™7210（1.3Ghz，64核）上运行，其中96GB RAM的DDR4与CentOS 7.2一起安装。

上述结果中的数字将在稍后与英特尔®优化Caffe *的结果进行比较。

在此之前，让我们来看看VTune™结果，以便详细观察Caffe *的表现



### VTune 剖析

Intel® VTune™ Amplifier是一款现代处理器性能分析器，能够快速分析”hotspots“并帮助调整目标应用。 您可以从以下链接中找到英特尔®VTune™放大器的详细信息：

Intel® VTune™ Amplifier : <https://software.intel.com/en-us/intel-vtune-amplifier-xe>

我们在本文中使用英特尔®VTune™放大器来查找具有最高总CPU利用时间的功能。 此外，OpenMP线程如何工作。

 

### VTune结果分析

 

![img](https://software.intel.com/sites/default/files/managed/5c/97/Capture1.PNG)

我们在这里看到的是屏幕左侧列出的一些功能，这些功能占用了大部分CPU时间。 它们被称为“hotspots”，可以作为性能优化的目标函数。

在这种情况下，我们将关注'caffe :: im2col_cpu <float>'函数作为优化候选。

'im2col_cpu <float>'是执行直接卷积作为使用高度优化的BLAS库的GEMM操作的步骤之一。 在我们使用BVLC Caffe *训练Cifar 10模型的测试中，此功能占用了最大的CPU资源。

我们来看看这个函数的线程行为。 在VTune™中，您可以选择一个功能并过滤其他工作负载，以仅观察指定功能的工作负载。

![img](https://software.intel.com/sites/default/files/managed/36/a2/Capture2.PNG)

在上面的结果中，我们可以看到函数的CPI（每指令周期数）是0.907，并且该函数仅使用一个单个线程进行整个计算。

英特尔VTune放大器提供的更直观的数据就在这里。

![img](https://software.intel.com/sites/default/files/managed/45/a8/Capture3.PNG)

此“CPU使用率直方图”提供同时运行的CPU数量的数据。 训练过程使用的CPU数量似乎约为25.该平台有64个物理内核和英特尔®超线程技术，因此它拥有256个CPU。 这里的CPU使用率直方图可能意味着该进程没有有效的线程。

但是，我们不能仅仅确定这些结果是“坏的”，因为我们没有设置任何性能标准或期望的性能来进行分类。 我们将在稍后将这些结果与英特尔®优化Caffe *的结果进行比较。

现在让我们转向英特尔®优化Caffe *。

 

### 如何安装 Intel® Optimized Caffe*

安装英特尔®优化Caffe *的基本步骤与BVLC Caffe *相同。

当从Git克隆英特尔®优化的Caffe *时，您可以使用以下替代方法：

```bash
git clone https: //github.com/intel/caffe
```

此外，还需要安装英特尔®MKL才能发挥英特尔®优化Caffe *的最佳性能。

请下载并安装英特尔®MKL。 英特尔免费提供MKL，无需技术支持或获得许可费即可获得一对一的私人支持。 英特尔®优化Caffe *的默认BLAS库设置为MKL。

 Intel® MKL : <https://software.intel.com/en-us/intel-mkl>

下载英特尔®优化Caffe *并安装MKL后，在Makefile.config中，确保选择MKL作为BLAS库并指向BLL_INCLUDE和BLAS_LIB的MKL include和lib文件夹

```bash
BLAS :=mkl

BLAS_INCLUDE := /opt/intel/mkl/include
BLAS_LIB := /opt/intel/mkl/lib/intel64
```

如果在编译英特尔®优化Caffe *期间遇到“libstdc ++”相关错误，请安装“libstdc++ - static”。 例如 ：

```bash
sudo yum install libstdc++- static
```

### 优化因素和曲调

在我们运行和测试示例的性能之前，我们需要更改或调整一些选项以优化性能。

- 使用'mkl'作为BLAS库：在Makefile.config中指定'BLAS:= mkl'并配置MKL的include和lib位置的位置。

- 设置CPU利用率限制：

```bash
echo "100" sudo tee /sys/devices/system/cpu/intel_pstate/min_perf_pct
echo "0" | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
```

 - 将'engine：“MKL2017”放在train_val.prototxt或solver.prototxt文件的顶部，或者使用此选项和caffe工具：-engine“MKL2017”

 - 当前实现使用OpenMP线程。默认情况下，OpenMP线程的数量设置为CPU核心数。每个线程都绑定到一个内核以获得最佳性能结果。但是，可以通过OpenMP环境变量（如KMP_AFFINITY，OMP_NUM_THREADS或GOMP_CPU_AFFINITY）提供正确的配置来使用自己的配置。对于下面的示例运行，已使用'OMP_NUM_THREADS = 64'。

 - 英特尔®优化Caffe *编辑了原始BVLC Caffe *代码的许多部分，以实现与OpenMP *更好的代码并行化。根据在后台运行的其他进程，调整OpenMP *使用的线程数通常很有用。对于Intel Xeon Phi™产品系列单节点，我们建议使用OMP_NUM_THREADS = numer_of_cores-2。

 - 请同时参考：[Intel Recommendation to Achieve the best performance ](https://github.com/intel/caffe/wiki/Recommendations-to-achieve-best-performance)

如果由于OS过于频繁地移动线程而观察到过多的开销，则可以尝试调整OpenMP * affinity环境变量：


```bash
KMP_AFFINITY=compact,granularity=fine
```

 

### 测试样例

对于英特尔®优化Caffe *，我们运行相同的示例，将结果与之前的结果进行比较。

```bash
cd $CAFFE_ROOT
./data/cifar10/get_cifar10.sh
./examples/cifar10/create_cifar10.sh             
```
```bash
./build/tools/caffe time \
    --model=examples/cifar10/cifar10_full_sigmoid_train_test_bn.prototxt \
    -iterations 1000
```


### 对比

以上示例的结果如下：

同样，用于测试的平台是 : Xeon Phi™ 7210 ( 1.3Ghz, 64 Cores ) with 96GB RAM, CentOS 7.2

首先，我们一起来看看BVLC Caffe *和英特尔®优化Caffe *，

![img](https://software.intel.com/sites/default/files/managed/a1/80/Picture1.png)  --> ![img](https://software.intel.com/sites/default/files/managed/ed/6f/Picture2.png)

为了便于比较，请参阅下表。 列出了每层采用的持续时间（以毫秒为单位），在第5列中，我们说明了每层英特尔®优化Caffe *比BVLC Caffe *快多少倍。 除了相对的bn层，您可以观察到显着的性能改进。 Bn代表“批量标准化”，它需要相当简单的计算，具有小的优化潜力。 Bn前向层显示更好的结果，Bn后向层显示比原始结果慢2~3％的结果。 由于线程开销，这里可能会出现更糟糕的性能。 总体而言，在这种情况下，英特尔®优化Caffe *的性能提升了约28倍。


|          | Direction        | BVLC (ms) | Intel (ms) | Performance Benefit (x) |
| -------- | ---------------- | --------- | ---------- | ----------------------- |
| conv1    | Forward          | 40.2966   | 1.65063    | 24.413                  |
| conv1    | Backward         | 54.5911   | 2.24787    | 24.286                  |
| pool1    | Forward          | 162.288   | 1.97146    | 82.319                  |
| pool1    | Backward         | 21.7133   | 0.459767   | 47.227                  |
| bn1      | Forward          | 1.60717   | 0.812487   | 1.978                   |
| bn1      | Backward         | 1.22236   | 1.24449    | 0.982                   |
| Sigmoid1 | Forward          | 132.515   | 2.24764    | 58.957                  |
| Sigmoid1 | Backward         | 17.9085   | 0.262797   | 68.146                  |
| conv2    | Forward          | 125.811   | 3.8915     | 32.330                  |
| conv2    | Backward         | 239.459   | 8.45695    | 28.315                  |
| bn2      | Forward          | 1.58582   | 0.854936   | 1.855                   |
| bn2      | Backward         | 1.2253    | 1.25895    | 0.973                   |
| Sigmoid2 | Forward          | 132.443   | 2.2247     | 59.533                  |
| Sigmoid2 | Backward         | 17.9186   | 0.234701   | 76.347                  |
| pool2    | Forward          | 17.2868   | 0.38456    | 44.952                  |
| pool2    | Backward         | 27.0168   | 0.661755   | 40.826                  |
| conv3    | Forward          | 40.6405   | 1.74722    | 23.260                  |
| conv3    | Backward         | 79.0186   | 4.95822    | 15.937                  |
| bn3      | Forward          | 0.918853  | 0.779927   | 1.178                   |
| bn3      | Backward         | 1.18006   | 1.18185    | 0.998                   |
| Sigmoid3 | Forward          | 66.2918   | 1.1543     | 57.430                  |
| Sigmoid3 | Backward         | 8.98023   | 0.121766   | 73.750                  |
| pool3    | Forward          | 12.5598   | 0.220369   | 56.994                  |
| pool3    | Backward         | 17.3557   | 0.333837   | 51.989                  |
| ipl      | Forward          | 0.301847  | 0.186466   | 1.619                   |
| ipl      | Backward         | 0.301837  | 0.184209   | 1.639                   |
| loss     | Forward          | 0.802242  | 0.641221   | 1.251                   |
| loss     | Backward         | 0.013722  | 0.013825   | 0.993                   |
| Ave.     | Forward          | 735.534   | 21.6799    | 33.927                  |
| Ave.     | Backward         | 488.049   | 21.7214    | 22.469                  |
| Ave.     | Forward-Backward | 1223.86   | 43.636     | 28.047                  |
| Total    |                  | 1223860   | 43636      | 28.047                  |


这种优化可能的原因有很多：

 -  SIMD的代码矢量化
 -  查找hotspot功能，降低功能复杂性和计算量
 -  CPU /系统特定的优化
 -  减少线程移动
 -  高效的OpenMP *利用率

另外，让我们比较一下BVLC Caffe和英特尔®优化Caffe *之间的VTune结果。

我们将简单地研究如何有效地利用im2col_cpu函数。

![img](https://software.intel.com/sites/default/files/managed/fd/1e/Capture2.PNG)

BVLC Caffe *的im2col_cpu函数的CPI为0.907，并且是单线程的。

![img](https://software.intel.com/sites/default/files/managed/4f/19/Capture4.PNG)

对于英特尔®优化Caffe *，im2col_cpu的CPI为2.747，由OMP Workers提供多线程。

这里CPI率增加的原因是矢量化带来了更高的CPI率，因为每条指令的延迟更长，多线程可以在等待其他线程完成工作时spinning。但是，在此示例中，矢量化和多线程的优势超过了延迟和开销，并且毕竟带来了性能改进。

VTune建议CPI率接近2.0在理论上是理想的，对于我们的情况，我们实现了该函数的正确CPI。 Cifar 10示例的训练工作量是为每次迭代处理32 x 32像素图像，因此当这些工作负载分解为多个线程时，它们中的每一个都可能是一个非常小的任务，这可能导致多线程的转换开销。对于较大的图像，我们会看到较短的spinning时间和较小的CPI率。

在这种情况下，整个过程的CPU使用率直方图也显示出更好的线程结果。

![img](https://software.intel.com/sites/default/files/managed/bf/89/Capture3.PNG)

 

![img](https://software.intel.com/sites/default/files/managed/83/0d/Capture5.PNG)

 



### 推荐链接

BVLC Caffe* Project : [http://caffe.berkeleyvision.org/ ](http://caffe.berkeleyvision.org/)

BVLC Caffe* Git : [https://github.com/BVLC/caffe ](http://caffe.berkeleyvision.org/)

 

Intel® Optimized Caffe* Introduction : <https://software.intel.com/en-us/videos/what-is-intel-optimized-caffe>

Intel® Optimized Caffe* Git : <https://github.com/intel/caffe>

Intel® Optimized Caffe* Recommendations for the best performance : [https://github.com/intel/caffe/wiki/Recommendations-to-achieve-best-performance ](https://github.com/intel/caffe/wiki/Recommendations-to-achieve-best-performance)

Intel® Optimized Caffe* Modern Code Techniques : <https://software.intel.com/en-us/articles/caffe-optimized-for-intel-architecture-applying-modern-code-techniques>

 



### 总结

英特尔®优化Caffe *是采用现代代码技术的英特尔架构的定制Caffe *版本。

在英特尔®优化Caffe *中，英特尔利用优化工具和英特尔®性能库，执行标量和串行优化，实现矢量化和并行化。

 

有关编译器优化的更完整信息，请参阅我们的 [Optimization Notice](https://software.intel.com/en-us/articles/optimization-notice#opt-en).