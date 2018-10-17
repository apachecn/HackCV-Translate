# Benefits of Intel® Optimized Caffe* in comparison with BVLC Caffe*

原文链接：[Benefits of Intel® Optimized Caffe* in comparison with BVLC Caffe*](https://software.intel.com/en-us/articles/comparison-between-intel-optimized-caffe-and-vanilla-caffe-by-intel-vtune-amplifier?from=hackcv&hmsr=hackcv.com&utm_medium=hackcv.com&utm_source=hackcv.com)

### Overview

 This article introduces Berkeley Vision and Learning Center (BVLC) Caffe* and  a custom version of Caffe*, Intel® Optimized Caffe*. We explain why and how Intel® Optimized Caffe* performs efficiently on Intel® architecture via Intel® VTune™ Amplifier and the time profiling option of Caffe* itself.

###  

### Introduction to BVLC Caffe* and Intel® Optimized Caffe*

[Caffe](http://caffe.berkeleyvision.org/)* is a well-known and widely used machine vision based Deep Learning framework developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu/)). It is an open-source framework and is evolving currently. It allows users to control a variety options such as libraries for BLAS, CPU or GPU focused computation, CUDA, OpenCV*, MATLAB and Python* before you build Caffe* through 'Makefile.config'. You can easily change the options in the configuration file and BVLC provides intuitive instructions on their project web page for developers. 

Intel® Optimized Caffe* is an Intel-distributed customized Caffe* version for Intel architecture. Intel® Optimized Caffe* offers all the goodness of main Caffe* with the addition of Intel architecture-optimized functionality and multi-node distributor training and scoring. Intel® Optimized Caffe* makes it possible to more efficiently utilize CPU resources.

To see in detail how Intel® Optimized Caffe* has changed in order to optimize itself to Intel Architectures, please refer this page : <https://software.intel.com/en-us/articles/caffe-optimized-for-intel-architecture-applying-modern-code-techniques>

In this article, we will first profile the performance of BVLC Caffe* with Cifar 10 example and then will profile the performance of Intel® Optimized Caffe* with the same example. Performance profile will be conducted through two different methods.

Tested platform : Xeon Phi™ 7210 ( 1.3Ghz, 64 Cores ) with 96GB RAM, CentOS 7.2

\1. Caffe* provides its own timing option for example : 

| `1`  | `./build/tools/caffe ``time` `\` |
| ---- | -------------------------------- |
|      |                                  |

| `2`  | `    ``--model=examples/cifar10/cifar10_full_sigmoid_train_test_bn.prototxt \` |
| ---- | ------------------------------------------------------------ |
|      |                                                              |

| `3`  | `    ``-iterations 1000` |
| ---- | ------------------------ |
|      |                          |

\2. Intel® VTune™ Amplifier :  Intel® VTune™ Amplifier is a powerful profiling tool that provides advanced CPU profiling features with a modern analysis interface.  <https://software.intel.com/en-us/intel-vtune-amplifier-xe>

 

 

### How to Install BVLC Caffe*

Please refer the BVLC Caffe project web page for installation : <http://caffe.berkeleyvision.org/installation.html>

If you have Intel® MKL installed on your system, it is better using MKL as BLAS library. 

In your Makefile.config , choose BLAS := mkl and specify MKL address. ( The default set is BLAS := atlas )

In our test, we kept all configurations as they are specified as default except the CPU only option. 

 

### Test example

In this article, we will use 'Cifar 10' example included in Caffe* package as default. 

You can refer BVLC Caffe project page for detail information about this exmaple : <http://caffe.berkeleyvision.org/gathered/examples/cifar10.html>

You can simply run the training example of Cifar 10 as the following : 

| `1`  | `cd $CAFFE_ROOT` |
| ---- | ---------------- |
|      |                  |

| `2`  | `./data/cifar10/get_cifar10.sh` |
| ---- | ------------------------------- |
|      |                                 |

| `3`  | `./examples/cifar10/create_cifar10.sh` |
| ---- | -------------------------------------- |
|      |                                        |

| `4`  | `./examples/cifar10/train_full_sigmoid_bn.sh` |
| ---- | --------------------------------------------- |
|      |                                               |

First, we will try the Caffe's own benchmark method to obtain its performance results as the following:

| `1`  | `./build/tools/caffe ``time` `\` |
| ---- | -------------------------------- |
|      |                                  |

| `2`  | `    ``--model=examples/cifar10/cifar10_full_sigmoid_train_test_bn.prototxt \` |
| ---- | ------------------------------------------------------------ |
|      |                                                              |

| `3`  | `    ``-iterations 1000` |
| ---- | ------------------------ |
|      |                          |

as results, we got the layer-by-layer forward and backward propagation time. The command above measure the time each forward and backward pass over a batch f images. At the end it shows the average execution time per iteration for 1,000 iterations per layer and for the entire calculation. 

![img](https://software.intel.com/sites/default/files/managed/c1/9d/Picture1.png)

This test was run on Xeon Phi™ 7210 ( 1.3Ghz, 64 Cores ) with 96GB RAM of DDR4 installed with CentOS 7.2.

The numbers in the above results will be compared later with the results of Intel® Optimized Caffe*. 

Before that, let's take a look at the VTune™ results also to observe the behave of Caffe* in detail. 

 

### VTune Profiling

Intel® VTune™ Amplifier is a modern processor performance profiler that is capable of analyzing top hotspots quickly and helping tuning your target application. You can find the details of Intel® VTune™ Amplifier from the following link :

Intel® VTune™ Amplifier : <https://software.intel.com/en-us/intel-vtune-amplifier-xe>

We used Intel® VTune™ Amplifier in this article to find the function with the highest total CPU utilization time. Also, how OpenMP threads are working. 

 

### VTune result analysis

 

![img](https://software.intel.com/sites/default/files/managed/5c/97/Capture1.PNG)

What we can see here is some functions listed on the left side of the screen which are taking the most of the CPU time. They are called 'hotspots' and can be the target functions for performance optimization. 

In this case, we will focus on 'caffe::im2col_cpu<float>' function as a optimization candidate. 

'im2col_cpu<float>' is one of the steps in performing direct convolution as a GEMM operation for using highly optimized BLAS libraries. This function took the largest CPU resource in our test of training Cifar 10 model using BVLC Caffe*. 

Let's take a look at the threads behaviors of this function. In VTune™, you can choose a function and filter other workloads out to observe only the workloads of the specified function. 

![img](https://software.intel.com/sites/default/files/managed/36/a2/Capture2.PNG)

On the above result, we can see the CPI ( Cycles Per Instruction ) of the function is 0.907 and the function utilizes only one single thread for the entire calculation.

One more intuitive data provided by Intel VTune Amplifier is here. 

![img](https://software.intel.com/sites/default/files/managed/45/a8/Capture3.PNG)

This 'CPU Usage Histogram' provides the data of the numbers of CPUs that were running simultaneously. The number of CPUs the training process utilized appears to be about 25. The platform has 64 physical core with Intel® Hyper-Threading Technology so it has 256 CPUs. The CPU usage histogram here might imply that the process is not efficiently threaded. 

However, we cannot just determine that these results are 'bad' because we did not set any performance standard or desired performance to classify. We will compare these results with the results of Intel® Optimized Caffe* later.

 

Let's move on to Intel® Optimized Caffe* now.

 

### How to Install Intel® Optimized Caffe*

 The basic procedure of installation of  Intel® Optimized Caffe* is the same as BVLC Caffe*. 

When clone  Intel® Optimized Caffe* from Git, you can use this alternative : 

| `1`  | `git clone https:``//github.com/intel/caffe` |
| ---- | -------------------------------------------- |
|      |                                              |

 

Additionally, it is required to install  Intel® MKL to bring out the best performance of  Intel® Optimized Caffe*. 

Please download and install  Intel® MKL. Intel offers MKL for free without technical support or for a license fee to get one-on-one private support. The default BLAS library of  Intel® Optimized Caffe* is set to MKL.

 Intel® MKL : <https://software.intel.com/en-us/intel-mkl>

After downloading Intel® Optimized Caffe* and installing MKL, in your Makefile.config, make sure you choose MKL as your BLAS library and point MKL include and lib folder for BLAS_INCLUDE and BLAS_LIB

| `1`  | `BLAS :=mkl` |
| ---- | ------------ |
|      |              |

| `2`  |      |
| ---- | ---- |
|      |      |

| `3`  | `BLAS_INCLUDE := /opt/intel/mkl/include` |
| ---- | ---------------------------------------- |
|      |                                          |

| `4`  | `BLAS_LIB := /opt/intel/mkl/lib/intel64` |
| ---- | ---------------------------------------- |
|      |                                          |

 

If you encounter 'libstdc++' related error during the compilation of  Intel® Optimized Caffe*, please install 'libstdc++-static'. For example :

| `1`  | `sudo yum install libstdc++-``static` |
| ---- | ------------------------------------- |
|      |                                       |

 

 

 

### Optimization factors and tunes

Before we run and test the performance of examples, there are some options we need to change or adjust to optimize performance.

- Use 'mkl' as BLAS library : Specify 'BLAS := mkl' in Makefile.config and configure the location of your MKL's include and lib location also.

- Set CPU utilization limit : 

  | `1`  | `echo ``"100"` `| sudo tee /sys/devices/``system``/cpu/intel_pstate/min_perf_pct` |
  | ---- | ------------------------------------------------------------ |
  |      |                                                              |

  | `2`  | `echo ``"0"` `| sudo tee /sys/devices/``system``/cpu/intel_pstate/no_turbo` |
  | ---- | ------------------------------------------------------------ |
  |      |                                                              |

- Put 'engine:"MKL2017" ' at the top of your train_val.prototxt or solver.prototxt file or use this option with caffe tool : -engine "MKL2017"

- Current implementation uses OpenMP threads. By default the number of OpenMP threads is set to the number of CPU cores. Each one thread is bound to a single core to achieve best performance results. It is however possible to use own configuration by providing right one through OpenMP environmental variables like KMP_AFFINITY, OMP_NUM_THREADS or GOMP_CPU_AFFINITY. For the example run below , 'OMP_NUM_THREADS = 64' has been used.

- Intel® Optimized Caffe* has edited many parts of original BVLC Caffe* code to achieve better code parallelization with OpenMP*. Depending on other processes running on the background, it is often useful to adjust the number of threads getting utilized by OpenMP*. For Intel Xeon Phi™ product family single-node we recommend to use OMP_NUM_THREADS = numer_of_cores-2.

- Please also refer here : [Intel Recommendation to Achieve the best performance ](https://github.com/intel/caffe/wiki/Recommendations-to-achieve-best-performance)

If you observe too much overhead because of too frequent movement of thread by OS, you can try to adjust OpenMP* affinity environment variable : 

| `1`  | `KMP_AFFINITY=compact,granularity=fine` |
| ---- | --------------------------------------- |
|      |                                         |

 

### Test example

 For Intel® Optimized Caffe* we run the same example to compare the results with the previous results. 

| `1`  | `cd $CAFFE_ROOT` |
| ---- | ---------------- |
|      |                  |

| `2`  | `./data/cifar10/get_cifar10.sh` |
| ---- | ------------------------------- |
|      |                                 |

| `3`  | `./examples/cifar10/create_cifar10.sh` |
| ---- | -------------------------------------- |
|      |                                        |

| `1`  | `./build/tools/caffe ``time` `\` |
| ---- | -------------------------------- |
|      |                                  |

| `2`  | `    ``--model=examples/cifar10/cifar10_full_sigmoid_train_test_bn.prototxt \` |
| ---- | ------------------------------------------------------------ |
|      |                                                              |

| `3`  | `    ``-iterations 1000` |
| ---- | ------------------------ |
|      |                          |

 

### Comparison

 The results with the above example is the following :

Again , the platform used for the test is : Xeon Phi™ 7210 ( 1.3Ghz, 64 Cores ) with 96GB RAM, CentOS 7.2

first, let's look at the BVLC Caffe*'s and Intel® Optimized Caffe* together, 

![img](https://software.intel.com/sites/default/files/managed/a1/80/Picture1.png)  --> ![img](https://software.intel.com/sites/default/files/managed/ed/6f/Picture2.png)

to make it easy to compare, please see the table below. The duration each layer took in milliseconds has been listed, and on the 5th column we stated how many times Intel® Optimized Caffe* is faster than BVLC Caffe* at each layer. You can observe significant performance improvements except for bn layers relatively. Bn stands for "Batch Normalization" which requires fairly simple calculations with small optimization potential. Bn forward layers show better results and Bn backward layers show 2~3% slower results than the original. Worse performance can occur here in result of threading overhead. Overall in total, Intel® Optimized Caffe* achieved about 28 times faster performance in this case. 

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

 

Some of many reasons this optimization was possible are :

- Code vectorization for SIMD 
- Finding hotspot functions and reducing function complexity and the amount of calculations
- CPU / system specific optimizations
- Reducing thread movements
- Efficient OpenMP* utilization

 

Additionally, let's compare the VTune results of this example between BVLC Caffe and Intel® Optimized Caffe*. 

Simply we will looking at how efficiently im2col_cpu function has been utilized. 

![img](https://software.intel.com/sites/default/files/managed/fd/1e/Capture2.PNG)

BVLC Caffe*'s im2col_cpu function had CPI at 0.907 and was single threaded. 

![img](https://software.intel.com/sites/default/files/managed/4f/19/Capture4.PNG)

In case of Intel® Optimized Caffe* , im2col_cpu has its CPI at 2.747 and is multi threaded by OMP Workers. 

The reason why CPI rate increased here is vectorization which brings higher CPI rate because of longer latency for each instruction and multi-threading which can introduce spinning while waitning for other threads to finish their jobs. However, in this example, benefits from vectorization and multi-threading exceed the latency and overhead and bring performance improvements after all.

VTune suggests that CPI rate close to 2.0 is theoretically ideal and for our case, we achieved about the right CPI for the function. The training workload for the Cifar 10 example is to handle 32 x 32 pixel images for each iteration so when those workloads split down to many threads, each of them can be a very small task which may cause transition overhead for multi-threading. With larger images we would see lower spining time and smaller CPI rate.

CPU Usage Histogram for the whole process also shows better threading results in this case. 

![img](https://software.intel.com/sites/default/files/managed/bf/89/Capture3.PNG)

 

![img](https://software.intel.com/sites/default/files/managed/83/0d/Capture5.PNG)

 

###  

### Useful links

BVLC Caffe* Project : [http://caffe.berkeleyvision.org/ ](http://caffe.berkeleyvision.org/)

BVLC Caffe* Git : [https://github.com/BVLC/caffe ](http://caffe.berkeleyvision.org/)

 

Intel® Optimized Caffe* Introduction : <https://software.intel.com/en-us/videos/what-is-intel-optimized-caffe>

Intel® Optimized Caffe* Git : <https://github.com/intel/caffe>

Intel® Optimized Caffe* Recommendations for the best performance : [https://github.com/intel/caffe/wiki/Recommendations-to-achieve-best-performance ](https://github.com/intel/caffe/wiki/Recommendations-to-achieve-best-performance)

Intel® Optimized Caffe* Modern Code Techniques : <https://software.intel.com/en-us/articles/caffe-optimized-for-intel-architecture-applying-modern-code-techniques>

 

###  

### Summary

Intel® Optimized Caffe* is a customized Caffe* version for Intel Architectures with modern code techniques.

In Intel® Optimized Caffe*, Intel leverages optimization tools and Intel® performance libraries, perform scalar and serial optimizations, implements vectorization and parallelization. 

 

 

For more complete information about compiler optimizations, see our [Optimization Notice](https://software.intel.com/en-us/articles/optimization-notice#opt-en).