# NEW OPTIMIZATIONS IMPROVE DEEP LEARNING FRAMEWORKS FOR CPUS

原文链接：[NEW OPTIMIZATIONS IMPROVE DEEP LEARNING FRAMEWORKS FOR CPUS](https://www.nextplatform.com/2017/10/13/new-optimizations-improve-deep-learning-frameworks-cpus/?from=hackcv&hmsr=hackcv.com&utm_medium=hackcv.com&utm_source=hackcv.com)

![brainonachip](https://3s81si1s5ygj3mzby34dq6qf-wpengine.netdna-ssl.com/wp-content/uploads/2015/04/brainonachip.jpg)



Today, most machine learning is done on processors. Some would say that acceleration of learning has to be done on GPUs, but for most users that is not good advice for several reasons. The biggest reason is now the Intel Xeon SP processor, formerly codenamed “Skylake.”

Up until recently, the software for machine learning has been often more optimized for GPUs than anything else. A series of efforts by Intel have changed that – and when coupled with Platinum version of the Intel Xeon SP family, the top performance gap is closer to 2X than it is to 100X. This may stun some, but it is well documented and not all that surprising when we understand the underlying architectures. With such closeness in performance, use of a GPU accelerator is more of a luxury than a necessity – and there are better choices emerging for ‘luxury’ when we really need it.

Make no mistake however, ‘accelerators’ can have an advantage in performance and/or power consumption when machine learning is all we need. I’ll come back to that with “*What if we only do machine learning?*” at the end of this article. Since most of us need more than a “machine learning only” server, I’ll focus on the reality of how Intel Xeon SP Platinum processors remain the best choice for servers, including servers needing to do machine learning as part of their workload.

### WHINE, WHINE, WHINE – WHERE ARE THE BENCHMARKS?

Intel engineers will tell you that frameworks for deep learning have been highly biased to be optimized for GPUs and not CPUs. So, Intel did something about it – and the lack of CPU optimizations in these frameworks has been addressed today by optimization efforts by Intel to add CPU optimizations to frameworks which were already optimized for GPUs.

The results speak for themselves. TensorFlow benchmarks, with CPU optimizations added, see CPU performance gain as much as 72X (see Intel blog titled *TensorFlow Optimizations on Modern Intel Architecture*). Similarly, Caffe benchmarks, with CPU optimization added see CPUs gain as much as 82X (see Intel blog titled [*Benefits of Intel Optimized Caffe in comparison with BVLC Caffe*](https://software.intel.com/en-us/articles/comparison-between-intel-optimized-caffe-and-vanilla-caffe-by-intel-vtune-amplifier)). That just a start. The website for Torch (torch.ch) proclaims “Torch is a scientific computing framework with wide support for machine learning algorithms that puts GPUs first.” Intel offers an alternative branch, which lets us choose to have CPUs first when we choose to use CPUs.  I’ve personally used this repository for my own work, and I know it helps a lot.

Later in this article, I go through the frameworks and libraries one-by-one and supply links where to download, and details on benchmark results thus far.

The most important benchmarks, of course, are your own programs. So, I advise you to compare results when using frameworks and libraries that offer CPU optimizations and GPU optimizations. Thanks to Intel, you can do both now.

This is not obvious unless you know that deep learning frameworks, tools, and libraries exist that are optimized for CPUs. In fact, the most popular frameworks have versions that are well optimized for CPUs, in particular – Intel Xeon SP processors.  Here is a partial run down of key software for accelerating deep learning on Intel Xeon Platinum processor versions enough that the best performance advantage of GPUs is closer to 2X than to 100X.

### DEEP LEARNING FRAMEWORKS WE KNOW AND LOVE

All of these frameworks have been optimized for both Intel Math Kernel Library (Intel MKL) and Intel Advanced Vector Extensions (Intel AVX).

- TensorFlow is a leading deep learning and machine learning framework created by Google. Tensorflow optimizations for processors are available for Linux as a [wheel installable through pip](https://software.intel.com/en-us/articles/intel-optimized-tensorflow-wheel-now-available). Intel performance tests show performance gains of up to 72X for CPUs over the base version of TensorFlow without these performance optimizations. For more information on the optimization work that made this possible, as well as performance data, see the [*blog post titled TensorFlow Optimizations on Modern Intel Architecture*](https://software.intel.com/en-us/articles/tensorflow-optimizations-on-modern-intel-architecture).
- Caffe is one of the most popular community applications for image recognition. Intel has contributed to an optimized fork dedicated to improving Caffe performance when running on CPUs. It is available from <https://github.com/BVLC/caffe/tree/intel>. Some performance tests showing that adding optimizations for CPUs yields as much as 82X – see the blog [*Benefits of Intel Optimized Caffe in comparison with BVLC Caffe*.](https://software.intel.com/en-us/articles/comparison-between-intel-optimized-caffe-and-vanilla-caffe-by-intel-vtune-amplifier)
- Torch is a popular framework for deep learning. There is no reason to use the standard Torch on a CPU without applying CPU optimizations. Use the *Intel Software Optimization for Torch* which is dedicated to improving Torch performance when running on CPU, in particular Intel Xeon Scalable processors. It is available from <https://github.com/intel/torch>. I’ve been personally using this on Intel processors (I use: install.sh icc off mkl noskip) and on Intel Xeon Phi processors (I use: install.sh icc avx512 mkl noskip). The team is very open to feedback, and has proven responsive to questions and feedback I have offered.
- Theano is an open source Python library, popular with machine learning programmers, to help define, optimize, and evaluate mathematical expressions involving multi-dimensional arrays efficiently. CPU optimizations are available that improves performance on CPU devices, in particular Intel Xeon Scalable processors and Intel Xeon Phi, and is available at <https://github.com/intel/theano>.
- Neon is a Python-based deep learning framework designed for ease of use and extensibility on modern deep neural networks and is committed to best performance on all hardware. Neon was created by Nervana, which was acquired by Intel. Learn more about it, including optimizations on all hardware, at <https://www.intelnervana.com/neon/>.

### DEEP LEARNING MATH LIBRARIES

- Python, and its libraries, is perhaps *the* most popular basis for machine learning applications. The accelerated version of Python has gained widespread adoption in the last few year – and is available for download directly, or via Conda, or via yum or apt-get, or Docker images. There is no excuse to be running vanilla un-accelerated Python. Every machine that I develop on has these accelerations for Python installed. Look at <https://software.intel.com/distribution-for-python> for all the information you need to know to use it. There is a nice piece titled *Overcome Python Performance Barriers for Machine Learning* in [Parallel Universe Magazine](https://software.intel.com/intel-parallel-universe-magazine), Issue 26, starting on page 33.
- BigDL is a distributed deep learning library for Apache Spark. With BigDL, users can write their deep learning applications as standard Apache Spark programs, which can directly run on top of existing Apache Spark or Hadoop clusters. Modeled after Torch, BigDL provides comprehensive support for deep learning, including numeric computing (via Tensor) and high level neural networks; in addition, users can load pre-trained Caffe or Torch models into Spark programs using BigDL. Intel has been reported to claim that processing in BigDL is “orders of magnitude faster than out-of-box open source Caffe, Torch, or TensorFlow on a single-node Xeon processor (i.e., comparable with mainstream GPU).” It is available from <https://github.com/intel-analytics/BigDL>. There is also a nice article on BigDL in [Parallel Universe Magazine](https://software.intel.com/intel-parallel-universe-magazine), Issue 28, starting on page 57.
- MXNet is an open-source, deep learning framework available from <https://github.com/apache/incubator-mxnet>.
- Intel MKL-DNN is an open source, performance-enhancing library for accelerating deep learning frameworks on CPUs with information with the [Intel MKL-DNN Overview blog](https://software.intel.com/articles/intel-mkl-dnn-part-1-library-overview-and-installation.).

In addition to the frameworks and libraries noted above, the Intel Data Analytics Acceleration Library (DAAL) is an open source library of optimized algorithmic building blocks for data analysis stages most commonly associated with solving Big Data problems. The library is designed for use popular data platforms including Hadoop, Spark, R, and Matlab. It is available from <https://software.intel.com/intel-daal>. There is also a good article in *Parallel Universe Magazine*, Issue 28, starting on page 26, titled *Solving Real-World Machine Learning Problems with Intel Data Analytics Acceleration Library*.

### WHAT IF WE ONLY DO MACHINE LEARNING?

While Intel Xeon Scalable processors may be the best solution when we justify a server supporting a variety of workloads, what if we want to take a leap and buy a “machine learning only” server or supercomputer?

My best advice “be sure you really know what you need” and be aware that things are really changing in the field. I do not mean to dissuade any one, but it is difficult to guess all the options we will have even a year from now. I have no doubt that the reality is that accelerators for machine learning will shift from GPUs to FPGAs, ASICs, and products with ‘neural’ in their descriptions. The CPU of choice in all these solutions where you have to support a variety of workloads will remain Intel Xeon processors.

Choices for accelerators are getting more diverse. High-core count CPUs (the Intel Xeon Phi processors – in particular the upcoming “Knights Mill” version), and FPGAs (Intel Xeon processors coupled with Intel/Altera FPGAs), offer highly flexible options excellent price/performance and power efficiencies. An Intel Xeon Phi processor-based system can train, or learn an AlexNet image classification system, up to 2.3 times faster than a similarly configured system using Nvidia GPUs. (see *Inside Intel: The Race for Faster Machine Learning*). Intel has shown that the Intel Xeon Phi Processor delivers up to nine times more performance per dollar versus a hosted GPU solution, and up to eight times more performance per watt. Coming soon are more products that are purpose built for AI from Intel Nervana.

It’s an exciting time to be a computer geek, and machine learning is nothing if it is not fun. It is great to see all the options available to build super-fast machines for machine learning.

### FOUNDATION FOR MACHINE LEARNING

The Xeon SP processors, particularly the Platinum processors, offer outstanding performance for machine learning, while giving us more versatility than any other solution. If and when we are ready to add acceleration, Intel Xeon Scalable processors still serve as the core of a versatile system with accelerators – and the choice of what those accelerators can be is growing quickly. Either way, relying on Skylake processors and their excellent support for machine learning gives us the best combination of performance and versatility in one package.

Learn more:

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

*James Reinders is an independent consultant in high performance computing and parallel programming. Reinders was most recently the parallel programming model architect for Intel’s HPC business, and was a key contributor to the design and implementation of the ASCI Red and Tianhe-2A massively parallel supercomputers.*