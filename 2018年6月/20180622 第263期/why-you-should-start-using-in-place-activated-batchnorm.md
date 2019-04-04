# Why you should start using In-Place Activated BatchNorm

In-Place Activated BatchNorm (InPlace-ABN) is memory efficient replacement for BatchNorm + Activation step. BN + Relu + Conv2d is an integral part of basic building blocks of modern network architectures (UNet, LinkNet, ResNet, etc.).

**TL;DR**: InPlace-ABN can save up to **50%** of GPU memory required to train deep neural network models.

All credits of InPlace-ABN goes to Mapillary: [https://arxiv.org/pdf/1712.02616.pdf](https://arxiv.org/pdf/1712.02616.pdf), [https://github.com/mapillary/inplace_abn](https://github.com/mapillary/inplace_abn). In this post I want to encourage you to try it and get immediate benefit of reduced memory footprint when training your model.

### What is InPlace-ABN?

Inplace-ABN is novel approach to reduce the memory required for training deep networks. I’m not going to dive deep into implementation details (that’s probably a topic for dedicated post on Medium;I encourage you to read [original article ](https://arxiv.org/pdf/1712.02616.pdf)which explains approach in detail). Very briefly, this method proposes a way to reduce amount of memory required to do back-propagation from activation and batch-norm layers up to 50%.

### How to use InPlace-ABN?

First of all, you need [PyTorch 0.4](https://pytorch.org/) and CUDA 9.0+ libraries installed. As of time of writing, I did not encounter implementation of InPlace-ABN for TF or MXNet. Please drop a comment below if you find it.

Linux users, as first-class citizens, can just type:



If you have all the dependencies this will compile and install python bindings to CUDA-optimized and CPU implementation for forward and backward routines to your active python environment. Unfortunately, Windows users needs to perform additional steps to build and install this package, which I will describe in next chapter.

Let’s see how we can adopt this module for U-Net architecture. A basic building block of U-Net is a so-called “double convolution” module which is Conv2d+BN+Relu repeated twice:



The same block with InplaceABN would look very similar:



And that’s it! Training time remains pretty much same (authors reports of ~2% overhead) and accuracy is claimed to not worse than classic BN+Activate approach:

> We observe consistent speed advantages in favor of our method when comparing against CHECKPOINTING, with the actual percentage difference depending on block’s metaparameters. As we can see, INPLACE-ABN induces computation time increase between 0.8 − 2% over STANDARD while CHECKPOINTING is almost doubling our overheads. (Source: https://arxiv.org/pdf/1712.02616.pdf)

But the memory footprint reduced drastically:

**Vanilla U-Net on 1080Ti (11GB)**

* BN+Relu: Batch size 3 of 1024x1024x3

* InplaceABN: Batch size 4 of 1024x1024x3

That is 30% less memory for free. I trained both U-Net and “U-Net with ABN” on [Inria Aerial Image Labeling Dataset](https://project.inria.fr/aerialimagelabeling/) to see if there is any difference in their performance:

As you may see, U-Net with InplaceABN performs a bit better according to validation loss. Convergence trend of both models are very similar which indicates InplaceABN does not introduce instability or gradient explosions and can be safely used as drop-in replacement of classic BatchNorm+ReLU.

![](https://cdn-images-1.medium.com/max/1600/1*ZzAifvcVkn4EY2UDWvcsLw.png)

Inplace ABN offers couple activation functions to your choice:

* Leaky ReLU

* ELU

* None

As you may notice, **there is no ReLU support**. That was done on purpose, since proposed method requires that activation function to be revertible (e.g having activation value one can revert input signal).

### Building on Windows

Windows has never been a top-priority platform for deep learning frameworks. Even PyTorch added official Windows support in 0.4. So it’s very common to encounter pitfalls during building libraries like this. Fortunately I already went through this minefield and going to share step-by-step guide:

#### Prerequsities:

* Visual Studio 2017

* NVIDIA CUDA 9.2

* Pytorch 0.4

* Text editor of your choice

Unfortunately I was not able to compile this library with VS 2015 and CUDA 9.1. So VS 2017 and CUDA 9.2 are hard requirements here. Before we start building something, we have to patch CUDA headers ;)

Open **host_config.h** which can be found at **“C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.2\include\crt”** and change line 131 to look as below:

The reason of this change is dictated by the fact that Microsoft C compiler version changes too fast for CUDA authors. In particular, _MSC_VER macro has version that is unknown for CUDA. After making this fix we are able to build InplaceABN:

#### Steps

1. Clone [https://github.com/mapillary/inplace_abn](https://github.com/mapillary/inplace_abn)

2. Open “x64 Native Tools Command Prompt for VS 2017”. This starts a new cmd interpreteur within VS environment.

3. Navigate to directory where you cloned inplace_abn repository

4. python setup.py install

Last step is the most critical one, it can fails for many reasons (wrong VS version, wrong CUDA version, wrong Pytorch version), so read error messages carefully.

We are almost here! As you may know, on Windows Python extensions are regular DLL’s. Our inplace_abn package depends on Pytorch.dll and ATen.dll, so if you try to import it, it will fail with cryptic “Dll load failed” message.

5. Add location of Pytorch.dll to %PATH%

Fortunately this is solvable, yet with a bit “hacky” solution. If you know how to make it in more elegant way — please post your solution in comments. To fix “Dll load failed”, we have to modify our PATH environment variable and add location of Pytorch.dll and ATen.dll. For Anaconda, it can be found at “**c:\Anaconda3\envs\kaggle\Lib\site-packages\torch\lib”**:

![](https://cdn-images-1.medium.com/max/1600/1*uvuLvl6ZBQw0_qfi19SPrA.png)

That’s it! Now all runtime dependencies should be resolved.

### Ok, what’s next?

Start hacking! With this module one can fit modern architectures even on memory-challenged GPUs like 1050 or have more batches on 1080’s or utilize efficient inter-GPU synchronization in ABN block.

I want to take a chance and highlight few projects in Github that already using InPlaceABN. First one is my personal project for studying & evaluating networks models on binary segmentation problem:

1. [https://github.com/BloodAxe/segmentation-networks-benchmark](https://github.com/BloodAxe/segmentation-networks-benchmark/blob/master/lib/models/unet_abn.py)

Second project has network definition and weights for second place solution in [CVPR 2018 DeepGlobe Building Extraction Challenge](https://competitions.codalab.org/competitions/18544) which is also using InplaceABN:

[2. https://github.com/ternaus/TernausNetV2](https://github.com/ternaus/TernausNetV2)

### Conclusion

Inplace-ABN researched by Mapillary offers a memory-efficient module for BN+Activation that can be used a drop-in replacement in existing neural network models and reduce memory footprint up to 50%.

* Paper: [https://arxiv.org/pdf/1712.02616.pdf](https://arxiv.org/pdf/1712.02616.pdf)

* Official implementation: [https://github.com/mapillary/inplace_abn](https://github.com/mapillary/inplace_abn)

Author would like to thank Open Data Science community (ods.ai) for many valuable discussions and educational help in the growing field of machine/deep learning.

