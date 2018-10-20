# Google Says Machine Learning Chips Make AI Faster and More Efficient

原文链接：[Google Says Machine Learning Chips Make AI Faster and More Efficient](https://singularityhub.com/2017/04/23/google-says-machine-learning-chips-make-ai-faster-and-more-efficient/?from=hackcv&hmsr=hackcv.com&utm_medium=hackcv.com&utm_source=hackcv.com)

Google recently bared the inner workings of its dedicated machine learning chip, the TPU, marking the latest skirmish in the arms race for AI hardware supremacy.

Shorthand for Tensor Processing Unit, the chip has been tailored for use with Google’s open-source machine learning library TensorFlow, and has been in use in Google’s data centers since 2015. But earlier this month the company [finally provided performance figures](https://arxiv.org/abs/1704.04760) for the device.

The company says the current generation of TPUs are designed for inference — using an already trained neural network to carry out some kind of function, like recognizing voice commands through Google Now. On those tasks, the firm says the TPU is 15 to 30 times faster than contemporary GPUs and CPUs, and equally important, they are 30 to 80 times more power-efficient.

For context, CPUs, or central processing units, are the processors that have been at the heart of most computers since the 1960s. But they are not well-suited to the incredibly high computational requirements of modern machine learning approaches, in particular deep learning.

In the late 2000s, researchers discovered that graphics cards were better suited for the highly parallel nature of these tasks, and GPUs, or graphics processing units, became the de facto technology for implementing neural networks. But as Google’s use of machine learning continued to expand, they wanted something custom built for their needs.

“The need for TPUs really emerged about six years ago, when we started using computationally expensive deep learning models in more and more places throughout our products. The computational expense of using these models had us worried,” lead engineer Norm Jouppi [writes in a blog post](https://cloudplatform.googleblog.com/2017/04/quantifying-the-performance-of-the-TPU-our-first-machine-learning-chip.html).

“If we considered a scenario where people use Google voice search for just three minutes a day and we ran deep neural nets for our speech recognition system on the processing units we were using, we would have had to double the number of Google data centers!”

Nvidia, for its part, says the comparison isn’t entirely fair. Google compared its TPU against a server-class Intel Haswell CPU and an Nvidia K80 GPU, but there have been two generations of Nvidia GPUs since then. Intel has kept quiet, but Haswell is also three generations old.

“While NVIDIA’s Kepler-generation GPU, architected in 2009, helped awaken the world to the possibility of using GPU-accelerated computing in deep learning, it was never specifically optimized for that task,” the company says in [a blog post](https://blogs.nvidia.com/blog/2017/04/10/ai-drives-rise-accelerated-computing-datacenter/).

To make their point, this was accompanied by their own benchmarks, which pointed to their latest P40 GPU being twice as fast. But importantly, the TPU still blows it out of the water on power consumption, and it wouldn’t be surprising that Google is already readying or even using a new generation of TPUs that improve on this design.

That said, it isn’t going to upend the chip market. Google won’t be selling the TPU to competitors and it is entirely focused on inferencing. Google still uses copious amounts of Nvidia’s GPUs for training, which explains the muted nature of the company’s rebuttal.

Google is also probably one of the few companies in the world with the money and the inclination to build a product from scratch in a completely new domain. But it is also one of the world’s biggest processor purchasers, so the fact that it has decided the only way to meet its needs is to design its own is a warning sign for chip makers.

Indeed, that appears to be part of the idea. “Google’s release of this research paper is intended to raise the level of discussion amongst the machine learning community and the chip makers that it is time for an off-the-shelf merchant solution for running inference at scale,” [writes Steve Patterson in *NetworkWorld*](http://www.networkworld.com/article/3190122/hardware/6-reasons-why-google-built-its-own-ai-chip.html)*.*

This is probably not too far off, analyst Karl Freund [writes in *Forbes*](https://www.forbes.com/sites/moorinsights/2017/04/13/googles-tpu-for-ai-is-really-fast-but-does-it-matter/amp/). “Given the rapid market growth and thirst for more performance, I think it is inevitable that silicon vendors will introduce chips designed exclusively for machine learning.”

Nvidia is unlikely to let its market leading position slip, and later this year Intel will release the first chips powered by the machine learning-focused [Nervana technology it acquired last August](https://www.technologyreview.com/s/602137/intel-buys-a-startup-to-catch-up-in-deep-learning/). Even mobile players are getting in on the act.

Arm’s Dynamiq microarchitecture will allow customers to [build AI accelerators directly into chips](http://www.theverge.com/2017/3/21/14998100/arm-new-dynamiq-microarchitecture-ai-chip-design) to bring native machine learning to devices like smartphones. And Qualcomm’s Project Zeroth has released a software development kit that can [run deep learning programs on devices like smartphones and drones](http://www.theverge.com/2016/5/2/11538122/qualcomm-deep-learning-sdk-zeroth)featuring its Snapdragon processors.

Google’s release of the TPU may be just a gentle nudge to keep them heading in the right direction.

Image Credit: [Shutterstock](http://www.shutterstock.com/)