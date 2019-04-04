# Highlights from the TensorFlow Developer Summit, 2018

Posted by [Sandeep Gupta](https://twitter.com/thesandeepgupta), Product Manager for TensorFlow, on behalf of the TensorFlow team.

Today, we’re holding the second [TensorFlow Developer Summit](https://www.tensorflow.org/dev-summit/) at the Computer History Museum in Mountain View, CA! The event brings together over 500 TensorFlow users in-person and thousands tuning into the livestream at TensorFlow events around the world. The day is filled with new product announcements along with technical talks from the TensorFlow team and guest speakers.

![](https://cdn-images-1.medium.com/max/1600/0*dwmeeMeY5htdHbvy.)

Machine learning is solving challenging problems that impact everyone around the world. Problems that we thought were impossible or too complex to solve are now possible with this technology. Using TensorFlow, we’ve already seen great advancements in many different fields. For example:

* Astrophysicists are using TensorFlow to analyze large amounts of data from the Kepler mission to [discover new planets](https://research.googleblog.com/2018/03/open-sourcing-hunt-for-exoplanets.html).

* Medical researchers are using ML techniques with TensorFlow to assess a person’s [cardiovascular risk of a heart attack and stroke](https://research.googleblog.com/2018/02/assessing-cardiovascular-risk-factors.html).

* Air Traffic Controllers are using TensorFlow to [predict flight routes through crowded airspace](http://www.eurocontrol.int/publications/traffic-prediction-improvements-tpi-factsheet-and-technical-documentation) for safe and efficient landings.

* Engineers are using TensorFlow to analyze auditory data in the rainforest to [detect logging trucks and other illegal activities](https://www.blog.google/topics/machine-learning/fight-against-illegal-deforestation-tensorflow/).

* Scientists in Africa are using TensorFlow to detect diseases in Cassava plants to improving yield for farmers.



We’re excited to see these amazing uses of TensorFlow and are committed to making it accessible to more developers. This is why we’re pleased to announce new updates to TensorFlow that will help improve the developer experience!

### We’re making TensorFlow easier to use

Researchers and developers want a simpler way of using TensorFlow. We’re integrating a more intuitive programming model for Python developers called [eager execution](https://www.tensorflow.org/programmers_guide/eager) that removes the distinction between the construction and execution of computational graphs. You can develop with eager execution and then use the same code to generate the equivalent graph for training at scale using the Estimator high-level API. We’re also announcing a new method for [running Estimator models](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/distribute/README.md) on multiple GPUs on a single machine. This allows developers to quickly scale their models with minimal code changes.

As machine learning models become more abundant and complex, we want to make it easier for developers to share, reuse, and debug them. To help developers share and reuse models, we’re announcing [TensorFlow Hub](http://tensorflow.org/hub), a library built to foster the publication and discovery of modules (self-contained pieces of TensorFlow graph) that can be reused across similar tasks. Modules contain weights that have been pre-trained on large datasets, and may be retrained and used in your own applications. By reusing a module, a developer can train a model using a smaller dataset, improve generalization, or simply speed up training. To make debugging models easier, we’re also releasing a new interactive [graphical debugger plug-in](https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/debugger/README.md) as part of the TensorBoard visualization tool that helps you inspect and step through internal nodes of a computation graph in real-time.

![](https://cdn-images-1.medium.com/max/1600/0*mTewjcnWRlVVK2rT.)

Model training is only one part of the machine learning process and developers need a solution that works end-to-end to build real-world ML systems. Towards this end, we’re announcing the roadmap for TensorFlow Extended (TFX) along with the launch of TensorFlow Model Analysis, an open-source library that combines the power of TensorFlow and Apache Beam to compute and visualize evaluation metrics. The components of TFX that have been released thus far (including [TensorFlow Model Analysis](https://github.com/tensorflow/model-analysis), [TensorFlow Transform](https://github.com/tensorflow/transform), [Estimators](http://tensorflow.org/programmers_guide/estimators), and [TensorFlow Serving](https://github.com/tensorflow/serving)) are well integrated and let developers prepare data, train, validate, and deploy TensorFlow models in production.

![](https://cdn-images-1.medium.com/max/1600/0*194_INsq197WvIf1.)

### TensorFlow is available in more languages and platforms

Along with making TensorFlow easier to use, we’re announcing that developers can use TensorFlow in new languages. [TensorFlow.js](https://js.tensorflow.org) is a new ML framework for JavaScript developers. Machine learning in the browser using TensorFlow.js opens exciting new possibilities, including interactive ML and support for scenarios where all data remains client-side. It can be used to build and train modules entirely in the browser, as well as import TensorFlow and Keras models trained offline for inference using WebGL acceleration. The [Emoji Scavenger Hunt game](https://emojiscavengerhunt.withgoogle.com/) is a fun example of an application built using TensorFlow.js.

![](https://cdn-images-1.medium.com/max/1600/0*V4HYbZt28PHZZ3aD.)

We also have some exciting news for Swift programmers: [TensorFlow for Swift](https://www.tensorflow.org/community/swift) will be open sourced this April. TensorFlow for Swift is not your typical language binding for TensorFlow. It integrates first-class compiler and language support, providing the full power of graphs with the usability of eager execution. The project is still in development, with more updates coming soon!

We’re also sharing the latest updates to [TensorFlow Lite](https://www.tensorflow.org/mobile/tflite/), TensorFlow’s lightweight, cross-platform solution for deploying trained ML models on mobile and other edge devices. In addition to existing support for Android and iOS, we’re announcing support for Raspberry Pi, increased support for ops/models (including custom ops), and describing how developers can easily use TensorFlow Lite in their own apps. The TensorFlow Lite core interpreter is now only 75KB in size (vs 1.1 MB for TensorFlow) and we’re seeing speedups of up to 3x when running quantized image classification models on TensorFlow Lite vs. TensorFlow.

For hardware support, TensorFlow now has [integration with NVIDIA’s TensorRT](https://developers.googleblog.com/2018/03/tensorrt-integration-with-tensorflow.html). TensorRT is a library that optimizes deep learning models for inference and creates a runtime for deployment on GPUs in production environments. It brings a number of optimizations to TensorFlow and automatically selects platform specific kernels to maximize throughput and minimizes latency during inference on GPUs.

For users who run TensorFlow on CPUs, our partnership with Intel has delivered [integration with a highly optimized Intel MKL-DNN](https://github.com/tensorflow/tensorflow/pull/16474) open source library for deep learning. When using Intel MKL-DNN, we observed up to 3x inference speedup on various Intel CPU platforms.

The list of platforms that run TensorFlow has grown to include Cloud TPUs, which were [released in beta](https://cloudplatform.googleblog.com/2018/02/Cloud-TPU-machine-learning-accelerators-now-available-in-beta.html) last month. The Google Cloud TPU team has already delivered a strong 1.6X performance increase in ResNet-50 performance since launch. These improvements will be available to TensorFlow users with the 1.8 release soon.

### Enabling new applications and domains using TensorFlow

Many data analysis problems are solved using statistical and probabilistic methods. Beyond deep learning and neural network models, TensorFlow now provides state-of-the-art methods for Bayesian analysis via the [TensorFlow Probability API](https://github.com/tensorflow/probability/). This library contains building blocks like probability distributions, sampling methods, and new metrics and losses. Many other classical ML methods also have increased support. As an example, boosted decision trees can be easily trained and deployed using [pre-made high-level classes](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/estimator/canned/boosted_trees.py).

Machine learning and TensorFlow have already helped solve challenging problems in many different fields. Another area where we see TensorFlow having a big impact is in genomics, which is why we’re releasing [Nucleus](https://www.github.com/google/nucleus), a library for reading, writing, and filtering common genomics file formats for use in TensorFlow. This, along with [DeepVariant](https://github.com/google/deepvariant/blob/r0.5/README.md), an open-source TensorFlow based tool for genome variant discovery, will help spur new research and advances in genomics.

### Expanding community resources and engagement

These updates to TensorFlow aim to benefit and grow the community of users and contributors — the thousands of people who play a part in making TensorFlow one of the most popular ML frameworks in the world. To continue to engage with the community and stay up-to-date with TensorFlow, we’ve launched the new official [TensorFlow blog](http://blog.tensorflow.org) and the [TensorFlow YouTube channel](http://youtube.com/tensorflow).

We’re also making it easier for our community to collaborate by launching [new mailing lists](http://tensorflow.org/community/lists) and [Special Interest Groups](https://tensorflow.org/community/contributing#special_interest_groups) designed to support open-source work on specific projects. To see how you can be a part of the community, visit the [TensorFlow Community](https://tensorflow.org/community) page and as always, you can follow TensorFlow on [Twitter](http://twitter.com/tensorflow) for the latest news.

We’re incredibly thankful to everyone who has helped make TensorFlow a successful ML framework in the past two years. Thanks for attending, thanks for watching, and remember to use #MadeWithTensorFlow to share how you are solving impactful and challenging problems with machine learning and TensorFlow!

