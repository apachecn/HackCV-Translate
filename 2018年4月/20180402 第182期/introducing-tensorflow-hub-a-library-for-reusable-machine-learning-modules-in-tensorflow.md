# Introducing TensorFlow Hub: A Library for Reusable Machine Learning Modules in TensorFlow

Posted by [Josh Gordon](http://twitter.com/random_forests), Developer Advocate for TensorFlow

One of the things that’s so fundamental in software development that it’s easy to overlook is the idea of a repository of shared code. As programmers, libraries immediately make us more effective. In a sense, they change the problem solving process of programming. When using a library, we often think of programming in terms of building blocks — or modules — that can be tied together.

How might a library look for a machine learning developer? Of course, in addition to sharing code, we’d also like to share pretrained models. Sharing a pretrained model makes it possible for a developer to customize it for their domain, without having access to the computing resources or the data used to train the model originally on hand. For example, [NASNet](https://research.googleblog.com/2017/11/automl-for-large-scale-image.html) took thousands of GPU-hours to train. By sharing the learned weights, a model developer can make it easier for others to reuse and build upon their work.

![](https://cdn-images-1.medium.com/max/1600/0*iOf29hLvkfK0UN6n.)

It’s the idea of a library for machine learning developers that inspired [TensorFlow Hub](http://tensorflow.org/hub), and today we’re happy to share it with the community. TensorFlow Hub is a platform to publish, discover, and reuse parts of machine learning modules in TensorFlow. By a module, we mean a self-contained piece of a TensorFlow graph, along with its weights, that can be reused across other, similar tasks. By reusing a module, a developer can train a model using a smaller dataset, improve generalization, or simply speed up training. Let’s look at a couple examples to make this concrete.

### Image Retraining

As a first example, let’s look at a technique you can use to train an image classifier, starting from only a small amount of training data. Modern image recognition models have millions of parameters, and of course, training one from scratch requires a large amount of labeled data and computing power. Using a technique called [Image Retraining](https://www.tensorflow.org/tutorials/image_retraining), you can train a model using a much smaller amount of data, and much less computing time. Here’s how this looks in TensorFlow Hub.



The basic idea is to reuse an existing image recognition module to extract features from your images, and then train a new classifier on top of these. As you can see above, TensorFlow Hub modules can be instantiated from a URL (or, from a filesystem path) while a TensorFlow graph is being constructed. There are variety of [modules](http://tensorflow.org/hub/modules) on TensorFlow Hub for you to choose from, including various flavors of NASNet, [MobileNet](https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html) (including its recent V2), Inception, ResNet, and others. To use a module, you [import](http://tensorflow.org/hub/installation) TensorFlow Hub, then copy/paste the module’s URL into your code.

![](https://cdn-images-1.medium.com/max/1600/0*00s77XkzFPFzAt-B.)

Each module has a defined interface that allows it to be used in a replaceable way, with little or no knowledge of its internals. In this case, this module has a method that you can use to retrieve the expected image size. As a developer, you need only provide a batch of images in the correct shape, and call the module on them to retrieve the feature representation. This module takes care of preprocessing your images for you, so you can go directly from a batch of images to a feature representation in a single step. From here, you can learn a linear model, or other type of classifier, on top of these.

In this case, notice the module we’re using is hosted by Google, and is versioned (so you can rely on the module not changing as you work on your experiments). Modules can be applied like an ordinary Python function to build part of the graph. Once exported to disk, modules are self-contained, and can be used by others without access to the code and data used to create and train it (though of course you can publish those, too).

### Text Classification

Let’s take a look at a second example. Imagine you’d like to train a model to classify movie reviews as positive or negative, starting with only a small amount of training data (say, on the order of several hundred positive and negative movie reviews). Since you have a limited number of examples, you decide to leverage a dataset of word embeddings, previously trained on a much larger corpus. Here’s how this looks using a TensorFlow Hub.



As before, we start by selecting a [module](http://tensorflow.org/hub/modules/text). TensorFlow Hub has a variety of text modules for you to explore, including Neural network language models in a variety of languages (EN, JP, DE, and ES), as well as Word2vec trained on Wikipedia, and NNLM embeddings trained on Google News.

![](https://cdn-images-1.medium.com/max/1600/0*a9kWTkQOji3VfTmI.)

In this case, we’ll use a module for word embeddings. The code above downloads a module, uses it to preprocess a sentence, then retrieves the embeddings for each token. This means you can go directly from a sentence in your dataset to a format suitable for a classifier in a single step. The module takes care of tokenizing the sentence, and other logic like handling out-of-vocabulary words. Both the preprocessing logic and the embeddings are encapsulated in a module, making it easier to experiment with various datasets of word embeddings, or different preprocessing strategies, without having to substantially change your code.

![](https://cdn-images-1.medium.com/max/1600/0*hCPZRkenidXHOOas.)

If you’d like to try this out, use this [tutorial](http://tensorflow.org/tutorials/text_classification_with_tf_hub) to take it for a spin, and to learn how TensorFlow Hub modules work with TensorFlow Estimators.

### Universal Sentence Encoder

We’ve also shared a TensorFlow Hub module for something new! Below is an example using the Universal Sentence Encoder. It’s a sentence-level embedding module trained on a wide variety of datasets (in other words, “universal”). Some of the things it’s good at are semantic similarity, custom text classification, and clustering.

![](https://cdn-images-1.medium.com/max/1600/1*ack_mbSYP96g3Yu5YhlUrQ.png)

As in image retraining, relatively little labeled data is required to adapt the module to your own task. Let’s try it on a restaurant reviews, for example.



Check out the this [tutorial](http://tensorflow.org/tutorials/text_classification_with_tf_hub) to learn more.

### Other Modules

TensorFlow Hub is about more than image and text classification. On the website, you’ll also find a couple [modules](http://tensorflow.org/hub/modules/other) for Progressive GAN and [Google Landmarks Deep Local Features](https://github.com/tensorflow/models/tree/master/research/delf).

### Considerations

There are a couple of important considerations when using TensorFlow Hub modules. First, remember that modules contain runnable code. Always use modules from a trusted source. Second, as in all of Machine Learning, [fairness](http://ml-fairness.com/) is an [important](https://research.googleblog.com/2016/10/equality-of-opportunity-in-machine.html) consideration. Both of the examples we’ve shown above leverage large pre-trained datasets. When reusing such a dataset, it’s important to be mindful of what data it contains (and whether there are any existing biases there), and how these might impact the product you are building, and its users.

### Next steps

We hope you find TensorFlow Hub useful in your projects! To get started, head to [tensorflow.org/hub](http://tensorflow.org/hub). If you run into any bugs, you can file an [issue on GitHub](https://github.com/tensorflow/hub/issues). To stay in touch, you can star the [GitHub project](https://github.com/tensorflow/hub), and follow TensorFlow on [Twitter](http://twitter.com/tensorflow). Thanks for reading!

