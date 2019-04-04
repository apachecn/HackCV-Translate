# MXNet for PyTorch users in 10 minutes

Author: [Mu Li](https://github.com/mli), Principal Scientist at Amazon
Translated from: [https://zh.mxnet.io/blog/learn-mxnet-for-pytorch-users](https://zh.mxnet.io/blog/learn-mxnet-for-pytorch-users)

[PyTorch](https://pytorch.org/) has quickly established itself as one of the most popular deep learning framework due to its easy-to-understand API and its completely imperative approach. But you might not be aware that MXNet includes the [Gluon API](https://gluon-crash-course.mxnet.io/) which gives you the simplicity and flexibility of PyTorch, whilst allowing you to hybridize your network to leverage performance optimizations of the symbolic graph.

In the next 10 minutes, we’ll show you a quick comparison between the two frameworks and show you how small the learning curve can be when switching from one to the other. We use the example of image classification on MNIST dataset.

![](https://cdn-images-1.medium.com/max/1600/0*Ncynv-83j8ESn3DZ.png)

### Installation

PyTorch uses conda for installation by default, for example:



For MXNet we use pip. We can also use the `--pre` flag to install the nightly version:



### Multidimensional matrix

For multidimensional matrices, PyTorch follows Torch’s naming convention and refers to “tensors”. MXNet follows NumPy’s conventions and refers to “ndarrays”. Here we create a two-dimensional matrix where each element is initialized to 1. Then we add 1 to each element and print.

* PyTorch:





* MXNet:





The main difference apart from the package name is that the MXNet’s shape input parameter needs to be passed as a tuple enclosed in parentheses as in NumPy.

### Model training

Let’s look at a slightly more complicated example below. Here we create a Multilayer Perceptron (MLP) to train a model on the MINST data set. We divide the experiment into 4 sections.

### 1 - Read data

We download the MNIST data set and load it into memory so that we can read batches one by one.

* PyTorch:



* MXNet:



The main difference here is that MXNet uses `transform_first` to indicate that the data transformation is done on the first element of the data batch, the MNIST picture, rather than the second element, the label.

### 2 — Creating the model

Below we define a Multilayer Perceptron (MLP) with a single hidden layer and 10 units in the output layer.

* PyTorch:



* MXNet:



We used the `Sequential` container to stack layers one after the other in order to construct the neural network. MXNet differs from PyTorch in the following ways:

* In MXNet, there is no need to specify the input size, it will be automatically inferred.

* You can specify activation functions directly in fully connected and convolutional layers.

* You need to create a `name_scope` to attach a unique name to each layer: this is needed to save and reload models later.

* You need to explicitly call the model initialization function.

With a `Sequential` block, layers are executed one after the other. To have a different execution model, with PyTorch you can inherit `nn.Module` and then customize how the `.forward()` function is executed. Similarly, in MXNet you can inherit `nn.Block` to achieve similar results.

### 3 — Loss function and optimization algorithm

* PyTorch:



* MXNet:



Here we pick a cross-entropy loss function and use the Stochastic Gradient Descent algorithm with a fixed learning rate of 0.1.

### 4 — Training

Finally we implement the training algorithm. Note that the results for each run may vary because the weights will get different initialization values and the data will be read in a different order due to shuffling.

* PyTorch





* MXNet





Some of the differences in MXNet when compared to PyTorch are as follows:

* You don’t need to put the input into `Variable` (This is not necessary anymore since PyTorch 0.4.0), but you need to perform the calculation within the`mx.autograd.record()` scope so that it can be automatically differentiated in the backward pass.

* It is not necessary to clear the gradient every time as with Pytorch’s `trainer.zero_grad() `because by default the new gradient is written in, not accumulated.

* You need to specify the update step size (usually batch size) when performing`.step()` on the trainer.

* You need to call `.asscalar()` to turn a multidimensional array into a scalar.

* In this sample, MXNet is twice as fast as PyTorch. Though you need to be cautious with such toy comparisons.

### Next steps

* Follow more detailed MXNet tutorials: [http://mxnet.incubator.apache.org/tutorials/index.html](http://mxnet.incubator.apache.org/tutorials/index.html)

* Feel free to create an issue on the [github repo](https://github.com/apache/incubator-mxnet) or create a question on the [forum](https://discuss.mxnet.io/) if you are missing some features from PyTorch in MXNet!

* Follow the [Gluon 60 minutes crash-course](https://gluon-crash-course.mxnet.io/)

