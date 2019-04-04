![](https://cdn-images-1.medium.com/max/1600/1*YbSWOBCsgXXFa709FCuPaQ.png)

# Rediscovering Semantic Segmentation

The code and ideas discussed here resulted from some amazing collaboration with [Prathmesh Dali](https://medium.com/@prathmesh.dali) and [Safwan Ahmad Siddiqi](https://medium.com/@safwan.ahmad.siddiqi).


emantic Segmentation is a machine learning technique that learns to identify the extents of individual objects in an image. Semantic segmentation gives machine learning systems the human-like ability to understand the contents of an image. It enables machine learning algorithms to locate the precise boundaries of objects, be it cars and pedestrians in a street image or heart, liver and kidneys in a medical image.

There are some excellent articles on the topic of semantic segmentation, perhaps the most comprehensive one is this blog:

[**A 2017 Guide to Semantic Segmentation with Deep Learning**
In this post, I review the literature on semantic segmentation. Most research on semantic segmentation use natural/real…blog.qure.ai](http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review)[](http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review)

Unlike most other articles on semantic segmentation, the aim of this blog post is to describe how to build a small semantic segmentation network that can be quickly trained and can be used to experiment with semantic segmentation .

This post explains how to reuse some layers of a convolution neural network (CNN) , trained to classify MNIST digits, and build a fully connected network(FCN) upon them, that can semantically segment multi-digit images.

The dataset for semantic segmentation has been built by copying more than one 28px*28px MNIST digits to a 64px*84px image.

### Background


here are different types of semantic segmentation networks and the focus here is on Fully Convolution Networks(FCNs). The first FCN was proposed in [this](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) paper from Berkely. FCNs are built by extending normal convolution networks (CNN) and thus have more parameters and take longer to train than the latter. The work described here stemmed from an effort to build an FCN that is small enough to be trained on a typical laptop in a few minutes. The idea was to first build a dataset containing multiple [MNIST](https://www.kaggle.com/c/digit-recognizer) digits in every image. The [code](https://github.com/farhanhubble/udacity-connect/blob/master/segmented-generator.ipynb) used to generate this derived dataset is here. Let us call it M2NIST (multi-digit MNIST) to avoid any confusion.

### M2NIST

Every image in M2NIST is grayscale (single channel), 64x84 pixels in size, and contains up to 3 digits from MNIST dataset. A typical image can look like this:

![](https://cdn-images-1.medium.com/max/1600/1*Mwgvjm5knKGQlIwYP6NCvw.png)

The labels for the M2NIST dataset are segmentation masks. A segmentation mask is a binary image (pixel values 0 or 1),with the same height and width as the multi-digit image but with 10 channels, one for every digit from 0 to 9. The **k-th**channel in the mask has only those pixels set to 1 that coincide with the location of digit **k**in the input multi-digit. If digit **k**is not present in the multi-digit, the **k-th**channel in the mask has all its pixels set to 0. On the other hand, if the multi-digit contains more than one instance of the the **k-th**digit, the **k-th**channel will have all those pixels set to 1 that happen to coincide with either of the instances in the multi-digit. For example the mask for the multi-digit above looks like this:

![](https://cdn-images-1.medium.com/max/1600/1*nPBRHxd5TyPrKWmauviO9A.png)

To keep things easy the M2NIST dataset combines digits from MNIST and does not perform any transform, for example, rotation or scaling. M2NIST does ensures that the digits do not overlap.

### The Idea Behind FCNs

The idea behind FCNs is very simple. Like CNNs, FCNs use a cascade of convolution and pooling layers. The convolution and maxpooling layers reduce the spatial dimension of an input image and combine local patterns to generate more and more abstract ‘features’. This cascade is called an encoder as raw input is encoded into more abstract, encoded, features.

In a CNN, the encoder is followed by a few fully-connected layers that mix together the local features produced by the encoder into global predictions that tell a story about the presence or absence of objects of our interest.

> CNN = Encoder + Classifier

![](https://cdn-images-1.medium.com/max/1600/1*NQQiyYqJJj4PSYAeWvxutg.png)

In an FCN, we are interested in predicting masks. A mask has **n**channels if there are **n**classes of objects that could be present in an input image. The pixel at row **r**and column **c**in the **k-th**channel of the mask,****predicts the probability of the pixel with coordinates **(r,c)**in the input belonging to class **k**. This is also known as pixel-wise dense prediction. Because the total probability of belonging to different classes for any pixel should add up to 1, the sum of values at **(r,c)**from channel 1 to **n**have sum equal to 1.

![](https://cdn-images-1.medium.com/max/1600/1*P1ooLjeSwhxeJGyFawCvaQ.png)

Let us understand how FCNs achieve pixel-wise dense prediction. FCNs first, gradually, expand the output features from the encoder stage using transpose convolution. Transpose convolution re-distributes the features back to pixel positions they came from. To understand how transpose convolution works, refer to this excellent post:

[**Up-sampling with Transposed Convolution**
If you’ve heard about the transposed convolution and got confused what it actually means, this article is written for…towardsdatascience.com](https://towardsdatascience.com/up-sampling-with-transposed-convolution-9ae4f2df52d0)[](https://towardsdatascience.com/up-sampling-with-transposed-convolution-9ae4f2df52d0)

It is important to stress that transpose convolution **does not**undo convolution. It merely redistributes the output of some convolution in a fashion that is consistent with, but in the opposite direction of, the way in which convolution combines multiple values.

![](https://cdn-images-1.medium.com/max/1600/1*4a4OjlszAvi7-vqjOT0PoA.png)

The expansion or up-sampling, as it is called, is repeated, using multiple transpose convolutions, until the features have the same height and width as the input image. This essentially gives us features for every pixel position and constitutes the decoder stage of an FCN.

> FCN = Encoder + Decoder

![](https://cdn-images-1.medium.com/max/1600/0*eyUKBp0N2g8FQdX9.png)

The output of the decoder is a volume with shape **HxWxC**,****where **H**and**W**are the dimensions of the input image and **C**is a hyper-parameter. The **C**channels are then combined into **n**channels in a pixel-wise fashion, **n**being the number of object classes we care about. The pixel-wise combination of features values is done using normal 1x1 convolution. 1x1 convolutions are commonly used for this kind of [‘dimension reduction’](https://stats.stackexchange.com/a/194450/69501).

In most cases we have **C > n**so it makes sense to call this operation a dimension reduction. It is also worth mentioning that, in most implementations, this dimension reduction is applied to the output of the encoder stage instead of the decoder’s output. This is done to [reduce the size of the network](https://arxiv.org/pdf/1409.4842.pdf).

Whether the encoder’s output is up-sampled by the decoder and then the decoder’s output dimension is reduced to **n**OR the encoder’s output dimension is immediately reduced to **n**and then the decoder up-samples this output, the final result has shape **HxWxn**. A Softmax classifier is then applied pixel-wise to predict the probability of each pixel belonging to each of the **n**classes.

> To take a concrete example, suppose the encoder’s output has shape 14x14x512, as in the FCN diagram above, and the number of classes, n, is 10. One option is to first reduce the thickness dimension using 1x1 convolutions. This gives us a 14x14x10 volume which is then up-sampled to 28x28x10, 56x56x10 and so on, until the output has shape HxWx10. The second option is to up-sample first, which gives us 28x28x512, 56x56x512 and so on until we reach HxWx512 and then use 1x1 convolution to reduce the thickness to HxWx10. Clearly the second option consumes more memory as all the intermediate outputs with thickness 512 will use more memory than intermediate outputs with thickness 10 that are produced with the first approach.

With the encoder-decoder architecture in mind, let us see how to reuse parts of a CNN as the encoder for an FCN.

### Repurposing an MNIST Classifier

Typically, FCNs are built by extending existing CNN classification networks e.g. Vgg, Resnet or GoogLeNet. Not only are these architectures reused, their pre-trained weights are reused too, which significantly reduces the training time of the FCN.

The recipe for converting a CNN into an FCN is described in the original [paper](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) as:

> We decapitate each net by discarding the final classifier layer, and convert all fully connected layers to convolutions.

The CNN used to build our FCN has a simple convolution-maxpooling-convolution-maxpooling-dense-dense architecture. The CNN architecture and training code can be found [here](https://github.com/farhanhubble/udacity-connect/blob/master/mnist.ipynb). The trained network is saved so that it can be reused. The network is defined like this:











To ‘decapitate’ the network, we remove the final classifier layer named **dense10**. The only remaining fully-connected layer named **dense32**is then replaced by a 1x1 convolution layer. This is something we have not discussed so far but is done in the original paper. In the code listed above, this amounts to removing the **flatten**and **dense32**layers and inserting a new 1x1 convolution with output thickness set to 32. This is equivalent to discarding everything after the last maxpooling layer **pool2**and adding the 1x1 convolution layer.

The code for building the **initial version**of our FCN is on [Github](https://github.com/farhanhubble/udacity-connect/blob/4408cc1e8917f37e287d09177d6e4585bfe164ff/FCN-mnist.ipynb) (The [latest code](https://github.com/farhanhubble/udacity-connect/) looks different but the gist is same). In the excerpt below, the output of the last maxpooling is extracted (via`get_tensor_by_name() `), it is then fed to a 1x1 convolution with output thickness 32. This convolution is the ‘replacement’ for the **dense32**layer found in the original CNN. Next the thickness is reduced to 10, once again using 1x1 convolution. This is the dimension reduction discussed earlier.











This finishes the encoder stage of our FCN. To build the decoder stage, we need to think about how and how much to scale the encoder’s output width and height.

Although the convolution and maxpooling in the encoder come from a CNN for classifying MNIST images of size 28x28, they can be fed any image of any size. Convolution and maxpooling do not care about the height and width of their input, dense layers do but they have already been gotten rid of by decapitating the last dense layer and converting all other dense layers to 1x1 convolutions.

When a 64x84x1 M2NIST image is fed to the encoder stage, the first convolution layer(from the original CNN) having kernel size **k=5**, stride **s=1,**and output depth **f=8,**produces an output with shape 60x80x8. The maxpooling layer with **k=2**and **s=2** halves the size to 30x40x8. The next convolution with **k=3**,**s=1**,**f=8**produces an output with 28x38x8 and the size is again halved to 14x19x8 by the next maxpooling layer. To summarize:

> the part of the FCN borrowed from the CNN ingests an image with shape 64x84x1 and outputs features with shape 14x19x8.

The next layer in the encoder (the replacement for **dense32**)****is a 1x1 convolution with output thickness**f=32**. It recombines 14x19x8 features into new features with shape 14x19x32.

The thickness of these features is then reduced (dimension reduction). This employs 1x1 convolution with thickness **f=10**. So the final features coming out of the encoder have shape 14x19x10. These features are then up-sampled by the decoder stage until their shape becomes to 64x84x10.

> The decoder has to up-sample 14x19x10 features to 64x84x10 features.

The up-sampling is done in stages to avoid ugly patterns in the final output (mask). In our (early) implementation, the features were up-sampled from 14x19x10 to 30x40x10 and then up-sampled again to 64x84x10.

Up-sampling is done with transpose convolution which, like convolution, takes kernel size **k,**stride **s**, and number of filters (thickness) **f**as parameters. The number of filters is **f=10**for both transpose convolution operations, since we are not changing the thickness.

The stride is decided from the ratio of final and initial dimensions. For the first transpose convolution the ratio of heights (30/14) and widths(40/19) both is 2 so **s=2**is used. In the second transpose convolution, the ratios are 64/30 and 84/40, so again **s=2**is used.

Deciding the kernel size is slightly tricky and involves some experimentation. For the first transpose convolution, using **k=1**exactly doubles the dimension from 14x19x10 to 28x38x10. To get to 30x40x10 **k=2** and **k=3** were tried but fell short. Finally **k=4** worked. For the second transpose convolution, kernel size was found out to be **k=6**.

![](https://cdn-images-1.medium.com/max/1600/1*Mc6HEB3ILCHtWd1nGedoXg.png)

The code for the decoder is exactly two lines of Tensorflow API calls:





To perform pixel wise probability computation, the output of the decoder is fed to a Softmax layer. The softmax is applied along the thickness (channels).







The FCN is trained using cross entropy cost function for 100–400 epochs on a Laptop with Nvidia 1050Ti GPU. The typical training time is of the order of a few minutes with 2000 samples.

This initial design had a high bias problems which was fixed in a later iteration. In addition there were a few logical and programming bugs that caused the network to perform sub-optimally. Here’s a snapshot from the best performing early design:

![](https://cdn-images-1.medium.com/max/1600/1*QZjq16_ePON6hfepx2TDhQ.png)

![](https://cdn-images-1.medium.com/max/1600/1*Z2YPQI1ipJxQ7VDzfIu2Gw.png)

After fixing the shortcomings in the network, it was able to perform near perfect segmentation. For example, here is the predicted output:

![](https://cdn-images-1.medium.com/max/1600/1*zRAw-tFNwo-DH0O6GZH1Yw.png)

### Takeaways and More

The idea for this project came when teaching Semantic Segmentation during a [Udacity](https://medium.com/@udacity) connect program.

It took around two weeks to do all the research and experimentation to get acceptable results. It was worth reinventing the wheel because the small footprint network enables hundreds or even thousands of experiments that would otherwise have been impossible, at least without massive computation power.

Some of the experiments already done have given clues about which knobs to turn when tuning an FCN. Expect a follow up post with the nitty-gritty and gotchas of building the final version as well as covering ‘skip connections’ that have not been explored in this post.

The full code is available at [https://github.com/farhanhubble/udacity-connect/](https://github.com/farhanhubble/udacity-connect/). Feel free to fork.

Please leave comments below and follow 100 Shades of Machine Learning if you liked this post.

