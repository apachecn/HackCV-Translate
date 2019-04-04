# NeuroNuggets: Age and Gender Estimation

Today, we begin a new series of posts that we call NeuroNuggets. On February 15, right on time, we released the first version of the [NeuroPlatform](http://platform.neuromation.io/). So far it is still in alpha, and it will take a lot of time to implement everything that we have planned. But even now, there are quite a few cool things you can do. In the NeuroNuggets series, we will present these cool things one by one, explaining not only the technicalities of how to run something on the platform but also the main ideas behind every model. This is also my chance to present my new deep learning team hired at our new office in St. Petersburg, Russia.

In this post, we present our first installment: the age and gender estimation model. This is the simplest neural architecture among our demos, but even this network will have quite a few tricks to explain. And it is my pleasure to introduce Rauf Kurbanov, one of our first hires in St. Petersburg, with whom we have co-authored this post:

![](https://cdn-images-1.medium.com/max/1600/0*bMoHBU4jodqQ39s_.)

### Who hired a nerd?

AI researchers tend to question the nature of intuitive. As soon as you ask how a computer can do the same thing that seems too easy for humans, you see that what is “intuitively clear” for us can be very hard to formalize. Our visual perception of human age and gender is a good example of such a subtle quality.

To us AI nerds, Eliezer Yudkowsky is familiar both as an AI safety researcher and the author of the most popular Harry Potter fanfic ever (we heartily recommend “[Harry Potter and the Method of Rationality](http://hpmor.com)”, HPMoR for short, to everyone). And the Harry Potter series features a perfect example for this post, a fictional artifact that appears intuitively clear but is hard to reproduce in practice:

![](https://cdn-images-1.medium.com/max/1600/0*_OTbejyYOqAjWFT0.)

Albus [Dumbledore](http://harrypotter.wikia.com/wiki/Albus_Dumbledore) had placed an [Age Line](http://harrypotter.wikia.com/wiki/Age_Line) around the [Goblet of Fire](http://harrypotter.wikia.com/wiki/Goblet_of_Fire) to prevent anyone under the age of seventeen from approaching it. Age Line magic was so advanced that even an Ageing Potion could not fool it. Even Yudkowsky did not really dig into the mechanics of Age Line in his meticulous manner in HPMoR but today we will give it a try; and while we are on the subject, we will give a shot to gender recognition as well. As usual in computer vision, we begin with convolutional neural networks.

### Convolutional neural networks

A neural network, as the name suggests, is a machine learning approach which is in a very abstract way modeled after how the brain processes information. It is a network of learning units called artificial neurons, or perceptrons. During training, the neurons learn how to convert input signals (say, the picture of a cat) into corresponding output signals (say, the label “cat” in this case), training automated recognition from real life examples.

Virtually all computer vision nowadays is based on convolutional neural networks. Very roughly speaking, CNNs are multilayer (deep) neural networks where each layer processes the image in small windows, extracting local features. Gradually, layer by layer, local features become global, able to draw their inputs from a larger and larger portion of the original image. Here is how it works in a very simple CNN (picture taken from [this tutorial](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/), which we recommend to read in full):

![](https://cdn-images-1.medium.com/max/1600/0*ArVMZKJOFf3JMVpn.)

In the end, after several (sometimes several hundred) layers we get global features that “look at” the whole original image, and they can now be combined in relatively simple ways to obtain class labels (recognize whether it is a dog, cat, boat, or Harry Potter).

Technically, a convolutional neural network is a neural network with convolutional layers, and a convolutional layer is a transformation that applies a certain kernel (filter) to every point in the input (a “picture” with multiple channels in every pixel, i.e., a three-dimensional tensor) and generate filtered output by sliding the kernel over the input.

Let us consider a simple example of a filter: edge detection in images. In this case, the input for edge detection is an image, and each pixel in the image is defined by three numbers: the intensities of red, green, and blue in the color of that pixel. We construct a special kernel which will be applied to every pixel in the image; the output is a new “image” that shows the results of this kernel. Basically, the kernel here is a small matrix. Here’s how it works:

![](https://cdn-images-1.medium.com/max/1600/0*gfpyXOIenRB9fqrq.)

The kernel is sliding over every pixel in the image and the output value increases whenever there is an edge, an abrupt change of colors. In the figure above, after multiplying this simple matrix element-wise to every 3x3 window in the image we get a very nice edge detection result.

Once you understand filters and kernels, it becomes quite simple to explain convolutional layers in neural networks. You can think of them as vanilla convolutions, as in the edge detection example above, but now we are learning convolutional kernels end-to-end when training the networks. That is, we do not have to invent these small matrices by hand anymore but can automatically learn matrices that extract the best features for a specific task.

### The model pipeline

Age and gender estimation sounds like a traditional machine learning task: binary classification for the genders (it might stir up some controversy but yeah, our models live in a binary world) and regression for the ages. But before we can begin to solve these problems, we need to find the faces on the photo! Classification will not work on the picture as a whole because it might, for example, contain several faces. Therefore, the age and gender estimation problem is usually broken down into two steps, face detection and age/gender estimation for the detected faces:

![](https://cdn-images-1.medium.com/max/1600/0*KbPeF1P7rCxSF4UK.)

In the model that you can find on the NeuroPlatform, these steps are performed independently and are not trained end-to-end, so let us discuss each of them in particular.

### Face detection

Face detection is a classic problem in computer vision. It was solved quite successfully even before the deep learning revolution, in early 2000s, by the so-called [Viola-Jones algorithm](https://link.springer.com/article/10.1023/B:VISI.0000013087.49260.fb). It was one of the most famous application of Haar cascades as features; but those days are long gone…

Today, face detection is not treated as a separate task that requires individual approaches. It is also solved by convolutional neural networks. To be honest, since the advent of deep learning it has long become clear that CNNs kick ass at object detection, and therefore we expect modern solution to an old problem to be based on CNNs as well. And we are not wrong.

But in real world machine learning, you should also consider other properties beside detection accuracy such as simplicity and inference speed. If a simpler approach works well enough, it might not be worth it to introduce very complicated models to gain a few percentage points (remind me to tell you about the Netflix prize challenge results later). Therefore, in the NeuroPlatform demo we use a more classical approach to face detection while keeping CNNs for the core age/gender recognition task. But we can learn a lot from classical computer vision too.

In short, the face detection model can be described as an SVM on HOG + SIFT feature representation. HOG and SIFT representations are hand-crafted features, the result of years of experience in building image recognition systems. These features recognize gradient orientations in localized portions of an image and perform a series of deterministic image transformations. It turns out this representation works quite well with kernel methods such as support vector machines (SVM).

### Data augmentation

![](https://cdn-images-1.medium.com/max/1600/0*g6hrmFYocr0ZQdj4.)

Here at Neuromation, we are big fans of using synthetic data for computer vision. Usually, this means that we generate sophisticated synthetic datasets from 3D computer graphics and even look towards using Generative Adversarial Networks for synthetic data generation in the future. But let us not forget the most basic tool for increasing the datasets: data augmentation.

Since we have already extracted faces on the previous step, it is enough to augment only the faces, not the whole image. In the demo, we are using standard augmentation tricks such as horizontal/vertical shifts and mirroring alongside with a more sophisticated one of randomly erasing patches of the image.

### Age estimation

To predict the age, we apply a deep convolutional neural network to the face image detected on the previous processing stage. The method in the demo uses the [Wide Residual Network](https://arxiv.org/abs/1605.07146) (WRN) architecture which beat Inception architecture on mainstream object detection datasets, achieving convergence on the same task twice faster. Before we explain what residual networks are, we begin with a brief historical reference.

### The ImageNet challenge

The [ImageNet](http://www.image-net.org/) project is a large visual [database](https://en.wikipedia.org/wiki/Database) designed to be used in [visual object recognition ](https://en.wikipedia.org/wiki/Outline_of_object_recognition)research. The [deep learning](https://en.wikipedia.org/wiki/Deep_learning) revolution of the 2010s started in computer vision with a dramatic advance in solving the ImageNet Challenge. Results on ImageNet were recognized not only within the AI community but across the entire industry, and ImageNet has become and still remains the most popular general purpose computer vision dataset. Without getting into too much details, let us just take a glance at a comparison plot between the winners of a few first years:

![](https://cdn-images-1.medium.com/max/1600/0*bjyeZqx9VNTszOWk.)

On the plot, the horizontal axis shows how computationally intensive a model is, the circle size indicates the number of parameters, and the vertical axis shows image classification accuracy on ImageNet. As you can see, ResNet architectures show some of the best results while staying on the better side of efficiency as well. What is their secret sauce?

### Residual connections

It is well known that deeper models perform better than shallower models, they are more expressive. But optimization in deep models is a big problem: deeper models are harder to optimize due to the peculiarities of how gradients propagate from the top layers to the bottom layers (I hope one day we will explain it all in detail). Residual connections are an excellent solution to this problem: they add connections “around” the layers, and the gradient flow is now able to “skip” excessive layers during backpropagation, resulting in much faster convergence and better training:

![](https://cdn-images-1.medium.com/max/1600/0*LxRAWICvlu8GQGPx.)

Essentially the only difference with Wide Residual Networks is that in the original paper the tradeoff between width and depth of architecture was studied more carefully resulting in more efficient architecture with better convergence speed.

### DEX on NeuroPlatform

We have wrapped the DEX model into a docker container and uploaded in to the NeuroPlatform. Let’s get started! First, enter Neuromation login page [https://mvp.neuromation.io/](https://mvp.neuromation.io/)

![](https://cdn-images-1.medium.com/max/1600/0*0Z5AH-i0zDY0hUvq.)

On your dashboard on the NeuroPlatform, you can see how much NTK is left on your balance and spend them in three sections:

* AI Models,

* Datasets,

* Generators.

![](https://cdn-images-1.medium.com/max/1600/0*G7IZB5Rnhm1KN2tU.)

Today we are dealing with the Age and Gender estimator, an AI model available at the NeuroMarket. Let us purchase our first model! We enter AI models and buy the Age&Gender model:

![](https://cdn-images-1.medium.com/max/1600/0*Jd00m3M2c9Uru6kR.)

![](https://cdn-images-1.medium.com/max/1600/0*noDno5LotSYgVOO6.)

Then we request a new instance; it may take a while:

![](https://cdn-images-1.medium.com/max/1600/0*A3kkc376rZvfYn3-.)

And here we go! We can now try the model on a demo interface:

![](https://cdn-images-1.medium.com/max/1600/0*IHEaGegaIiXDkcQU.)

![](https://cdn-images-1.medium.com/max/1600/0*BB2m-6pnY_YuRsVg.)

It does tend to flatter me a bit.

Sergey NikolenkoChief Research Officer, Neuromation

Rauf KurbanovSenior Researcher, Neuromation

