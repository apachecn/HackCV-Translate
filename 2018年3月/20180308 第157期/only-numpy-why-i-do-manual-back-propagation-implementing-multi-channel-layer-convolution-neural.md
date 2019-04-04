# Only Numpy: (Why I do Manual Back Propagation) Implementing Multi Channel/Layer Convolution Neural Network on Numpy with Interactive Code

So, I made a post about understanding back propagation on Max Pooling Layer as well as Transpose Convolution. The next step would be to use those knowledge to make a Multi Channel/Layer CNN, so….lets do that! Also, just FYI, I am using momentum optimizer.

Before reading this post, I recommend reading this Quora question, “[Why is Geoffrey Hinton suspicious of backpropagation and wants AI to start over?](https://www.quora.com/Why-is-Geoffrey-Hinton-suspicious-of-backpropagation-and-wants-AI-to-start-over)” or this blog post “[Why we should be Deeply Suspicious of BackPropagation](https://medium.com/intuitionmachine/the-deeply-suspicious-nature-of-backpropagation-9bed5e2b085e)”. Both are very interesting.

Also, please note that it would be better if you already understand well about back propagating via transpose convolution and max pooling layer, since I won’t go into detail in this post.

**Training Data and Declaring Hyper Parameter**

![](https://cdn-images-1.medium.com/max/1600/1*Mmr_QP-SBlxazmI3tDDr8A.png)

As seen above, all of the kernels are (3*3) matrix, and we are performing a very simple binary task on “[Recognizing hand-written digits](http://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html)” data set only for 0 and 1 images.

**Network Architecture**

Input → Image that have (8*8) Dimension

Red Star → Layer 1 with two different channels
Red Circle → Activation and Max Pooling Layer Applied to Layer 1

Blue Star → Layer 2 with four difference channels
Blue Circle → Activation and Max Pooling operation Applied to Layer 2

Green Star → Layer 3 with Fully Connected Weight (W3) Dimension of (16*28)
Green Circle → Activation Function Applied to Layer 3

Pink Star → Layer 4 with Fully Connected Weight (W4) Dimension of (28*36)
Pink Circle → Activation Layer Applied to Layer 4

Black Star → Layer 5 with Fully Connected Weight (W5) Dimension of (36*1)
Black Circle → Activation Layer Applied to Layer 5

Black Box → Cost Function using the L2 Norm

**Forward Feed Operation Implemented**

Each Color represents the operation of that layer, two things to note here.

1. I performed zero padding before every convolution layer to preserve dimension.

2. I used variety of different activation functions for each layer.

**Back Propagation Respect to W5, W4 and W3 Implemented**

Standard Back propagation with appropriate derivative for each activation functions, for standard fully connected layers, nothing special.

**Back Propagation respect to all of W2 (W2a, W2b, W2c, and W2d) Implemented**

First line of code (Underlined Green) → Back Propagating from previous layer 
Green Boxed Region → Calculating Gradient Respect to W2a, W2b, W2c, and W2d in their respective order.

**Back Propagation Respect to all of W1 (W1a and W1b) Implemented**

First Red Box → Gradient Respect to W1a
Second Red Box → Gradient Respect to W2a

**Training and Results**

Pretty good, results the final results are not too bad. As seen above the number of epoch was set to 500. And the Network classified every image on test set except for one.

**Breaking the Math in Back Propagation**

So above is the proper back propagation, where we perform element wise multiplication between Derivative from previous layer and Derivative of activation function. However, out of curiosity I decided to do something like below.

Blue Line → Simple Element Wise multiplication
Green Line → Only Performing Transpose on Calculated Derivative after Max Pooling Layer 
Red Line → Only Performing Transpose on Mask of Max Pooling Layer.

For simplicity, I will call the above network as broken back prop net.

I didn’t expecting network to perform well after the change, I was just curious. However, lets take a look at the results shown below. I did three different experiments with different Hyper Parameters each time. (Only changing the dimension of last fully connected layers.)

**Result 1: Both Network have weight Dimension of W3(16*18), W4(18*20), and W5(20,1)**

Red Box → Result of Broken back prop net. 
Blue Box → Result of Original net

As seen, for the first experiment, both network have 100 accuracy on test data, however broken net had a lower cost at epoch 250.

**Result 2: Both Network have weight Dimension of W3(16*48), W4(48*56), and W5(56,1)**

Red Box → Result of Broken Net
Blue Box → Result of Original Net
Green Star → Where the network predicted wrong

This time it was interesting, broken net had a higher cost, however had 100 accuracy on test data. While original net had lower cost with some predictions being wrong on test data.

**Result 3: Forgot the Dimension of Weight (LOL) but they had the same Dimension for both Network.**

Red Box → Result of Broken Net
Blue Box → Result of Original Net

Again, both network had 100 accuracy on test data. While Broken Net having higher cost.

**Interactive Code: Original Net**

Note: The online compiler does not have “from sklearn import datasets”, so I was not able to just copy and paste the code that I used on my laptop. So I copied four training examples that represent hand written digit of 0,1,1,0 in their respective order, and adjusted the hyper parameters as well.

To [access the original net code, please click on this link.](https://repl.it/@Jae_DukDuk/Multi-Channel-and-Layer-Original)

**Interactive Code: Broken Net**

Note: The online compiler does not have “from sklearn import datasets”, so I was not able to just copy and paste the code that I used on my laptop. So I copied four training examples that represent hand written digit of 0,1,1,0 in their respective order, and adjusted the hyper parameters as well.

To [access the Broken net code, please click here.](https://repl.it/@Jae_DukDuk/Multi-Channel-and-Layer-Broken)

#### Final Words

Results like this fascinates me, and this is the reason why I do manual back propagation. Even Dr. Hinton is [suspicious of back propagation](https://www.quora.com/Why-is-Geoffrey-Hinton-suspicious-of-backpropagation-and-wants-AI-to-start-over) and wants AI to start over again. Thou I don’t know about starting over LOL, but I think it is crucial for us to gain a deeper understanding of back propagation, and try creative methods to see if other methods work as good back prop.

If any errors are found, please email me at jae.duk.seo@gmail.com.

Meanwhile follow me on my twitter [here](https://twitter.com/JaeDukSeo), and visit [my website](https://jaedukseo.me/), or my [Youtube channel](https://www.youtube.com/c/JaeDukSeo) for more content. I also did comparison of Decoupled Neural Network [here if you](https://becominghuman.ai/only-numpy-implementing-and-comparing-combination-of-google-brains-decoupled-neural-interfaces-6712e758c1af) are interested.

**References**

1. Perez, C. E. (2017, September 16). Why we should be Deeply Suspicious of BackPropagation. Retrieved January 29, 2018, from [https://medium.com/intuitionmachine/the-deeply-suspicious-nature-of-backpropagation-9bed5e2b085](https://medium.com/intuitionmachine/the-deeply-suspicious-nature-of-backpropagation-9bed5e2b085)e

2. Recognizing hand-written digits¶. (n.d.). Retrieved January 29, 2018, from [http://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html](http://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html)

3. Bhatia, R. (2018, January 18). Back-propagation: Is It The Achilles Heel Of Today’s AI. Retrieved January 29, 2018, from [https://analyticsindiamag.com/back-propagation-is-it-the-achilles-heel-of-todays-ai/](https://analyticsindiamag.com/back-propagation-is-it-the-achilles-heel-of-todays-ai/)

4. 2018. [Online]. Available: [https://www.quora.com/Why-is-Geoffrey-Hinton-suspicious-of-backpropagation-and-wants-AI-to-start-over.](https://www.quora.com/Why-is-Geoffrey-Hinton-suspicious-of-backpropagation-and-wants-AI-to-start-over.) [Accessed: 29- Jan- 2018].

#### This story is published in The Startup, Medium’s largest entrepreneurship publication followed by 298,432+ people.

#### Subscribe to receive our top stories here.

