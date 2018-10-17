# Introduction to Information Theory and Why You Should Care

原文链接：[Introduction to Information Theory and Why You Should Care](https://recast.ai/blog/introduction-information-theory-care/?from=hackcv&hmsr=hackcv.com&utm_medium=hackcv.com&utm_source=hackcv.com)

Let’s talk about information – what is it? Can we compress it and how much? What are the limits on the communication of information? We’ll try to answer these questions by looking at  the revolutionary work of one man – Claude E. Shannon, and the men and women who followed in his footsteps. Most importantly, we’ll try to answer the question ‘is this important for machine learning and why?’. I hope that by the time you finish reading this post you too will be convinced that information theory can be very beneficial to anybody who’s interested in developing machine learning systems. Let’s start by discussing the general way the word “information” was understood before Shannon’s work in 1948. Then we will be ready to understand the revolutionary nature of this work.

# **A Brief History of Information Theory**

In the first half of the 20th century, the world was quickly becoming more and more connected through different types of **analogue communication**. This included public radio broadcasts as well as 2-way radio communication (such as for ships or aircraft). A general understanding of the term “information” was lacking at the time, and the communication of each type of data (such as voice, pictures, film etc.) was based on its own theory and practices. However, the main understanding was: for any given communication channel, communication has two characteristics – rate (the amount of data that can be transmitted in a given period of time) and reliability. Increasing each of these would invariably come at the expense of the other. When computer science started to emerge as a field, the general understanding was that this trade-off would stay true in this case as well. (Note: in this post I chose to simplify matters by treating information theory as the science of digital communication, ignoring analogue aspects.)

Consider the following channel, called the BEC (binary erasure channel): each bit, when transmitted over the channel, can either be received on the other side error-free (with probability 1-p) or be erased (with probability p):

 

[![img](https://recast.ai/blog/wp-content/uploads/2017/09/BEC-e1506504023977.png)](https://recast.ai/blog/wp-content/uploads/2017/09/BEC.png)

*Binary Erasure Channel (BEC). Each bit is erased with probability p*

Assume that we want to transmit the bit “0”. If we send it once, there is a significant chance (p) that it won’t survive. We can try sending it twice, and reduce that probability to ![p^2](https://s0.wp.com/latex.php?latex=p%5E2&bg=ffffff&fg=000000&s=0), but we will pay a heavy price in rate – we will have to use the channel twice in order to transmit one bit of information. Of course, we can continue to reduce the probability of erasure by sending the same bit again and again, but the price in rate would continue to increase. Shannon showed that this trade-off can be broken, and we will come back to the BEC later and see how.

Let’s start exploring Shannon’s results and information theory as a whole now. To understand the math in this post, you need some basic concepts of probability theory, but don’t worry about that too much, I will explain everything intuitively.

# **Basic Concepts**

## **Important quantities and their meaning**

Most practitioners of machine learning already use some concepts from information theory, sometimes without even knowing it. Here are a few basic quantities you may already be familiar with.

### **Entropy**

The entropy of a random variable X, usually referred to as ![H(X)](https://s0.wp.com/latex.php?latex=H%28X%29&bg=ffffff&fg=000000&s=0) ,can be calculated through ![H(X) = - \sum\limits_{x \in \mathcal{X}} P_X(x)\log P_X(x)](https://s0.wp.com/latex.php?latex=H%28X%29+%3D+-+%5Csum%5Climits_%7Bx+%5Cin+%5Cmathcal%7BX%7D%7D+P_X%28x%29%5Clog+P_X%28x%29&bg=ffffff&fg=000000&s=0). The entropy can be thought of as a measure of the “mess” inherent in the variable X – given the size of the alphabet (the number of different values X can take), a uniform distribution over the alphabet maximizes the entropy, while a known value (![X = x](https://s0.wp.com/latex.php?latex=X+%3D+x&bg=ffffff&fg=000000&s=0)) gives ![H(X) = 0](https://s0.wp.com/latex.php?latex=H%28X%29+%3D+0&bg=ffffff&fg=000000&s=0). For example, consider the Bernoulli distribution, defined over an alphabet of size 2: ![P_X(0) = p, \quad P_X(1) = 1-p](https://s0.wp.com/latex.php?latex=P_X%280%29+%3D+p%2C+%5Cquad+P_X%281%29+%3D+1-p&bg=ffffff&fg=000000&s=0). The entropy is maximized for ![p = \frac{1}{2}](https://s0.wp.com/latex.php?latex=p+%3D+%5Cfrac%7B1%7D%7B2%7D&bg=ffffff&fg=000000&s=0) (“uniform” distribution) and minimized for ![p = 0](https://s0.wp.com/latex.php?latex=p+%3D+0&bg=ffffff&fg=000000&s=0) or ![p = 1](https://s0.wp.com/latex.php?latex=p+%3D+1&bg=ffffff&fg=000000&s=0) (certainty). Notice also that for discrete random variables, the entropy cannot be negative:

 

[![img](https://recast.ai/blog/wp-content/uploads/2017/09/Binary_entropy_plot.png)](https://recast.ai/blog/wp-content/uploads/2017/09/Binary_entropy_plot.png)

*Entropy of a Bernoulli random variable X, as a function of X’s probability of being 1*

 

Sometimes we want to measure the “mess” that is left after some information is already known. In order to do that, we can use the conditioned version of the entropy, defined as follows: ![H(X|Y) = - \sum\limits_{(x,y) \in \mathcal{X}\times\mathcal{Y}} P_{XY}(x,y) \log P_{X|Y}(x|y)](https://s0.wp.com/latex.php?latex=H%28X%7CY%29+%3D+-+%5Csum%5Climits_%7B%28x%2Cy%29+%5Cin+%5Cmathcal%7BX%7D%5Ctimes%5Cmathcal%7BY%7D%7D+P_%7BXY%7D%28x%2Cy%29+%5Clog+P_%7BX%7CY%7D%28x%7Cy%29&bg=ffffff&fg=000000&s=0). Here, Y represents the information we already know. Note though, that in the calculation we don’t assume the actual value of Y, instead we average over it: ![H(X|Y) = \mathbb{E}[H(X|Y=y)]](https://s0.wp.com/latex.php?latex=H%28X%7CY%29+%3D+%5Cmathbb%7BE%7D%5BH%28X%7CY%3Dy%29%5D&bg=ffffff&fg=000000&s=0).

### **Mutual information**

The mutual information is a measure defined over two random variables. It helps us gain insight about the information that one piece of data (or one random variable) carries about the other. Looking at the mathematical definition of the mutual information: ![I(X;Y) = \sum\limits_{(x,y) \in \mathcal{X}\times\mathcal{Y}} P_{XY}(x,y) \log \frac{P_{XY}(x,y)}{P_X(x)P_Y(y)}](https://s0.wp.com/latex.php?latex=I%28X%3BY%29+%3D+%5Csum%5Climits_%7B%28x%2Cy%29+%5Cin+%5Cmathcal%7BX%7D%5Ctimes%5Cmathcal%7BY%7D%7D+P_%7BXY%7D%28x%2Cy%29+%5Clog+%5Cfrac%7BP_%7BXY%7D%28x%2Cy%29%7D%7BP_X%28x%29P_Y%28y%29%7D&bg=ffffff&fg=000000&s=0), it can be seen that in fact it gives us insight about the answer to the question “how far are X and Y from being independent from each other?”. Through some simple algebra it can be shown that ![I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)](https://s0.wp.com/latex.php?latex=I%28X%3BY%29+%3D+H%28X%29+-+H%28X%7CY%29+%3D+H%28Y%29+-+H%28Y%7CX%29&bg=ffffff&fg=000000&s=0). In other words, the mutual information is the difference between the “mess” inherent in X, and the “mess” left in X after knowing Y (and the same is also true when the variables change roles). Note that while both the mutual information of two variables and their correlation give hints about their relationship, they look at the question from two very different angles. Here’s a picture of different two-dimensional scatters and the corresponding values of mutual information:

[![img](https://recast.ai/blog/wp-content/uploads/2017/09/Mutual_Information_Examples.svg_.png)](https://recast.ai/blog/wp-content/uploads/2017/09/Mutual_Information_Examples.svg_.png)

*Values of Mutual Information for different 2-variable probability distributions. In each sub-figure, the scatter represents a distribution over the variables, represented by the X and Y axis (source).*

You can think about the correlation of the X and Y axis for each of these pictures and see for yourselves that sometimes correlation and mutual information behave very differently (for example: what would be the correlation between the axis for a narrow straight line? And for a circle? You can go [here](https://commons.wikimedia.org/wiki/File:Correlation_examples.png) to find out.)

### **KL Divergence**

When we want to compare two probability distributions, ![P_X](https://s0.wp.com/latex.php?latex=P_X&bg=ffffff&fg=000000&s=0) and ![Q_X](https://s0.wp.com/latex.php?latex=Q_X&bg=ffffff&fg=000000&s=0), one way of doing so is through the Kullback-Leibler (KL) divergence (sometimes referred to as the KL distance, although it is not a mathematical distance): ![\mathcal{D}(P_X||Q_X) = \sum\limits_{x \in \mathcal{X}} P_X(x) \log \frac{P_X(x)}{Q_X(x)}](https://s0.wp.com/latex.php?latex=%5Cmathcal%7BD%7D%28P_X%7C%7CQ_X%29+%3D+%5Csum%5Climits_%7Bx+%5Cin+%5Cmathcal%7BX%7D%7D+P_X%28x%29+%5Clog+%5Cfrac%7BP_X%28x%29%7D%7BQ_X%28x%29%7D&bg=ffffff&fg=000000&s=0). This quantity turns out to be very useful in many cases. For example, it constitutes the penalty to pay in code length when trying to encode a source ![X \sim P_X](https://s0.wp.com/latex.php?latex=X+%5Csim+P_X&bg=ffffff&fg=000000&s=0) as if it was distributed according to ![Q_X](https://s0.wp.com/latex.php?latex=Q_X&bg=ffffff&fg=000000&s=0). One of the reasons the KL divergence is so useful is the property ![\mathcal{D}(P_X||Q_X) \geq 0](https://s0.wp.com/latex.php?latex=%5Cmathcal%7BD%7D%28P_X%7C%7CQ_X%29+%5Cgeq+0&bg=ffffff&fg=000000&s=0), with equality if and only if ![P_X \equiv Q_X](https://s0.wp.com/latex.php?latex=P_X+%5Cequiv+Q_X&bg=ffffff&fg=000000&s=0). Note however, that ![\mathcal{D}(P_X||Q_X) \neq \mathcal{D}(Q_X||P_X)](https://s0.wp.com/latex.php?latex=%5Cmathcal%7BD%7D%28P_X%7C%7CQ_X%29+%5Cneq+%5Cmathcal%7BD%7D%28Q_X%7C%7CP_X%29&bg=ffffff&fg=000000&s=0), which is why the KL divergence is not a mathematical distance. Also, the mutual information is in fact just a private case of the KL divergence, where we compare the joint distribution of X and Y with their product distribution: ![I(X;Y) = \mathcal{D}(P_{XY}||P_XP_Y)](https://s0.wp.com/latex.php?latex=I%28X%3BY%29+%3D+%5Cmathcal%7BD%7D%28P_%7BXY%7D%7C%7CP_XP_Y%29&bg=ffffff&fg=000000&s=0).

If you are interested in digging deeper into the KL divergence, as well as other quantities mentioned above and more, take a look at [this blog post](http://colah.github.io/posts/2015-09-Visual-Information/). It contains neat visualisations and explains the relation of all of this to coding theory.

## **Let’s play**

Now that we know the three most basic quantities in information theory, let’s have some fun with them! Here are some important concepts to remember:

### **Conditioning reduces entropy**

![H(X|Y) \leq H(X)](https://s0.wp.com/latex.php?latex=H%28X%7CY%29+%5Cleq+H%28X%29&bg=ffffff&fg=000000&s=0). This concept is very intuitive – the “mess” in X when Y is known cannot be bigger than the “mess” in X without knowing Y. If that were the case, we could always just “ignore” our knowledge of Y! As for the proof,we already have all the tools to do it. The mutual information between X and Y is just the KL divergence between ![P_{XY}](https://s0.wp.com/latex.php?latex=P_%7BXY%7D&bg=ffffff&fg=000000&s=0) and ![P_XP_Y](https://s0.wp.com/latex.php?latex=P_XP_Y&bg=ffffff&fg=000000&s=0). As such, ![I(X;Y) = H(X) - H(X|Y) \geq 0](https://s0.wp.com/latex.php?latex=I%28X%3BY%29+%3D+H%28X%29+-+H%28X%7CY%29+%5Cgeq+0&bg=ffffff&fg=000000&s=0) and thus ![H(X) \geq H(X|Y)](https://s0.wp.com/latex.php?latex=H%28X%29+%5Cgeq+H%28X%7CY%29&bg=ffffff&fg=000000&s=0).

### **Chain rule**

The chain rule is quite easy to prove through basic logarithmic properties – try it! It tells us that the joint “mess” in X and Y is exactly equal to the “mess” in X in addition to the mess in Y,when X is already known: ![H(X,Y) = H(X) + H(Y|X)](https://s0.wp.com/latex.php?latex=H%28X%2CY%29+%3D+H%28X%29+%2B+H%28Y%7CX%29&bg=ffffff&fg=000000&s=0).

### **Data Processing Theorem**

If you only remember one thing from this blog post, I hope this is it. Let’s start with the mathematical formulation and leave the explanation for later:

Let ![X \leftrightarrow Y \leftrightarrow Z](https://s0.wp.com/latex.php?latex=X+%5Cleftrightarrow+Y+%5Cleftrightarrow+Z&bg=ffffff&fg=000000&s=0) form a Markov chain, in that order. Then ![I(X;Z) \leq \min [I(X;Y), I(Y;Z)]](https://s0.wp.com/latex.php?latex=I%28X%3BZ%29+%5Cleq+%5Cmin+%5BI%28X%3BY%29%2C+I%28Y%3BZ%29%5D&bg=ffffff&fg=000000&s=0).

But what does that mean? We say that ![X \leftrightarrow Y \leftrightarrow Z](https://s0.wp.com/latex.php?latex=X+%5Cleftrightarrow+Y+%5Cleftrightarrow+Z&bg=ffffff&fg=000000&s=0) form a Markov chain, if Y is the only “connection” between X and Z. In other words, given Y, knowing also Z doesn’t give any additional information about X. In that case, the mutual information between X and Z is smaller than both ![I(X;Y)](https://s0.wp.com/latex.php?latex=I%28X%3BY%29&bg=ffffff&fg=000000&s=0) and ![I(Y;Z)](https://s0.wp.com/latex.php?latex=I%28Y%3BZ%29&bg=ffffff&fg=000000&s=0). An interesting private case of this theorem is to consider ![X = f(Y)](https://s0.wp.com/latex.php?latex=X+%3D+f%28Y%29&bg=ffffff&fg=000000&s=0). In this case,![X \leftrightarrow Y \leftrightarrow Z](https://s0.wp.com/latex.php?latex=X+%5Cleftrightarrow+Y+%5Cleftrightarrow+Z&bg=ffffff&fg=000000&s=0) do form a Markov chain, and the significance of this is that by processing data, we can never gain information about a hidden quality. For example, assume that Y represents possible pictures in a dataset and Z represents the main object in the picture (the alphabet of Z is thus “cat”, “dog”, “ball”, “car” etc.). By processing Y (for example through a convolutional neural network) it is **impossible** to create new information about Z, all we can hope for is to lose as little as possible. How come convolutional neural networks work so well then, you may ask? It is because they can **extract** the important information from the picture in order to make the classification, but they can never **create** it.

### **Asymptotic Equipartition Property**

One could argue that this property is single handedly responsible for making the magic happen in information theory. It tells us that given a large series of independent and identically distributed (i.i.d.) experiments, ![X_1, X_2, \ldots X_n \sim P_X(x)](https://s0.wp.com/latex.php?latex=X_1%2C+X_2%2C+%5Cldots+X_n+%5Csim+P_X%28x%29&bg=ffffff&fg=000000&s=0), their empirical entropy will be very close to the theoretical entropy of ![P_X](https://s0.wp.com/latex.php?latex=P_X&bg=ffffff&fg=000000&s=0): ![-\frac{1}{n}\log P_X(X_1,X_2, \ldots X_n) \to H(X)](https://s0.wp.com/latex.php?latex=-%5Cfrac%7B1%7D%7Bn%7D%5Clog+P_X%28X_1%2CX_2%2C+%5Cldots+X_n%29+%5Cto+H%28X%29&bg=ffffff&fg=000000&s=0) (in probability). This means that, given a large series of such experiments, all of the probability will be divided only between a relatively small set of sequences, which we call **typical**. Let’s take an example: Imagine a **non-fair** coin, that lands on heads with probability 0.7. Throwing this coin 100 times, clearly the probability of any specific sequence is very small. Nevertheless, almost all of the probability will be inside the set of results that have about (but not necessarily exactly) 70 heads and 30 tails. If we continue and throw the coin 1000 times, the probability of the set that contains only results with about 700 heads would be even closer to 1. Implementing a small python code of 1000 such experiments, each with 1000 coin tosses, only one experiment resulted in a number of heads not between 650 and 750:

```
import random

experiments = 1000 # number of complete experiments
toss_per_experiment = 1000  #number of tosses per experiment
delta = 50 # the tolerance
prob_heads = 0.7 # the probability the coin lands on heads
count = 0

for exper in range(experiments):
  heads = 0
  for toss in range(toss_per_experiment):
    if random.random() <= prob_heads:
      heads += 1
  if (heads <= toss_per_experiment * prob_heads - delta) or (heads >= toss_per_experiment * prob_heads + delta):
    count += 1

print(count)
```

I invite you to play with the parameters yourselves – try to separately change the number of experiments, the tosses per experiment and the tolerance ‘delta’ and see what happens.

# **Capacity**

The original paper by Shannon from 1948 is packed full of important and interesting results (go take a look yourself, it is very pleasantly written. See a full citation and a link at the final section of this post). Arguably the two most important results in this paper can be referred to as the **Source Coding Theorem** and the **Channel Coding Theorem**.

## **Source Coding Theorem**

n random variables ![X_1, X_2, \ldots, X_n](https://s0.wp.com/latex.php?latex=X_1%2C+X_2%2C+%5Cldots%2C+X_n&bg=ffffff&fg=000000&s=0), all independently distributed by ![P_X(x)](https://s0.wp.com/latex.php?latex=P_X%28x%29&bg=ffffff&fg=000000&s=0), can be compressed into any number of bits that is strictly larger than ![nH(X)](https://s0.wp.com/latex.php?latex=nH%28X%29&bg=ffffff&fg=000000&s=0) with negligible risk of information loss as ![n \to \infty](https://s0.wp.com/latex.php?latex=n+%5Cto+%5Cinfty&bg=ffffff&fg=000000&s=0).

But how can this be achieved, from a theoretical point of view? Well, the asymptotic equipartition property tells us that all of the probability is in the typical set. Thus, it makes no sense to distribute codewords in our code to sequences that are **not** in this set. And what is the size of the typical set? It contains about ![2^{nH(X)}](https://s0.wp.com/latex.php?latex=2%5E%7BnH%28X%29%7D&bg=ffffff&fg=000000&s=0) sequences, so it makes sense that we would not need more than about ![nH(X)](https://s0.wp.com/latex.php?latex=nH%28X%29&bg=ffffff&fg=000000&s=0) bits in order to represent these sequences, and them only!

This result takes us back to the intuitive explanation about the nature of the entropy: For each of the random variables ![X_1, X_2, \ldots, X_n](https://s0.wp.com/latex.php?latex=X_1%2C+X_2%2C+%5Cldots%2C+X_n&bg=ffffff&fg=000000&s=0), the “mess” inherent in it is ![H(X)](https://s0.wp.com/latex.php?latex=H%28X%29&bg=ffffff&fg=000000&s=0). Doesn’t it make sense then, that in order to compress the result of each of these “experiments”, all we need to do is to dissipate the inherent “mess”? Note however that this is only true in the limit where n is very large – if we only want to compress the result of a few experiments we may need more resources.

## **Channel Coding Theorem**

The noisy channel coding theorem states that any communication channel has a capacity – a maximum rate of communication (in bits per channel use, for example) that can be transmitted on the channel reliably (with the probability of error being as low as we want it to be!). This is true for **any channel**, no matter how noisy it may be. Of course, if the channel does not let the signal pass at all, the capacity is zero. Let’s take a simple example: Consider the **binary erasure channel** (BEC) that we have already seen:

[![img](https://recast.ai/blog/wp-content/uploads/2017/09/BEC-e1506504023977.png)](https://recast.ai/blog/wp-content/uploads/2017/09/BEC.png)

*Binary Erasure Channel. Despite erasures, reliable communication is possible under the channel capacity*

Clearly, sending n bits (and assuming n is large), we can assume that about ![n(1-p)](https://s0.wp.com/latex.php?latex=n%281-p%29&bg=ffffff&fg=000000&s=0) of these bits would arrive safely. Unfortunately, if we choose this “blind” strategy, we cannot know in advance which of the bits would be dropped. Amazingly enough, the capacity of this channel is in fact ![1-p](https://s0.wp.com/latex.php?latex=1-p&bg=ffffff&fg=000000&s=0)! This means that if, for n channel uses, we are willing to contend with only transmitting ![n(1-p)](https://s0.wp.com/latex.php?latex=n%281-p%29&bg=ffffff&fg=000000&s=0) bits of information instead of the n bits we can physically “push” into the channel, we can guarantee that these ![n(1-p)](https://s0.wp.com/latex.php?latex=n%281-p%29&bg=ffffff&fg=000000&s=0) bits will arrive safely to the other side! It is important, however, to remember the big difference between the amount of information (measured in bits) and the number of bits on the channel, which is equal to the number of channel uses – We would still send n bits on the channel during the communication, but the information embedded in them would only be equivalent to ![n(1-p)](https://s0.wp.com/latex.php?latex=n%281-p%29&bg=ffffff&fg=000000&s=0) bits. These ![n(1-p)](https://s0.wp.com/latex.php?latex=n%281-p%29&bg=ffffff&fg=000000&s=0) bits of information, however, would arrive safely to the other side.

# **Information Theory in Machine Learning (or: Why Should I Care?)**

Congratulations on making it all the way down here! I hope I succeeded in my mission to convey these principles in an intuitive way, and that the math that I did have to include was understandable enough. Your prize for getting here is to find out – why does all of this matter for machine learning?

Before anything else, in my opinion the intuition and basic understanding of concepts that comes with knowing a little about information theory is the most valuable lesson to take into the world of machine learning. Looking at any ML problem as the problem of creating a **channel** from the original data-point at the input to an answer at the output, for which the **information** it conveys about the desired property of the data is maximized, would allow you  to look at many different problems from a different angle than usual. The quantities presented above also turn out to be helpful in many different situations. Consider for example a situation where we would like to compare two unsupervised clustering algorithms over the same data, in order to check if they give similar results or not. Why not use the mutual information between the results of each of the algorithms, where X is the random variable that represents the cluster chosen for any data point by the first algorithm and Y represents the second? Another option is to use the conditioned entropy (![H(X|Y)](https://s0.wp.com/latex.php?latex=H%28X%7CY%29&bg=ffffff&fg=000000&s=0) or ![H(Y|X)](https://s0.wp.com/latex.php?latex=H%28Y%7CX%29&bg=ffffff&fg=000000&s=0)) in order to see how much “mess” is left when guessing the result of clustering by one algorithm, while the result of clustering by the other is already known (these two methods of comparison are very similar but there is a difference, can you spot it?).

In the remainder of this section, let’s try to take a closer look at some interesting results in ML, that were the direct result of a connection with information theory:

## **Training a Decision Tree**

One of the most popular algorithms for training decision trees is based on the principle of maximizing the **information gain**. Although given a different name, the information gain that corresponds to each feature is exactly the mutual information between that feature and the labels of the data points (You can go see for yourself [here](https://en.wikipedia.org/wiki/Decision_tree_learning)). This actually makes a lot of sense – when trying to decide which is the best feature to split the tree on, why not choose the one that gives the most information about the result? In other words, why not choose the one that, after splitting, would dissipate as much of the “mess” as possible? What happens after the split? How do we continue building the tree? Each node after the split can be represented by a new random variable, and we can start the whole process again for each of the resulting nodes.

Considering this process, we may also be able to gain some understanding about another very important issue – regularization. Decision trees are one of the models that requires the most regularization, as experience tells us that completely “free” trees would overfit almost every time. Let’s consider this issue through the data processing theorem: What we wish to do is to increase the mutual information between the actual label of a data point (let’s call the label Z to stay consistent with the data processing theorem above) and the predicted label X. Unfortunately, in order to predict the label, we can only use the attributes Y. We do our best by increasing the mutual information between X and Y, but according to the theorem this does not guarantee an increase also in ![I(X;Z)](https://s0.wp.com/latex.php?latex=I%28X%3BZ%29&bg=ffffff&fg=000000&s=0). ![I(X;Y)](https://s0.wp.com/latex.php?latex=I%28X%3BY%29&bg=ffffff&fg=000000&s=0), you may remember, only constitutes an **upper bound** over ![I(X;Z)](https://s0.wp.com/latex.php?latex=I%28X%3BZ%29&bg=ffffff&fg=000000&s=0). Thus, in the first steps, the information gain is significant and there’s a good chance that it contributes, at least in part, to a gain also in ![I(X;Z)](https://s0.wp.com/latex.php?latex=I%28X%3BZ%29&bg=ffffff&fg=000000&s=0), which is what we really want. As the information gain becomes less and less significant as the tree grows, the hope of increasing ![I(X;Z)](https://s0.wp.com/latex.php?latex=I%28X%3BZ%29&bg=ffffff&fg=000000&s=0) diminishes and instead all we get is overfitting to the data. Hence, it is better to truncate the tree when the information gain becomes insignificant.

## **Clustering by Compression**

Another interesting example is that of clustering by compression. The main idea here is to use popular compression algorithms, like the ones responsible for ZIP, RAR, Gif and more, that present good performance especially for files that have repetitive features, in order to determine which category a data point belongs to. We do so by appending the new data point to each of the files representing the classes, and choosing the one that is most performant in compressing the data point.

This approach takes advantage of the Lempel Ziv (LZ) family of compression algorithms, (see for example [here](https://en.wikipedia.org/wiki/LZ77_and_LZ78)). While these algorithms come in many different variations, the main idea stays the same: Going over the document to be compressed from beginning to end, at each point known passages are encoded through a reference to a previous appearance, while completely new information is added to the “dictionary”, in order to be available for use when a similar segment is encountered again. The way this dictionary is created and managed may differ between members of the LZ family of algorithms, but the main idea stays the same.

Using this type of algorithm to compress a file, it is clear that the type of documents that would benefit the most out of this type of compression are **long, repetitive documents.** That is because the Source Coding Theorem tells us that lossless compression is bounded from below by ![nH(X)](https://s0.wp.com/latex.php?latex=nH%28X%29&bg=ffffff&fg=000000&s=0) and the entropy, which represents mess, is much bigger for random files than it is for repetitive ones. For these repetitive documents, the algorithm would have enough “time” to learn the patterns in them, and then use them again and again in order to save the information in an efficient manner.

It turns out that the fact that similarities in a file make for good compression can be used for classification. For supervised classification (where the classes exist and contain a significant amount of data as “examples”), a new data-point can be appended to any of the existing files (where each file represents a “class”), and the declared class for the data-point is the one that was successful in compressing the data-point the most, relative to its original size (in bits). Note that since we append the data-point to the end of each of the documents, all the information in the existing documents should already exist in the respective “dictionaries” when the compression algorithm reaches the new data-point. Thus, if there are similarities between any of the existing documents and the new data-point, they will be automatically used in order to create good compression.

Considering unsupervised classification (or **clustering**), a similar approach can still be helpful. Using the level of “successfulness” of joint compression of different combinations of data-points, clusters can be created such as the data-points within each class compress well together. Of course, in this case there are some more questions to answer, mainly having to do with the vast amount of combinations to test and the complexity of the final clustering algorithm, but these problems can be addressed, as was done for example in [this very complete work](https://arxiv.org/abs/cs/0312044). The advantage of this clustering approach is that there is no need to predefine the characteristics to be used. Taking for example the problem of the clustering of music files, other approaches would require us to first define and extract different characteristics of the files, such as beat, pitch, name of artist and so on. Here, all we need to do is to check which files compress well together. It is important to remember, however, the **No Free Lunch Lemma**: The whole magic here is contained in the compression process, thus understanding the specific compression algorithm chosen is imperative. How is the dictionary built? How is it used? What is a “long enough” document to be compressed by it? etc. These specificities can determine the type of similarities the compression is susceptible to use, and thus the characteristics that control the clusterization.

## [![🤖](https://s.w.org/images/core/emoji/11/svg/1f916.svg) Happy bot building ![🤖](https://s.w.org/images/core/emoji/11/svg/1f916.svg)](https://recast.ai/blog/build-your-first-bot-with-recast-ai/?utm_source=blog&utm_medium=article)

# **Where Can I Learn More?**

The world of Information Theory is vast and there’s a lot to learn. If this post gave you the desire to do so (and I hope it did), here are some good places to start:

If you are interested in the biography and achievements of Claude E. Shannon, I invite you to take a look at this blog post, which I found to be very interesting: <https://www.technologyreview.com/s/401112/claude-shannon-reluctant-father-of-the-digital-age/>

In addition, you can find a newly published biography of Shannon here: <https://www.amazon.com/Mind-Play-Shannon-Invented-Information/dp/1476766681/ref=sr_1_1?ie=UTF8&qid=1503066163&sr=8-1&keywords=a+mind+at+play>

If it is Information Theory itself that interests you, I invite you to start from Shannon’s original paper, which is quite comprehensive and also easy to read:

- C. E. Shannon, “A mathematical theory of communication,” in *The Bell System Technical Journal*, vol. 27, no. 3, pp. 379-423, July 1948.
  <http://math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf>

In addition, there are a few books that are also a good place to start. Most students start with one of these, as far as I know:

- Cover, T. M., & Thomas, J. A. (2012). *Elements of information theory*. John Wiley & Sons.
- Gallager, R. G. (1968). *Information theory and reliable communication* (Vol. 2). New York: Wiley.

A book that is a bit harder to start with but may be worth the work for its mathematical completeness is the following one:

- Csiszar, I., & Körner, J. (2011). *Information theory: coding theorems for discrete memoryless systems*. Cambridge University Press.

If you are specifically interested in Lempel-Ziv compression, there are a lot of papers you could start with. Try this one:

- Ziv, J., & Lempel, A. (1977). A universal algorithm for sequential data compression. *IEEE Transactions on information theory*, *23*(3), 337-343.

Other resources are included in the body of this post, and of course the world of information theory is rich in interesting results and you can probably find what your heart desires quite easily. Have fun and thank you for reading.

#### Want to build your own conversational bot? Get started with Recast.AI !

![img](https://recast.ai/wp-content/uploads/2017/03/bouton-e1490803506575.png)