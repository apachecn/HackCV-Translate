# Neural Network Learning Part #1: Feed-Forward



![](https://cdn-images-1.medium.com/max/1600/0*5Qx3mrCUP_lE8orl.jpg)

> This article is part of the series “Refresher Guide on Neural Network”. Its index can be found here:https://medium.com/@prakhar.verma7/refresher-guide-on-neural-networks-438c678df575

As discussed in the previous part, a neural network consists of a number of layers and each layer consists of a number of neurons. The weights are randomly initialized at the starting and with each step in the training it starts learning and the values gets updated.

Learning of neural network mainly consists of 2 parts :

1. Feed-forward pass

2. Back-propagation

In this article we will focus on feed-forward pass only.

### Feed-Forward

Feed-forward defines the output of the network. In feed-forward the data moves in only one direction, forward. It moves from the input nodes, through the hidden nodes (if any), to the output nodes. There are no cycles or loops in the network.

### Terminologies

1. **Bias:** Bias is an additional constants of each neuron which is added to the weight before it is send to the activation function. It helps the model represent patterns that do not necessarily pass through the origin. Just like weight it is learned by the model.

2. **Activation Function:**Activation function defines the output of the node given a set of****inputs. The main role of the activation function is to make the model non-linear and map the input space to a different output space.
**E.g.**: ReLU, ELU, Sigmoid, etc.

### Linear Algebra

To understand the inner working of the Neural Network a quick refresher of linear algebra is what I strongly recommend as it will really help you understand the below mathematics.

A course you can follow is : [https://www.youtube.com/watch?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab](https://www.youtube.com/watch?v=kjBOesZCoqc&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)

Article for a quick refresher: [https://towardsdatascience.com/linear-algebra-for-deep-learning-f21d7e7d7f23](https://towardsdatascience.com/linear-algebra-for-deep-learning-f21d7e7d7f23)

### Examples

#### 1. Basic Calculation

![](https://cdn-images-1.medium.com/max/1600/1*rX2lSnCCdPE2R_XOhUGqdg.png)

**z(1) = w * a(0) + ba(1) = f(z(1))**

a = activity
w = weight
b = bias
f = activation function

#### 2. Neuron with multiple inputs

Now let’s consider a neuron which has 3 inputs X1, X2 and X3 with W1, W2 and W3 weights respectively.

![](https://cdn-images-1.medium.com/max/1600/1*st2AYkdOlrDxWKlrEnrh2w.png)

**z(1) = w1*x1 + w2*x2 + w3*x3 + ba(1) = f(z(1))**

> Generally the output of a neuron can be calculated as :a = f( ∑[w(i) . x(i)] + b)

### More specific Formula

More specifically the formula can be written as:

**W(new) = Transpose[W(old)] . X + b**

X = vector of input images
W = vector of weights

### Simple Neural Network

Let us create a network with 2 layers (1 hidden + 1 output). Input layer has 2 neurons, hidden layer has 4 neurons and output 1 neuron.

![](https://cdn-images-1.medium.com/max/1600/1*u-Q84goWP7Ga4VA0zsc43Q.png)

#### Summary of the network

Total layers : 2
Input Neurons : 2
Hidden Neurons : 4
Output Neuron : 1
Total weights : 12
Total biases : 5

#### Calculation

z1 = w1.x1 + w5.x2 + b1
a1 = f(z1)

z2 = w2.x1 + w6.x2 + b2
a2 = f(z2)

z3 = w3.x1 + w7.x2 + b3
a3 = f(z3)

z4 = w4.x1 + w8.x2 + b4
a4 = f(z4)

z5 = w9. a1 + w10.a2 + w11.a3 + w12.a4 + b5
a5 = f(z5)

Thus the output of the network is:
**y = a5**

### Code



### Conclusion

And, using feed-forward, we got the output from the network. Next we will compare the network’s output with the desired output i.e.ground truth, introduce cost function, back-propagation & see how the weight gets updated.

Follow me on Medium and keep an eye on the index page which can be found here to know about the next articles:

[**Refresher Guide on Neural Networks**
A guide to refresh concepts related to neural network, its components and other conceptsmedium.com](https://medium.com/@prakhar.verma7/refresher-guide-on-neural-networks-438c678df575)[](https://medium.com/@prakhar.verma7/refresher-guide-on-neural-networks-438c678df575)

