# **tensorflow**

原文链接：[tensorflow](https://github.com/nicodjimenez/nicodjimenez.github.io/blob/master/_posts/2017-10-08-tensorflow.markdown)

# Introduction

Every few months I enter the following query into Google: “Tensorflow sucks” or “f*** Tensorflow”, hoping to find like-minded folk on the internet. Unfortunately, although Tensorflow has been around for about two years, I still cannot find a bashing of Tensorflow that leaves me fully satisfied.

Although I suppose it’s possible I might be asking the wrong search engine, I think there’s a different force at work here: Google envy. The phenomenon known as “Google deep envy” is the following set of assumptions made by engineers across the world:

- People who work at Google are more intelligent and competent than yourself
- If you learn Tensorflow you could get a deep learning job at Google! (keep deep dreaming young fellow)
- If your mediocre startup uses Tensorflow and you blog about its virtues maybe Google will want to buy it
- If you don’t “get” Tensorflow's unintuitive design, you’re just dumb

Let's leave our assumptions behind us for now and give Tensorflow an honest look.

When Tensorflow first came out, we were promised an end to the endless nightmare of poorly designed or poorly maintained deep learning frameworks. (e.g. <https://github.com/BVLC/caffe/issues>). What we got instead, was the deep learning framework equivalent of Java (write once, run everywhere), but less fun to work with, and with a purely declarative paradigm. Yuck.

Where did things go wrong? In trying to build a tool to satisfy everyone’s needs, it seems that Google built a product that does a so-so job of satisfying anyone's needs.

For researchers, Tensorflow is hard to learn and hard to use. Research is all about flexibility, and lack of flexibility is baked into Tensorflow at a deep level.

Want to extract the values of intermediate layers of a neural net? You’ll need to define a graph, and then execute it with the data passed in as a dictionary, and oh don’t forget to add the intermediate layers as outputs of the graph, or else you won’t be able to retrieve their values. Ok, that hurt, but it’s doable.

Want to execute layers conditionally, such as an RNN that stops whenever an end-of-sentence (EOS) token is produced? Someone using Pytorch will be on their 3rd failed AI startup by the time you're done with that.

For machine learning practitioners such as myself, Tensorflow is not a great choice either. The declarative nature of the framework makes debugging much more difficult. The advantage of being able to run models on Android or iOS looks great until you see how big the framework binaries are (20MB+), or you try to look at the nearly non-existent C++ documentation, or you want to do any kind of conditional network execution, which is super useful in low resource situations such as mobile.

# Comparisons with other frameworks

It is true that the developers of Tensorflow are deep learning superstars. However, the original developer of Tensorflow that is probably most widely known and respected, Yangquing Jia, has recently left Google to join Facebook, where his Caffe2 project is quietly picking up steam: (<https://github.com/caffe2/caffe2/graphs/contributors>, <https://github.com/caffe2/caffe2/issues>). Unlike Tensorflow, Caffe2 allows the user to execute a layer on a piece of data in one line of code. Radical!

In addition, Pytorch is quickly developing popularity amongst top AI researchers. Torch users, although nursing RSI injuries from writing Lua code to perform simple string operations, simply aren’t deserting in droves to Tensorflow -- they are switching to Pytorch. It appears that Tensorflow is just not good enough for top AI labs. Sorry, Google.

The most interesting question to me is why Google chose a purely declarative paradigm for Tensorflow in spite of the obvious downsides of this approach. Did they feel that encapsulating all the computation in a single computation graph would simplify executing models on their TPU’s so they can cut Nvidia out of the millions of dollars to be made from cloud hosting of deep learning powered applications? It’s difficult to say. Overall, Tensorflow does not feel like a pure open source project for the common good. Which I would have no problem with, had their design been sound. In comparison with beautiful Google open source projects out there such as Protobuf, Golang, and Kubernetes, Tensorflow falls dramatically short.

While declarative paradigms are great for UI programming, there are many reasons why it is a problematic choice for deep learning.

Take the React Javascript library as an example, the standard choice today for interactive web applications. In React, the complexity of how data flows through the application makes sense to be hidden from the developer, since Javascript execution is generally orders of magnitudes faster than updates to the DOM. React developers don't want to worry about the mechanics of how state is propagated, so long as the end user experience is “good enough”.

On the other hand, in deep learning, a single layer can literally execute billions of FLOP’s! And deep learning researchers care very much about the mechanics of how computation is done and want fine control because they are constantly pushing the edge of what’s possible (e.g. dynamic networks) and want easy access to intermediate results.

# A concrete example

Let's look at a simple example of training a model to multiply its input by 3.

First, let's look at the Tensorflow example:

{% highlight python %} import tensorflow as tf import numpy as np X = tf.placeholder("float") Y = tf.placeholder("float") W = tf.Variable(np.random.random(), name="weight") pred = tf.multiply(X, W) cost = tf.reduce_sum(tf.pow(pred-Y, 2)) optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost) init = tf.global_variables_initializer() with tf.Session() as sess: sess.run(init) for t in range(10000): x = np.array(np.random.random()).reshape((1, 1, 1, 1)) y = x * 3 (_, c) = sess.run([optimizer, cost], feed_dict={X: x, Y: y}) print c {% endhighlight %}

Now let's look at a Pytorch example that does the same thing:

{% highlight python %} import numpy as np import torch from torch.autograd import Variable model = torch.nn.Linear(1, 1) loss_fn = torch.nn.MSELoss(size_average=False) optimizer = torch.optim.SGD(model.parameters(), lr=0.01) for t in range(10000): x = Variable(torch.from_numpy(np.random.random((1,1)).astype(np.float32))) y = x * 3 y_pred = model(x) loss = loss_fn(y_pred, y) optimizer.zero_grad() loss.backward() optimizer.step() print loss.data[0] {% endhighlight %}

Although the Pytorch example is one less line of code, the operations are much more explicit, and the syntax follows the actual learning process much more closely inside the training loop:

1. Forward pass of input
2. Generate loss
3. Compute gradients
4. Backprop

whereas in Tensorflow the core operation is a magic `sess.run` call.

Why would you want to write more lines of code to end up with something more difficult to understand and maintain? Pytorch's interface is objectively much better than Tensorflow's. It's not even close.

# Conclusion

With Tensorflow, Google has created a framework that is simultaneously too low level to use comfortably for rapid prototyping, yet too high level to use comfortably in cutting edge research or in production environments that are resource constrained.

Let's be honest, when you have about half a dozen open source high-level libraries out there built on top of your already high-level library to make your library usable, you know something has gone terribly wrong:

- <http://tflearn.org/>
- <https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim>
- <https://github.com/fchollet/keras>
- <https://github.com/tensorflow/skflow>

Note: I will concede that Tensorboard (Tensorflow's monitoring tool) is a really good idea. If you want a beautiful monitoring solution for your machine learning project that includes advanced model comparison features, check out Losswise ([https://losswise.com](https://losswise.com/)). I developed it to allow machine learning developers such as myself to decouple tracking their model's performance from whatever machine learning library they use, and to implement many awesome features that I wanted which Tensorboard does not provide.