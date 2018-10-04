## Recurrent Neural Networks

原文链接：[Recurrent Neural Networks](https://freecontent.manning.com/recurrent-neural-networks/)

Save 37% on *Machine Learning with TensorFlow.* Just enter code **fccshukla** into the discount code box at checkout at [manning.com](https://www.manning.com/).

------

![img](https://freecontent.manning.com/wp-content/uploads/Shukla_RNN_00a.png)

Back in school, I remember the sigh of relief when one of my midterm exams was made up of only true-or-false questions. I can’t be the only one that assumed half the answers would be “true” and the other half would be “false.”

I figured out answers to most of the questions, and left the rest to random guessing. I did something clever, a strategy that you might have employed as well. After counting my number of “true” answers, I realized a disproportionate amount of “false” answers were lacking. Most of my guesses were “false” to balance the distribution.

It worked. I sure felt sly in the moment. What’s this feeling of craftiness that makes us feel confident in our decisions, and how can we give a neural network the same power?

One answer to this question is to use context to answer questions. Contextual cues are important signals that can improve the performance of machine learning algorithms. For example, imagine you want to examine an English sentence and tag the part of speech of each word.

The naive approach is to individually classify each word as a “noun,” “adjective,” and so on, without acknowledging its neighboring words. Consider trying that technique on the words in *this* sentence. The word “trying” was used as a verb, but depending on the context, you can also use it as an adjective, making parts-of-speech tagging a *trying* problem.

A better approach is to consider the context. To provide context to a neural network we can use an architecture called a recurrent neural network.

**Introduction to recurrent neural networks**

To understand recurrent neural networks, let’s first look at a simple architecture shown in figure 1. It takes as input a vector *X(t)* and generates an output a vector *Y(t)*, at some time (*t)*. The circle in the middle represents the hidden layer of the network.

------

![img](https://freecontent.manning.com/wp-content/uploads/Shukla_RNN_01.png)

**Figure 1** A neural network with the input and output layer labeled as X(k) and Y(k), respectively

------

With enough input/output examples, you can learn the parameters of the network in TensorFlow. For instance, let’s refer to the input weights as a matrix Win, and the output weights as a matrix Wout. Assume there’s one hidden layer, referred to as a vector Z(t).

As shown in figure 2, the first half of the neural network is characterized by the function Z(t) = X(t) * Win, and the second half of the neural network takes the form Y(t) = Z(t) * Wout. Equivalently, if you prefer, the whole neural network is the function Y(t) = (X(t) * Win) * Wout.

------

![img](https://freecontent.manning.com/wp-content/uploads/Shukla_RNN_02.png)

**Figure 2** The hidden layer of a neural network can be thought of as a hidden representation of the data, which is encoded by the input weights and decoded by the output weights.

------

After spending nights fine-tuning the network, you probably want to start using your learned model in a real-world scenario. Typically, that implies you’ll be calling the model multiple times, maybe even repeatedly one after another, as visualized in figure 3.

------

![img](https://freecontent.manning.com/wp-content/uploads/Shukla_RNN_03.png)

**Figure 3** Often we end up running the same neural network multiple times, without using knowledge about the hidden states of the previous runs.

------

At each time *t*, when calling the learned model, this architecture doesn’t consider knowledge about the previous runs. It’s like predicting stock-market trends by only looking at data from the current day. A better idea is to exploit overarching patterns from a week’s worth or months’ worth of data.

A recurrent neural network (RNN) is different from a traditional neural network because it introduces a transition weight *W* to transfer information across time. Figure 4 shows the three weight matrices that must be learned in a RNN.

------

![img](https://freecontent.manning.com/wp-content/uploads/Shukla_RNN_04.png)

**Figure 4** A recurrent neural network architecture can use the previous states of the network to its advantage.

------

Diagrams are nice, but you’re here to get your hands dirty. Let’s get right to it! The next section shows how to use TensorFlow’s built-in RNN models. We’ll use this RNN on real world timeseries data to predict the future!

**Implementing a recurrent neural network**

As we implement the RNN, we’ll use TensorFlow to do much of the heavy lifting. You won’t need to manually build up a network as shown earlier in figure 4, because the TensorFlow library already supports some robust RNN models.

Reference For TensorFlow library information on RNN, please see https://www.tensorflow.org/tutorials/recurrent

One type of RNN model is called Long Short-Term Memory (LSTM). I admit, it’s a fun name. It means exactly what it sounds like, too: short-term patterns aren’t forgotten in the long-term.

The precise implementation detail of LSTM isn’t in the scope of this article. Trust me, a thorough inspection of the LSTM model would distract us, because there’s no definite standard yet. This is where TensorFlow comes in to the rescue. It takes care of how the model is defined, allowing you to use it out-of-the-box.

Further reading For understanding how to implement LSTM from scratch, I suggest the following explanation: https://apaszke.github.io/lstm-explained.html

Let’s begin by writing our code in a new file, called `simple_regression.py`. Import the relevant libraries, as shown in listing 1.

Listing 1 Import relevant libraries

```
 import numpy as np
 import tensorflow as tf
 from tensorflow.contrib import rnn
  
  
```

Now, define a class called `SeriesPredictor`. The constructor, as shown in listing 2, sets up model hyper-parameters, weights, and the cost function.

Listing 2 Define a class and its constructor

```
 class SeriesPredictor:
     def __init__(self, input_dim, seq_size, hidden_dim=10):
  
         self.input_dim = input_dim //#A
         self.seq_size = seq_size  //#A
         self.hidden_dim = hidden_dim  //#A
  
         self.W_out = tf.Variable(tf.random_normal([hidden_dim, 1]), name='W_out') //#B
         self.b_out = tf.Variable(tf.random_normal([1]), name='b_out')  //#B
         self.x = tf.placeholder(tf.float32, [None, seq_size, input_dim]) //#B
         self.y = tf.placeholder(tf.float32, [None, seq_size]) //#B
  
         self.cost = tf.reduce_mean(tf.square(self.model() - self.y)) //#C
         self.train_op = tf.train.AdamOptimizer().minimize(self.cost) //#C
  
         self.saver = tf.train.Saver()  //#D
  
```

\#A Hyper-parameters

\#B Weight variables and input placeholders

\#C Cost optimizer

\#D Auxiliary ops

Next, let’s use TensorFlow’s built-in RNN model called BasicLSTMCell. The hidden dimension of the cell is the dimension of the hidden state that gets passed through time. We can run this cell with data using the `rnn.dynamic_rnn` function, to retrieve outputs results. Listing 3 details how to use TensorFlow to implement a predictive model using LSTM.

Listing 3 Define the RNN model

```
     def model(self):
         """
         :param x: inputs of size [T, batch_size, input_size]
         :param W: matrix of fully-connected output layer weights
         :param b: vector of fully-connected output layer biases
         """
         cell = rnn.BasicLSTMCell(self.hidden_dim)  #A
         outputs, states = tf.nn.dynamic_rnn(cell, self.x, dtype=tf.float32) #B
         num_examples = tf.shape(self.x)[0]
         W_repeated = tf.tile(tf.expand_dims(self.W_out, 0), [num_examples, 1, 1])
           #C
         out = tf.matmul(outputs, W_repeated) + self.b_out
         out = tf.squeeze(out)
         return out
  
```

\#A Create a LSTM cell

\#B Run the cell on the input to obtain tensors for outputs and states

\#C Compute the output layer as a fully connected linear function

With a model and cost-function defined, we can now implement the training function, which learns the LSTM weights given example input/output pairs. As listing 4 shows, you open a session and repeatedly run the optimizer on the training data.

By the way You can use cross-validation to figure out how many iterations to train the model. In our case here, we assume a fixed number of epocs.

After training, save the model to file to load it later.

Listing 4 Train the model on a dataset

```
     def train(self, train_x, train_y):
         with tf.Session() as sess:
             tf.get_variable_scope().reuse_variables()
             sess.run(tf.global_variables_initializer())
             for i in range(1000):  #A
                 _, mse = sess.run([self.train_op, self.cost], feed_dict={self.x: train_x, self.y: train_y})
                 if i % 100 == 0:
                     print(i, mse)
             save_path = self.saver.save(sess, 'model.ckpt')
             print('Model saved to {}'.format(save_path))
  
```

\#A Run the train op 1000 times

Let’s say all went well, and our model has successfully learned parameters. Next, we’d like to evaluate the predictive model on other data. Listing 5 loads the saved model, and runs the model in a session by feeding in some test data. If a learned model doesn’t perform well on testing data, then we can try tweaking the number of hidden dimensions of the LSTM cell.

Listing 5 Test the learned model

```
     def test(self, test_x):
         with tf.Session() as sess:
             tf.get_variable_scope().reuse_variables()
             self.saver.restore(sess, './model.ckpt')
             output = sess.run(self.model(), feed_dict={self.x: test_x})
             print(output)
  
```

It’s done! But to convince ourselves that it works, lets make up some data and try to train the predictive model. In listing 6, we’ll create input sequences, called `train_x`, and corresponding output sequences, called `train_y`.

Listing 6 Train and test on some dummy data

```
 if __name__ == '__main__':
     predictor = SeriesPredictor(input_dim=1, seq_size=4, hidden_dim=10)
     train_x = [[[1], [2], [5], [6]],
                [[5], [7], [7], [8]],
                [[3], [4], [5], [7]]]
     train_y = [[1, 3, 7, 11],
                [5, 12, 14, 15],
                [3, 7, 9, 12]]
     predictor.train(train_x, train_y)
  
     test_x = [[[1], [2], [3], [4]],  #A
               [[4], [5], [6], [7]]]  #B
     predictor.test(test_x)
  
```

\#A predicted result should be 1, 3, 5, 7

\#B predicted result should be 4, 9, 11, 13

You can treat this predictive model as a black-box, and train it with real-world timeseries data for prediction.

Thanks for reading!