## 循环神经网络

原文链接：[Recurrent Neural Networks](https://freecontent.manning.com/recurrent-neural-networks/)

 只需结账时在[manning.com](https://www.manning.com/)的优惠码上输入"fccshukla"，[使用TensorFlow进行机器学习](https://www.manning.com/books/machine-learning-with-tensorflow)就可以优惠37%。

------

![img](https://freecontent.manning.com/wp-content/uploads/Shukla_RNN_00a.png)

我记得当我在学校的时候，有一场期中考试只是由判断题组成，我松了一口气。我不是唯一假设一半的选项选择“正确”，一半的选“错误”的人。

我从大部分的答案中用概率来推测答案，我做了你可能觉得很棒的机智策略。在数完我“正确“的答案后，我发现我”错误“的答案有点少，为了保持平衡，其他剩下我的答案我就蒙”错误“。

这非常有效，在这事我觉得自己很狡猾。这种感觉让我们对决策充满信心，我们怎样才可以赋予神经网络同样能力呢？

解决这个问题的一个方案是通过上下文来回答问题，上下文线索是提高机器学习算法性能的重要信号。比如，想象一下如果你想检查一个英语句子并标记每个单词的词性。

天真的方法使把每个单词分类为名词、形容词等，而且不考虑它相邻的单词，考虑一下如果在这个句子里尝试标注每个单词，"trying"一般是动词，但是当它放入语境中，你也可以用作形容词，使词性标注称为一个难题。

一个更好的方法就是考虑上下文，为了给神经网络提供上下文，我们可以使用一种称为神经网络的架构。

**介绍循环神经网络**

为了理解循环神经网络，我们先看一个简单的架构，图一。 以*X(t)*的速度输入，输出一个*Y(t)*的速度，在某一时刻 (*t)*，中间的圆圈代表网络的隐藏层。

------

![img](https://freecontent.manning.com/wp-content/uploads/Shukla_RNN_01.png)

**Figure 1** 输入输出层分别标记为 X(k) 、 Y(k)的神经网络

------

有足够的输入输出的示例，你可以了解TensorFlow中的网络参数。比如，我们将输入权值称为矩阵Win，输出权值称为矩阵Wout，假设有个隐藏层，称为向量 Z(t)。

如图二所示，神经网络前半部特征为函数 Z(t) = X(t) * Win, 后半部特征为 Y(t) = Z(t) * Wout. 同样的，如果你愿意，整个神经网络就可以表示为函数 Y(t) = (X(t) * Win) * Wout.

------

![img](https://freecontent.manning.com/wp-content/uploads/Shukla_RNN_02.png)

**Figure 2** 神经网络的隐藏层可以看作使数据的隐表示，由输入权值进行编码，输出权值进行解码

------

在花了几个晚上对网络进行微调后，你可能希望开始学习在实际生活场景中能运用到的模型。通常，这意味着，你会多次调用模型，甚至一个接着一个的重复调用，如图三所示。

------

![img](https://freecontent.manning.com/wp-content/uploads/Shukla_RNN_03.png)

**Figure 3** 通常我们最终会多次运行相同的神经网络，而不会使用有关先前运行的隐藏状态的知识。

------

在每一个时刻 *t*,当调用神经网络模型的时候，这个架构不会考虑我们之前的知识。就像我们我们预测股市趋势只看当天的数据。一个更好的想法使利用一周或数月的数据挖掘出总体模式。

一个循环神经网络不同于传统的神经网络，因为它引用了一个过渡权W来跨时间传递信息，图四展示了RNN中必须学习的三个权重矩阵。

------

![img](https://freecontent.manning.com/wp-content/uploads/Shukla_RNN_04.png)

**Figure 4** 循环神经网络体系结构可以充分利用网络的原有状态。

------



下一部分展示如何使用TensorFlow来构建RNN模型，我们将用现实世界的时间序列数据来使用此RNN去预测未来！

**实现循环神经网络**

我们使用TensorFlow来完成大部分繁重的工作来实现循环神经网络。你不需要如图4那样手动构建网络，TensorFlow的库已经提供强大的RNN的模型。

请参见TensorFlow的RNN库的信息，请访问 https://www.tensorflow.org/tutorials/recurrent

RNN的一种类型被称为长短期记忆(LSTM),这是一个很有趣的名称，意思实质上就和它的名称一样，从长远来看短期记忆不会被遗忘。

关于LSTM具体实现细节不在这篇文章的讨论范围内，相信我，对LSTM的彻底分析会分散我们的注意力，因为现在还没有明确的标准。这个时候TensorFlow来拯救我们了，它负责定义模型的方式，允许你开箱使用。

为了进一步理解如何从实现LSTM，我建议看如下的解释： https://apaszke.github.io/lstm-explained.html

我们开始在一个 `simple_regression.py`的新文件中写我们的代码。导入依赖包，如listing 1所示。

Listing 1 导入依赖包

```
 import numpy as np
 import tensorflow as tf
 from tensorflow.contrib import rnn
  
  
```

现在定义一个class叫`SeriesPredictor`. 结构如listing 2所示，设置超参数、权值和成本函数。

Listing 2 定义一个class和它的结构

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

\#A 超参数

\#B 权值变量和输入占位符

\#C  成本优化

\#D 辅助操作

接下来，我们使用TensorFlow的内置RNN模型BasicLSTMCell。cell的隐藏维度是通过时间传递的隐藏状态的维度。我们可以使用` rnn.dynamic_rnn` 函数来运行带有数据的cell，以检索出结果 Listing 3描述了如何通过TensorFlow使用LSTM实现预测模型。

Listing 3 定义RNN模型

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

\#A 创建 LSTM cell

\#B 在输入上运行cell以获取输出和状态

\#C 将输出层计算为完全连接的线性函数

定义完模型和成本函数，我们现在可以实现训练函数，通过例子的输入输出可以学习LSTM的权值，如listing 4所示，打开会话并在训练数据上重复运行优化程序。

顺便一说，你可以使用交叉验证去算出训练模型迭代了多少次，在我们的例子，我们假设有一定数量的周期。

训练完后，保存模型为文件，以便以后加载。

Listing 4 在数据集上训练模型

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

\#A 运行训练1000次

让我们说一切顺利，我们的模型已成功学习参数。接下来，我们想评估其他数据的预测模型。Listting 5加载之前保存的模型，在一个会话中和一些测试数据运行模型。如果学习模型在这些测试数据中测试的结果不理想，我们可以尝试调整LSTM cell的隐藏维度。

Listing 5 测试学习模型

```
     def test(self, test_x):
         with tf.Session() as sess:
             tf.get_variable_scope().reuse_variables()
             self.saver.restore(sess, './model.ckpt')
             output = sess.run(self.model(), feed_dict={self.x: test_x})
             print(output)
  
```

已经完成！但是为了让我们相信它有效，我们可以编制一些数据并尝试训练预测模型，在listing 6中，我们将创建名为`train_x`的输入序列和相应的输出序列 `train_y`.

Listing 6 训练和测试一些虚拟数据

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

\#A 预测的结果应该为 1, 3, 5, 7

\#B 预测的结果应该为 4, 9, 11, 13

您可以将此预测模型视为黑盒，并使用实际时间序列数据对其进行预测。

感谢阅读！