# IOS中的计算机视觉——目标检测

原文链接：[COMPUTER VISION IN IOS – OBJECT DETECTION](https://sriraghu.com/2017/07/12/computer-vision-in-ios-object-detection/?from=hackcv&hmsr=hackcv.com&utm_medium=hackcv.com&utm_source=hackcv.com)

## 什么是目标检测？

在之前，我写了一篇叫做“目标识别”的博客，介绍了如何用iPhone去实现它。（IOS中的计算机视觉——目标检测）那么现在，我写这篇博客是要介绍些什么呢？在这篇文章中，我将会探讨目标检测。

在目标识别的问题中，我们通过训练深度神经网络去识别在一张图片中是否有目标物品的出现。在目标检测中，深度神经网络不仅可以识别图片中有什么物体，还可以检测到图片当中目标的具体位置。如果咪表检测可以被应用于实时运行中，那么在自动驾驶方面的很多问题都可以很容易的解决。

在这篇文章中，让我们快速的了解以下我们如何使用Apple的机器学习框架CoreML，以及如何在iPhone7中实现目标检测的app。我们将从熟悉CoreML的相关工具开始，虽然这一开始会使你感到有很多困惑，但是你只要挺过去了就会有收获。

Apple给我们提供了很好的工具，使我们能够方便的将任意模型从我们选择的库中转换为CoreML模型。所以，在我开始正式介绍这篇博客的主要内容前，我想先感谢CoreML的团队，因为他们使我将一个本来我以为得花一年时间做的项目，变成了一周完成。

这篇博客的主要内容是实时运行目标检测（再IPHONE7上）并可以嵌入到驾驶应用程序中。所以，当我说我想要检测可能出现我车前的物体时，可能是汽车、行人、卡车等。而现在我只想检测其中的汽车。多亏了Udacity的自驾车微学位的开设，因为这个有很多微学位的项目在github上为开源项目。

目标检测在深度学习的探索中吸引了很大的注意力，所以有很多的论文在探讨这个话题。然而一些论文仅仅关注于准确性，一些论文集中于实时运行的表现。如果你想要进一步了解这个领域，你可以了解下列提及的方面的论文： R-CNN, Fast R-CNN, Faster R-CNN, YOLO, YOLO-9000, SSD, MobileNet SSD。

### 什么是最好的神经网络？

我在上文提到了7种不同的神经网络。那么上列的哪个神经网络最好实现？🤔这是个困难的问题，尤其是当你不想浪费你的时间用暴力方法用实现每个神经网络观察它的结果。所以，让我们对选择最好的神经网络实现做一些分析。在目标检测的过程中究竟发生了什么？

- **预处理**：从相机中获取图片，在将其放入神经网络前先进行一些图像处理（缩放和调整大小）。
- **处理**：这是很重要的一步。我们将预处理后的图片放入CoreML的模型中，然后得到它生成的结果。
- **后处理**：由CoreML模型生成的结果将在MLMultiArray中，我们需要对数组做一定的处理去得到目标的位置，这就是他的预测精度。

当我使用移动手机时，我应该会关注于实时运行的神经网络。所以，我可以停止考虑那些在GPU配备的机器上不在正确的FPS上执行的神经网络。

### 规则1：

如果一个神经网络可以在电脑（GPU或者CPU)上实时运行,那么这个神经网络值得一试。

这个规则可以在上面的选择中将R-CNN和Fast-RCNN去除了。虽然现在还有5种神经网络，但是他们可以划分为以下几类：Faster R-CNN,YOLO和SSD。和Fsater R-CNN比起来，YOLO和SSD在实时运行方面都表现的更好。（比较这些模型的运行时间。）现在我们只需要在两个中做选择：是YOLO还是SSD呢。我在Apple的市场上Coremltools版本是3.0的时候开始做这个目标识别的项目。这些还没有广泛适用的文档，这被支持Keras 1.2.2以及Caffe v1.0。CoreML的团队持续的更新着coremltools，目前它已经到了Keras 2.0.4支持的v4.0,所以我应该选择一个神经网络有着简单的层（只有卷积），而不是繁琐的操作，如去卷积、扩张卷积和深度卷积。这样分析一下，YOLO战胜了我个人最喜欢的SSD。

### YOLO – You Only Look Once:

在这篇文章中，我将会实现Tiny YOLO v1.0,我在将来有时间还会实现YOLO v2.0。让我们先熟悉下我们将要使用的Tiny YOLO v1包含9个卷积层而后接3个全连接层 ，包含4500万个参数。

![mode_yolo_plot](https://sriraghublog.files.wordpress.com/2017/07/mode_yolo_plot.jpg?w=700)

这和只有1500万参数的Tiny Yolo v2.0相比它显得有些大。这个神经网络的输入是一个448×448×3RGB的图像，而输出是一个长度为1470的矢量。该矢量可以划分为3个部分：有一定可能性的、确定的、箱形坐标系。这三个部分的内部又可以划分为49个小的个体对应于图像上的每个单元的检测。

![net_output](https://sriraghublog.files.wordpress.com/2017/07/net_output.png?w=700)

理论就说到这里了，现在让我们动手写代码。

**事先准备**：如果你是第一次读这篇博客，请先看我先前的这篇博客—[Computer Vision in iOS – CoreML+Keras+MNIST](https://sriraghu.com/2017/07/06/computer-vision-in-ios-coremlkerasmnist/) —在你的Mac上准备好工作环境。因为训练一个YOLO网络需要耗费大量的时间和努力，所以我们将使用预先训练的权重来设计CoreML Tiny YOLO v1模型。你可以在这个[链接](https://drive.google.com/file/d/0B1tW VtY7onibmdQWE1zVERxcjQ/view)下载权重。

在你从上面的链接下载了权重之后，新创建一个随意命名的主目录，并将您下载的权重文件移到该目录下。

- 让我们引用一些必要的库

```
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import cv2

import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Flatten, Dense, Activation, Reshape
```

- 定义 Tiny YOLO v1.0模型

```
def yolo(shape):
    model = Sequential()
    model.add(Conv2D(16,(3,3),strides=(1,1),input_shape=shape,padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32,(3,3),padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
    model.add(Conv2D(64,(3,3),padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
    model.add(Conv2D(128,(3,3),padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
    model.add(Conv2D(256,(3,3),padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
    model.add(Conv2D(512,(3,3),padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
    model.add(Conv2D(1024,(3,3),padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(1024,(3,3),padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(1024,(3,3),padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dense(4096))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(1470))
    return model
```

- 现在让我们写一个有用的方法把我们下载的“yolo-tiny.weights”文件中的权重加载到模型中。

```
# Helper function to load weights from weights-file into YOLO model
def load_weights(model,yolo_weight_file):
    data = np.fromfile(yolo_weight_file,np.float32)
    data=data[4:]

    index = 0
    for layer in model.layers:
        shape = [w.shape for w in layer.get_weights()]
        if shape != []:
            kshape,bshape = shape
            bia = data[index:index+np.prod(bshape)].reshape(bshape)
            index += np.prod(bshape)
            ker = data[index:index+np.prod(kshape)].reshape(kshape)
            index += np.prod(kshape)
            layer.set_weights([ker,bia])
```

- 我们使用的预先训练好的权重是以Theano作为图像尺寸处理的后端。所以，我们必须要将Theano设置为后端，然后再把权重值加载到模型中。

```
# Load the initial model
keras.backend.set_image_dim_ordering('th')
shape = (3,448,448)
model = yolo(shape)
print "Theano mode summary: \n",model.summary()
load_weights(model,'./yolo-tiny.weights')
```

### 层的尺寸：

我上面提到的模型是符合Theano的图片大小排序的。如果你想知道这是什么意思，那么让我给你介绍一下3D可视化！一个普通的1D信号是用矢量或者1D的数组，其尺寸就是矢量或数组的长度。图像就是2D数组或矩阵表示的信号。图像的尺寸为矩阵的宽✖高。当我们提到卷积层时，我们就是在讨论三维的数据结构。卷积层就是二维矩阵的叠加而成。这个新的尺寸叫做层的深度。做个类比，RGB图像是三个二维矩阵的组合。（宽✖高✖3）在图像中我们称之为通道，而在卷积层中我们称之为深度。好的，这就解释清楚了。但是为什么我们要讨论图片大小排序呢?深度学习中有两个主要的库，Theano和TensorFlow。Keras是由Theano和TensorFlow构建而成的，使我们能灵活的调用这些库。Apple的coremltools支持Keras以TensorFlow作为后端。TensorFlow对于图片尺寸的处理是宽✖高✖深度，而Theano处理图片尺寸是深度✖宽✖高。所以，如果你想要将我们的模型从以Theano为后端的Keras变成CoreML的模型。你首先需要将后端变成TensorFlow。当我们考虑图片尺寸和权重矩阵尺寸时会发现问题很复杂，但是我们将这些权重值放到另一个模型Keras2.0.4.时问题就很好解决了。

真正的挑战是将这个模型的后端从theano转化为tensorflow，这一步进行的并不顺利。幸运的是我找到了更好的代码能将权重从theano层转化为tensorflow层。但是如果我们仔细观察模型会发现，它是卷积层和全连接层的组合。所以，在接入全连接层之前，我们要将卷积层的结果压平。这个问题十分困难，我们不能自动化生成，除非我们的大脑能够轻易的可视化3D和4D维度。为了能够简单容易的进行调试，让我们把这一个模型分成三个部分。

- **第一部分**—包含所有的卷积层。
- **第二部分**—包含使第一部分输出结果平坦化的所有操作。
- **第三部分**—包含将第二部分的结果接入全连接层。

让我们先定义模型的第一部分和第三部分。

```
# YOLO Part 1
def yoloP1(shape):
    model = Sequential()
    model.add(Conv2D(16,(3,3),strides=(1,1),input_shape=shape,padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32,(3,3),padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
    model.add(Conv2D(64,(3,3),padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
    model.add(Conv2D(128,(3,3),padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
    model.add(Conv2D(256,(3,3),padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
    model.add(Conv2D(512,(3,3),padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
    model.add(Conv2D(1024,(3,3),padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(1024,(3,3),padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(1024,(3,3),padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    return model

# YOLO Part 3
def yoloP3():
    model = Sequential()
    model.add(Dense(256,input_shape=(50176,)))
    model.add(Dense(4096))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(1470))
    return model
```

- 让我们用将TensorFlow为后端的Keras初始化三个网络。他们分别使：YOLO 第一部（model_p1）,YOLO 第三部分（model_p3）以及YOLO Full(model_full)。同时为了测试这几个网络是否正常运行，我们同时定义第一和第三部分模型用Theano作为后端。

```
# Let us get Theano backend edition of Yolo Parts 1 and 3
model_p1_th = yoloP1(shape)
model_p3_th = yoloP3()
# Tensorflow backend edition
keras.backend.set_image_dim_ordering('tf')
shape = (448,448,3)
model_p1 = yoloP1(shape)
model_p3 = yoloP3()
model_full = yolo(shape)
```

- 在前面我们已经提到了以Theano和TensorFlow为后端的卷积层的图片尺寸不同。所以让我们写一个程序将Theano ‘model’的权重转化为TensorFlow的“model_full”。

```
# Transfer weights from Theano model to TensorFlow model_full
for th_layer,tf_layer in zip(model.layers,model_full.layers):
    if th_layer.__class__.__name__ == 'Convolution2D':
        kernel, bias = th_layer.get_weights()
        kernel = np.transpose(kernel,(2,3,1,0))
        tf_layer.set_weights([kernel,bias])
    else:
        tf_layer.set_weights(th_layer.get_weights())
```

- 在进行下个阶段前，让我们简单的测试一下是否Theano的'model'和TensorFlow的“model_full”的结果是否一致。为了达到这样的目的，我们读取一张图片，处理它，然后将它放入模型中并预测它的输出结果。

```
# Read an image and pre-process it
im = cv2.imread('test1.jpg')
plt.imshow(im[:,:,::-1])
plt.show()
im = cv2.resize(im,(448,448))
im = 2*im.astype(np.float32)/255. - 1
im = np.reshape(im,(1,448,448,3))
im_th = np.transpose(im,(0,3,1,2))

# Theano
output_th = model.predict(im_th)
# TensorFlow
output_tf = model_full.predict(im)

# Distance between two predictions
print 'Difference between two outputs:\nSum of Difference =', np.sum(output_th-output_tf),'\n2-norm of difference =',np.linalg.norm(output_th-output_tf)
```

- 通过运行上面的代码，我发现对于同一张图片，theano和tensorflow的输出结果有很大的不同。如果输出结果一致的话，“Sum of Difference”和“2-norm of difference”应该等于0。
- 既然直接转换没有用，那么让我们转向模型部件的设计。首先，让我们从Tiny YOLO 的第一部分开始。

```
# Theano
model_p1_th.layers[0].set_weights(model.layers[0].get_weights())
model_p1_th.layers[3].set_weights(model.layers[3].get_weights())
model_p1_th.layers[6].set_weights(model.layers[6].get_weights())
model_p1_th.layers[9].set_weights(model.layers[9].get_weights())
model_p1_th.layers[12].set_weights(model.layers[12].get_weights())
model_p1_th.layers[15].set_weights(model.layers[15].get_weights())
model_p1_th.layers[18].set_weights(model.layers[18].get_weights())
model_p1_th.layers[20].set_weights(model.layers[20].get_weights())
model_p1_th.layers[22].set_weights(model.layers[22].get_weights())

# TensorFlow
model_p1.layers[0].set_weights(model_full.layers[0].get_weights())
model_p1.layers[3].set_weights(model_full.layers[3].get_weights())
model_p1.layers[6].set_weights(model_full.layers[6].get_weights())
model_p1.layers[9].set_weights(model_full.layers[9].get_weights())
model_p1.layers[12].set_weights(model_full.layers[12].get_weights())
model_p1.layers[15].set_weights(model_full.layers[15].get_weights())
model_p1.layers[18].set_weights(model_full.layers[18].get_weights())
model_p1.layers[20].set_weights(model_full.layers[20].get_weights())
model_p1.layers[22].set_weights(model_full.layers[22].get_weights())

# Theano
output_th = model_p1_th.predict(im_th)
# TensorFlow
output_tf = model_p1.predict(im)

# Dimensions of output_th and output_tf are different, so apply transpose on output_th
output_thT = np.transpose(output_th,(0,2,3,1))

# Distance between two predictions
print 'Difference between two outputs:\nSum of Difference =', np.sum(output_thT-output_tf),'\n2-norm of difference =',np.linalg.norm(output_thT-output_tf)
```

- 通过运行上面的代码，我们可以发现两个的输出完全匹配。也就是说我们成功的完成了我们模型第一部分的设计！现在让我们去设计模型的第三部分。通过仔细的观察model_p3和model_p3_th的情况，我们容易发现这两种模型是类似的。因此，对于给定的输入，两个模型都会给我们相同的固定输出。但是这些模型的输入是什么呢？理想情况下的输入应该是来自Yolo的第二部分，但是Yolo的第二部分只是一个给任意多维的输入它的输出结果都是一个一维矢量。假设我们已经序列化了model_p1_th的输出，那么model_p3和model_p3_th会给我们相似的结果。

```
# Theano
model_p3_th.layers[0].set_weights(model.layers[25].get_weights())
model_p3_th.layers[1].set_weights(model.layers[26].get_weights())
model_p3_th.layers[3].set_weights(model.layers[28].get_weights())

# TensorFlow
model_p3.layers[0].set_weights(model_full.layers[25].get_weights())
model_p3.layers[1].set_weights(model_full.layers[26].get_weights())
model_p3.layers[3].set_weights(model_full.layers[28].get_weights())

# Design the input
input_p3 = np.reshape(np.ndarray.flatten(output_th),(1,50176))

# Theano
output_th = model_p3_th.predict(input_p3)
# TensorFlow
output_tf = model_p3.predict(input_p3)

# Distance between two predictions
print 'Difference between two outputs:\nSum of Difference =', np.sum(output_th-output_tf),'\n2-norm of difference =',np.linalg.norm(output_th-output_tf)
```

- 在运行上面的代码时我们可以观察到model_p3和model_p3_th给了我们一样的结果。
- YOLO的第二部分在哪里？耐心些，我们现在就要设计它了。在设计YOLO第二部分前，让我们讨论一下维度。我已经提过了YOLO的第二部分只是一个简单的平坦化的层。是什么使这变得困难？如果你还记得的话，整个网络都是以Theano为后端设计的，只有这些权重值需要我们的模型以Tensorflow为后端。为了更好的理解flatten层的操作，我写了一些代码让你去理解。通过运行下列代码你会找到我们的model_full给出的结果比model给出的结果奇怪的原因。

```
# Let us build a simple 3(width) x 3(height) x 3(depth) matrix and assume it as an output from Part 1
A = np.reshape(np.asarray([i for i in range(1,10)]),(3,3))
B = A + 10
C = A + 100

print 'A =\n',A,'\n\nB =\n',B,'\n\nC =\n',C

part1_output_tf = np.dstack((A,B,C))
print '\n\nTensorFlow\'s model_p1 output (assume) = \n',part1_output_tf

part1_output_th = np.transpose(part1_output_tf,(2,0,1))
print '\n\nTheano\'s model_p1_th output (assume) = \n',part1_output_th

print '\n\nDesired input for model_p3 =\n', part1_output_th.flatten()
print '\n\nActual input for model_p3 =\n', part1_output_tf.flatten()
```

- 现在我们了解了flatten层的应用不像我们预想的那么容易。下面有一些方法关于我们应该如何实现平坦化层。
  - **想法1**—获取第一部分的输出作为MLMultArray，并在IOS应用程序的CPU上应用自定义的平坦化操作。操作的消耗太大了！
  - **想法2**—使用Keras设置一个以 Permute 层加上Flatten层的模型并将它转化为CoreML模型。如果可以做到就设计成一个单一的模型。
  - **想法3**—