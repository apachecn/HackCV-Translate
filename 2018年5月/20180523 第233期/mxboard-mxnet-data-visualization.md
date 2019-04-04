# MXBoard — MXNet Data Visualization

Author: [Wu Jun](https://github.com/reminisce), Amazon AI Software Engineer
Translated from: [https://zh.mxnet.io/blog/mxboard](https://zh.mxnet.io/blog/mxboard)

### Preface

Deep neural networks are notoriously difficult to design and train. It usually involves a large number of tweaking and adjustments, modifying the network structure, and trying various optimization algorithms and hyper-parameters. From a theoretical perspective, the mathematical foundations of deep neural networks architectures remain largely incomplete and techniques are often based on generalization of empirical results.

Data visualizations, thanks to their intrinsic visual nature, can partially compensate the above deficiencies and paint a higher level picture to guide researchers during training of deep neural networks. For example, if the gradient’s data distribution can be drawn in real time during model training, the phenomenon of vanishing gradients or exploding gradients can be quickly detected and corrected.

![](https://cdn-images-1.medium.com/max/1600/1*Nb8IrYdjcAEqIsWNaKv0KA.png)

Another example, being able to visualize word embeddings help to clearly see that words are aggregated into different manifolds in a lower dimensional space that maintains contextual proximity. Another useful visualization is data clustering: projecting high-dimensional data into a lower-dimensional space using for example the T-SNE algorithm. There are a large amount of data visualization that can be used in the context of deep learning to help understand better the training process and the data itself.

The emergence of [TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard) has brought powerful visualizations to [TensorFlow](https://github.com/tensorflow/tensorflow) ‘s users. We have had feedback from many different users, including corporate ones, that they started using TensorFlow because of the rich feature set offered in TensorBoard. Can this powerful tool be made available to other deep learning frameworks? Thanks to the[ TeamHG-Memex](https://github.com/TeamHG-Memex/tensorboard_logger) efforts and their **tensorboard_logger**, we now have a transparent interface to write custom data to the event file format that are then consumed by TensorBoard.

It is based on this foundation that we have developed [MXboard](https://github.com/awslabs/mxboard), a python package for recording MXNet data frames and displaying them in TensorBoard. To install [MXBoard](https://github.com/awslabs/mxboard) follow these simple [instructions](https://github.com/awslabs/mxboard).

**Note: Please note that MXNet 1.2.0 is required to use all the features of MXBoard.** **Before the official release of MXNet 1.2.0, please install MXNet nightly version:**`pip install --pre mxnet`

### MXBoard Quick Start Guide

MXBoard supports most of the data types in TensorBoard:

![](https://cdn-images-1.medium.com/max/1600/0*dvSKMAJkyMU1XegP.png)

The MXBoard API is designed to follow the [tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch) API. All record APIs are defined in a class called `SummaryWriter` . This class contains information such as the file path of the record files, the frequency of writing, the queue size, etc. To record a new data point of a specific data type, be it a scalar or an image for example, you only need to call the corresponding API on the `SummaryWriter` object.

For example, we want to draw a data distribution diagram with a gradually decreasing standard deviation of normal distribution. First define a `SummaryWriter` object as follows:



Then in each loop, we create an `NDArray` with values drawn from normal distribution. We then pass the `NDArray` to the summary writer `add_histogram()` function, specifying the number of `bin` and the loop index `i` which will be the index of our data point. Finally, as with any file descriptors used in Python, it is good practice to close the file handle of the `SummaryWriter` using `.close()`.



In order to visualize the plotted diagram, on the terminal, enter the working directory, and type the following command to start TensorBoard:



Then enter `127.0.0.1:8888` in the browser's address bar. Click `HISTOGRAM` and you will see the following rendering:

![](https://cdn-images-1.medium.com/max/1600/0*WDqEtZV8_Vh5CxZK.png)

### Real-world MXBoard

Using what we learnt in the above section let’s try to accomplish the following two tasks:

1. Monitoring supervised learning training

2. Get insights on convolutional neural networks inner workings

### Training MNIST model

Let’s use the [MNIST](https://github.com/apache/incubator-mxnet/blob/master/example/gluon/mnist.py) dataset from the [Gluon vision API](https://mxnet.incubator.apache.org/api/python/gluon/data.html#module-mxnet.gluon.data.vision) and let’s use MXBoard to record in real-time:

* The cross-entropy loss

* The validation and training accuracy

* Gradient data distribution

All of them are good indicators of the progress of the training.

First, we define a `SummaryWriter` object:



The `flush_secs=5` is added here to specify that we want to write the records to the log file every five seconds so that we can track the real-time progress of the training in the browser.

Then we record the cross-entropy loss at the end of each batch:



At the end of each epoch, we record the gradient as `HISTOGRAM` data type and record the training and test accuracy as `SCALAR` types.



Then we simultaneously run the Python training script and TensorBoard to visualize the training in the browser in real-time.

To reproduce this experiment, you can find the fully worked out solution code available [here](https://github.com/reminisce/mxboard-demo) on Github.

![](https://cdn-images-1.medium.com/max/1600/0*37yi6vDp2rSqdqR8.png)

![](https://cdn-images-1.medium.com/max/1600/0*oA0BFAqg4Xkk6r4E.png)

### Visualization of convolutional filters and feature maps

Visualizing the convolutional filters and feature maps as images is useful for two reasons:

1. When training has converged, convolutional filters exhibits clear pattern detection features, lines and distinctive colors. Convolutional filters that do not converge or overfit the model will display a lot of noise.

2. Observing the RGB rendition of filters and feature maps can help give us an understanding of the features that are learnt and considered meaningful for the network, typically edge and color detection.

Here we use three pre-trained CNN models from the [MXNet Model Zoo](https://mxnet.incubator.apache.org/model_zoo/index.html), the Inception-BN , Resnet-152 , and VGG16. The filters of the first convolutional layer are visualized directly in TensorBoard, alongside the resulting feature maps when applied to a black swan image. Notice how networks can have different convolutional kernel sizes.

* Inception-BN

![](https://cdn-images-1.medium.com/max/1600/0*TTWb0Z7dwRsUU98Y.png)

* Resnet-152

![](https://cdn-images-1.medium.com/max/1600/0*l2qD0N5bDnp8t9-G.png)

* VGG16

![](https://cdn-images-1.medium.com/max/1600/0*V_FfwwYtTHXH8QPW.png)

You can see that the filters of the three models exhibit pretty good smoothness and regularity, usual signs of a model that has converged. The colored filters are mainly responsible for extracting color-based features in the image. The gray-colored images are responsible for extracting general patterns and outline features of the objects in the image.

For the full implementation and further analysis, check the code [here](https://github.com/reminisce/mxboard-demo).

### Visual image embedding

The last example is equally interesting. Embedding is a key concept used in several machine learning domains, including computer vision and Natural Language Processing (NLP). It is the representation of higher-dimensional data into a lower-dimensional space. In a traditional image classification setting, the output of the penultimate layer of a convolutional neural network is usually connected to a fully connected layer with a Softmax activation that is used to predict the class or category that the image belongs to. If we strip the network of this classification layer we are left with a network that outputs a vector of features for each example, usually 512 or 1024 features per example. This is called the embedding of our image. We can call to MXBoard `add_embedding()` API to observe the distribution of the embeddings of the dataset projected down into 2D or 3D space. Pictures with similar visual features are clustered together.

Here we randomly select 2304 images from the validation, calculate their embeddings using Resnet-152, add the embedding to the MXBoard log file and visualize them:

![](https://cdn-images-1.medium.com/max/1600/0*N5Q8ZX799vy82A8X.gif)

The embeddings of 2304 images are projected on a 3D space using the [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) algorithm by default. However the clustering effect is not obvious. This is because the PCA algorithm cannot maintain the spatial relationship between the original data points.Therefore, we use the t-[SNE](https://lvdmaaten.github.io/tsne/&xid=17259,15700023,15700124,15700149,15700168,15700173,15700186,15700189,15700201&usg=ALkJrhi8AewPeHi5ReaHseaVJdAS-dEoRw) algorithm provided by the TensorBoard interface to get a better visualization of the embeddings. Constructing the optimal projection is a dynamic process:

![](https://cdn-images-1.medium.com/max/1600/0*l7y4nF9LwD28ZBf7.gif)

After convergence of the t-SNE algorithm, it can be clearly seen that the dataset is divided into several clusters.

![](https://cdn-images-1.medium.com/max/1600/0*BZlf_MC0AP_vZvfC.png)

Finally, we can use the TensorBoard UI to verify the correctness of the image classification. We enter “dog” in the upper right corner of the TensorBoard GUI. All pictures of the validation dataset classified as “dog” tag will be highlighted. We also see that the clustering derived from the T-SNE projection follows closely the class boundaries.

![](https://cdn-images-1.medium.com/max/1600/0*HfBMRPJpOsOjB_3Z.gif)

All codes and instructions available [here](https://github.com/reminisce/mxboard-demo) .

### Conclusion

After this MXBoard tutorial, we can see that visualizations are a powerful tool in supervising the training of models and getting insights in the principles of deep learning. MXBoard provides MXNet with a simple, minimally intrusive, easy-to-use, centralized visualization solution for scientific and production environments. Best of all, all you need to use it is a browser.

Special thanks to [Zheng Zihao](https://github.com/zihaolucky) for providing technical support during the development of the project!

