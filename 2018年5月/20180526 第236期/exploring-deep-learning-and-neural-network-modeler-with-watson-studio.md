# Exploring Deep Learning and Neural Network Modeler with Watson Studio

![](https://cdn-images-1.medium.com/max/1600/1*NSi_6E6RaNodwv268g0EvQ.png)

It was too much fun creating my own convolutional neural network with [Watson Studio](https://dataplatform.ibm.com/)! [The Neural Network Modeler](https://www.youtube.com/watch?v=xLcHbF8GM9c) provides expressive and intuitive graphical tools for building powerful deep learning models. It integrates well with all aspects of [Watson Machine Learning](https://datascience.ibm.com/docs/content/analyze-data/ml-overview.html). Application developers, analysts, and data scientists use WML to integrate machine learning into their workflow seamlessly. For example, you can save and deploy models as a REST API with a few clicks, minimizing the time it takes to put machine learning models to work. In this example, I used the familiar [MNIST data set](http://yann.lecun.com/exdb/mnist/) (images of hand-written digits) and created a three layers deep CNN. Then, I exported the algorithm in TensorFlow / Keras python code before training my model using NVIDAI GPUs. I also used the downloaded code to train my model locally in my Jupyter Macbook environment. In both cases, I got some very nice results. Of course, training with GPUs in the cloud is lightning fast.

The images of digits we’re classifying look like this:

![](https://cdn-images-1.medium.com/max/1600/1*Ft2rLuO82eItlvJn5HOi9A.png)

To data scientists and AI developers : go and use it. And for all other enthusiasts, the best way to learn AI is to create it yourself 🙂 !

First, to access the Watson Studio environment, you need your IBM ID. Visit [IBM Cloud](https://console.bluemix.net/) and then create your free account. **Everything I’ve used here is free**, so you can learn, develop your models and deploy them for productive use quickly and painlessly.

![](https://cdn-images-1.medium.com/max/1600/1*FYGeqC5Sq8j2Ob7a0M5Eow.png)

Once you have the access to IBM Cloud, you will need to source three services:****[Cloud Object Storage](https://console.bluemix.net/catalog/services/cloud-object-storage) (to store your data sets, trained models and training results), [Machine Learning](https://console.bluemix.net/catalog/services/machine-learning) (to train your models and benefit from ultrafast GPUs) and [Watson Studio](https://console.bluemix.net/catalog/services/watson-studio) (to create and manage your DL models, collaborate with your team, and more).

![](https://cdn-images-1.medium.com/max/1600/1*WTJ2ODgRhuC1FbTtmn4vKA.jpeg)

When this is done, check in your Watson Dashboard that you have all these three services properly created. You can check this from [within the Watson Studio Admin Console](https://dataplatform.ibm.com/console/overview)****or from [within your IBM Cloud Dashboard](https://console.bluemix.net/dashboard/apps).

The next step is to **create buckets in your Cloud Object Storage (COS)** and **upload your data**. My recommendation is to **create two buckets** (you can also call them containers for your data objects), **one for the data** and **one for the models and results**. For larger models, more granularity is needed, since the Watson Machine Learning (WML) will load the data from the training data bucket, so you want to avoid unnecessary waiting time and memory consumption for the data which is not used in your training runs.

![](https://cdn-images-1.medium.com/max/1600/1*cwbwXWED6IY9KKYh73HOkw.jpeg)

Here is how it looks like when you create both buckets and upload the data:

![](https://cdn-images-1.medium.com/max/1600/1*gr2ANg2IwVkkWnBNhOIbLQ.jpeg)

![](https://cdn-images-1.medium.com/max/1600/1*UrAJ6MTNjGxUaB2lBTEuAA.jpeg)

When this is done, you can launch Watson Studio and create your project. I named mine MNIST-LZRVC. You will see on the right hand side your Cloud Object Storage (COS). Watson Studio will automatically create connection to your COS service.

![](https://cdn-images-1.medium.com/max/1600/1*iwZXaolo7hC_s-O2VkSsqw.png)

Next, in the assets tab of Watson Studio, go to the Modeler Flow and add a new flow. Name it like your project but with some unique ID so that you can recognize when you start modifying and experimenting with different architectures.

![](https://cdn-images-1.medium.com/max/1600/1*4--fDCMJJAyMRBWFz672lA.png)

When you do this, you will land on the design canvas for modeling your neural networks. You can click on the left upper button to open the side toolbar with different groups of NN building blocks.

![](https://cdn-images-1.medium.com/max/1600/1*XORNr7iIZBrol_h2H2fQeQ.png)

These building blocks allow you to construct the whole deep neural structure from the scratch quickly and easily. There are also many samples available in the platform. Of course, you need to know what you are doing, but there are quite a lot of very good learning resources on the web if you want to learn more. In my particular case of this image recognition example, I have connected my input data with three convolutional + ReLU activation + pooling layers, then to a fully connected layer and passed the resulting feature representation to softmax classifier with 10 categories (for 10 different digits). In the end, I connected everything with the cross-entropy loss function and used Adam adaptive learning-rate optimizer. I’m measuring the performance of my model on the validation data set using the accuracy metric.

I have experimented with different number of layers. One is clearly not enough, but three layers give very good results, even after only two or three training epochs. Of course, you need to be careful with the hyperparameters like number and sizes of filters, strides, channels, etc.

![](https://cdn-images-1.medium.com/max/1600/1*q9KwAnEAWnZBb6quh9r1sw.png)

When you have done this the most creative part of your project, connect your model with the COS bucket that contains your data set. Click on the three dots of upload your files in this new data bucket. I’m using pickled data in three separate files for training, validation and test.

![](https://cdn-images-1.medium.com/max/1600/1*UAnNufVrh_FessQ3FqIcOA.jpeg)

When your model is ready, take a look at the two arrow-like icons in the toolbar. They allow you to download the code or launch your model.

I will first download this model on my laptop, because I want to play with it.

![](https://cdn-images-1.medium.com/max/1600/1*CAF7XbPdDiTdPFsHLUZvDw.jpeg)

When you select this download button you will get the choice of different formats. Pay attention — with Watson Studio y**ou can create training code for multiple deep learning frameworks**. This is the beauty — using IBM’s Fabric for Deep Leaarning (FfDL), you are able to run your deep learning algorithms on TensorFlow (with or without Keras), PyTorh and Caffe, all that within one unified training and deployment cloud environment.

So, I continue to download my model. I’m interested to see how it is written.

![](https://cdn-images-1.medium.com/max/1600/0*IfKPoAXD24lI_nLc.jpeg)

And here it is, I’m running it locally in my laptop python. No changes to the code needed !

![](https://cdn-images-1.medium.com/max/1600/1*VOVKj8WD6SAxRE3HvdrGvA.jpeg)

I also like very much Jupyter notebook environment, so I just copy-pasted my code into it. If you would like to check it out, please download all files from my github repository.

[https://github.com/LZRVC/mnist-with-keras-and-watson-studio](https://github.com/LZRVC/mnist-with-keras-and-watson-studio)

I’m adding here a couple of screenshots to show the training and prediction with this Jupyter notebook. One training epoch takes 2 minutes, and I’m getting 96% of accuracy on the included test data set !

![](https://cdn-images-1.medium.com/max/1600/1*n9QKDbi-GbR_yffnuqgBYQ.jpeg)

Here you can see the structure of the model :

![](https://cdn-images-1.medium.com/max/1600/1*cbRJJtV_uKh7smx4lUE6vw.jpeg)

And here is the prediction part. For the previously unseen images, I’m opening them in grayscale, adding the contrast, normalizing, resizing and reshaping them. Then, I use model.predict() method on this image and extract the predictions from the classifier. This is an array of 10 probabilities, where in the end we select the highest probability for the image to deliver our prediction.

![](https://cdn-images-1.medium.com/max/1600/1*PAucW5jdUYv8VVgO3eEIuw.jpeg)

Note : I noticed inaccurate predictions for number 7, when I write it with a dash across the numeral, and for all numbers when they are written with thin pencil. The training data seems to be of a similar type, so the real-world results are a bit skewed.

Now, going back to Watson Studio. I want to upload my model directly from the design canvas to Watson Machine Learning engine.

First, we will need to associate WML to our project. Select your project, and Setting menu tab option. Scroll down to see the Associated services, and add Machine Learning

![](https://cdn-images-1.medium.com/max/1600/1*QGv8bwAd6Wuc_c5tvww29w.jpeg)

![](https://cdn-images-1.medium.com/max/1600/1*3AulHsmIdPOfWeJlfm_zYA.jpeg)

Now, go to your neural network modeler part of Watson Studio and click on the other icon on the toolbar to publish your model to WML.

![](https://cdn-images-1.medium.com/max/1600/1*fUfY6L25UvEyLxkskjpFig.jpeg)

We will now create an experiment (in other words, prepare to launch the training run):

![](https://cdn-images-1.medium.com/max/1600/1*yElK68ICFek6gWnONwT9kg.jpeg)

Enter a name of your training run (mind using IDs and some meaningful abbreviations so that you can trace back where the results are coming from). Select and confirm your COS buckets for the input data and results, and create your training definition to choose the number of GPUs and if you would like to do hyperparameter optimization (Select “None” — I will describe this in one of the next posts in the future).

![](https://cdn-images-1.medium.com/max/1600/1*wUHFwO8EFa_aF2TmqyGn0Q.jpeg)

When you click on Create and Run, WML will automatically start your training process. You can monitor the progress, review the log files as the system works. There is some delay until your job lands on the GPUs, and here after 5 minutes 25 seconds I got the results. This elapsed time is actually not bad, having in mind that I left 10 epochs specified in my model definition. One training epoch took 18 seconds, and there was also some queuing of my job before it started.

![](https://cdn-images-1.medium.com/max/1600/1*0vA7Jg3E4x71r9yE_2kvVw.jpeg)

Here in the results page you can see the location of your trained model. In the next blog post I will introduce more sophisticated concepts showing WML can be used for your Deep Learning projects, including Hyperparameter Optimization.

![](https://cdn-images-1.medium.com/max/1600/1*9fYylHQAGwlcGKTzMKpbHg.jpeg)

### Conclusion

Watson Studio, Watson Machine Learning, and Cloud Object Storage are, as you could see from this tutorial, very sophisticated, well integrated, extremely flexible and easy-to-use tools for data scientists and AI developers. DLaaS provides excellent features that make IBM Watson Studio second to none. [Get started for free today!](https://www.ibm.com/cloud/watson-studio)

Sasha Lazarevic, IBM
May 21, 2018

[https://www.linkedin.com/in/lzrvc/](https://www.linkedin.com/in/lzrvc/)
[https://github.com/LZRVC/mnist-with-keras-and-watson-studio](https://github.com/LZRVC/mnist-with-keras-and-watson-studio)

