# How to easily Detect Objects with Deep Learning on Raspberry Pi



Disclaimer: I’m building [nanonets.com](https://nanonets.com/objectdetection/?utm_source=medium.com&utm_medium=content&utm_campaign=How to easily Detect Objects with Deep Learning on RaspberryPi&utm_content=top) to help build ML with less data and no hardware

> If you’re impatient scroll to the bottom of the post for the Github Repos

![](https://cdn-images-1.medium.com/max/1600/1*YJbdykJRHFlzlIXWwn0nIA.gif)

### Why Object Detection?, Why Raspberry Pi?

The raspberry pi is a neat piece of hardware that has captured the hearts of a generation with ~15M devices sold, with hackers building even [cooler projects](http://www.trustedreviews.com/opinion/best-raspberry-pi-projects-pi-3-pi-zero-2949390) on it. Given the popularity of Deep Learning and the [Raspberry Pi Camera](https://www.raspberrypi.org/products/camera-module-v2/) we thought it would be nice if we could detect any object using Deep Learning on the Pi.

Now you will be able to detect a photobomber in your selfie, someone entering Harambe’s cage, where someone kept the Sriracha or an Amazon delivery guy entering your house.

### What is Object Detection?

20M years of evolution have made human vision fairly evolved. The human brain has [30% of it’s Neurons work on processing vision (as compared with 8 percent for touch and just 3 percent for hearing)](http://discovermagazine.com/1993/jun/thevisionthingma227). Humans have two major advantages when compared with machines. One is stereoscopic vision, the second is an almost infinite supply of training data (an infant of 5 years has had approximately 2.7B Images sampled at 30fps).

![](https://cdn-images-1.medium.com/max/1600/1*4tPwx3wG720gOmIOaONOEQ.jpeg)

To mimic human level performance scientists broke down the visual perception task into four different categories.

1. **Classification**, assigns a label to an entire image

2. **Localization**, assigns a bounding box to a particular label

3. **Object Detection**, draws multiple bounding boxes in an image

4. **Image segmentation**, creates precise segments of where objects lie in an image

Object detection has been good enough for a variety of applications (even though image segmentation is a much more precise result, it suffers from the complexity of creating training data. It typically takes a human annotator 12x more time to segment an image than draw bounding boxes; this is more anecdotal and lacks a source). Also, after detecting objects, it is separately possible to segment the object from the bounding box.

#### Using Object Detection:

Object detection is of significant practical importance and has been used across a variety of industries. Some of the examples are mentioned below:

![](https://cdn-images-1.medium.com/max/1600/1*ZUGVScHbBgmmzO82bALIZQ.jpeg)

### How do I use Object Detection to solve my own problem?

Object Detection can be used to answer a variety of questions. These are the broad categories:

1. **Is an object present** in my Image or not? eg is there an intruder in my house

2. **Where is an object** in the image? eg when a car is trying to navigate it’s way through the world, its important to know where an object is.

3. **How many objects** are there in an image? Object detection is one of the most efficient ways of counting objects. eg How many boxes in a rack inside a warehouse

4. **What are the different types of objects** in the Image? eg Which animal is there in which part of the Zoo?

5. **What is the size of an object?**Especially with a static camera, it is easy to figure out the size of an object. eg What is the size of the Mango

6. **How are different objects interacting with each other?**eg****How does the formation on a football field effect the result?

7. **Where is an object with respect to time (Tracking an Object). eg**Tracking a moving object like a train and calculating it’s speed etc.

### Object Detection in under 20 Lines of Code

![](https://cdn-images-1.medium.com/max/1600/1*I4vKwR9X33DoNz36I1IooQ.jpeg)

There are a variety of models/architectures that are used for object detection. Each with trade-offs between speed, size, and accuracy. We picked one of the most popular ones: [YOLO](https://pjreddie.com/darknet/yolo/) (You only look once). and have shown how it works below in under 20 lines of code (if you ignore the comments).

**Note: This is pseudo code, not intended to be a working example. It has a black box which is the CNN part of it which is fairly standard and shown in the image below.**

You can read the full paper here: [https://pjreddie.com/media/files/papers/yolo_1.pdf](https://pjreddie.com/media/files/papers/yolo_1.pdf)

![](https://cdn-images-1.medium.com/max/1600/1*hV1SLRRZ-5ySyARb2P0uXA.png)



### How do we build a Deep Learning model for Object Detection?

#### The workflow for Deep Learning has 6 Primary Steps Broken into 3 Parts

1. Gathering Training Data

2. Training the model

3. Predictions on New Images

![](https://cdn-images-1.medium.com/max/1600/1*hUOIe8skkgMQx68-279z_A.jpeg)

### Phase 1 — Gather Training Data

#### Step 1. Collect Images (at least 100 per Object):

For this task, you probably need a few 100 Images per Object. Try to capture data as close to the data you’re going to finally make predictions on.

![](https://cdn-images-1.medium.com/max/1600/1*ZqUXpif7jgmAsIwX7ZFdrQ.png)

#### Step 2. Annotate (draw boxes on those Images manually):

Draw bounding boxes on the images. You can use a tool like [labelImg](https://github.com/tzutalin/labelImg). You will typically need a few people who will be working on annotating your images. This is a fairly intensive and time consuming task.

![](https://cdn-images-1.medium.com/max/1600/1*osRdxUvKXSaOHX-9VyGbCQ.png)

### Phase 2 — Training a Model on a GPU Machine

#### Step 3. Finding a Pretrained Model for Transfer Learning:

You can read more about this at [medium.com/nanonets/nanonets-how-to-use-deep-learning-when-you-have-limited-data-f68c0b512cab](http://medium.com/nanonets/nanonets-how-to-use-deep-learning-when-you-have-limited-data-f68c0b512cab). You need a pretrained model so you can reduce the amount of data required to train. Without it, you might need a few 100k images to train the model.

[You can find a bunch of pretrained models here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

#### Step 4. Training on a GPU (cloud service like AWS/GCP etc or your own GPU Machine):

![](https://cdn-images-1.medium.com/max/1600/1*b1-9TBSK6GUMGWd27wcLvQ.png)

#### Docker Image

The process of training a model is unnecessarily difficult to simplify the process we created a docker image would make it easy to train.

To start training the model you can run:



#### Please refer to this link for details on how to use

The docker image has a run.sh script that can be called with the following parameters





You can find more details at:

[**NanoNets/RaspberryPi-ObjectDetection-TensorFlow**
RaspberryPi-ObjectDetection-TensorFlow - Object Detection using TensorFlow on a Raspberry Pigithub.com](https://github.com/NanoNets/RaspberryPi-ObjectDetection-TensorFlow)[](https://github.com/NanoNets/RaspberryPi-ObjectDetection-TensorFlow)

**To train a model you need to select the right hyper parameters.**

**Finding the right parameters**

The art of “Deep Learning” involves a little bit of hit and try to figure out which are the best parameters to get the highest accuracy for your model. There is some level of black magic associated with this, along with a little bit of theory. [This is a great resource for finding the right parameters](https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607).

**Quantize Model (make it smaller to fit on a small device like the Raspberry Pi or Mobile)**

Small devices like Mobile Phones and Rasberry PI have very little memory and computation power.

Training neural networks is done by applying many tiny nudges to the weights, and these small increments typically need floating point precision to work (though there are research efforts to use quantized representations here too).

Taking a pre-trained model and running inference is very different. One of the magical qualities of Deep Neural Networks is that they tend to cope very well with high levels of noise in their inputs.

**Why Quantize?**

Neural network models can take up a lot of space on disk, with the original AlexNet being over 200 MB in float format for example. Almost all of that size is taken up with the weights for the neural connections, since there are often many millions of these in a single model.

The Nodes and Weights of a neural network are originally stored as 32-bit floating point numbers. The simplest motivation for quantization is to shrink file sizes by storing the min and max for each layer, and then compressing each float value to an eight-bit integer.The size of the files is reduced by 75%.

![](https://cdn-images-1.medium.com/max/1600/0*Ey92vYBh1Wq2uHfH.png)

**Code for Quantization:**





> Note: Our docker image has quantization built into it.

### Phase 3: Predictions on New Images using the Raspberry Pi

#### Step 5. Capture a new Image via the camera

You need the Raspberry Pi camera live and working. Then capture a new Image

![](https://cdn-images-1.medium.com/max/1600/1*tMcyYPmB8aCJYXSS8Y2I8A.jpeg)

For instructions on how to install checkout this [link](https://thepihut.com/blogs/raspberry-pi-tutorials/16021420-how-to-install-use-the-raspberry-pi-camera)



#### Step 6. Predicting a new Image

**Download Model**

Once your done training the model you can download it on to your pi. To export the model run:



Then download the model onto the Raspberry Pi.

**Install TensorFlow on the Raspberry Pi**

Depending on your device you might need to change the installation a little















**Run model for predicting on the new Image**



### Performance Benchmarks on Raspberry Pi

The Raspberry Pi has constraints on both Memory and Compute (a version of Tensorflow Compatible with the Raspberry Pi GPU is still not available). Therefore, it is important to benchmark how much time do each of the models take to make a prediction on a new image.

### Workflow with NanoNets:

![](https://cdn-images-1.medium.com/max/1600/1*m5grJCpQ6Dk6Ee-JEBIPPg.jpeg)

#### We at NanoNets have a goal of making working with Deep Learning super easy. Object Detection is a major focus area for us and we have made a workflow that solves a lot of the challenges of implementing Deep Learning models.

### How NanoNets make the Process Easier:

#### 1. No Annotation Required

We have removed the need to annotate Images, we have expert annotators who will **annotate your images for you**.

#### 2. Automatic Best Model and Hyper Parameter Selection

We **automatically train the best mode**l for you, to achieve this we run a battery of model with different parameters to select the best for your data

#### 3. No Need for expensive Hardware and GPUs

NanoNets is **entirely in the cloud** and runs without using any of your hardware. Which makes it much easier to use.

#### 4. Great for Mobile devices like the Raspberry Pi

Since devices like the Raspberry Pi and mobile phones were not built to run complex compute heavy tasks, you can outsource the workload to our cloud which does all of the compute for you

### Here is a simple snippet to make prediction on an image using the NanoNets API



### Build your Own NanoNet

![](https://cdn-images-1.medium.com/max/1600/1*D0woyU-XyyqlUsNP1ToOBA.png)

### You can try building your own model from:

### 1. Using a GUI (also auto annotate Images): https://nanonets.com/objectdetection/

### 2. Using our API: https://github.com/NanoNets/object-detection-sample-python

#### Step 1: Clone the Repo



#### Step 2: Get your free API Key

Get your free API Key from [http://app.nanonets.com/user/api_key](http://app.nanonets.com/user/api_key)

#### Step 3: Set the API key as an Environment Variable



#### Step 4: Create a New Model



> Note: This generates a MODEL_ID that you need for the next step

#### Step 5: Add Model Id as Environment Variable



#### Step 6: Upload the Training Data

Collect the images of object you want to detect. You can annotate them either using our web UI (https://app.nanonets.com/ObjectAnnotation/?appId=YOUR_MODEL_ID) or use open source tool like [labelImg](https://github.com/tzutalin/labelImg). Once you have dataset ready in folders, `images` (image files) and `annotations` (annotations for the image files), start uploading the dataset.



#### Step 7: Train Model

Once the Images have been uploaded, begin training the Model



#### Step 8: Get Model State

The model takes ~2 hours to train. You will get an email once the model is trained. In the meanwhile you check the state of the model



#### Step 9: Make Prediction

Once the model is trained. You can make predictions using the model



### Code (Github Repos)

#### Github Repos to Train a model:

1. [Tensorflow Code for model Training and Quantization](https://github.com/NanoNets/RaspberryPi-ObjectDetection-TensorFlow)

2. [NanoNets Code for model Training](https://github.com/NanoNets/IndianRoadsObjectDetectionDataset)

#### Github Repos for Raspberry Pi to make Predictions (ie Detecting New Objects):

1. [Tensorflow Code for making Predictions on the Raspberry Pi](https://github.com/NanoNets/TF-OD-Pi-Test)

2. [NanoNets Code for making Predictions on the Raspberry Pi](https://gist.github.com/sjain07/a30388035c0b39b53841c501f8262ee2)

#### Datasets with Annotations:

1. [Cars on Indian Roads sees, dataset for extracting vehicles from Images of Indian Roads](https://github.com/NanoNets/IndianRoadsObjectDetectionDataset)

2. [Coco Dataset](http://cocodataset.org/#download)

