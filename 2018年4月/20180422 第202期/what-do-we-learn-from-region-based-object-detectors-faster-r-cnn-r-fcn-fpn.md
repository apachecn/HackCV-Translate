# What do we learn from region based object detectors (Faster R-CNN, R-FCN, FPN)?

In this series, we will take a comprehensive journey on object detection. In Part 1 here, we cover the region based object detectors including Fast R-CNN, Faster R-CNN, R-FCN and FPN. In part 2, we will study the single shoot detectors. In part 3, we cover the performance and some implementation issues. By studying them in one context, we study what is working, what matters and where can be improved. Hopefully, by studying how we get here, it will give us more insights on where we are heading.

[Part 1](https://medium.com/@jonathan_hui/what-do-we-learn-from-region-based-object-detectors-faster-r-cnn-r-fcn-fpn-7e354377a7c9): What do we learn from region based object detectors (Faster R-CNN, R-FCN, FPN)?

[Part 2](https://medium.com/@jonathan_hui/what-do-we-learn-from-single-shot-object-detectors-ssd-yolo-fpn-focal-loss-3888677c5f4d): What do we learn from single shot object detectors (SSD, YOLO), FPN & Focal loss?

[Part 3](https://medium.com/@jonathan_hui/design-choices-lessons-learned-and-trends-for-object-detections-4f48b59ec5ff): Design choices, lessons learned and trends for object detections?

### Sliding-window detectors

Since AlexNet won the 2012 ILSVRC challenge, the use of the CNN for classification has dominated the field. One brute force approach for object detection is to slide windows from left and right, and from up to down to identify objects using classification. To detect different object types at various viewing distances, we use windows of varied sizes and aspect ratios.

We cut out patches from the picture according to the sliding windows. The patches are warped since many classifiers take fixed size images only. However, this should not impact the classification accuracy since the classifier are trained to handle warped images.

![](https://cdn-images-1.medium.com/max/1600/1*A7DE4HKukbXpQqwvCaLOEQ.jpeg)

The warped image patch is fed into a CNN classifier to extract 4096 features. Then we apply a SVM classifier to identify the class and another linear regressor for the boundary box.

![](https://cdn-images-1.medium.com/max/1600/1*BYSA3iip3Cdr0L_x5r468A.png)

Below is the pseudo code. We create many windows to detect different object shapes at different locations. To improve performance, one obvious solution is to reduce the number of windows.



### Selective Search

Instead of a brute force approach, we use a region proposal method to create **regions of interest (ROIs)**for object detection. In **selective search**(**SS**), we start with each individual pixel as its own group. Next, we calculate the texture for each group and combine two that are the closest. But to avoid a single region in gobbling others, we prefer grouping smaller group first. We continue merging regions until everything is combined together. In the first row below, we show how we grow the regions, and the blue rectangles in the second rows show all possible ROIs we made during the merging.

![](https://cdn-images-1.medium.com/max/1600/1*_8BNWWwyod1LWUdzcAUr8w.png)

### R-CNN

R-CNN makes use of a region proposal method to create about 2000 **ROI**s (regions of interest). The regions are warped into fixed size images and feed into a CNN network individually. It is then followed by fully connected layers to classify the object and to refine the boundary box.

Here is the system flow.

![](https://cdn-images-1.medium.com/max/1600/1*ciyhZpgEvxDm1YxZd1SJWg.png)

With far fewer but higher quality ROIs, R-CNN run faster and more accurate than the sliding windows.



#### Boundary box regressor

Region proposal methods are computation intense. To speed up the process, we often pick a less expensive region proposal method to create ROIs followed by a linear regressor (using fully connected layers) to refine the boundary box further.

![](https://cdn-images-1.medium.com/max/1600/1*rvPyjhiVQnOm3yOqSDUKuA.jpeg)

### Fast R-CNN

R-CNN needs many proposals to be accurate and many regions overlap with each other. **R-CNN is slow in training & inference.**If we have 2,000 proposals, each of them is processed by a CNN separately, i.e. we repeat feature extractions 2,000 times for different ROIs.

Instead of extracting features for each image patch from scratch, we use a **feature extractor** (a CNN) to extract features for the whole image first. We also use an external region proposal method, like the selective search, to create ROIs which later combine with the corresponding feature maps to form patches for object detection. We warp the patches to a fixed size using **ROI pooling** and feed them to fully connected layers for classification and **localization** (detecting the location of the object). By not repeating the feature extractions, Fast R-CNN cuts down the process time significantly.

Here is the network flow:

![](https://cdn-images-1.medium.com/max/1600/1*fLMNHfe_QFxW569s4eR7Dg.jpeg)

In the pseudo-code below, the expensive feature extraction is moving out of the for-loop, a significant speed improvement since it was executed for all 2000 ROIs. Fast R-CNN is 10x faster than R-CNN in training and 150x faster in inferencing.



One major takeaway for Fast R-CNN is that the whole network (the feature extractor, the classifier, and the boundary box regressor) are trained end-to-end with**multi-task losses** (classification loss and localization loss). This improves accuracy.

**ROI Pooling**

Because Fast R-CNN uses fully connected layers, we apply **ROI pooling** to warp the variable size ROIs into in a predefined size shape.

Let’s simplify the discussion by transforming 8 × 8 feature maps into a predefined 2 × 2 shape.

* Top left below: our feature maps.

* Top right: we overlap the ROI (blue) with the feature maps.

* Bottom left: we split ROIs into the target dimension. For example, with our 2×2 target, we split the ROIs into 4 sections with similar or equal sizes.

* Bottom right: find the maximum for each section and the result is our warped feature maps.

![](https://cdn-images-1.medium.com/max/1600/1*LLP4tKGsYGgAx3uPfmGdsw.png)

So we get a 2 × 2 feature patch that we can feed into the classifier and box regressor.

### Faster R-CNN

Fast R-CNN depends on an external region proposal method like selective search. However, those algorithms run on CPU and they are slow. In testing, Fast R-CNN takes 2.3 seconds to make a prediction in which 2 seconds are for generating 2000 ROIs.



Faster R-CNN adopts similar design as the Fast R-CNN except it replaces the region proposal method by an internal deep network and the ROIs are derived from the feature maps instead. The new region proposal network (**RPN**) is more efficient and run at 10 ms per image in generating ROIs.

![](https://cdn-images-1.medium.com/max/1600/1*F-WbcUMpWSE1tdKRgew2Ug.png)

The network flow is similar but the region proposal is now replaced by a convolutional network (RPN).

**Region proposal network**

The region proposal network (**RPN**) takes the output feature maps from the first convolutional network as input. It slides 3 × 3 filters over the feature maps to make class-agnostic region proposals using a convolutional network like ZF network (below). Other deep network likes VGG or ResNet can be used for more comprehensive feature extraction at the cost of speed. The ZF network outputs 256 values, which is feed into 2 separate fully connected layers to predict a boundary box and 2 objectness scores. The **objectness**measures whether the box contains an object. We can use a regressor to compute a single objectness score but for simplicity, Faster R-CNN uses a classifier with 2 possible classes: one for the “have an object” category and one without (i.e. the background class).

For each location in the feature maps, RPN makes **k** guesses. Therefore RPN outputs 4×k coordinates and 2×k scores per location. The diagram below shows the 8 × 8 feature maps with a 3× 3 filter, and it outputs a total of 8 × 8 × 3 ROIs (for k = 3). The right side diagram demonstrates the 3 proposals made by a single location.

![](https://cdn-images-1.medium.com/max/1600/1*smu6PiCx4LaPwGIo3HG0GQ.jpeg)

Here, we get 3 guesses and we will refine our guesses later. Since we just need one to be correct, we will be better off if our initial guesses have different shapes and size. Therefore, Faster R-CNN does not make random boundary box proposals. Instead, it predicts offsets like 𝛿x, 𝛿y that are relative to the top left corner of some reference boxes called **anchors**. We constraints the value of those offsets so our guesses still resemble the anchors.

![](https://cdn-images-1.medium.com/max/1600/1*yF_FrZAkXA3XKFA-sf7XZw.png)

To make k predictions per location, we need k anchors centered at each location. Each prediction is associated with a specific anchor but different locations share the same anchor shapes.

![](https://cdn-images-1.medium.com/max/1600/1*RJoauxGwUTF17ZANQmL8jw.png)

Those anchors are carefully pre-selected so they are diverse and cover real-life objects at different scales and aspect ratios reasonable well. This guides the initial training with better guesses and allows each prediction to specialize in a certain shape. This strategy makes early training more stable and easier.

Faster R-CNN uses far more anchors. It deploys 9 anchor boxes: 3 different scales at 3 different aspect ratio. Using 9 anchors per location, it generates 2 × 9 objectness scores and 4 × 9 coordinates per location.

![](https://cdn-images-1.medium.com/max/1600/1*PszFnq3rqa_CAhBrI94Eeg.png)

> Anchors are also called priors or default boundary boxes in different papers.

### Performance for R-CNN methods

As shown below, Faster R-CNN is even much faster.

![](https://cdn-images-1.medium.com/max/1600/1*fO2MSeQxIVVUUp6csJ8oWg.jpeg)

### Region-based Fully Convolutional Networks (R-FCN)

Let’s assume we only have a feature map detecting the right eye of a face. Can we use it to locate a face? It should. Since the right eye should be on the top-left corner of a facial picture, we can use that to locate the face.

![](https://cdn-images-1.medium.com/max/1600/1*gqxSBKVla8dzwADKgADpWg.jpeg)

If we have other feature maps specialized in detecting the left eye, the nose or the mouth, we can combine the results together to locate the face better.

So why we go through all the trouble. In Faster R-CNN, the detector applies multiple fully connected layers to make predictions. With 2,000 ROIs, it can be expensive.



R-FCN improves speed by reducing the amount of work needed for each ROI. The region-based feature maps above are independent of ROIs and can be computed outside each ROI. The remaining work is much simpler and therefore R-FCN is faster than Faster R-CNN.



Let’s consider a 5 × 5 feature map **M** with a blue square object inside. We divide the square object equally into 3 × 3 regions. Now, we create a new feature map from M to detect the top left (TL) corner of the square only. The new feature map looks like the one on the right below. Only the yellow **grid cell** [2, 2] is activated.

![](https://cdn-images-1.medium.com/max/1600/1*S0enLblW1t7VK19E1Fs4lw.png)

Since we divide the square into 9 parts, we can create 9 feature maps each detecting the corresponding region of the object. These feature maps are called **position-sensitive score maps**because each map detects (scores) a sub-region of the object.

![](https://cdn-images-1.medium.com/max/1600/1*HaOHsDYAf8LU2YQ7D3ymOg.png)

Let’s say the dotted red rectangle below is the ROI proposed. We divide it into 3 × 3 regions and ask how likely each region contains the corresponding part of the object. For example, how likely the top-left ROI region contains the left eye. We store the results into a 3 × 3 vote array in the right diagram below. For example, vote_array[0][0] contains the score on whether we find the top-left region of the square object.

![](https://cdn-images-1.medium.com/max/1600/1*Ym6b1qS0pXpeRVMysvvukg.jpeg)

This process to map score maps and ROIs to the vote array is called **position-sensitive** **ROI-pool**. The process is extremely close to the ROI pool we discussed before. We will not cover it further but you can refer to the future reading section for more information.

![](https://cdn-images-1.medium.com/max/1600/1*K4brSqensF8wL5i6JV1Eig.png)

After calculating all the values for the position-sensitive ROI pool, the class score is the average of all its elements.

![](https://cdn-images-1.medium.com/max/1600/1*ZJiWcIl2DUyx1-ZqArw33A.png)

Let’s say we have **C** classes to detect. We expand it to C + 1 classes so we include a new class for the background (non-object). Each class will have its own 3 × 3 score maps and therefore a total of (C+1) × 3 × 3 score maps. Using its own set of score maps, we predict a class score for each class. Then we apply a softmax on those scores to compute the probability for each class.

The following is the data flow. For our example, we have k=3 below.

### Our journey so far

We start with the basic sliding window algorithm.



Then we try to reduce the number of windows and move as much work as possible outside the for-loop.



In [Part 2](https://medium.com/@jonathan_hui/what-do-we-learn-from-single-shot-object-detectors-ssd-yolo-fpn-focal-loss-3888677c5f4d), we go even further to completely remove the for-loop. Single shot detectors make object detections in single shot without a separate step of region proposal.

### Further reading on FPN, R-FCN and Mask R-CNN

Both FPN and R-FCN are more complex than we described here. For further study, please refer to:

* [Feature Pyramid Networks (FPN) for object detection.](https://medium.com/@jonathan_hui/understanding-feature-pyramid-networks-for-object-detection-fpn-45b227b9106c)

* [Region-based Fully Convolutional Networks (R-FCN)](https://medium.com/@jonathan_hui/understanding-region-based-fully-convolutional-networks-r-fcn-for-object-detection-828316f07c99).

[**Image segmentation with Mask R-CNN**
In a previous article, we discuss the use of region based object detector like Faster R-CNN to detect objects. Instead…medium.com](https://medium.com/@jonathan_hui/image-segmentation-with-mask-r-cnn-ebe6d793272)[](https://medium.com/@jonathan_hui/image-segmentation-with-mask-r-cnn-ebe6d793272)

### Resources

[Detectron](https://github.com/facebookresearch/Detectron): Facebook Research’s implementation of the Faster R-CNN and Mask R-CNN using Caffe2.

The official implementation for the [Faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn) in MATLAB.

[Faster R-CNN](https://github.com/endernewton/tf-faster-rcnn) implementation in TensorFlow.

[R-FCN](https://github.com/msracver/Deformable-ConvNets) implementation in MXNet.

[R-FCN ](https://github.com/daijifeng001/R-FCN)implementation in Caffe and MATLAB.

[R-FCN](https://github.com/xdever/RFCN-tensorflow) implementation in TensorFlow.

