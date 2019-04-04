# Using Deep Learning to automatically rank millions of hotel images



![](https://cdn-images-1.medium.com/max/1600/1*nbVih54ky78GCPz0k52bVw.png)

At [idealo.de](https://www.idealo.de/) (the leading price comparison website in Europe and one of the largest portals in the German e-commerce market) we provide one of the best hotel price comparisons available on the market. For each hotel we receive dozens of images and face the challenge of choosing the most “attractive” image for each offer on our offer comparison pages, [as photos can be just as important for bookings as reviews](https://www.tripadvisor.com/TripAdvisorInsights/w613). Given that we have millions of hotel offers, we end up with more than 100 million images for which we need an “attractiveness” assessment.

We addressed the need to automatically assess image quality by implementing an aesthetic and technical image quality classifier based on Google’s research paper “[NIMA: Neural Image Assessment](https://arxiv.org/pdf/1709.05424.pdf)”. NIMA consists of two Convolutional Neural Networks (CNN) that aim to predict the aesthetic and technical quality of images, respectively. The models are trained via transfer learning, where [ImageNet](http://www.image-net.org/) pre-trained CNNs are fine-tuned for each quality classification tasks.

In this article, we will present our training approach and insights that we’ve gained throughout the process. We will then try to shed some light on what the trained models actually learned by visualising the convolutional filter weights and output nodes of our trained models.

We’ve published the trained models and code on [GitHub](https://github.com/idealo/image-quality-assessment/tree/master/models/MobileNet). The provided code allows one to use any of the pre-trained CNNs in [Keras](https://keras.io/applications/), so we are looking forward to contributions that explore other CNNs for image quality assessments 😃

### Training

The aesthetic and technical classifiers were trained in a transfer learning setup. We used the [MobileNet architecture](https://arxiv.org/abs/1704.04861) with ImageNet weights, and replaced the last dense layer in MobileNet with a dense layer that outputs to 10 classes (scores 1 to 10).

#### Earth Mover’s Loss

A special feature of NIMA is the use of the Earth Mover’s Loss (EML) as the loss function, contrary to the Categorical Cross Entropy (CCE) loss, that is generally applied in Deep Learning classification tasks. The EML can be understood as the amount of “earth” that needs to be moved to make two probability distributions equal. A useful attribute of this loss function is that it captures the inherent order of the classes. For our image quality ratings, the scores 4, 5, and 6 are more related than 1, 5, and 10, i.e. we would like to punish a prediction of 4 more if the true score is 10 than when the true score is 5. CCE does not capture this relationship, and it is often not required in object classifications task (e.g. misclassifying a tree as a dog is as bad as classifying it as a cat).

In order to use the EML we need for each image a distribution of ratings across all ten score classes. For the [AVA](https://github.com/ylogx/aesthetics) dataset, which is used to train the aesthetic classifications, these distribution labels are available. For the [TID2013](http://www.ponomarenko.info/tid2013.htm) dataset, used for the technical classifications, we inferred the distribution from the mean score given for each image. For more details on our distribution inference check out our [GitHub repo](https://github.com/idealo/image-quality-assessment/blob/master/data/TID2013/get_labels.py).

#### Fine-tuning stages

We train the models in a two stage process:

1. We start by training only the last dense layer with a higher learning rate to ensure that the newly added random weights are adjusted to the ImageNet convolutional weights. Without this burn-in period you risk juggling around the convolutional weights at training start and consequently slowing down the training process.

2. After the burn-in period we train all weights in the CNN with a low learning rate.

For both the aesthetic and technical model the train and validation losses level out after 5 and 25 epochs, respectively. This is a good indicator that the newly added weights have learned to classify aesthetics and technical quality as good as possible, and it is time to start training all weights.

For the aesthetic classifier we see a significant drop in loss once we start training also the convolutional weights (dashed lines in left graph above), indicating that we are adjusting the convolutional weights quite a bit for the aesthetic classification task. For the technical classifier the drop in loss is smaller, which at first is counter-intuitive, as the technical image quality should be object agnostic, and the ImageNet weights are optimised to recognise objects. The small drop might be due to the very small learning rate that is required to regularise training on the small TID2013 dataset.

You can find all hyper-parameters used for training on our [GitHub repo](https://github.com/idealo/image-quality-assessment/tree/master/models/MobileNet).

### Results

The above predictions show that the aesthetic classifier correctly ranks the images from very aesthetic (leftmost image with sunset) to least aesthetic (boring hotel room on the right). Similarly for technical classifications, the classifier predicts higher scores for undistorted images (first and fourth image from left), versus images with jpeg compression (second and fifth) or blur (third and sixth).

### Visualisations

In order to gain a better understanding as to how the CNN assesses aesthetic image quality, we used the [Lucid](https://github.com/tensorflow/lucid) package to visualise the learned convolutional filter weights and output nodes in Aesthetic MobileNet. The awesome blog post [Feature Visualization](https://distill.pub/2017/feature-visualization/) provides a great interactive overview of state-of-the-art CNN visualisation techniques.

Earlier convolutional layers are generally associated with simpler structures, like edges, wave patterns, and grids. The images above show patterns associated to six filters in layer 23 of MobileNet - the six images in the top row are generated from the original MobileNet ImageNet weights (ImageNet MobileNet), whereas the bottom row images are generated from the MobileNet weights fine-tuned on the AVA dataset for aesthetic ratings (Aesthetic MobileNet). From the filter visualisations we can see that the earlier convolutional filters are not much affected throughout fine-tuning, as they are very similar to the original ones.

For mid convolutional filters at layer 51, the learned shapes are more complex, and resemble interwoven structures like fur or a grid with eyes. Even at this level, Aesthetic MobileNet is very similar to ImageNet MobileNet.

The later convolutional layers show even more complex structures that resemble animals and tree like shapes. We can see that the filters for Aesthetic MobileNet differ significantly from the ImageNet ones, as they seem to be less focussed on objects, e.g. no animal shapes in the fourth filter from the left.

We also generated visualisations for the output nodes of Aesthetic MobileNet , which represent the probabilities for scores 1 to 10. The visualisations thus show a “representative” image that is associated to each score.

It is difficult to interpret the output node visualisations, just as much as it is difficult to define aesthetics. If anything, the visualisations for lower scores seem to be less colourful and diversified, whereas higher scores are associated with more colourful and dramatic shapes. The image for score 10 seems to resemble a landscape with a sky background, a motive generally associated with high aesthetics.

### Summary

In this article, we presented our business challenge to automatically assess the quality of images. We showed that the trained aesthetic and technical models successfully rank images according to aesthetics and technical quality. We further explored the learned CNN weights of the aesthetic model by visualising the convolutional filters and output nodes, and concluded that fine-tuning primarily affects later convolutional weights.

Fine-tuning deep neural networks is a great strategy to tackle many computer vision problems that businesses face. However, the classifications of these models, with their millions of parameters, are generally difficult to interpret, and we hope to have shed some light on this black box with our visualisation analysis.

Please let me know if you found this article useful (👏🏻) so others can find it too, and share it with your friends. You can follow me here on Medium ([Christopher Lennan](https://medium.com/@christopher.lennan)) or on Twitter ([@chris_lennan](https://twitter.com/chris_lennan)) to stay up-to-date with my work. Thanks a lot for reading!

