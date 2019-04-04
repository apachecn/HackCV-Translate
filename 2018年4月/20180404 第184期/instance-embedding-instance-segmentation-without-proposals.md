![](https://cdn-images-1.medium.com/max/1600/1*73E15g6xOD8leNHppZZycw.png)

# Instance Embedding: Segmentation Without Proposals

In this article I will review 3 papers in the field of instance segmentation. They are different from the mainstream proposal-based Faster-RCNN based approach like [Mask-RCNN](https://arxiv.org/abs/1703.06870) or [MaskLab](https://arxiv.org/abs/1712.04837) and the latest [PANet](https://arxiv.org/abs/1803.01534), achieving state-of-the-art results on multiple datasets ([CityScapes](https://www.cityscapes-dataset.com/), [COCO](http://cocodataset.org/), [MVD](https://www.mapillary.com/dataset/vistas)). See tutorial on Mask-RCNN [here](http://kaiminghe.com/iccv17tutorial/maskrcnn_iccv2017_tutorial_kaiminghe.pdf).

There are three fundamental flaws in a proposal-based instance segmentation architecture. First, two objects may share the same bounding box, or a very similar boxes. In this case, the mask head, has no way of telling which object to pick in the box. This is a serious problem with stringy like object that have low fill rate in their bounding box (e.g. bicycles and chairs). Second, there is nothing in the architecture preventing a pixel to be shared between two instance. Third, the number of instances is limited by the number of proposals processed by the network (usually hundreds).

![](https://cdn-images-1.medium.com/max/1600/0*hKOJOX99Mxg_O3Yr.)

More over, the architecture is complex and hard to tune and “debug”. In object detection, a precursor to this problem, there are already successes being made to use a simpler, single-stage, architectures e.g. [RetinaNet](https://arxiv.org/abs/1708.02002).

With instance embedding, each object is assigned a “color” in a n-dimensional space. The network processes the image and produces a dense output, same size as the input image. Each pixel in the output of the network is a point in the embedding space. Pixels that belong to the same object are close in the embedding space while pixels that belong to different objects are distant in the embedding space. Parsing the image embedding space involves some sort of clustering algorithm.

### Paper 1: Semantic Instance Segmentation with a Discriminative Loss Function

[Bert De Brabandere](https://arxiv.org/find/cs/1/au:+Brabandere_B/0/1/0/all/0/1), [Davy Neven](https://arxiv.org/find/cs/1/au:+Neven_D/0/1/0/all/0/1), [Luc Van Gool](https://arxiv.org/find/cs/1/au:+Gool_L/0/1/0/all/0/1) [https://arxiv.org/abs/1708.02551](https://arxiv.org/abs/1708.02551)
[https://github.com/DavyNeven/fastSceneUnderstanding](https://github.com/DavyNeven/fastSceneUnderstanding)

![](https://cdn-images-1.medium.com/max/1600/0*PYT_oRjBL-4tGm0B.)

**The Loss.** This paper uses a contrastive loss function complied from 3 parts:

(1) A pull force. Penalizing the distance of all elements of the same instance from their mean. That is, taking all pixels of an instance and calculating their mean. The pull force will draw all pixel embeddings of the same instance to the same point. In short, reducing the variance of the embedding per instance.

(2) A push force. Taking all the center points (in the embedding space, not the spatial center) and pushing them farther apart.

(3) Regularization. Centers shouldn’t be too far from the origin.

![](https://cdn-images-1.medium.com/max/1600/1*n7d5zeLPcxRq_2NpQ1X90g.png)

Alpha and beta are used with the value of 1 and gamma is set to be 0.001. Both deltas are thresholds for the pull and push forces.

**Parsing**. After obtaining the semantic segmentation map (car, dog, computer, …) we subdivide each class mask to instances. This is done by picking a random unassigned point in the semantic mask and iteratively applying the mean-shift algorithm to find the mean point of the instance.

The first hypothesis for the mean is the embedding of the random pixel which was initially picked. Then a set of points is expanded around that point (in the embedding space) and their mean is then again calculated and the process repeats until the change to the mean is not significant. In my experience, it take no more than 10 iterations for the algorithm to converge. And most of the time 3–4 iterations are enough.

The radius used to expand the instance mask in embedding space is the same as the pull threshold. Theoretically, if the test error is 0, and the minimum distance between centers is at least twice as large as the pull threshold for the variance component we can use these thresholds to parse the image. All points at a distance of no greater than the pull threshold should belong to the same instance. Since the test error is almost never 0, the mean-shift algorithm is used to find the center of a high density part of the embedding.



**Source of error**

![](https://cdn-images-1.medium.com/max/1600/1*eAgv5h2BhHwWumPP0bzgzQ.png)

These results show where most of the error comes from on the Cityscapes Dataset. if the semantic segmentation is not predicated and the ground truth is rather used, the AP50 results jumps from 40.2 to 58.5. If the actual center is also used and not estimated using mean-shift, the score gains almost another 20 points, reaching to 77.8. Current state of the art results without pretraining on COCO is 57.1 using [PANet](https://arxiv.org/abs/1803.01534) (see [dashboard](https://www.cityscapes-dataset.com/benchmarks/.)). Same as using the semantic segmentation ground truth. We learn that the embedding itself is probably pretty good.

### Example Embedding

Below is an example instance embedding produces by a network trained by, yours truly. It is used to solve the problem presented by the [Data Science Bowl 2018](https://www.kaggle.com/c/data-science-bowl-2018), currently running on Kaggle. The purpose is to find cell nuclii in medical images.

The top-left image is the original image. Middle-top image is the semantic segmentation (here with only two classes, background and foreground). The rest of the images are the first 7 channels of 64 of embedding space. It is evident from the embedding that the network learned channels that spatially differentiate the nuclei. Example for diagonal or horizontal coding. Some encode the distance from center of the image. However, inside an instance the color is homogeneous. This is giving us some insight to how the network learned to segment instances.

![](https://cdn-images-1.medium.com/max/1600/0*2U0aA1GvO-m3Rw8Y.)

### Paper 2: Semantic Instance Segmentation via Deep Metric Learning

[Alireza Fathi](https://arxiv.org/find/cs/1/au:+Fathi_A/0/1/0/all/0/1), [Zbigniew Wojna](https://arxiv.org/find/cs/1/au:+Wojna_Z/0/1/0/all/0/1), [Vivek Rathod](https://arxiv.org/find/cs/1/au:+Rathod_V/0/1/0/all/0/1), [Peng Wang](https://arxiv.org/find/cs/1/au:+Wang_P/0/1/0/all/0/1), [Hyun Oh Song](https://arxiv.org/find/cs/1/au:+Song_H/0/1/0/all/0/1), [Sergio Guadarrama](https://arxiv.org/find/cs/1/au:+Guadarrama_S/0/1/0/all/0/1), [Kevin P. Murphy](https://arxiv.org/find/cs/1/au:+Murphy_K/0/1/0/all/0/1) 
[https://arxiv.org/abs/1703.10277](https://arxiv.org/abs/1703.10277)

The main contribution in this paper is a seediness score which is learned for each pixel. The score tells us if the pixel is a good candidate to expand a mask around. In the previous paper the seed was chosen at random and then the center was refined using the mean-shift algorithm. Here, only one expansion is made.

![](https://cdn-images-1.medium.com/max/1600/0*S382qDDi2S8nFUVd.)

The paper propose to learn several possible seeds for each pixel. We learn a seed for each radius (in embedding space) and class. So if we have C classes and we learn T bandwidths (radii) we have CxT seed “proposals” per pixel. For each pixel only the proposal with the highest score is considered.

**Embedding Loss.**In this paper, the embedding is penalized for pairs of pixels. We consider pairs that are of the same instance and pairs from different instances.

![](https://cdn-images-1.medium.com/max/1600/0*C0JEC4d2sHPRMPIb.)

The paper uses a modified logistic function that transforms the euclidean distance in embedding space to the [0, 1] domain. Pairs that are close in the embedding space will be assigned a value close to 1 by the function, pairs that are distant will approach 0.

Naturally, logloss is used as a loss function. Instances sizes may vary so, in order to mitigate this imbalance issue, pairs are weighted with respect to the size of the instance they are a part of.

![](https://cdn-images-1.medium.com/max/1600/1*aklQ2hqQpbJ9u-S-p95rzw.png)

**Seediness Loss.**For each pixel, the model learns several seediness scores. One score for each combination of bandwidth (radius in embedding space) and class. Since the seediness score is close but not the same as semantic segmentation, the ground truth for each is determined every time the embedding is evaluated. A mask is expanded around the embedding of a pixel and if the IoU with a ground truth instance exceeds a certain threshold, the pixel is considered as a seed for the class of the instance. The loss will then penalize a low seediness score for this class.

![](https://cdn-images-1.medium.com/max/1600/0*26PsDHQzweWbcccn.)

Only 10 or so seeds are evaluated per image in each batch, picked randomly. Several such models are learned, one for each bandwidth. The wider the bandwidth, the larger the object. In a way, the bandwidth that received the highest score, is the model’s way to convey it’s estimation to the instance size (with respect to the distances in the embedding space).

**Training Procedure.**The paper uses ResNet-101 backbone pretrained on the COCO dataset. Training starts with no classification/seediness predication i.e. λ=0 and progresses to 0.2 as the embedding is more stable.

![](https://cdn-images-1.medium.com/max/1600/1*25y46E_ezGr94BlNK6yU9A.png)

The backbone is evaluated at different scales (0.25, 0.5, 1, 2) and the concatenated results are fed to the seediness and embedding heads.

**Parsing**. The procedure pretty straight forward since the seeds learned. The paper proposes a procedure to pick the best seed set for an image. It optimizes for a high seediness score on one hand and diversity in the embedding space on the other.

![](https://cdn-images-1.medium.com/max/1600/0*1hPGRNNFwr0d5WYG.)

Seeds are chosen iteratively, each new seed is chosen to be distant in the embedding space from the previously selected seeds. The first seed selected is the pixel with the highest seediness score in the image. The second one will be the seed that on one hand has a high seediness score and on the other hand is not close in the embedding space. The balance between the two requirements is controlled using the parameter alpha. Alpha is a to be tuned, the range tested for this parameter is between 0.1 and 0.6. Unlike NMS, diversity in embedding space is encouraged, rather than spatial diversity.

![](https://cdn-images-1.medium.com/max/1600/0*-ZNb-CVraFaUCFE_.)

### Paper 3: Recurrent Pixel Embedding for Instance Grouping

[Shu Kong](https://arxiv.org/find/cs/1/au:+Kong_S/0/1/0/all/0/1), [Charless Fowlkes](https://arxiv.org/find/cs/1/au:+Fowlkes_C/0/1/0/all/0/1)
[https://arxiv.org/abs/1712.08273](https://arxiv.org/abs/1712.08273)
[https://github.com/aimerykong/Recurrent-Pixel-Embedding-for-Instance-Grouping](https://github.com/aimerykong/Recurrent-Pixel-Embedding-for-Instance-Grouping)

![](https://cdn-images-1.medium.com/max/1600/0*RFwW286zQuYOeCPp.)

This paper proposes to have the embedding on a n-sphere and to measure proximity of pixels using the cosine distance. However, the main contribution is this paper is the recurrent grouping model, based on a modified version of the Gaussian Blurring Mean-Shift (GBMS) algorithm.

GBMS is an iterative algorithm like the simple mean-shift algorithm used to find instance centers in the first paper. In this version, all the pixels are considered to be potential seeds. All pixels are updated at each iteration with respect to the density around them. Moving toward a “center of gravity”, as if the embedding space of the image was a nebula producing planets. The farther points are from each other, the less of the effect they will have on one another. The distance is controlled by the bandwidth of the Gaussian, it’s standard deviation, as is clear from the algorithm below.

![](https://cdn-images-1.medium.com/max/1600/0*NBDyeIcrSvhvqx42.)

For GBMS there are cubic convergence guarantees so eventually we should get very dense, almost point-like, clusters after applying the transform several times. For more on GBMS see [here](http://www.cs.cmu.edu/~aarti/SMLRG/miguel_slides.pdf).

In order to incorporate the algorithm in the network, it has be expressed using operations on matrices.

![](https://cdn-images-1.medium.com/max/1600/0*3z21-wzUPCuRZ9kc.)

Simply applying the algorithm described above is does not make sense since the embedding are on a sphere and their proximity is measured using the cosine transform. The affinity matrix, describing the distances between all points is calculated using the following transformation:

![](https://cdn-images-1.medium.com/max/1600/0*sIDr2K3mkoiqLeNR.)

Measuring distances on the sphere, rather than using the L2 norm. In addition, after applying a GBMS step, it is required to normalize the resulting embeddings so they will be on the unit sphere.

![](https://cdn-images-1.medium.com/max/1600/0*hs7RhCoBikkGdSzw.)

**Training.** Pairwise pixel loss is used, similarly to the previous paper with a threshold on the distance required from dissimilar pairs (alpha). Each pair is evaluated using a calibrated cosine distance which ranges [0,1] instead of [-1, -1].

![](https://cdn-images-1.medium.com/max/1600/1*slRREbiolW6BFjeNNGGGpw.png)

The loss is back-propagated through each application of the recurrent grouping model. Later stages of application will surface only very difficult cases. The authors compare this property to hard negative mining used in the training of Faster-RCNN, for example.

![](https://cdn-images-1.medium.com/max/1600/0*optGvm68SkyMuzbW.)

The authors are using 0.5 as value for alpha in the paper. Notice that the size of the instance is used to re-balance the loss between small and large instances.

**Parsing.** After several applications of the grouping module, the clusters should be very dense, picking values at random should produce good enough seeds.

For practical purposes, it makes sense to use only some of the pixels in the GBMS steps since computing the similarity matrix might prove prohibitively expensive. The amount of pixels taken is a speed/accuracy trade-off consideration.

### Other approaches

Instance embedding is not the only alternative to proposal based networks. Here are some papers that use other methods to solve the problem of instance segmentation

* **End-to-End Instance Segmentation with Recurrent Attention**
[https://arxiv.org/abs/1605.09410](https://arxiv.org/abs/1605.09410)

* **Deep Watershed Transform for Instance Segmentation**
[https://arxiv.org/abs/1611.08303](https://arxiv.org/abs/1611.08303)

* **Associative Embedding: End-to-End Learning for Joint Detection and Grouping**
[http://ttic.uchicago.edu/~mmaire/papers/pdf/affinity_cnn_cvpr2016.pdf](http://ttic.uchicago.edu/~mmaire/papers/pdf/affinity_cnn_cvpr2016.pdf)

* **SGN: Sequential Grouping Networks for Instance Segmentation**
[https://www.cs.toronto.edu/~urtasun/publications/liu_etal_iccv17.pdf](https://www.cs.toronto.edu/~urtasun/publications/liu_etal_iccv17.pdf)

### Summary

Compared to proposal based solutions, the results from these papers are not competitive. We have reviewed 3 papers with different solution approaches to the loss and the parsing.

(1) **Semantic Instance Segmentation with a Discriminative Loss Function**
Used a non-pairwise loss function. Producing far richer gradients using all the pixels in the image.

(2) **Semantic Instance Segmentation via Deep Metric Learning**
Introduces a seediness model, helping us to classify and pick the best seeds at the same time, optimizing for speed.

(3) **Recurrent Pixel Embedding for Instance Grouping**
GBMS, a variant of mean-shift, was used inside the network in both training and parsing. Creates very dense clusters.

These approaches could probably be combined and refined to produce far better results. They are simpler and possibly faster than proposal based approaches while avoiding the fundamental flaws mentioned in the intro to this paper at the same time.

**Contact:** me@barvinograd.com

**Slides:**[https://goo.gl/iTC9aS](https://goo.gl/iTC9aS)



