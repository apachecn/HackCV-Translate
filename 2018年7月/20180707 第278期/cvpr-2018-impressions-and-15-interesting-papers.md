# CVPR 2018. Impressions and 15 interesting papers

This [CVPR](http://cvpr2018.thecvf.com/) was great in the sense of meeting new people and de-virtualize with others. Regarding new cool results — not so much. I mean, lots of good papers, but nothing that ground-breaking, at least in areas of my interest.

0) If you present graphics-related poster, e.g. debluring, super-resolution, etc, DO NOT PRINT IT ON CLOTH. Especially on coarse fabric. 
[Scale-recurrent Network for Deep Image Deblurring](https://arxiv.org/abs/1802.01770) is nice work, but you cannot see anything due to cloth texture :(

![](https://cdn-images-1.medium.com/max/1600/1*0NQ5dgJ_SkrI7SyOp3YTnQ.jpeg)

1) GANs, domain adaptation, autonomous driving and very minor architectural tricks are everywhere.

2) East Europe is still badly represented, but grows. Hungary, Ukraine, Romania, Poland, Turkey. Not many papers, but at least some and the number is increasing. That is great.

3) Workshop “[Being good CVPR citizen](https://www.cc.gatech.edu/~parikh/citizenofcvpr/)” was surprisingly cool. Especially, “What is good research” by Vladlen Koltun and “Calendar, not lists” by Devi Parikh. All slides and videos are available at link above

4) Graph neural networks of various flavors were discovery for me. Good tutorial by [Michael Bronstein](http://www.inf.usi.ch/bronstein/teaching_tutorial.html)

5) Workshop paper “[Markov chain neural networks](https://arxiv.org/abs/1805.00784)”. Simple idea — add additional random input variable to control desired output and then you can control conditionally control output at test time.

![](https://cdn-images-1.medium.com/max/1600/1*3jRVScdL06relqzN0kGwcQ.jpeg)

6) Three versions of differentiable SLAM: 
 — [MapNet: An Allocentric Spatial Memory for Mapping Environments](http://www.robots.ox.ac.uk/~joao/mapnet/) from Oxford
 — NVIDIA MapNet “[Geometry Aware learning of Maps for Camera Localizaton](https://arxiv.org/abs/1712.03342)”. 
 — DeepSLAM from Salakhutdinov‏ group: [Global Pose Estimation with Attention-based RNNs](https://arxiv.org/abs/1802.06857)

7) RANSAC is still improving. Two cool papers: [Latent RANSAC](https://arxiv.org/abs/1802.07045) and [Graph-Cut RANSAC](https://arxiv.org/abs/1706.00984)

8) Workshop on [large scale landmark recognition](https://landmarkscvprw18.github.io/): key take-away messages: global features work good enough, but pooling and ensembling could be improved. If you need local features — go for DELF. Here are twitter-translations:
[https://twitter.com/ducha_aiki/status/1008815959291777025](https://twitter.com/ducha_aiki/status/1008815959291777025)
[https://twitter.com/ducha_aiki/status/1008818384664838145](https://twitter.com/ducha_aiki/status/1008818384664838145)

![](https://cdn-images-1.medium.com/max/1600/1*RPdNR0BlPAMSfBa9W6mBUQ.jpeg)

8) In contrast, [InLoc: Indoor Visual Localization with Dense Matching and View Synthesis](https://arxiv.org/abs/1803.10368) and [Benchmarking 6DOF Outdoor Visual Localization in Changing Conditions](https://arxiv.org/abs/1707.09092) papers say different: local features are cool, but you need to be dense, not sparse

9) Local features-related papers thread — [https://twitter.com/ducha_aiki/status/1009192898061979648](https://twitter.com/ducha_aiki/status/1009192898061979648)

Other papers I like:

* [Perception-distortion trade-off](https://arxiv.org/abs/1711.06077). You cannot have, e.g. both nice details AND be close to ground truth in your reconstruction if information was lost, need to select. The next question is how to get algorithm given your priorities and where exactly trade-off lies.

* [A PID Controller Approach for Stochastic Optimization of Deep Networks](http://www4.comp.polyu.edu.hk/~cslzhang/paper/CVPR18_PID.pdf)
Parallel between SGD and PID controller, which enables to re-use known control theory approaches for learning.

* [Adversarial metric learning](https://arxiv.org/abs/1802.03170): turn easy negatives into hard by GAN style-feature transfer. One of the few non-boring GAN papers.

* [Learning by Asking Questions](https://arxiv.org/abs/1712.01238). Network learns to perform visual question answering by learning to ask questions itself.

* 2 papers about rank-based losses: two papers about optimizing average precision: here is a [closed-form solution of average precision and its differentiable formulation](https://arxiv.org/abs/1804.05312). 
[Efficient Optimization for Rank-based Loss Functions](https://arxiv.org/abs/1604.08269): authors don`t use close form, but propose efficient evaluation by exploiting quick-search-like algorithm.

* [Self-supervised Learning of Geometrically Stable Features Through Probabilistic Introspection](https://arxiv.org/abs/1804.01552) learning features by predicting matchability.

* [Analyzing Filters Toward Efficient ConvNet](http://openaccess.thecvf.com/content_cvpr_2018/html/Kobayashi_Analyzing_Filters_Toward_CVPR_2018_paper.html): steerable filters on first layers, 1st level DCT for fully-connected, BoW-based pooling for the pre-FC layers. Improved VGGNet and ResNet.

* [Learning to Extract a Video Sequence From a Single Motion-Blurred Image ](http://openaccess.thecvf.com/content_cvpr_2018/html/Jin_Learning_to_Extract_CVPR_2018_paper.html)New motion deblur task formulation: it is better and easier to extract frame sequence, which produced blurred image, than single deblured image.

* Atlanta world: if you have 4 kinds of dominant orientations in your 2D world, use 4D coordinate systems, not minimal 2. 
Paper: [Globally Optimal Inlier Set Maximization for Atlanta Frame Estimation](http://openaccess.thecvf.com/content_cvpr_2018/html/Joo_Globally_Optimal_Inlier_CVPR_2018_paper.html)

P.S. Another interesting report from CVPR 2018 [https://olgalitech.wordpress.com/2018/06/30/cvpr-2018-recap-notes-and-trends/](https://olgalitech.wordpress.com/2018/06/30/cvpr-2018-recap-notes-and-trends/)

