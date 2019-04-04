# It’s a no-brainer! Deep learning for brain MR images

Written by Atli Kosson and Henrik Marklund

![](https://cdn-images-1.medium.com/max/1600/1*PPcY2o1SCZKVaQs8BNNGVw.png)

Finding cats and dogs in pictures using deep learning is easy! Finding tumors and lesion in the brain using deep learning is harder, but we are getting there.

When planning the treatment and tracking the progression of various brain diseases, locating the exact regions affected is important. Consider the case of brain tumors. When deciding whether or not to perform surgery, it is crucial to know exactly where the tumor is located. If the tumor is located in some area that can’t be removed, without affecting vital surrounding normal tissue, the brain stem for instance, you can’t operate. When treating patients with multiple sclerosis (MS), it is important to be able to locate lesions. This enables doctors to track disease progression and to determine if a treatment is working. Magnetic Resonance Imaging (MRI) images can be used to image the brain in 3D but a highly specialized doctor still has to review the resulting images and manually mark the affected volume. This is a difficult and time consuming task. Algorithms that could at least partially automate the process would be very valuable. Also, this would enable the creation of large scale datasets on tumors and lesions. These large datasets are currently lacking but much needed for research.

![](https://cdn-images-1.medium.com/max/1600/1*8XpuBtLuVGc12ZfLAa_h4A.jpeg)

In this post we will discuss this segmentation task in detail and cover:

* The segmentation problem

* The data from MR scans

* Three ML challenges with brain segmentation

* Brain Tumor Segmentation Challenge (BraTS)

* Case-study on lesion segmentation reaching human performance

#### The segmentation problem

In semantic segmentation, we want to determine the class (type of object) of each pixel in an image. Consider the image of a cat below. It has a cat, region with grass, some trees and the sky in it. On the right is an image which specifies the class of each pixel.

![](https://cdn-images-1.medium.com/max/1600/1*YLMhuTGxQDIcPIVCtNtkCQ.jpeg)

Deep neural networks are able to perform very well on this kind of segmentation. Architectures often involve multiple convolutional layers and pooling layers (see Stanford’s [CS231n](http://cs231n.stanford.edu/) for details). These layers compress the image into a small neural representation of the image. This representation is then fed through a series of upsampling or deconvolutional layers until we end up with an image that is the same size as the original image. The final image has multiple channels, one for each type of object that we can classify. Each channel specifies whether the object corresponding to the channel is present at each location in the image. An example architecture by [Badrinarayanan et al. (2015)](https://arxiv.org/pdf/1511.00561.pdf) can be seen below (image retrieved from their paper).

![](https://cdn-images-1.medium.com/max/1600/1*erm427klBUNcrc2OqmKEvA.png)

Segmenting medical images comes along with some additional challenges. The main challenge is the size of datasets for the task, which are typically very small (hundreds) compared to very large datasets of more common images (millions of images). This makes it difficult to train very deep architectures, so the architectures used for the medical segmentation task are usually simpler. It is also a much harder task to determine the class of each pixel: human experts only agree on perhaps 80% of lesion pixels. Even with an autopsy it would be difficult to determine the exact "ground truth" segmentation. Furthermore, the shape, size and appearance of tumors and lesions and can vary widely and they can have relatively “soft” boundaries compared to object in traditional images. Finally, the images are three dimensional.

#### MR Images: what are they?

The data used for segmenting tumors and lesions come from MR images. The patient is inserted into a tunnel that is essentially a solenoid with a strong magnetic field inside. This causes all protons in the body to ‘align’ themselves so their quantum spin is the same. A pulse of oscillating magnetic field is then used to disrupt this alignment. When the protons return to equilibrium they send out an electromagnetic wave. The wave signals are recorded and through a series of steps, a set of images (slices) of the brain can be gathered. Different kinds of tissue, for instance white matter (axons, more fatty) and grey matter (neuron bodies and dendrites, less fatty) have different chemical composition which causes them to emit different signals. Importantly, depending on the type of stimulation (i.e. sequences) used to disrupt the protons, different images will be obtained. These different images will be sensitive to different kinds of tissue. Four common sequences that are obtained are T1, T1 with contrast (T1C), T2 and FLAIR. [Video: Doctor going through MR images looking for tumors.](https://www.youtube.com/watch?v=Z2Ulvt1RWiw&feature=youtu.be) If you’d like to learn more, [please take a look at our group’s detailed blogpost on MR images](https://medium.com/stanford-ai-for-healthcare/dont-just-scan-this-deep-learning-techniques-for-mri-52610e9b7a85)

![](https://cdn-images-1.medium.com/max/1600/1*eTkBMyqdg9JodNcG_O4-Kw.jpeg)

### Three challenges with brain images

Well tuned deep architectures with tens or hundreds of layers have been shown to perform very well on large datasets like ImageNet. The datasets available for brain segmentation tasks seem to be too small to train these deep architectures. Most of the medical architectures only have few layers, often less than five. Data augmentation is used but is often limited to just flipping the images. Apart from this size constraint we will now discuss three additional challenges with lesion and tumor segmentation.

**Challenge 1: Memory intensive 3D data.**

Because of the extra dimension, 3D convolutional networks are more memory intensive than 2D networks. In a 3D convolutional network, it is not only the input image that is larger, but also the representations after each layer in the network. These image representations need to be cached for back propagation, consuming extensive memory.

Therefore, the volumes from the MR images are rarely used directly as input to the machine learning model. Rather, the images are sliced and split into chunks before training. The goal is to balance the tradeoff between computational efficiency and making use of contextual information (nearby tissue provides important contextual information). There are many different approaches:

1. Create 2D slices (very common).

2. Train on smaller patches of the MR images.

3. When classifying each pixel, extract patches of 2–3 different sizes surrounding the pixel [(Havaei et al., 2017)](https://www.sciencedirect.com/science/article/pii/S1361841516300330?via=ihub).

4. Two-pathway models: take into account the local tissue just around the voxel (pixel but in 3D), as well as taking into account the global aspect of location [(Havaei et al., 2017)](https://www.sciencedirect.com/science/article/pii/S1361841516300330?via=ihub).

5. Sample smaller 3D chunks and use a 3D convolutional net ([Kamnitsas et al](http://Efficient multi-scale 3D CNN with fully connected CRF for accurate brain lesion segmentation)).

The model below uses approaches 1, 2 and 4.

![](https://cdn-images-1.medium.com/max/1600/1*ko3LIqWZYntj9O2ygThqIg.jpeg)

The current trend is to do less preprocessing and input more of the data at once. This seems to be the consequence of datasets’ growing size and the general trend in deep learning.

**Challenge 2: Smoothing voxel-wise predictions**

Most brain segmentation models work with small regions at a time and the prediction for each pixel is made independently of the predictions for nearby pixels. This kind of model doesn't take into account the relation between nearby pixels, for instance an individual healthy pixel in the middle of a tumor is very unlikely. We can use post-processing methods to smooth the output of the model.

![](https://cdn-images-1.medium.com/max/1600/1*la9Pb0DicI5XjUaRUIc1zw.png)

1. Conditional random fields (CRFs) are probabilistic graphical models that model the conditional probability distribution over some output given an input (see figure above). In brain segmentation the inputs are the prediction distributions for each pixel. The output is an assigned label for each pixel. The CRF captures the likelihood of each possible combination of assignments to the output pixels. It breaks the joint probability distribution into factors that model the joint distribution of an output pixel, the adjacent output pixels and the corresponding input distribution. We can do maximum likelihood estimation to find the most probable configuration of the output nodes given the inputs.

2. Use the pixel-wise-probabilities as input to a second neural network. This network is usually a shallow convolutional network. In other areas RNNs have largely replaced CRFs but they do not seem to be used for brain segmentation, perhaps because they might be prone to overfitting on the relatively small datasets available.

With larger input patches and deeper models the receptive field used for classifying each pixel becomes larger (see [Garcia-Garcia](https://arxiv.org/abs/1704.06857) for a discussion). This makes the inputs for two nearby pixels more similar than for smaller receptive fields. This in turn decreases the probability of mislabeling individual pixels and therefore the need for this kind of post-processing.

**Challenge 3: Missing data**

An additional challenge is when one of the sequences is missing. Different hospitals may have different practices and not the same procedure when performing MRIs or other. For instance, the FLAIR modality, which has useful contrast properties for finding lesions, may be missing. Several methods have been proposed:

1. Build several models trained on the different sequences (e.g one model specifically for T1 and FLAIR in combination).

2. Impute the missing data. Predict the missing data based on the available data (requires training of a model for each combination of modality).

3. Build one model that takes an arbitrary number of sequences. [Havaei et al (2016)](https://arxiv.org/abs/1607.05194) do this by having a convolutional pipeline for each sequence. The feature maps from each modality is then merged by computing the mean and the variance. Each pipeline learns “to separately map each modality into an embedding common to all modalities”. The key idea is that the expectation of the mean and variance does “not depend on the number of terms (modalities)”. Thus having more modalities, or sequences, available lets us estimate these statistics better, but missing one modality will not completely throw off the model.

### BraTS Challenge

When training ML algorithms, the data is crucial and so is understanding the data. We will now take a deeper look at a common dataset used in brain tumor segmentation: the BraTS Challenge 2015 dataset (Brain Tumor Segmentation Challenge).

![](https://cdn-images-1.medium.com/max/1600/0*nWWUY0WPLb3Kx-ot.)

**Dataset size:**220 subjects with high grade tumors. 54 subjects with low grade tumors.

**Input:**For each subject we are given four MR images, i.e four three-dimensional volumes (FLAIR, T1W, T1C and T2).

**Labels:**Each example has been segmented and labelled by 5 raters (certified doctors). Each voxel are labelled: (1) Healthy, (2) Edema, (3) Necrosis, (4) Non-enhancing tumor, (5) Enhancing tumor.

The machine learning task is then to label each voxel with one of these 5 labels.

#### BraTS Evaluation

When comparing performance in the BRATS challenge, evaluation is based on three different combinations of the labels: complete, core and enhancing. You train your algorithm on all five labels, but when you evaluate you evaluate as if it was three different binary classification tasks. For instance, the category complete includes all affected area. To perform well in this category, it does not matter if your algorithm confuses Necrosis with Edema for instance.

![](https://cdn-images-1.medium.com/max/1600/1*iOvPId9PHQso0IFeg43SDw.png)

**The Dice Score**

Due to the huge class imbalance, a regular accuracy score does not say a lot. Thus, for each category, the common F1 score is used (in the literature known as the [Dice score](http://smial.sri.utoronto.ca/LV_Challenge/Evaluation.html)). It measures the overlap between manual segmentation (the combination of expert raters’ opinion: the fused score) and the machine learning segmentation.

![](https://cdn-images-1.medium.com/max/1600/0*g_GONWZ5U_jR5jL2.)

A dice score of 1 indicates perfect agreement with consensus expert rating.

![](https://cdn-images-1.medium.com/max/1600/1*DGvt4kbR9_I8M8ulX43vBA.png)

### Case-study: Location Sensitive Deep Convolutional Neural Networks for Segmentation of White Matter Hyperintensities

![](https://cdn-images-1.medium.com/max/1600/0*lFid3vQ2zJirALhn.)

Seeing the whole process, from data preprocessing, to architecture design, to final performance is a helpful task. We’ll carry out this process by looking at a paper by [Ghafoorian et al](https://arxiv.org/abs/1610.04834). on brain lesion segmentation. This paper is exciting as they develop a neural architecture that almost achieves human performance segmenting white matter lesions for multiple sclerosis, Alzheimer's and similar diseases.

**Data**
A total of 378 training, 42 dev and 50 independent test images (each with two modalities: T1 and FLAIR). All sets were segmented by a human expert creating the “ground truth” (lesion / not lesion). The independent test set was also annotated by a second expert.

**Preprocessing**
The two imaging modes were spatially aligned. Other non-brain structures (the skull, eyes etc.) were removed. Bias field correction was applied and the image intensities for each patient were normalized to a range between 0 and 1.

**Inputs**
The model works with 2D slices. When classifying each pixel, patches of three different sizes (32 × 32, 64 × 64 and 128 × 128) surrounding the pixel are extracted. Each patch has two channels (T1 and FLAIR). The larger patches are downsampled to 32 × 32.

![](https://cdn-images-1.medium.com/max/1600/1*rkGPmkhmUQ5t9DCfENmUcA.png)

**Model**

![](https://cdn-images-1.medium.com/max/1600/0*mYw5nEWD40L6waoB.)

The model takes in the three patches. Each patch is then processed by a series of convolutional layers (with an identical structure for each patch). The results are then combined in a fully connected layer. Additional spatial features are added to this fully connected layer. These are the 3D location of the target pixel, the in-plane distance to the left ventricle, the right ventricle and the cortex, the distance to the midsagittal brain surface and finally the prior probability of a lesion occurring at this location. This is fed into another fully connected layer and then finally into a binary classifier. All nonlinearities are ReLUs.

![](https://cdn-images-1.medium.com/max/1600/1*7er9N_k9qRcUVZbMHpegqA.png)

**Training** 
Because normal non-lesion pixels (negative samples) are much more common, the lesion pixels (positive samples) are oversampled during training. In this way around 50% training patches will correspond to a lesion. To prevent overfitting, they apply dropout regularization to the fully connected layers. Dropout probability is of 0.3.

**Performance**
The model achieves a Dice score of 0.795 on the test set which is very close to the score of the other human expert, 0.805. Thus, the difference in scoring between the model and an expert is close to the difference in scorings between two human experts.

![](https://cdn-images-1.medium.com/max/1600/0*j-JqqZR9OCJo3skR.)

### Final thoughts

Semantic segmentation through deep learning clearly has a lot of potential in neurology and neurosurgery. It will enable us to diagnose various brain diseases and track their progression which is imperative for effective treatment. Moreover, automatic segmentation would let us create larger datasets of MR images segmented with tumors, lesions etc. These larger datasets would help us research these diseases in depth, studying, for instance, tumors progression over time or for use in clinical trials. This could potentially lead to new drugs and treatments. This is part of the larger transformation of healthcare by AI that will make healthcare smarter, more effective, and more affordable.

### Acknowledgments

We are extremely grateful to Matthew Lungren MD MPH, Assistant Professor of Radiology and Bhavik Patel, MD, MBA, Assistant Professor of Radiology at the Stanford University Medical Center for providing valuable feedback. We also want to thank Pranav Rajpurkar, Jeremy Irvin, Chris Lin and Jessica Wetstone from the [Stanford ML Group](https://stanfordmlgroup.github.io/) and Christopher Bucknell, Machine Learning student at Stanford for their comments.

### References

Badrinarayanan, V., Kendall, A., & Cipolla, R. (2015). Segnet: A deep convolutional encoder-decoder architecture for image segmentation. arXiv preprint arXiv:1511.00561.

Brosch, T., Tang, L. Y., Yoo, Y., Li, D. K., Traboulsee, A., & Tam, R. (2016). Deep 3D convolutional encoder networks with shortcuts for multiscale feature integration applied to multiple sclerosis lesion segmentation. IEEE transactions on medical imaging, 35(5), 1229–1239.

CS228, Probabilistic Graphical Models, lecture notes on Conditional Random Fields (CRF), [https://ermongroup.github.io/cs228-notes/representation/undirected/](https://ermongroup.github.io/cs228-notes/representation/undirected/)

Menze, B. H., Jakab, A., Bauer, S., Kalpathy-Cramer, J., Farahani, K., Kirby, J., … & Lanczi, L. (2015). The multimodal brain tumor image segmentation benchmark (BRATS). IEEE transactions on medical imaging, 34(10), 1993–2024 [http://ieeexplore.ieee.org/document/6975210/](http://ieeexplore.ieee.org/document/6975210/)

Havaei, M., Guizard, N., Chapados, N., & Bengio, Y. (2016, October). HeMIS: Hetero-modal image segmentation. In International Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 469–477). Springer International Publishing.

Havaei, M., Davy, A., Warde-Farley, D., Biard, A., Courville, A., Bengio, Y., … & Larochelle, H. (2017). Brain tumor segmentation with deep neural networks. Medical image analysis, 35, 18–31.

Kamnitsas, K., Ledig, C., Newcombe, V. F., Simpson, J. P., Kane, A. D., Menon, D. K., … & Glocker, B. (2017). Efficient multi-scale 3D CNN with fully connected CRF for accurate brain lesion segmentation. Medical image analysis, 36, 61–78.

Garcia-Garcia, A., Orts-Escolano, S., Oprea, S., Villena-Martinez, V., & Garcia-Rodriguez, J. (2017). A review on deep learning techniques applied to semantic segmentation. arXiv preprint arXiv:1704.06857.

Ghafoorian, M., Karssemeijer, N., Heskes, T., van Uden, I., Sanchez, C., Litjens, G., … & Platel, B. (2016). Location sensitive deep convolutional neural networks for segmentation of white matter hyperintensities. arXiv preprint arXiv:1610.04834.

Pereira, S., Pinto, A., Alves, V., & Silva, C. A. (2016). Brain tumor segmentation using convolutional neural networks in MRI images. IEEE transactions on medical imaging, 35(5), 1240–1251.

