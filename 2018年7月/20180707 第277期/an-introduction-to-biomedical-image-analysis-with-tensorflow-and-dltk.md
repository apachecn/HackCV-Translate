# An Introduction to Biomedical Image Analysis with TensorFlow and DLTK

By Martin Rajchl, S. Ira Ktena and Nick Pawlowski — Imperial College London

[DLTK](https://dltk.github.io/), the Deep Learning Toolkit for Medical Imaging extends [TensorFlow](https://www.tensorflow.org/) to enable deep learning on biomedical images. It provides specialty ops and functions, implementations of models, [tutorials](https://github.com/DLTK/DLTK/tree/master/examples/tutorials) (as used in this blog) and [code examples for typical applications](https://github.com/DLTK/DLTK/tree/master/examples/applications).

This blog post serves as a quick introduction to deep learning with biomedical images, where we will demonstrate a few issues and solutions to current engineering problems and show you how to get up and running with a prototype for your problem.

![](https://cdn-images-1.medium.com/max/1600/1*pfe19FYqq9GKAUUgBBjXjg.jpeg)

### Overview

**What is biomedical image analysis and why is it needed?** Biomedical images are measurements of the human body on different scales (i.e. microscopic, macroscopic, etc.). They come in a wide variety of imaging modalities (e.g. a CT scanner, an ultrasound machine, etc.) and measure a physical property of the human body (e.g. radiodensity, the opacity to X-rays). These images are interpreted by domain experts (e.g. a radiologist) for clinical tasks (e.g. a diagnosis) and have a large impact on decision making of physicians.

![](https://cdn-images-1.medium.com/max/1600/1*1bNAG7ujkPbx64HWXKMoeg.png)

Biomedical images are typically volumetric images (3D) and sometimes have an additional time dimension (4D) and/or multiple channels (4-5D) (e.g. multi-sequence MR images). The variation in biomedical images is quite different from that of a natural image (e.g. a photograph), as clinical protocols aim to stratify how an image is acquired (e.g. a patient is lying on his/her back, the head is not tilted, etc.). In their analysis, we aim to detect subtle differences (i.e. some small region indicating an abnormal finding).

**Why computer vision and machine learning?** Computer vision methods have long been employed to automatically analyze biomedical images. The recent advent of deep learning has replaced many other machine learning methods, because it avoids the creation of hand-engineering features, thus removing a critical source of error from the process. Additionally, the fast inference speeds of GPU-accelerated fully networks, allows us scale analyses to unprecedented amounts of data (e.g. [10⁶ subject images](https://imaging.ukbiobank.ac.uk/)).

**Can we readily employ deep learning libraries for biomedical imaging?Why create DLTK?**The main reasons for creating [DLTK](http://dltk.github.io/) were to include speciality tools for this domain out of the box. While many deep learning libraries expose low-level operations (e.g. tensor multiplications, etc.) to the developers, a lot of the higher-level specialty operations are missing for their use on volumetric images (e.g. differentiable 3D upsampling layers, etc.), and due to the additional spatial dimension(s) of the images, we can run into memory issues (e.g. storing a single copy of a database of 1k CT images, with image dimensions of 512x512x256 voxels in float32 is ~268 GB). Due to the different nature of acquisition, some images will require special pre-processing (e.g. intensity normalization, bias-field correction, de-noising, spatial normalization/registration, etc).

### File formats, headers & reading images

While many vendors of imaging modalities produce images in the [DICOM](https://en.wikipedia.org/wiki/DICOM) standard format, saving volumes in series of 2D slices, many analysis libraries rely on formats more suited for computing and interfacing with medical images. We use the [NifTI ](https://nifti.nimh.nih.gov/nifti-1/)(or .nii format), originally developed for brain imaging, but widely used for most other volume images in both DLTK and for this tutorial. What this and other format saves is necessary information to reconstruct the image container and orient it in physical space.

For this, it requires specialty header information, and we will go through a few attributes to consider for deep learning:

* Dimensions and size store information about how to reconstruct the image (e.g. a volume into three dimensions with a size vector).

* Data type

* Voxel spacing (also the physical dimensions of voxels, typically in mm)

* Physical coordinate system origin

* Direction

**Why are these attributes important?** The network will train in the space of voxels, meaning we will create tensors of shape and dimensions [batch_size, dx, dy, dz, channels/features] and feed it to the network. The network will train in that voxel space and assume that all images (also unseen test images) are normalised in that space or might have issues to generalise. In that voxel space, the feature extractors (e.g. convolutional layers) will assume that voxel dimensions are isotropic (i.e. are the same in each dimension) and all images are oriented the same way.

However, since most images are depicting physical space, we need to transform from that physical space into a common voxel space:

If all images are oriented the same way (sometimes we require registration to spatially normalize images: check out [MIRTK](https://biomedia.doc.ic.ac.uk/software/mirtk/)), we can compute the scaling transform from physical to voxel space via



where all these information are vectors stored in the .nii header.

**Reading .nii images:** There are several libraries to read .nii files and access the header information and parse it to obtain a reconstructed image container as a [numpy](http://www.numpy.org/) array. We chose [SimpleITK](http://www.simpleitk.org/), a python wrapper around the [ITK](https://itk.org/) library, which allows us to import additional image filters for pre-processing and other tasks:



### Data I/O considerations

Depending on the size of the training database, there are several options to feed .nii image data into the network graph. Each of these methods has specific trade-offs in terms of speed and can be a bottleneck during training. We will go through and explain three options:

**In memory & feeding dictionaries:**We can create a tf.placeholder to the network graph and feed it via feed_dict during training. We read all .nii files from disk , process them in python (c.f. load_data()) and store all training examples in memory, where we feed from:



TLDR: this direct approach is typically the fastest and easiest to implement, as it avoids continuously reading the data from disk, however requires to keep the entire database of training examples (and validation examples) in memory, which is not feasible for larger databases or larger image files.

**Using a TFRecords database:** For most deep learning problems on image volumes, the database of training examples is too large to fit into memory. The TFRecords format allows to serialise training examples and store them on disk with quick write access (i.e. parallel data reads):



The format can directly interface with TensorFlow and can be directly integrated into a training loop in a tf.graph:



TLDR: TFRecords are fast means of accessing files from disk, but require to store yet another copy of the entire training database. If we are aiming to work with a database of several TB size, this could be prohibitive.

**Using native python generators:** Lastly, we can use python generators, creating a read_fn() to directly load the image data…



and tf.data.Dataset.from_generator() to queue the examples:



TLDR: It avoids creating additional copies of the image database, however is considerably slower than TFRecords, due to the fact that the generator cannot parallel read and map functions.

**Speed benchmarking & choosing a method:** We ran these three methods of reading .nii files to TensorFlow and compared the time required to load and feed a fixed-size example database. All codes and results can be found in here.

The obviously fastest method was feeding from memory via placeholders in 5.6 seconds, followed by TFRecords with 31.1 seconds and the un-optimised reading from disk using python generators with 123.5 seconds. However, as long as the forward/backward passes during training are the computational bottleneck, the speed of the data I/O is negligible.

### Data normalization

As with natural images, we can normalize biomedical image data, however the methods might slightly vary. The aim of normalization is to remove some variation in the data (e.g. different subject pose or differences in image contrast, etc.) that is known and so simplify the detection of subtle differences we are interested in instead (e.g. the presence of a pathology). Here, we will go over the most common forms of normalization:

**Normalization of voxel intensities:**This form is highly dependent on the imaging modality, the data was acquired with. Typical [zero-mean, unit variance normalization](https://github.com/DLTK/DLTK/blob/dev/dltk/io/preprocessing.py#L9) is standard for qualitative images (e.g. weighted brain MR images, where the contrast is highly dependent on acquisition parameters, typically set by an expert). If we employ such statistical approaches, we use statistics from a full single volume, rather than an entire database.

In contrast to this, quantitative imaging measures a physical quantity (e.g. radio-density in CT imaging, where the intensities are comparable across different scanners) and benefit from clipping and/or re-scaling, as [simple range normalisation](https://github.com/DLTK/DLTK/blob/dev/dltk/io/preprocessing.py#L39) (e.g. to [-1,1]).

![](https://cdn-images-1.medium.com/max/1600/1*UmKR5B3wM8mdhsnVsz01hg.png)

**Spatial normalisation:**Normalising for image orientation avoids that the model will have to learn all possible orientations, which largely reduces the amount of training images required (see the importance of header attributes to know what orientation an image is in). We additionally account for voxel spacing, which may vary between images, even when acquired from the same scanner. This can be done by resampling to an isotropic resolution:



If further normalisation is required, we can use medical image registration packages (e.g. [MIRTK](https://biomedia.doc.ic.ac.uk/software/mirtk/), etc.) and register the images into the same space, so that voxel locations between images correspond to each other. A typical step in analysing structural brain MR images (e.g. T1-weighted MR images) is to register all images in the training database to a reference standard, such as a mean atlas (e.g. the [MNI 305](https://www.mcgill.ca/bic/software/tools-data-analysis/anatomical-mri/atlases/mni-305) atlas). Depending on the degrees of freedom of the registration method, this can also normalise for size (affine registration) or shape (deformable registration). These two variants are rather rarely used, as they remove some of the information in the image (i.e. shape information or size information), that might be important for analysis (e.g. a large heart might be predictive of heart disease).

### Data augmentation

More often than not, there is a limited amount of data available and some of the variation is not covered. A few examples include:

* soft-tissue organs, where a wide range of normal shapes exist

* pathologies, such as cancer lesions, which can largely vary in shape and location

* free-hand ultrasound images, where a lot of possible views are possible

In order to properly generalise to unseen test cases, we augment training images by simulating a variation in the data we aim to be robust against. Similarly to normalisation methods, we distinguish between intensity and spatial augmentations:

Examples of intensity augmentations:

* Adding noise to training images generalise to noisy images

* Adding a random offset or contrast to handle differences between images

Examples of spatial augmentations:

* Flipping the image tensor in directions on where to expect symmetry (e.g. a left/right flip on brain scans)

* Random deformations, (e.g. for mimicking differences in organ shape)

* Rotations along axes (e.g. for simulating difference ultrasound view angles)

* Random cropping and training on patches

![](https://cdn-images-1.medium.com/max/1600/1*FgV1en0rLz5UFGQzdFdqvg.png)

Important notes on augmentation and data I/O: Depending on which augmentations are required or helpful, some operations are only available in python (e.g. [random deformations](https://github.com/DLTK/DLTK/blob/master/dltk/io/augmentation.py#L75)), meaning that if a reading method is used that uses raw TensorFlow (i.e. TFRecords or tf.placeholder), they will need to be pre-computed and stored to disk, thus largely increasing the size of the training database.

### Class balancing

Domain expert interpretations (e.g. manual segmentations or disease classes) are a requirement during supervised learning from medical images. Typically, the image-level (e.g. a disease class) or voxel-level (i.e. segmentation) labels are not available in the same ratio, which means that the network will not see an equal amount of examples from each class during training. This does not have a large effect on accuracy if the class ratios are somewhat similar (e.g. 30/70 for a binary classification case). However, since most losses are average costs on the entire batch, the network will first learn to correctly predict the most frequently seen class (e.g. background or normal cases, which are are typically more examples available of).

A class imbalance during training will have a larger impact on rare phenomena (e.g. small lesions in image segmentation) and largely impact the test accuracy.

To avoid this drop, there are two typical approaches to combat class imbalances in datasets:

* Class balancing via sampling: Here, we aim to correct the frequencies of seen examples during sampling. This can be done by a) sampling an equal amount from each class, b) under-sampling over-represented classes or c) over-sampling less frequent classes. In DLTK, we have an implementation for a), which can be found [here](https://github.com/DLTK/DLTK/blob/blog/dltk/io/augmentation.py#L120). We sample random locations in the image volume and consider an extracted example, if it contains the class we are looking for.

* Class balancing via loss function: In contrast to typical voxel-wise mean losses (e.g. categorical cross-entropy, L2, etc.), we can a) use a loss function that is inherently balanced (e.g. [smooth Dice loss](https://github.com/DLTK/DLTK/blob/master/dltk/core/losses.py#L51), which is a mean Dice-coefficient across all classes) or b) [re-weight the losses for each prediction by the class frequency](https://github.com/DLTK/DLTK/blob/master/dltk/core/losses.py#L10) (e.g. median-frequency re-weighted cross-entropy).

### Example application highlights

With all the basic knowledge provided in this blog post, we can now look into building full applications for deep learning on medical images with TensorFlow. We have implemented several typical applications using deep neural networks and will walk through a few of them to give you an insight on what problems you now can attempt to tackle.

Note: These example applications learn something meaningful, but were built for demo purposes, rather than high-performance implementations.

#### Example datasets

We provide [download and pre-processing scripts](https://github.com/DLTK/DLTK/tree/master/data) for all the examples below. For most cases (including the demos above), we used the [IXI brain database](http://brain-development.org/ixi-dataset/). For image segmentation, we downloaded the [MRBrainS13 challenge database](http://mrbrains13.isi.uu.nl/), which you will need to register for, before you can download it.

#### Image segmentation of multi-channel brain MR images

![](https://cdn-images-1.medium.com/max/1600/1*OjngIU6_yw_cIT-b8ConDA.png)

This image segmentation application learns to predict brain tissues and white matter lesions from multi-sequence MR images (T1-weighted, T1 inversion recovery and T2 FLAIR) on the small (N=5) MRBrainS challenge dataset. It uses a 3D U-Net-like network with residual units as feature extractors and tracks the Dice coefficient accuracy for each label in TensorBoard.

The code and instructions can be found [here](https://github.com/DLTK/DLTK/tree/master/examples/applications/MRBrainS13_tissue_segmentation).

#### Age regression and sex classification from T1-weighted brain MR images

![](https://cdn-images-1.medium.com/max/1600/1*s81FfZex1tzjAoxv7mSSYQ.png)

Two similar applications employing a scalable 3D ResNet architecture learn to predict the subject’s age (regression) or the subject’s sex (classification) from T1–weighted brain MR images from the IXI database. The main difference between this applications is the loss function: While we train the regression network to predict the age as a continuous variable with a L2-loss (the mean squared differences between the predicted age and the real age), we use a categorical cross-entropy loss to predict the class of the sex.

The code and instructions for these applications can be found here: [classification](https://github.com/DLTK/DLTK/tree/master/examples/applications/IXI_HH_sex_classification_resnet), [regression](https://github.com/DLTK/DLTK/tree/master/examples/applications/IXI_HH_age_regression_resnet).

#### Representation learning on 3T multi-channel brain MR images

![](https://cdn-images-1.medium.com/max/1600/1*lahtOexzm9GhpsHFYFMnPw.png)

Here we demo the use of a deep convolutional autoencoder architecture, a powerful tool for representation learning: The network takes a multi-sequence MR image as input and aims to reconstruct them. By doing so, it compresses the information of the entire training database in its latent variables. The trained weights can also be used for transfer learning or information compression. Note, that the reconstructed images are very smooth: This might be due to the fact that this application uses an L2-loss function or the network being to small to properly encode detailed information.

The code and instructions can be found [here](https://github.com/DLTK/DLTK/tree/master/examples/applications/IXI_HH_representation_learning_cae).

#### Simple image super-resolution on T1w brain MR images

![](https://cdn-images-1.medium.com/max/1600/1*K1YrnwlXdgY15zms5VS01A.png)

Single image super-resolution aims to learn how to upsample and reconstruct high-resolution images from low resolution inputs. This simple implementation creates a low-resolution version of an image and the super-res network learns to upsample the image to its original resolution (here the up-sampling factor is [4,4,4]). Additionally, we compute a linearly upsampled version to show the difference to the reconstructed image.

The code and instructions can be found [here](https://github.com/DLTK/DLTK/tree/master/examples/applications/IXI_HH_superresolution).

### Lastly…

We hope that this tutorial has helped you to ease into the topic of deep learning on biomedical images. If you found it helpful, we appreciate you sharing it and following DLTK on [github](https://github.com/DLTK/DLTK). If you require help with a similar problem, come to our [gitter.io chat](https://gitter.im/DLTK/DLTK) and ask us. Maybe some day we can host your application in the DLTK [model zoo](https://github.com/DLTK/models). Thanks for reading!

![](https://cdn-images-1.medium.com/max/1600/1*iIbVhflH6hX-DCcQlJzZjw.png)

### Resources

[Tutorial code](https://github.com/DLTK/DLTK/tree/master/examples/tutorials), [example applications](https://github.com/DLTK/DLTK/tree/master/examples/applications), [DLTK source](https://github.com/DLTK/DLTK)

