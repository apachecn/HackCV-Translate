# Multi-GPU Framework Comparisons

[GitHub Link](https://github.com/ilkarman/DeepLearningFrameworks#2-training-time-densenet-121-on-chestxray---image-recognition-multi-gpu)

For latest updates please follow the re-post on [technet ](https://blogs.technet.microsoft.com/machinelearning/)(appearing soon).

Having [previously examined a wide breadth of deep-learning frameworks](https://blogs.technet.microsoft.com/machinelearning/2018/03/14/comparing-deep-learning-frameworks-a-rosetta-stone-approach/), it was difficult to go into a lot of depth for each one. In this post I take **Tensorflow, PyTorch, MXNet, Keras, and Chainer and train a**[CheXNet](https://arxiv.org/pdf/1711.05225.pdf)**model.**

![](https://cdn-images-1.medium.com/max/1600/1*MG1DAWXhex9KZPlfHu72mQ.png)

The notebook-examples, train a data-parallel model (DenseNet121 pre-trained on ImageNet) across 4 V100s on [Azure’s Data Science Virtual Machine](https://azure.microsoft.com/en-us/services/virtual-machines/data-science-virtual-machines/) using native data-loaders to preprocess the images on-the-fly, along with image-augmentation and validation.

### Data

The **real**-dataset consists of 12120 PNGs of grayscale chest X-rays resized to (264px, 264px) and saved to disk. Framework native data-loaders are used to load, pre-process the data and perform some augmentations (random horizontal flip and random crop to 224px), on the fly.

The **synthetic**-dataset is just a numpy array of random (224, 224) matrices loaded into memory and used to benchmark how much loading the data (augmentation, and the validation stages) affect the training performance.

Initially I used the full-sized images (over 1000x1000px), however I noticed that even with asynchronous data-loaders, resizing the images down to 264 would bottleneck the training (even with one GPU). **Hence, the images are resized to 264 and saved to disk first. A possible further extension could be convert them to JPEGs to benefit from**[libjpeg-turbo](https://libjpeg-turbo.org/)**& pillow-simd**. Recently [NVidia have released](https://github.com/NVIDIA/dali) optimised libraries for loading and processing JPEGs. The data-loading pipeline can be neatly summarised with **torchvision.transforms**:

![](https://cdn-images-1.medium.com/max/1600/1*10MlKwLQMQOvvxaYRPIvhg.jpeg)

### Setup

In these notebooks we use [DenseNet121](https://arxiv.org/pdf/1608.06993.pdf) (w/ Imagenet weights), applying sigmoid activation and binary cross-entropy loss on the last fully-connected (14) layer to account for non mutually-exclusive labels. The model is trained for 5 epochs (for around 0.82 AUC on the test data-set) and the timings are reported below.

With Pytorch, Keras, Tensorflow and MXNet, to fully benefit from data-parallel mode involved manually increasing the batch-size by the number of GPUs (effectively running a bigger batch-size). For a fixed number of epochs this would mean GPU-count-fewer gradient updates, so I adopted a simple linear-scaling rule where the learning-rate was increased by the number of GPUs. Chainer, however, was the only framework to automatically scale the batch-size by the number of GPUs and adjust the optimiser.

### Overall Results

Please follow the GitHub link for updated results

I time how long it takes to match the same number of epochs (instead of the same evaluation metric, which would take longer).

![](https://cdn-images-1.medium.com/max/1600/1*BepwyNgXv_xF7gr-slz6KA.jpeg)

### Summary

The idea of this small comparison was to use relatively high-level (single-node) data-parallel wrappers and assess the performance. **For single-node multi-gpu training using distributed wrappers (such as**[Horovod ](https://github.com/uber/horovod)**or**[torch.distributed](https://pytorch.org/docs/stable/distributed.html)**) is probably likely to result in faster timings** (each process will bind to a single GPU and do multi-process distributed training on a single-node). This is something we are currently working on and will be the subject of the next blog post.

The way that most of the frameworks perform data-parallel training (notable exceptions are [1-bit SGD](https://www.microsoft.com/en-us/research/publication/1-bit-stochastic-gradient-descent-and-application-to-data-parallel-distributed-training-of-speech-dnns/)) is to: (i) replicate the model across GPUs, (ii) scatter input over batch to multiple GPUs, (iii) output[i]= replica[i](input[i]), (iv) gather output[i] to main gpu, (v) calculate loss and split gradients, (vi) backward via replicas and finally (vii) sum gradients among all replicas. In PyTorch all of this is handled with just a single call to: [torch.nn.data_parallel](https://pytorch.org/docs/stable/_modules/torch/nn/parallel/data_parallel.html)!

**Gathering the output to the main-gpu means that the API call to calculate the loss is less complex for the user**, simply: loss(output, target), since everything resides on cuda:0 (the main gpu). However, as seen from the gpu-utilisation charts this means the **loss calculation is done on gpu:0** and there is a small performance-improvement potential by not gathering the output, and instead scattering the target to multiple-gpus and performing the loss calculation on each one.

The**PyTorch team were incredible at offering support** (I think a key criteria for judging a framework) and thanks to their help you can check [this notebook](https://github.com/ilkarman/DeepLearningFrameworks/blob/master/notebooks/PyTorch_MultiGPU-Exp-Loss.ipynb) which overrides the default-API for PyTorch to calculate the loss on multiple-gpus and shaves off a few seconds (at the expense of a bit more code). This was also straight-forward to do in [Gluon](https://github.com/ilkarman/DeepLearningFrameworks/blob/master/notebooks/Gluon_MultiGPU.ipynb).

**PyTorch itself is a very flexible framework and that makes it much easier to override defaults**(another important criteria). For example; in my experience taking a [PyTorch version of Faster-RCNN](https://github.com/chenyuntc/simple-faster-rcnn-pytorch) and chopping off the ROI-head to leave only the RPN (useful for one-class object detection) is very easy to do. **In general I believe that support from contributors and ease of modification are much more important factors than just speed of training.**

**Something to look out for when working with flexible APIs like Gluon and PyTorch, however, is to not accidentally bottleneck the training process**. At the very least one should avoid mixing variables on the cpu (perhaps for logging loss) with variables on the gpu (the output of the model). And when really going for performance (and calculating loss on multiple-gpus, instead of gathering to one) it’s important to then log the list of losses (one per GPU) to a variable which resides on the same GPU (instead of logging all of the losses to gpu:0). For example with gluon we would create a variable to log loss like so: [mx.nd.zeros((1), ctx=c) for c in ctx] and with PyTorch: [torch.FloatTensor(1).fill_(0).cuda(i) for i in range(GPU_COUNT)]

Chainer uses a very interesting API that resembles a distributed wrapper more closely, and automatically scales the batch-size using [NCCL ](https://github.com/NVIDIA/nccl)behind the hood to achieve a very impressive looking gpu-utilisation chart. The **support from the community (particularly via slack) was fantastic** and the**notebook uses a**[modified ](https://github.com/ilkarman/DeepLearningFrameworks/pull/103)**CaffeFunction to load DenseNet** (combine batch-norm & scale into a single batch-norm layer and then remove unnecessary interim outputs). It would be great to see Chainer finalise (optimised) ONNX import to give users access to a much wider model-zoo, and deprecate the CaffeFunction.

**Below follows my summary at a more detailed level** for each framework along with a gpu-utilisation (top panel) and gpu-memory utilisation (bottom panel) visualisation using this [tool](https://github.com/msalvaris/gpu_monitor). With most of the frameworks we can see 5 spikes in the beginning (training + validation on real-data), a small spike for testing, and then a sustained period at the end running 5 epochs on synthetic data (without validation).

GPU:0 is the green line and it should be easy to spot which frameworks gather output to gpu:0 to calculate loss and which do it across multiple GPUs, by default.

It’s also interesting to see the small up-tick in the beginning which is probably auto-tune (selecting the optimal convolution-forward algorithm)

**I feel two factors could potentially help improve the performance for most frameworks**:

1. Bundling optimised binaries with the pip-install to handle common functions such as image-loading and resizing (to avoid the data-loader bottlenecking the GPU); for example the time to resize an image in PIL vs OpenCV is pretty large. Soumith has a [great gist](https://gist.github.com/soumith/01da3874bf014d8a8c53406c2b95d56b), which shows how to install pillow-simd with turbojpeg; to benefit from this I should convert my PNGs to JPEGs.

2. Optimising the validation-cycle after training (pre-fetching the data for it before training ends to ensure as little down-time as possible whilst memory is swapped) since over a large number of epochs this adds up & automatically using all GPUs for validation. You can see a example of this prefetching in the [optimised-PyTorch notebook](https://github.com/ilkarman/DeepLearningFrameworks/blob/master/notebooks/PyTorch_MultiGPU-Exp-Loss.ipynb) thanks to help from Adam.

Finally, an interesting discovery for me was that maximising GPU utilisation doesn’t necessarily translate to better performance:

* Increasing the batch size of the data-loader for validation beyond that of training (since we don’t store gradients we have more memory and can do this) reduced the performance, presumably because the GPUs had to wait for the full big batch to be ready before they could start. A potential way around this (with big enough CPU memory) would be to **call iter() on the validation data-set before the training loop starts** (so that prefetch would happen in the background), however there is the potential it would interfere with loading in the actual (training) phase.

* **Setting num_processes for the data-loader to cpu_count (24) was actually sub-optimal and for this example 6 seemed to be the fastest**(perhaps my processing was too light and opening/closing processes created over-head). Taking this to the extreme, for the synthetic data (when no pre-processing was needed) using just one process gave the fastest time.

### PyTorch (notebook)

![](https://cdn-images-1.medium.com/max/1600/1*69rwA9lOVGS5YAO_ihIWKA.jpeg)

* **PyTorch was the easiest framework to work with** and became my overall favourite at the end of this experiment. The **user-friendliness seems to come cost-free since it was one of the fastest frameworks.**

* The **GPU memory utilisation** resembles Chainer and Gluon

* The gpu utilisation chart for PyTorch is more GPU-0 intensive compared to Gluon for reasons mentioned above.

* The training code for PyTorch is super flexible, however this also places more responsibility on the user to not bottleneck training by accumulating certain variables (loss or predictions) in a sub-optimal way.

* **torchvision.transforms was very convenient and possessed the greatest number of transformations**; and so it’s great to see MXNet follow PyTorch with mxnet.gluon.data.vision.transforms. Their high-level API seems almost identical.

* Curiously MXNet’s data-loader was a couple of minutes faster than PyTorch’s on real-data despite possessing pretty much the same high-level API, which suggests that they have optimised some of the back-end functionality.

* I couldn’t notice a difference between using [Pillow or Pillow-SIMD](https://github.com/pytorch/vision#image-backend) for torchvision — **perhaps I should convert the PNGs to JPEGs?** Unfortunately I wasn’t able to find Intel’s IPP to get accimage to work (the last GitHub commit is Nov 2017 so I’m not sure if it’s being maintained). I feel it would be great to see a few common functions (image load and resize) bundled with the pip install as optimised binaries to speed up the data-loader.

### Keras (notebook)

![](https://cdn-images-1.medium.com/max/1600/1*u4ozTDWuJlFX289SEbrE5A.jpeg)

* **Running data-parallel with Keras using TF as a backend was straightforward**. The speed was pretty much equal to my raw-TF implementation (and on real-data benefited slightly from a faster data-loader).

* However, the GPU utilisation graph seems to highlight that the GPUs were not used to their full capacity (e.g. compared to PyTorch or MXNet).

* Unfortunately at time of writing there were issues with ModelCheckpoint when running multi-gpu so I omitted this for all notebooks.

* **Keras has a very fast data-loader**

* I found ImageDataGenerator.flow_from_directory() a bit too restrictive — sub-folders became labels and this becomes an issue for multi-label since images would be duplicated across folders. A bit hacky but I would just override the labels after creating them. **I feel that allowing the generator to accept a {file_name: label, } dictionary would make the whole process a lot easier.**

* Just before publishing this post I noticed that Keras 2.1.5 would send images to the ImageDataGenerator.preprocessing_function() instead of CHW arrays (Keras 2.1.4 and previous). Since most of the preprocessing code operates on arrays this meant I had to convert the image to an array to do processing and then convert it back to an image (to be later converted to an array again). I therefore decided to stick with Keras 2.1.4 — the alternative would be to inherit the Sequence class and write my own data-loader.

* Keras’ fit_generator allows the user to choose between multi-processing and multi-threading. I found that the latter was better for this experiment since resizing and random-cropping were not processor-heavy (since the images were resized down to 264 and saved to disk). I ran an experiment using the non-resized images (1000px+) and there, multi-processing proved a bit faster. Hence, I assume if the processing is light and the workload is mainly IO-bound then multi-threading would be faster. It would be great to get a bit more guidance on this.

### Tensorflow (notebook)

![](https://cdn-images-1.medium.com/max/1600/1*U0NSlTi8dbcBFq576eVzLg.jpeg)

* **I could not find an**[official implementation of DenseNet121](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models)**for Tensorflow which was very strange since it’s not a very recent model (and was available for Keras-TF)**, so I used a [custom-repo ](https://github.com/pudae/tensorflow-densenet)that created a version using tf-slim by pudae (who was very helpful).

* It’s highly likely that some optimised version of the code (perhaps using different APIs) would be much faster, however I reached out and didn’t receive any help on how to improve the notebooks.

* The **tf.data.Dataset** and**tf.estimator.Estimator**higher-level APIs are a very welcome (recent) addition to reduce the verbosity of Tensorflow code and I would recommend everyone to explore them fully.

* The TF data-loader I created ended up being slower than the Keras data-loader. I could not find many examples of a ‘good’ TF data-loader (perhaps due to a focus on tfrecords?) so wasn’t sure if this was because my implementation was lacking or Keras truly being faster. I noticed that with TF 1.8 it’s now possible to prefetch to GPU memory with tf.contrib.data.prefetch_to_device().

* In general **with TF there are many different APIs that can be used for the same purpose**. Perhaps they call the same backend and the high-level method is kept for backwards-compatibility, however I believe this becomes very confusing for the user to know if they are using the optimal function or something that should be deprecated. I can see why this is hard to avoid however, since TF requires a lot more low-level boiler-plate code than any other framework and this gets constantly improved so a lot more functions are altered (versus just a high-level function that utilises this).

* Creating a multi-gpu example proved very difficult and **involved writing a lot of low-level code** (e.g. average_gradients). The issue I have with writing lots of low-level code versus a high-level API is two-fold: 1) it’s difficult to tell if my implementation is efficient, 2) as the versions advance I would have to keep changing this code to make use of the performance improvements (rather than having this take place automatically behind the scenes). Providing a ‘good’ example would solve the first issue and not the second. Hence, it will be great to see multi-gpu incorporated as a flag in tf.estimator(). Since I get similar timings on synthetic data (avoiding the data-loader) with Keras and raw-TF, I want to assume I did the multi-gpu wrapping correctly.

* The processing was very gpu:0 heavy (as seen in the chart), which probably is creating a bottleneck somewhere.

* The is_training flag for tf.slim models [was very confusing for me](https://github.com/tensorflow/models/issues/3556#issuecomment-372328969), and seemed to only affect how batch normalisation was done. I think with is_training=False, TF would use saved mean and variance for batch-norm, otherwise it would accumulate it live. In the walkthrough notebook on their GitHub page is_training was set to True for validation, which is what I followed. **I believe it should be renamed to something else, e.g. batch_norm_use_saved**.

* At time of writing I could not set validation to occur after every epoch, instead tf.estimator.EvalSpec() had a trottle_sec method which would execute it after a certain number of seconds. This is why the GPU-utilisation only has 3 peaks and not 5 (I couldn’t guess the timing well and had 3 validations, not 5).

* The **GPU-utilisation chart shows a lot of down-time between training and validation**not present with other frameworks. This was because TF would save the model to disk after training and then load it from disk to begin validation and again to resume training. I’m not sure how to avoid this, since this cost TF quite a bit of time in the table above.

* **Hence, it is my personal opinion (as an outsider) that Google want people to use Tensorflow through Keras rather than raw-TF** (or at-least [tf.keras](https://www.tensorflow.org/api_docs/python/tf/keras), will this replace tf.estimator?). There is just **so much boiler-plate code that is constantly updated that it is very hard to keep your TF code optimal**unless you work with it daily. Also, there a lot more pre-trained models available for Keras. For the multi-node distributed training examples we are working on, we decided to not use raw-TF at all and **just use**[horovod](https://github.com/uber/horovod)**.**

### Chainer (notebook)

![](https://cdn-images-1.medium.com/max/1600/1*HlahtYkTAvgv0yCs00uaJw.jpeg)

* Chainer’s **gpu-utilisation was the highest and most consistent out of all the other frameworks**. It also **averaged one of the fastest times** (following some modification to default functions). The**gpu-utilisation was not gpu:0 dominant**, and appears to spread the load out evenly.

* Chainer did not have DenseNet121 available as a pretrained model at time of writing so I found a [Caffe version](https://github.com/shicai/DenseNet-Caffe) by shicai, which can be loaded using **chainer.links.caffe.CaffeFunction**.

* Chainer’s API was perhaps the most different out of all the frameworks and I enjoyed using it a lot. I think if Chainer improved the model-zoo (and the CaffeFunction import) or finalised ONNX import & export, this would attract a lot more people to the framework. I found the [chainercv ](https://github.com/chainer/chainercv)library also to be very useful for transforms for image-augmentation.

* To load the model I had to write a method that would**truncate the batch normalisation epsilon**param to 1e-5, this was a bit strange since the prototxt was already 1e-5 and I wonder if it’s some kind of import bug.

* The CaffeFunction method is pretty general and saves the results of all layers in the network (not just the last one), this bloats the memory a lot and meant I could only use a batch of 32. I created another method to override this that **stores only the layers necessary for DenseNet**(bear in mind there is some concatenation across groups) which reduced memory and let me increase the batch-size. [Another modification](https://github.com/ilkarman/DeepLearningFrameworks/pull/103) made to the function (by the community) **combines batch-norm and scale** into one batch-norm layer and improves the performance (relative to default) considerably

* Strangely the **test AUC metric using multi-gpu is considerably lower** than that for all other frameworks and I’m not sure why. Adopting the same linear-scaling rule harmed this even more so I left the learning-rate untouched under the assumption that the optimiser is somehow automatically scaled. I need to investigate this further.

* From the graph above (and in the code) we can see that **one-GPU only is used for validation**, if all four could be used (similar to PyTorch) then this is likely to improve the performance further

* **Chainer utilises NCCL for single-node multi-gpu communication** (and chainermn is used for distributed chainer, across nodes). For further details see this [chart](http://corochann.com/wp-content/uploads/2017/06/chainer-install-diagram-800x393.png).

### MXNet Gluon (notebook)

![](https://cdn-images-1.medium.com/max/1600/1*WpGjMCAa1alv3JzsxfSc7Q.jpeg)

* The Gluon-API bears very strong resemblance to PyTorch and is thus very convenient and user-friendly to use.

* The GPU utilisation chart looks very good (in particular the **down spikes during training+validation are much smaller compared to other frameworks**, which may be a strength of the data-loader).

* Gluon required a little tinkering to figure out the optimal number of batches to send asynchronously before blocking with a print-loss method, without this you would receive an out-of-memory error.

### Acknowledgements

Many thanks to [Thomas Delteil](https://twitter.com/thdelteil) for creating and running the Gluon notebook, Soumith Chintala & Adam Paszke & Teng Li for PyTorch support, [Mathew Salvaris](https://twitter.com/msalvaris) for the gpu-monitor and [Shunta Saito](https://github.com/mitmul) for Chainer-related help

