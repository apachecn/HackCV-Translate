# Benchmarking Tensorflow Performance on Next Generation GPUs

As machine learning (ML) researchers and practitioners continue to explore the bounds of deep learning, the need for powerful GPUs to both train and run these models grows ever stronger. New models for object detection, image segmentation, and speech transcription continue to be refined, finding use in a variety of industries ranging from autonomous driving to home assistants.

To satisfy this demand for GPU-compute, both Amazon and Google recently added next generation Nvidia [Volta](https://aws.amazon.com/ec2/instance-types/p3/) and [P100](https://cloud.google.com/gpu/) GPUs to their instance types. [Paperspace](https://www.paperspace.com/)¹, another cloud GPU vendor, has also [added Volta GPU](https://www.paperspace.com/volta-gpu)s to its offerings. These P100 and Volta GPUs are the best GPUs currently available on the market and are at the cutting edge for performance for ML applications. Not only do these GPUs have superior performance relative to the older [K80](http://www.nvidia.com/object/tesla-k80.html) GPUs, they also come with 16GB of memory enabling even more expressive ML models and larger training minibatch sizes.

![](https://cdn-images-1.medium.com/max/1200/1*Ia0xN7HyvdLRgNQt7BIjJQ.jpeg)

To test how these modern GPUs perform on typical ML tasks, I trained a Faster R-CNN/resnet101 object detection model on Nvidia’s most recent GPUs. The object detection model was [implemented in Tensorflow](https://github.com/tensorflow/models/tree/master/research/object_detection) and operated on 300x300px image inputs, with training minibatch sizes of 10, 15, and 20 images.

The GPUs that were benchmarked:

* [Paperspace Volta](https://www.paperspace.com/volta-gpu) (16GB — $2.30/hour)

* [Google Cloud P100](https://cloud.google.com/gpu/) (16GB — $1.73/hour)

* [Amazon EC2 p3.2xlarge Volta](https://aws.amazon.com/ec2/instance-types/p3/) (16GB— $3.06/hour)

* [Nvidia 1080Ti](https://www.nvidia.com/en-us/geforce/products/10series/geforce-gtx-1080-ti/) (11GB — Personal Machine)

**Note:**This****benchmark focuses specifically on newer GPUs and thus excludes the older K80 and Quadro GPUs. These GPUs were [benchmarked last April](https://medium.com/initialized-capital/benchmarking-tensorflow-performance-and-cost-across-different-gpu-options-69bd85fe5d58).

#### Results

From a performance standpoint, Voltas are unsurprisingly the most powerful GPUs available today, outperforming both the Nvidia 1080Ti (~1.1-1.3x) and the P100 (~1.2-1.5x) by significant margins, despite the 1080Ti being only around 9 months old. This continues Nvidia’s rapid cadence of releasing increasingly powerful GPU architectures.

Notably, Amazon’s Volta instances did not perform as well as Paperspace’s Volta on the same training task. My own brief investigation into this suggests that slow I/O between the instance and the GPU may be to blame; comparing pure GPU benchmarks between Amazon and Paperspace shows similar performance.

From a cost perspective, the Paperspace Voltas offer good value for money; adjusting for cost, Google’s P100 is approximately 10% more expensive while Amazon finishes at a full 40% more.

#### What Should I Use?

* Heavy users, of course, should probably buy their own GPUs. Renting GPUs from cloud providers continues to be an expensive proposition, and buying your own GPU allows you to access the best hardware for lower cost— provided you can keep them utilized to amortize cost.

* Paperspace Voltas are a good value for users who do not wish to plunge into owning their own GPUs. For users with who only need a single GPU, using a Volta will provide a good increase in performance.

* Google’s P100s are the most flexible, allowing users to attach 1, 2, and 4 P100 GPUs (or up to 8 K80 GPUs) to any instance, allowing users to customize their CPU and GPU configurations to suit their computing needs. They also are competitive on a cost-adjusted basis despite the poorer performance of the P100.

* Amazon’s Voltas are more powerful than the Google’s P100 and also offer the ability to attach 1, 4, or 8 GPUs. However, users do not have the ability to customize the base instance type. They are also relatively expensive on a cost-adjusted basis. I would recommend using them only if you have a compelling need for 8 GPUs or need to be on EC2.

Building something interesting? [Initialized Capital](http://twitter.com/@initializedcap) would love to chat with you.

**¹ Disclosure:**Paperspace is an Initialized Capital portfolio company.

