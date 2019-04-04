# TensorFlow CPUs and GPUs Configuration

I try to load two neural networks in TensorFlow and fully utilize the power of GPUs. However, my GPUs only have 8GBs memory, which is quite small. So I need to use GPUs and CPUs at the same time. This article is mainly training to resolve this problem.

At first, TensorFlow uses `tf.ConfigProto()` to configure the session.



It can also take in parameters when running tasks by setting environmental variable `CUDA_VISIBLE_DEVICES.`-1 is set to not use GPU. The id maps to the ones shown in `nvidia-smi` command.



Another way is to use `export` to control all the environment



Now, back to the configuration in tensorflow.

### 1. Default Mode

TensorFlow default mode is to initialize all available GPUs. Given the following code:



The output is as follows:



















To better analyze how tensorflow is assigning resource, and to find out which devices your operations and tensors are assigned to, create the session with `log_device_placement` configuration option set to `True`.

### 2. Decide Using GPU or CPU

To not use GPU, a good solution is to not allow the environment to see any GPUs by setting the environmental variable `CUDA_VISIBLE_DEVICES.`



















The output is as follows:













































The parameter `device_count` which takes a dictionary to assign available GPU device number and CPU device number. For example, the following code can make tensorflow not use any gpu resource.





The following code can assign both cpu and gpu.



#### Configuring CPUs

To run Tensorflow on one single CPU thread



`device_count` limits the number of CPUs being used, not the number of cores or threads.

### 3. Configuring GPUs

In `ConfigProto()`, `gpu_options` is used to configure gpus.

#### Visible Devices

First, you need to check your machine’s available GPUs with `nvidia-smi` command. When using multiple gpus, we need to use `,` to separate them.



#### GPU Growth

The first is the `allow_growth` option, which attempts to allocate only as much GPU memory based on runtime allocations: **it starts out allocating very little memory, and as Sessions get run and more GPU memory is needed, we extend the GPU memory region needed by the TensorFlow process.**



### 4. Manual device placement

If you would like a particular operation to run on a device of your choice instead of what’s automatically selected for you, you can use `with tf.device` to create a device context such that all the operations within that context will have the same device assignment.



You will see that now `a` and `b` are assigned to `cpu:0`.



### 5. Using a single GPU on a multi-GPU system

If you have more than one GPU in your system, the GPU with the lowest ID will be selected by default. If you would like to run on a different GPU, you will need to specify the preference explicitly:



If the device you have specified does not exist, you will get `InvalidArgumentError`:



If you would like TensorFlow to automatically **choose an existing and supported device to run the operations in case the specified one doesn’t exist, you can set**`allow_soft_placement`**to**`True`**in the configuration option when creating the session.**



### 6. Using multiple GPUs

If you would like to run TensorFlow on multiple GPUs, you can construct your model in a **multi-tower fashion** where each tower is assigned to a different GPU. For example:



You will see the following output.



Solve our problem.

It looks like we need the multi-tower fashion. CIFAR-1o is a good example. The reason CIFAR-10 was selected was that it is complex enough to exercise much of TensorFlow’s ability to scale to large models.

### Training a Model Using Multiple GPU Cards

Modern workstations may contain multiple GPUs for scientific computation. TensorFlow can leverage this environment to run the training operation concurrently across multiple cards.

Training a model in a parallel, distributed fashion requires coordinating training processes. For what follows we term model replica to be one copy of a model training on a subset of data.

**Naively employing asynchronous updates of model parameters leads to sub-optimal training performance because an individual model replica might be trained on a stale copy of the model parameters.** Conversely, employing fully synchronous updates will be as slow as the slowest model replica.

In a workstation with multiple GPU cards, each GPU will have similar speed and contain enough memory to run an entire CIFAR-10 model. Thus, we opt to design our training system in the following manner:

* **Place an individual model replica on each GPU.**

* **Update model parameters synchronously by waiting for all GPUs to finish processing a batch of data.**

Here is a diagram of this model:

![](https://cdn-images-1.medium.com/max/1600/0*lkoteETW9g2BwXsD.png)

Note that each GPU computes inference as well as the gradients for a unique batch of data. This setup effectively permits dividing up a larger batch of data across the GPUs.

**This setup requires that all GPUs share the model parameters.** A well-known fact is that transferring data to and from GPUs is quite slow. For this reason, we decide to **store and update all model parameters on the CPU (see green box)**. A fresh set of model parameters is transferred to the GPU when a new batch of data is processed by all GPUs.

The GPUs are synchronized in operation. All gradients are accumulated from the GPUs and averaged (see green box). The model parameters are updated with the gradients averaged across all model replicas.

### Placing Variables and Operations on Devices

Placing operations and variables on devices requires some special abstractions.

The first abstraction we require is a function for computing inference and gradients for a single model replica. In the code we term this abstraction a “tower”. We must set two attributes for each tower:

* A unique name for all operations within a tower. `tf.name_scope` provides this unique name by prepending a scope. For instance, all operations in the first tower are prepended with `tower_0`, e.g. `tower_0/conv1/Conv2D`.

* A preferred hardware device to run the operation within a tower. `tf.device` specifies this. For instance, all operations in the first tower reside within `device('/gpu:0')` scope indicating that they should be run on the first GPU.

**All variables are pinned to the CPU and accessed via**`tf.get_variable`**in order to share them in a multi-GPU version.**See how-to on [Sharing Variables](https://www.tensorflow.org/programmers_guide/variable_scope).

A good example is here: [https://github.com/tensorflow/tensorflow/blob/r0.7/tensorflow/models/image/cifar10/cifar10_multi_gpu_train.py](https://github.com/tensorflow/tensorflow/blob/r0.7/tensorflow/models/image/cifar10/cifar10_multi_gpu_train.py)

Can try this out at first, see if it works.

