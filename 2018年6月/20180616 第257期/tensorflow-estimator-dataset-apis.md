# TensorFlow Estimator & Dataset APIs

![](https://cdn-images-1.medium.com/max/1600/1*VQGbvTlU3b68j7qzMHXbPQ.png)

When TensorFlow 1.3 was released the [Estimator, and related high-level APIs, caught my eye](https://medium.com/onfido-tech/higher-level-apis-in-tensorflow-67bfb602e6c0). This is almost a year ago and TensorFlow has had a few updates, with 1.8 the latest version at the time of writing this. Time to revisit these APIs and see how they evolved.

The Estimator and Dataset APIs have become more mature since TF 1.3. [The TensorFlow tutorials recommend to use them when writing TensorFlow programs](https://www.tensorflow.org/get_started/premade_estimators):

> We strongly recommend writing TensorFlow programs with the following APIs:

> Estimators, which represent a complete model. The Estimator API provides methods to train the model, to judge the model’s accuracy, and to generate predictions.

> Datasets, which build a data input pipeline. The Dataset API has methods to load and manipulate data, and feed it into your model. The Dataset API meshes well with the Estimators API.

The Estimator API provides a top-level abstraction and integrates nicely with other APIs such as the Dataset API to build input streams, and the Layers API to build model architectures. It’s even possible to construct an estimator from a Keras model with [one function](https://www.tensorflow.org/api_docs/python/tf/keras/estimator/model_to_estimator).

![](https://cdn-images-1.medium.com/max/1600/1*8e8Aq_GlJFy8tGuZx1F2IA.png)

In the following, I will give an overview of the API. [An accompanying repository with example code is provided here](https://github.com/peterroelants/tf_estimator_example).

### Estimator

The core of the [Estimator API](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator) has stayed stable, we can still create an estimator as follows:



* model_fn is the function initializing the model. This function is represented and fully defined by an [EstimatorSpec](https://www.tensorflow.org/api_docs/python/tf/estimator/EstimatorSpec) object which knows how to generate all outputs, how to train, and how to evaluate the model. This model function is still agnostic to how you implement the architecture, you can use [tf.layers](https://www.tensorflow.org/api_docs/python/tf/layers) or other libraries to implement your model architecture.

* config is a [RunConfig](https://www.tensorflow.org/api_docs/python/tf/estimator/RunConfig) object specifying how to run the estimator. It knows where to save the checkpoints, how many times to save the checkpoints, when to log, etc.

* params is an object holding the hyper-parameters of the model. In my previous post this was an [HParams](https://www.tensorflow.org/api_docs/python/tf/contrib/training/HParams) object. However in 1.8 HParams is not used anymore, and params can be a dictionary, or another object (an [argparse](https://docs.python.org/3/library/argparse.html) object in my [example code](https://github.com/peterroelants/tf_estimator_example/blob/master/src/mnist_estimator.py)).

After creating the Estimator we can train it using the [train_and_evaluate](https://www.tensorflow.org/api_docs/python/tf/estimator/train_and_evaluate) function:



* model_estimator is the Estimator.

* train_spec is an [TrainSpec](https://www.tensorflow.org/api_docs/python/tf/estimator/TrainSpec) object representing the training configuration and knows how to build the training data feeder.

* eval_spec is an [EvalSpec](https://www.tensorflow.org/api_docs/python/tf/estimator/EvalSpec) object representing the evaluation configuration and knows how to build the evaluation data feeder.

Note that this is different to the previous blogpost, where we used the TensorFlow [Experiment](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/Experiment) class, which is now deprecated, to run the training. Getting rid of the Experiment class makes everything less complicated. The training and evaluation input functions and hooks are now clearly separated into the TrainSpec and the EvalSpec, and you only have to call the train_and_evaluate function.

![](https://cdn-images-1.medium.com/max/1600/1*XIA2rgFA2BaBG5o9GpN-QQ.png)

### Dataset

The [Dataset API](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) has become fully mature and moved from [contrib](https://www.tensorflow.org/api_docs/python/tf/contrib) to the TensorFlow core library and it now allows you to build complex input pipelines. In the [accompanying code](https://github.com/peterroelants/tf_estimator_example/blob/master/src/mnist_estimator.py) this is used to build an input feeder that shuffles the data and repeats it for as long as needed with the correct batch size:



Note that in this example I’m just loading the [mnist data from the Keras API](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data). You can build much more [complex input pipelines](https://www.tensorflow.org/programmers_guide/datasets) with the Dataset API.

### Running code locally

You can run the code from the [repo](https://github.com/peterroelants/tf_estimator_example) locally by:



This should start a training and evaluation session. By using the Estimator API it also sets-up default logging and checkpoint saving, which we can visualize with [TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard):

![](https://cdn-images-1.medium.com/max/1600/1*Sou8xlCNja1iPT6uIwpiBA.png)

### Running code on Google Cloud ML Engine

Thanks to the abstraction of the configuration the Estimator API allows for it to train models easily on [Google Cloud’s ML Engine](https://cloud.google.com/ml-engine/):



> With tf.estimator.train_and_evaluate you can run the same code both locally and distributed in the cloud, on different devices and using different cluster configurations, and get consistent results without making any code changes

I provided a minimal example of [how to run the accompanying code on Google Cloud](https://github.com/peterroelants/tf_estimator_example#training-on-google-cloud). For example, you can train the code on the cloud by running:



In summary, the TensorFlow Estimator API, as well as the Dataset API, have matured a lot. They provide a nice abstraction layer to manage input data streams, models, and training/evaluation configurations.

