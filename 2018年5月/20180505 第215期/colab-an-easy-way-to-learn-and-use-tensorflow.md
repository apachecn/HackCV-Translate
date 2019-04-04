# Colab: An easy way to learn and use TensorFlow

![](https://cdn-images-1.medium.com/max/1600/1*g_x1-5iYRn-SmdVucceiWw.png)

Colaboratory is a hosted Jupyter notebook environment that is free to use and requires no setup. You may have already seen it in [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/), tensorflow.org’s [eager execution](https://colab.research.google.com/github/tensorflow/models/blob/master/samples/core/get_started/eager.ipynb) tutorial, or on various research articles (like [this one](https://distill.pub/2018/building-blocks/)). We wanted to offer 5 tips for using it:

**1. TensorFlow is already pre-installed**

When you create a new notebook on [colab.research.google.com](http://colab.research.google.com/), TensorFlow is already pre-installed and optimized for the hardware being used. Just `import tensorflow as tf`, and start coding.

**2. Setup your libraries and data dependencies in code cells**

Creating a cell with `!pip install` or `!apt-get` works as you’d expect. It also makes it easy for others to reproduce your setup.

To get in your training data, you can follow these tutorials for popular data sources: [BigQuery](https://colab.research.google.com/notebooks/bigquery.ipynb), [Drive, Sheets, or Google Cloud Storage](https://colab.research.google.com/notebooks/io.ipynb). You also have access to the shell with `!`, so `!wget`, `!pwd`, etc. might also help.

**3. Use it with Github**

If you have a nice .ipynb on Github, it’s easy to create a one-click link for your readers to start playing with it. Just add your Github path to colab.research.google.com/github/ . For example, [colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb](https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb) will load [this ipynb](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb) stored on Github.

![](https://cdn-images-1.medium.com/max/1600/1*ZpNn76K98snC9vDiIJ6Ldw.jpeg)

You can also easily save a copy of your Colab notebook to Github by using File > Save a copy to Github…

**4. Share and edit collaboratively**

Colab notebooks are just like Google Docs and Sheets. They are stored in Google Drive and can be shared, edited, and commented on collaboratively. Just click the Share button in the top right of any notebook that you’ve created.

**5. Hardware acceleration**

By default, Colab notebooks run on CPU. You can switch your notebook to run with GPU by going to Runtime > Change runtime type, and then selecting GPU. You can also have a Colab notebook use your local machine’s hardware by following these [instructions](https://research.google.com/colaboratory/local-runtimes.html).

For more tips, see our [welcome notebook](https://colab.research.google.com/notebooks/welcome.ipynb), read our [FAQ](https://research.google.com/colaboratory/faq.html), or find useful code snippets while using Colab (Help > Search code snippets..).

Thanks, and we hope you enjoy using TensorFlow and Colab!

