# PyTorch tutorial distilled

原文链接：[PyTorch tutorial distilled](https://towardsdatascience.com/pytorch-tutorial-distilled-95ce8781a89c?from=hackcv&hmsr=hackcv.com)

## Migrating from TensorFlow to PyTorch



![img](https://cdn-images-1.medium.com/max/2000/1*aqNgmfyBIStLrf9k7d9cng.jpeg)

When I first started study PyTorch, I drop it after a few days. It was hard for me to get core concepts of this framework comparing with the TensorFlow. That’s why I’ve put it on my “knowledge bookshelf” and forgot about it. But not so far ago a new version of PyTorch was released. So I’ve decided to give it a chance again. After a while, I understood that this framework is really easy to use and it makes me happy to code in PyTorch. In this post, I will try to explain core concepts of it clearly so that you will be motivated at least give it a try right now, not after a few years or more. We will cover some basic principles and some advanced stuff as learning rate schedulers, custom layers and more.

#### Resources

First that you should know about PyTorch it that [documentation](http://pytorch.org/docs/master/) and [tutorials](http://pytorch.org/tutorials/)are stored separately. Also sometimes they may don’t meet each other, because of fast development and version changes. So fill free to investigate [source code](http://pytorch.org/tutorials/). It’s very clear and straightforward. And it’s better to mention that there are exist awesome [PyTorch forums](https://discuss.pytorch.org/), where you may ask any appropriate question, and you will get an answer relatively fast. This place seems to be even more popular than StackOverflow for the PyTorch users.

#### PyTorch as NumPy

So let’s dive into PyTorch itself. The main building block of the PyTorch is the tensors. Really, they are very similar to the [NumPy ones](https://docs.scipy.org/doc/numpy-dev/user/quickstart.html). Tensors support a lot of the same API, so sometimes you may use PyTorch just as a drop-in replacement of the NumPy. You may ask what the reason is. The principal goal is that PyTorch can utilize GPU so that you can transfer your data preprocessing or any other computation hungry stuff to machine learning workhorse. And it’s very easy to convert tensors from NumPy to PyTorch and vice versa. Let’s check some examples in code:



<iframe width="700" height="250" data-src="/media/caf8def11adef8f02d682c26b30f288b?postId=95ce8781a89c" data-media-id="caf8def11adef8f02d682c26b30f288b" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Favatars3.githubusercontent.com%2Fu%2F9900548%3Fv%3D4%26s%3D400&amp;key=a19fcc184b9711e1b4764040d3dc5c07" class="progressiveMedia-iframe js-progressiveMedia-iframe" allowfullscreen="" frameborder="0" src="https://towardsdatascience.com/media/caf8def11adef8f02d682c26b30f288b?postId=95ce8781a89c" style="user-select: text !important; display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 666px;"></iframe>

#### From the tensors to the variables

Tensors are an awesome part of the PyTorch. But mainly all we want is to build some neural networks. What is about backpropagation? Of course, we can manually implement it, but what is the reason? Thankfully automatic differentiation exists. To support it PyTorch [provides variables](http://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_autograd.html) to you. Variables are wrappers above tensors. With them, we can build our computational graph, and compute gradients automatically later on. Every variable instance has two attributes: `.data` that contain initial tensor itself and `.grad` that will contain gradients for the corresponding tensor.



<iframe width="700" height="250" data-src="/media/214f557a06e55da09ba5bd2f2740b7cb?postId=95ce8781a89c" data-media-id="214f557a06e55da09ba5bd2f2740b7cb" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Favatars3.githubusercontent.com%2Fu%2F9900548%3Fv%3D4%26s%3D400&amp;key=a19fcc184b9711e1b4764040d3dc5c07" class="progressiveMedia-iframe js-progressiveMedia-iframe" allowfullscreen="" frameborder="0" src="https://towardsdatascience.com/media/214f557a06e55da09ba5bd2f2740b7cb?postId=95ce8781a89c" style="user-select: text !important; display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 578.987px;"></iframe>

You may note that we have manually computed and applied our gradients. It’s so tedious. Do we have some optimizer? Of course!



<iframe width="700" height="250" data-src="/media/1b9b56e531553ae43d9af79942cc1462?postId=95ce8781a89c" data-media-id="1b9b56e531553ae43d9af79942cc1462" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Favatars3.githubusercontent.com%2Fu%2F9900548%3Fv%3D4%26s%3D400&amp;key=a19fcc184b9711e1b4764040d3dc5c07" class="progressiveMedia-iframe js-progressiveMedia-iframe" allowfullscreen="" frameborder="0" src="https://towardsdatascience.com/media/1b9b56e531553ae43d9af79942cc1462?postId=95ce8781a89c" style="user-select: text !important; display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 578.987px;"></iframe>

Now all our variables will be updated automatically. But the main point that you should get from the last snippet: we still should manually zero gradients before calculating new ones. This is one of the core concepts of the PyTorch. Sometimes it may be not very obvious why we should do this, but on the other hand, we have full control over our gradients, when and how we want to apply them.

#### Static vs. dynamic computational graphs

Next main difference between PyTorch and TensorFlow is their approach to the graph representation. Tensorflow [uses a static graph](https://www.tensorflow.org/programmers_guide/graphs), that means that we define it once and after execute that graph over and over again. In PyTorch each forward pass defines a new computational graph. In the beginning, the distinction between those approaches not so huge. But dynamic graphs became very handful when you want to debug your code or define some conditional statements. You can use your favorite debugger as it is! Compare next two definitions of the while loop statements - the first one in TensorFlow and the second one in PyTorch:



<iframe width="700" height="250" data-src="/media/2e9029abd7bef658d7519be4e8176351?postId=95ce8781a89c" data-media-id="2e9029abd7bef658d7519be4e8176351" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Favatars3.githubusercontent.com%2Fu%2F9900548%3Fv%3D4%26s%3D400&amp;key=a19fcc184b9711e1b4764040d3dc5c07" class="progressiveMedia-iframe js-progressiveMedia-iframe" allowfullscreen="" frameborder="0" src="https://towardsdatascience.com/media/2e9029abd7bef658d7519be4e8176351?postId=95ce8781a89c" style="user-select: text !important; display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 578.987px;"></iframe>



<iframe width="700" height="250" data-src="/media/ca2584bf9ad710e712c2e227ed13f462?postId=95ce8781a89c" data-media-id="ca2584bf9ad710e712c2e227ed13f462" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Favatars3.githubusercontent.com%2Fu%2F9900548%3Fv%3D4%26s%3D400&amp;key=a19fcc184b9711e1b4764040d3dc5c07" class="progressiveMedia-iframe js-progressiveMedia-iframe" allowfullscreen="" frameborder="0" src="https://towardsdatascience.com/media/ca2584bf9ad710e712c2e227ed13f462?postId=95ce8781a89c" style="user-select: text !important; display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 234px;"></iframe>

It seems to me that second solution much easier than first one. And what do you think about it?

#### Models definition

Ok, now we see that it’s easy to build some if/else/while complex statements in PyTorch. But let’s revert to the usual models. The framework provides out of the box layers constructors very similar to [Keras](https://keras.io/) ones:

> The `nn` package defines a set of **Modules**, which are roughly equivalent to neural network layers. A Module receives input Variables and computes output Variables, but may also hold internal state such as Variables containing learnable parameters. The `nn` package also defines a set of useful loss functions that are commonly used when training neural networks.



<iframe width="700" height="250" data-src="/media/e26dadacc9034bc873a236617a196a5f?postId=95ce8781a89c" data-media-id="e26dadacc9034bc873a236617a196a5f" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Favatars3.githubusercontent.com%2Fu%2F9900548%3Fv%3D4%26s%3D400&amp;key=a19fcc184b9711e1b4764040d3dc5c07" class="progressiveMedia-iframe js-progressiveMedia-iframe" allowfullscreen="" frameborder="0" src="https://towardsdatascience.com/media/e26dadacc9034bc873a236617a196a5f?postId=95ce8781a89c" style="user-select: text !important; display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 514px;"></iframe>

Also if we want to build more complex models, we may subclass provided `nn.Module` class. And of course, these two approaches can be mixed with each other.



<iframe width="700" height="250" data-src="/media/76edc5e2f5e6d498bbab08aa2b21062d?postId=95ce8781a89c" data-media-id="76edc5e2f5e6d498bbab08aa2b21062d" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Favatars3.githubusercontent.com%2Fu%2F9900548%3Fv%3D4%26s%3D400&amp;key=a19fcc184b9711e1b4764040d3dc5c07" class="progressiveMedia-iframe js-progressiveMedia-iframe" allowfullscreen="" frameborder="0" src="https://towardsdatascience.com/media/76edc5e2f5e6d498bbab08aa2b21062d?postId=95ce8781a89c" style="user-select: text !important; display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 450px;"></iframe>

At the `__init__` method we should define all layers that will be used later. In the `forward` method, we should propose steps how we want to use already defined layers. Backward pass, as usual, will be computed automatically.

#### Self-defined layers

But what if we want to define some custom model with nonstandard backprop? Here is one example — XNOR networks:



![img](https://cdn-images-1.medium.com/max/1000/1*cjzIFgglAP9xGKg8mlRysQ.png)

I will not dive into details, more about this type of networks you may read in the [initial paper](https://arxiv.org/abs/1603.05279). All relevant to our issue is that backpropagation should be applied only to weights that less than 1 and greater than -1. In PyTorch it [can be implemented quite easy](http://pytorch.org/docs/master/notes/extending.html):



<iframe width="700" height="250" data-src="/media/48bf0fc8fecfe815810a138441674709?postId=95ce8781a89c" data-media-id="48bf0fc8fecfe815810a138441674709" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Favatars3.githubusercontent.com%2Fu%2F9900548%3Fv%3D4%26s%3D400&amp;key=a19fcc184b9711e1b4764040d3dc5c07" class="progressiveMedia-iframe js-progressiveMedia-iframe" allowfullscreen="" frameborder="0" src="https://towardsdatascience.com/media/48bf0fc8fecfe815810a138441674709?postId=95ce8781a89c" style="user-select: text !important; display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 730px;"></iframe>

So as you may see, we should only define exactly two methods: one for forward and one for backward pass. If we need access to some variables from the forward pass we may store them in the `ctx` variable. Note: in previous API forward/backward methods were not static and we stored required variables as `self.save_for_backward(input)` and access them as `input, _ = self.saved_tensors`.

#### Train model with CUDA

If was discussed earlier how we might pass one tensor to the CUDA. But if we want to pass the whole model, it’s ok to call `.cuda()` method from the model itself, and wrap each input variable to the `.cuda()` and it will be enough. After all computations, we should get results back with `.cpu()` method.



<iframe width="700" height="250" data-src="/media/a0754eaf7543b84f3a14bfabf1ada845?postId=95ce8781a89c" data-media-id="a0754eaf7543b84f3a14bfabf1ada845" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Favatars3.githubusercontent.com%2Fu%2F9900548%3Fv%3D4%26s%3D400&amp;key=a19fcc184b9711e1b4764040d3dc5c07" class="progressiveMedia-iframe js-progressiveMedia-iframe" allowfullscreen="" frameborder="0" src="https://towardsdatascience.com/media/a0754eaf7543b84f3a14bfabf1ada845?postId=95ce8781a89c" style="user-select: text !important; display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 450px;"></iframe>

Also, PyTorch supports direct devices allocation at the source code:



<iframe width="700" height="250" data-src="/media/a4d012ac9617d0a72a7ddd7b8f02e1bd?postId=95ce8781a89c" data-media-id="a4d012ac9617d0a72a7ddd7b8f02e1bd" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Favatars3.githubusercontent.com%2Fu%2F9900548%3Fv%3D4%26s%3D400&amp;key=a19fcc184b9711e1b4764040d3dc5c07" class="progressiveMedia-iframe js-progressiveMedia-iframe" allowfullscreen="" frameborder="0" src="https://towardsdatascience.com/media/a4d012ac9617d0a72a7ddd7b8f02e1bd?postId=95ce8781a89c" style="user-select: text !important; display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 406px;"></iframe>

Because sometimes we want to run the same model on the CPU and the GPU without code modification I propose some kind of wrapper:



<iframe width="700" height="250" data-src="/media/e7a51f45014201aef5f5a02c86ed1460?postId=95ce8781a89c" data-media-id="e7a51f45014201aef5f5a02c86ed1460" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Favatars3.githubusercontent.com%2Fu%2F9900548%3Fv%3D4%26s%3D400&amp;key=a19fcc184b9711e1b4764040d3dc5c07" class="progressiveMedia-iframe js-progressiveMedia-iframe" allowfullscreen="" frameborder="0" src="https://towardsdatascience.com/media/e7a51f45014201aef5f5a02c86ed1460?postId=95ce8781a89c" style="user-select: text !important; display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 514px;"></iframe>

#### Weight initialization

In TensorFlow weights initialization mainly are made during tensor declaration. PyTorch offers another approach — at first, tensor should be declared, and on the next step weights for this tensor should be changed. Weights can be initialized as direct access to the tensor attribute, as a call to the bunch of methods inside `torch.nn.init` package. This decision can be not very straightforward, but it becomes useful when you want to initialize all layers of some type with same initialization.



<iframe width="700" height="250" data-src="/media/0152c521ff9c9df1ed546564f8dd7431?postId=95ce8781a89c" data-media-id="0152c521ff9c9df1ed546564f8dd7431" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Favatars3.githubusercontent.com%2Fu%2F9900548%3Fv%3D4%26s%3D400&amp;key=a19fcc184b9711e1b4764040d3dc5c07" class="progressiveMedia-iframe js-progressiveMedia-iframe" allowfullscreen="" frameborder="0" src="https://towardsdatascience.com/media/0152c521ff9c9df1ed546564f8dd7431?postId=95ce8781a89c" style="user-select: text !important; display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 752px;"></iframe>

#### Excluding subgraphs from backward

Sometimes when you want to retrain some layers of your model or prepare it for the production mode, it’s great when you can disable autograd mechanics for some layers. For this purposes, [PyTorch provides two flags](http://pytorch.org/docs/master/notes/autograd.html): `requires_grad`and `volatile`. First one will disable gradients for current layer, but child nodes still can calculate some. The second one will disable autograd for current layer and for all child nodes.



<iframe width="700" height="250" data-src="/media/dfdac4bb40bb2190da57603515b22f73?postId=95ce8781a89c" data-media-id="dfdac4bb40bb2190da57603515b22f73" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Favatars3.githubusercontent.com%2Fu%2F9900548%3Fv%3D4%26s%3D400&amp;key=a19fcc184b9711e1b4764040d3dc5c07" class="progressiveMedia-iframe js-progressiveMedia-iframe" allowfullscreen="" frameborder="0" src="https://towardsdatascience.com/media/dfdac4bb40bb2190da57603515b22f73?postId=95ce8781a89c" style="user-select: text !important; display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 493px;"></iframe>

#### Training process

There are also exists some other bells and whistles in PyTorch. For example, you may use [learning rate scheduler](http://pytorch.org/docs/master/optim.html#how-to-adjust-learning-rate) that will adjust your learning rate based on some rules. Or you may enable/disable batch norm layers and dropouts with single train flag. If you want it’s easy to change random seed separately for CPU and GPU.



<iframe width="700" height="250" data-src="/media/a0e831526a0d22e2988647ad3f8ea1fc?postId=95ce8781a89c" data-media-id="a0e831526a0d22e2988647ad3f8ea1fc" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Favatars3.githubusercontent.com%2Fu%2F9900548%3Fv%3D4%26s%3D400&amp;key=a19fcc184b9711e1b4764040d3dc5c07" class="progressiveMedia-iframe js-progressiveMedia-iframe" allowfullscreen="" frameborder="0" src="https://towardsdatascience.com/media/a0e831526a0d22e2988647ad3f8ea1fc?postId=95ce8781a89c" style="user-select: text !important; display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 514px;"></iframe>

Also, you may print info about your model, or save/load it with few lines of code. If your model was initialized with [OrderedDict](https://docs.python.org/3/library/collections.html) or class-based model string representation will contain names of the layers.



<iframe width="700" height="250" data-src="/media/af9f29a3d4938c4255a98764ccfdd6c2?postId=95ce8781a89c" data-media-id="af9f29a3d4938c4255a98764ccfdd6c2" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Favatars3.githubusercontent.com%2Fu%2F9900548%3Fv%3D4%26s%3D400&amp;key=a19fcc184b9711e1b4764040d3dc5c07" class="progressiveMedia-iframe js-progressiveMedia-iframe" allowfullscreen="" frameborder="0" src="https://towardsdatascience.com/media/af9f29a3d4938c4255a98764ccfdd6c2?postId=95ce8781a89c" style="user-select: text !important; display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 621.987px;"></iframe>

As per PyTorch documentation saving model with `state_dict()` method is [more preferable](http://pytorch.org/docs/master/notes/serialization.html).

#### Logging

Logging of the training process is a pretty important part. Unfortunately, PyTorch has no any tools like tensorboard. So you may use usual text logs with [Python logging module](https://docs.python.org/3/library/logging.html) or try some of the third party libraries:

- [A simple logger for experiments](https://github.com/oval-group/logger)
- [A language-agnostic interface to TensorBoard](https://github.com/torrvision/crayon)
- [Log TensorBoard events without touching TensorFlow](https://github.com/TeamHG-Memex/tensorboard_logger)
- [tensorboard for pytorch](https://github.com/lanpa/tensorboard-pytorch)
- [Facebook visualization library wisdom](https://github.com/facebookresearch/visdom)

#### Data handling

You may remember [data loaders proposed in TensorFlow](https://www.tensorflow.org/api_guides/python/reading_data) or even tried to implement some of them. For me, it took about 4 hours or more to get some idea how all pipeline should work.



![img](https://cdn-images-1.medium.com/max/1000/1*S00VU2HiEjNZ35zlj2kqfw.gif)

Image source: TensorFlow docs

Initially, I thought to add here some code, but I think such gif will be enough to explain basic idea how all things happen.

PyTorch developers decided do not reinvent the wheel. They just use multiprocessing. To create your own custom data loader, it’s enough to inherit your class from `torch.utils.data.Dataset` and change some methods:



<iframe width="700" height="250" data-src="/media/7499f54f158531418c5d8fbc27c01f22?postId=95ce8781a89c" data-media-id="7499f54f158531418c5d8fbc27c01f22" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Favatars3.githubusercontent.com%2Fu%2F9900548%3Fv%3D4%26s%3D400&amp;key=a19fcc184b9711e1b4764040d3dc5c07" class="progressiveMedia-iframe js-progressiveMedia-iframe" allowfullscreen="" frameborder="0" src="https://towardsdatascience.com/media/7499f54f158531418c5d8fbc27c01f22?postId=95ce8781a89c" style="user-select: text !important; display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 1118.99px;"></iframe>

The two things you should know. First — image dimensions are different from TensorFlow. They are [batch_size x channels x height x width]. But this transformation can be made without you interaction by preprocessing step `torchvision.transforms.ToTensor()`. There are also a lot of useful utils in the [transforms package](http://pytorch.org/docs/master/torchvision/transforms.html).

The second important thing that you may use pinned memory on GPU. For this, you just need to place additional flag `async=True` to a `cuda()` call and get pinned batches from DataLoader with flag `pin_memory=True`. More about this feature [discussed here](http://pytorch.org/docs/master/notes/cuda.html#use-pinned-memory-buffers).

#### Final architecture overview

Now you know about models, optimizers and a lot of other stuff. What is the right way to merge all of them? I propose to split your models and all wrappers on such building blocks:



![img](https://cdn-images-1.medium.com/max/1000/1*A-cWYNur2lqDEhUF1_gdCw.png)

And here is some pseudo code for clarity:



<iframe width="700" height="250" data-src="/media/528d40b05ef5e7aab5007d10cf57018b?postId=95ce8781a89c" data-media-id="528d40b05ef5e7aab5007d10cf57018b" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Favatars3.githubusercontent.com%2Fu%2F9900548%3Fv%3D4%26s%3D400&amp;key=a19fcc184b9711e1b4764040d3dc5c07" class="progressiveMedia-iframe js-progressiveMedia-iframe" allowfullscreen="" frameborder="0" src="https://towardsdatascience.com/media/528d40b05ef5e7aab5007d10cf57018b?postId=95ce8781a89c" style="user-select: text !important; display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 730px;"></iframe>

#### Conclusion

I hope with this post you’ve understood main points of PyTorch:

- It can be used as drop-in replacement of Numpy
- It’s really fast for prototyping
- It’s easy to debug and use conditional flows
- There are lots of great tools out of the box

PyTorch is the fast-growing framework with an awesome community. And I think that today is the best day to try it out!