# 寻找Keras支持的最快的深度学习框架

原文链接：[Search for the fastest Deep Learning Framework supported by Keras](https://www.datasciencecentral.com/profiles/blogs/search-for-the-fastest-deep-learning-framework-supported-by-keras?from=hackcv&hmsr=hackcv.com&utm_medium=hackcv.com&utm_source=hackcv.com)

**摘要**：*Keras支持的流行深度学习框架的性能比较 - TensorFlow，CNTK，MXNet和Theano*

如果对数据科学家/工程师社区中流行的Keras和它的mindshare有任何疑问，那么您只需要查看它从所有主要的AI和Cloud使用者那里得到的支持。 目前，官方的Keras版本已经支持谷歌的TensorFlow和微软的CNTK深度学习库，除了支持如Theano等其他流行库之外。 去年，亚马逊网络服务公司宣布支持Apache MXNet，这是另一个功能强大的深度学习库，几周前，对MXNet的下一个候选版本增加了对Keras的支持。 截至目前，MXNet似乎只支持Keras v1.2.2而不是当前的Keras版本2.0.5。

[![img](https://api.ning.com/files/MhQM8NmP3L092R*BlOwPqEkMNCJKQ8w8IyczXKIuqt4thwiqpTjCkThcvOsj9kxwLNEdNHA-lZ4k6VkKNaAStWLWce05P7lo/keras_timeline.png?width=750)](https://api.ning.com/files/MhQM8NmP3L092R*BlOwPqEkMNCJKQ8w8IyczXKIuqt4thwiqpTjCkThcvOsj9kxwLNEdNHA-lZ4k6VkKNaAStWLWce05P7lo/keras_timeline.png)

虽然可以使用任何受支持的后端在生产中部署Keras模型，但开发人员和解决方案架构师应该记住，Keras本质上是不同DL于框架的高级API，尚不支持调整各个库提供的所有基础参数。因此，在希望对后端框架提供的所有参数进行微调的用例中，最好直接使用其中一个深度学习框架，而不是使用Keras作为顶层。当然，随着Keras和后端库中都添加了其他功能，这一点在将来可能会改变。但话说回来，Keras仍然是一个很好的工具，可以适用于大多数深度学习开发项目的早期阶段，因为它使数据科学家和工程师能够快速构建和测试复杂的深度学习模型。

 

Keras还允许开发人员在多个支持的深度学习框架中快速测试相对性能。Keras配置文件中的单个参数决定了将哪种深度学习框架用作后端。因此，您可以构建一个单独的模型且无需更改任何代码，您可以在TensorFlow、CNTK和Theano上运行它。对于MXNet，由于它目前只支持Keras ver1.2.2，因此需要对代码进行一些小小的更改，但这可能很快就会改变。这些单独的框架显然可以使用各个库中的功能来进一步的微调，但是Keras仍然提供了一个很好的机会来比较这些库之间的基本性能。



已经有一些文章比较了Keras支持的后端的相对性能，但是对于Keras或个别深度学习库的每个新版本，我们都看到了性能的显着提升。

那么，让我们看一下最近在不同深度学习框架中发布的最新版本是如何执行的。

 

让我们首先介绍用于测试的配置。

所有性能测试都是使用Nvidia Tesla K80 GPU在Azure NC6 VM上执行的。 使用的VM镜像是Ubuntu上的Azure DSVM（数据科学虚拟机）。除了其他数据科学工具之外，该镜像还预装了Keras，TensorFlow，Theano和MXNet。对于测试，所有包都更新到最新版本。为了使用MXNet，使用了较旧的Keras软件包1.2.2。有关Azure DSVM的其他[详细信息](https://docs.microsoft.com/en-us/azure/machine-learning/machine-learning-data-science-virtual-machine-overview)

 

**配置:**

 

由于每个框架的依赖性，我必须以三种配置运行测试，如下所示：

 

| **DL 框架**         | **软件配置**                                                 | **VM 配置**                                                  |
| ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **TensorFlow** | Keras: Version 2.0.8     <br>TensorFlow: Version 1.3.0     <br>CNTK: Version 2.1    <br>NVIDIA-CUDA Driver: v8.0.61      <br>CUDNN: v6.0.21 | Azure NC6 VM<br>GPU - Nvidia Tesla K80<br>vCPU - 6<br>Memory - 56GB<br>HDD - 380 GB<br>**MXNet** |
| **CNTK** | Keras: Version 1.2.2      <br>MXNet: Version 0.11.0      <br>NVIDIA-CUDA Driver: v8.0.61      <br>CUDNN: v6.0.21 |                                                              |
| **Theano**          | Keras: Version 2.0.8      <br>Theano: Version 0.9.0      <br>NVIDIA-CUDA Driver: v8.0.61      <br>CUDNN: v5.1.10 |                                                              |

 [![img](https://api.ning.com/files/MhQM8NmP3L2fpqEznI6t-sYr1dyvYCqeisHlfBd5VX*LvIQGBBsrXNPqzZpNeHrU8nscqTjTehorrm9x3Yh5*z7defiRXxZU/2screenshotfornvidia.png?width=643)](https://api.ning.com/files/MhQM8NmP3L2fpqEznI6t-sYr1dyvYCqeisHlfBd5VX*LvIQGBBsrXNPqzZpNeHrU8nscqTjTehorrm9x3Yh5*z7defiRXxZU/2screenshotfornvidia.png)

对于所有框架，使用最新的稳定版本进行测试。所有框架都有它们的下一个beta版本，它们声称可以提高性能，并且可能很适合用于研究目的，但是对于生产应用程序，首选是使用稳定的版本。因此，这些beta版本不包括在性能测试中。

**性能测试:**

为了比较DL框架的性能，我使用了下面描述的5种不同的测试模型。为了确保没有特定的框架得到任何特殊处理，所有的模型都来自GitHub上的[Keras/examples](https://github.com/fchollet/keras/tree/master/examples)。



测试代码/笔记本可以在我的GitHub仓库中找到 - https://github.com/jasmeetsb/deep-learning-keras-projects



注意：在两个测试中，MXNet被排除在外。这是因为MXNet还不支持较新的Keras函数，并且在MXNet上运行之前脚本需要进行重大更改。这可能会破坏这项工作的目的。即使是在MXNet上运行的3个测试也需要对脚本进行一些小小的更改，主要是因为在最近的版本中重命名了一些Keras函数。 

**1.** **测试 - CIFAR10 CNN**

​    **学习模型类型:** 卷积神经网络 (CNN)

​    **数据集/任务:**  CIFAR10 小图像数据集

 

​    **目标:** 将图像分为10类


​    就每轮的训练速度而言，TensorFlow略快于MXNet。

    在准确度/收敛速度方面，CNTK似乎在第25次迭代时略有优势，但到第50次迭代时，所有框架都显示出相似的精度。

 

[![img](https://api.ning.com/files/6wZBxJ-2hPUQ8GdRaRGGqfa5UbXDobWslvhpFvoBnjZetw2AbOYVac1VPjrCtkXSQJ3rAQ*ZC2bHAwjo7jH4YehYFwAw9eAJ/test1graph2.png?width=650)](https://api.ning.com/files/MhQM8NmP3L3DhiLVHPguTqYQu2mzg902KPWrpuz6J9ULYCrmf3s2rRr7w1GV6yHWjZKQ67oS-7DI*Mn0z7rDIcHf30xNuOo1/test1graph.png)

 

 [![img](https://api.ning.com/files/MhQM8NmP3L0cs0Qpj8Qc3p*wD0aNPgCHjrVAD8BEVJZmlfljLv0h7Q8jsh4Ywl31MjHNS6Q9kQyattwzXg2F7IKPx1SRV1Z*/test1linegraph.png?width=750)](https://api.ning.com/files/MhQM8NmP3L0cs0Qpj8Qc3p*wD0aNPgCHjrVAD8BEVJZmlfljLv0h7Q8jsh4Ywl31MjHNS6Q9kQyattwzXg2F7IKPx1SRV1Z*/test1linegraph.png)

 **2. 测试 - MNIST CNN**

​     **学习模型类型:** CNN

​     **数据集/任务:** MNIST 手写数字集

 

​     **目标:** 将图像分为10个类/数字

​     在此测试中，TensorFlow在训练速度方面优于其他框架，但在准确性/收敛速度方面，所有框架都展示了相似的特点。

 

[![img](https://api.ning.com/files/6wZBxJ-2hPVldddzDmUELt-C06QkvPSozIW4sVMDA6YFTh3X1DFktueyAD9BdcEKMZV62uo2oWIP1WgeqgzFRI3wt8bzlQcR/test2graph2.png?width=650)](https://api.ning.com/files/6wZBxJ-2hPVldddzDmUELt-C06QkvPSozIW4sVMDA6YFTh3X1DFktueyAD9BdcEKMZV62uo2oWIP1WgeqgzFRI3wt8bzlQcR/test2graph2.png)

 

[![img](https://api.ning.com/files/MhQM8NmP3L2UtEXhWuPqKIwzEGKwLZnAcyTykerQHQXZ6n7csou6iRp3b6QCVSh1tnqM9ykEcZlIVYRWsWRp6GdhBa*up82V/test2linegraph.png?width=750)](https://api.ning.com/files/MhQM8NmP3L2UtEXhWuPqKIwzEGKwLZnAcyTykerQHQXZ6n7csou6iRp3b6QCVSh1tnqM9ykEcZlIVYRWsWRp6GdhBa*up82V/test2linegraph.png)

**3.** **测试 - MNIST MLP**

​    **学习模型类型:** 多层感知器/Deep NN

​    **数据集/任务**: MNIST 手写数字集

 

​    **目标:** 将图像分为10个类/数字

​    在使用MNIST数据集的标准深度神经网络测试中，CNTK，TensorFlow和Theano获得了相似的分数（2.5 - 2.7 s / epoch），但MXNet以1.4s / epoch时间脱颖而出。 MXNet还在准确度/收敛速度方面展示了微小优势。

 

[![img](https://api.ning.com/files/6wZBxJ-2hPVm*usvaUvfEmaQMT-nGZ8nRvj3ozMQNBx71TLEegcs3zamYgs5QgDCqnVh3I*puz7wLjiN7hT7k2*Wh5A7ejEr/test3graph2.png?width=650)](https://api.ning.com/files/6wZBxJ-2hPVm*usvaUvfEmaQMT-nGZ8nRvj3ozMQNBx71TLEegcs3zamYgs5QgDCqnVh3I*puz7wLjiN7hT7k2*Wh5A7ejEr/test3graph2.png)

**![img](https://api.ning.com/files/MhQM8NmP3L2gtoNjDP4FcXuZnRRsJb8-y1u2Dyjs3TAemF5f9jFXsCjlxXLHC-olcejcPnN1qvAlcptBo3j8nHfhmwA9Ru5U/test3linegraph.png?width=750) 

**4. 测试 - MNIST RNN**

​    **学习模型类型:** 分层递归神经网络 (HRNN)

​    **数据集/任务:** MNIST 手写数字集

 

​    **目标:** 将图像分为10个类/数字

​    CNTK和MXNet在训练速度方面具有相似的性能（162-164 s/epoch），其次是TensorFlow，速度为179 s/epoch。 对于RNN模型，Theano表现似乎更差。

 [![img](https://api.ning.com/files/6wZBxJ-2hPV5-8RmbO0L9IkN9WnMYp9DEVAdjfr2DEQraFJfFXO*-wsQdtDK74CpS9OVQalKFHGudF6x8ZKd0pj6lQXDsOL3/test4graph2.png?width=650)](https://api.ning.com/files/6wZBxJ-2hPV5-8RmbO0L9IkN9WnMYp9DEVAdjfr2DEQraFJfFXO*-wsQdtDK74CpS9OVQalKFHGudF6x8ZKd0pj6lQXDsOL3/test4graph2.png)

 

 [![img](https://api.ning.com/files/MhQM8NmP3L3rjGsv9KJBRFt7HEPon8cQbLcOiowbfuqZH*i10VufmpVSRGbaL-uWqrSdKUITtOGv*cRofei54rxEvi23WzvL/test4linegraph.png?width=750)](https://api.ning.com/files/MhQM8NmP3L3rjGsv9KJBRFt7HEPon8cQbLcOiowbfuqZH*i10VufmpVSRGbaL-uWqrSdKUITtOGv*cRofei54rxEvi23WzvL/test4linegraph.png)

 

 

**5. 测试 - BABI RNN**

​    **学习模型类型:** 循环神经网络 (RNN)

​    **数据集/任务:** bAbi Project (<https://research.fb.com/downloads/babi/>)

 

​    **目标:**  根据一个故事和一个问题训练两个循环神经网络。然后查询合并后的向量以回答一系列bAbi任务。

​    **结果:** 由于Keras repo的样例脚本需要更改，MXNet被排除在外了。在9.5s/epoch时，TensorFlow和Theano的性能与CNTK相比提高了50%。

 [![img](https://api.ning.com/files/6wZBxJ-2hPXPlAiFsNE1a-3BztrYtEomdlQ4eQV6XYB9K9qdVi-e3bjQ3*sLZess2iG2ViXyGcDb8167KzfjUCj26CFY6kPH/test5graph2.png?width=650)](https://api.ning.com/files/6wZBxJ-2hPXPlAiFsNE1a-3BztrYtEomdlQ4eQV6XYB9K9qdVi-e3bjQ3*sLZess2iG2ViXyGcDb8167KzfjUCj26CFY6kPH/test5graph2.png)

 [![img](https://api.ning.com/files/MhQM8NmP3L0mVZq*9M4mahFptvz7dYL0ha0PQMNLraFYy17svyljVPkKxas1GL724*smfH3moRFwCGoDuMszDmQJgrZX7OT5/test5linegraph.png?width=750)](https://api.ning.com/files/MhQM8NmP3L0mVZq*9M4mahFptvz7dYL0ha0PQMNLraFYy17svyljVPkKxas1GL724*smfH3moRFwCGoDuMszDmQJgrZX7OT5/test5linegraph.png)

**总结:**

**![img](https://api.ning.com/files/MhQM8NmP3L3pORIsbBL-TykojVVo7SQAmoxf6pN03nZyhp23duqNgwkIuYG7xfg4G6Y-OVtsXddGmnICH0uUDLUc0NPS95Ha/summary.png)**

- TensorFlow在CNN测试中表现最佳，但在RNN测试案例中落后。
- CNTK在Babi RNN和MNIST RNN测试中表现明显优于TensorFlow和Theano，但在CNN测试案例中比TensorFlow慢。
- 在RNN测试中，MXNet的性能似乎比CNTK和TensorFlow略好，并且明显优于MLP测试中的所有框架，但事实上它不支持Keras v2功能，因此很难在不修改测试代码的情况下进行直接比较。 希望很快就能解决这个问题。
- 在深度神经网络（MLP）测试中，Theano在TensorFlow和CNTK上具有微弱优势。



**结论:** 从上面的结果可以看出，所有框架都有自己的优势，但目前还没有一个框架能够全面超越所有其他框架。CNTK在RNN用例、CNN和MXNet中的TensorFlow中表现出色，尽管展示了非常有前景的性能，但仍然有一些基础可以用来支持所有的Keras功能。正如开源世界中经常出现的情况一样，所有这些框架都在不断增强，从而提供更好的性能，并使它们更易于在生产环境中使用和部署。在考虑这些用于生产的深度学习框架时，性能是首要考虑因素，但在大多数情况下，您还需要考虑易于部署和作为这些工具的一部分的其他辅助工具，这些工具可以帮助您管理生产机器学习模型。这个讨论可能需要一个单独的文章，但我希望我上面的分析至少能给你提供一些额外的见解。

**免责声明:** *Jasmeet Bhatia是一名数据和AI解决方案架构师，目前在Microsoft工作。 在这篇文章中的观点是他本人的。*

*更新:包括关于使用稳定版本进行性能测试和使用更新的keras结果的说明。json参数image_data_forma*