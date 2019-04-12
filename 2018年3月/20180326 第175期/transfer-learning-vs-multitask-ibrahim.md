Training a deep neural network is a tedious process. More practical approaches includes re-using a trained networks for another task, and using the same network for number of tasks. In this article we discuss two important approaches. Transfer learning and Multi-task learning. 







![](https://media.licdn.com/dms/image/C4E12AQGvZRDDAEqp0A/article-inline_image-shrink_1000_1488/0?e=1560384000&v=beta&t=1PtMt1M731GEhMYqns2oTfPnH8MBr4dv9AmEv7cKJz0)



In Transfer learning, we would like to leverage the knowledge learned by a **source**task to help learning another **target**task. For example, a well-trained, rich image classification network could be leveraged for another image target related task. Another example, the knowledge learned by a network trained on simulated environment can be transferred to a network for the real environment. Basically, there are two basic scenarios for neural networks transfer learning: **Feature Extraction** and **Fine Tuning**. A well known example for transfer learning is to load the already trained large scale classification [VGG ](https://arxiv.org/abs/1409.1556)network that is able to classify images into one of 1000 classes, and use it for another task such as classification of special medical images. 



**1) Feature Extraction:**



In Feature extraction, a pre-trained network on a source task is used as a feature extractor for another target task by adding a simple classifier on top of the pre-trained network. Only the parameters of the added classifier are updated, while the pre-trained network parameters are frozen. This allows the new task to benefit from features learned from the source task. However, these features are more specialized for the source task.



**2) Fine Tuning:** 



Fine tuning allows modification of the pre-trained network parameters to learn the target task. Usually, a new randomly initialized layer is added above the pre-trained network. Parameters of the pre-trained network are updated but using a smaller learning rate to prevent major changes. It is normal to freeze the parameters of the bottom layers, the more generic layers, and only fine-tune some top layers, the more specific layers. Moreover, freezing some layers will reduce the number of trainable parameters and this could help to overcome the overfitting problem, especially when the available data for the target task is not large. Practically, fine tuning outperforms feature extraction as it enables optimizing pre-trained network for the new task.



**Transfer Learning Basic Scenarios:**



Basically, there are four scenarios for transfer learning depending on two main factors; 1) the size of target task dataset, 2) the similarity between the source and target tasks:



* **Case 1**: Target dataset is small and target task is similar to source task: In this case Feature extraction is used, because target dataset is small and training could cause model overfitting.

* **Case 2**: Target dataset is small and target task is different from source task: Here, we fine tune bottom, generic layers and remove higher, specific layers. In other words, we use feature extraction from early stages.

* **Case 3**: Target dataset is large and target task similar to source task: Here, we have large data, we can just train a network from scratch where the parameters are randomly initialized. However it would be better to make use of the pre-trained model to initialize the parameters and fine tune few layers. 

* **Case 4**: Target dataset is large and target task is different from source task: Here, we fine tune a large number of layers or even the entire network.







The main goal of multitask learning is to improve performance of a number of tasks simultaneously by optimizing all network parameters using samples from these tasks. For example, we would like to have one network that can classify an input face image as male or female, and at the same time can predict its age. Here we have two related tasks one is a binary classification task and the other is a regression task. It is clear that both tasks are related, and learning one should enhance learning the other. 



![](https://media.licdn.com/dms/image/C4E12AQE68ppEyvrIgw/article-inline_image-shrink_1500_2232/0?e=1560384000&v=beta&t=HrwUV3GEJOisyvjQ-EMALf90g2-b_iSeEgiZNdZjnWg)



An example of a simple network design could have a shared part between tasks and tasks specific heads. The shared part learns intermediate representations that are common between tasks that helps the learning of the tasks jointly. On the other hand, the specific heads learn how to use the shared representations for each specific task. 



 








> 
Transfer Learning and 
 Multitask learning are two vital approaches for Deep Learning 









Regards






