# From Research to Practice



**TL;DR:**

* **Get prepared**

* **Prepare your data**

* **Find an analogy between your application and the closest deep learning applications**

* **Create a simple baseline model**

* **Create visualization and debugging tools**

* **Fine tune your model**

* **End to end training, ensembles, and other complexities**

![](https://cdn-images-1.medium.com/max/1600/1*cpuB9pgtIEDiusfJCYwDMw.png)

Leslie Smith is a researcher at the US Naval Research Lab. The NRL isn’t top of mind when it comes to deep learning research, but I have really enjoyed reading Smith’s papers. I’ve found them to contain just as many impressive contributions as folks at Google/Facebook/Baidu.

This paper covers recommended steps for using deep learning to solve problems in new or adjacent domains. It does not introduce a novel architecture or propose a mind-bending new training technique, but I think its content is just as important for the growth of deep learning and is often overlooked by researchers.

Because the format of this paper is a bit unique, I’ll be covering it as a summary list of notes with some commentary interspersed rather than using our Paper Notes Template.

#### Phase 1: Get prepared

* **Do you need deep learning?** Do state-of-the-art machine learning/statistics techniques perform well enough for your use case?

* **What does success look like?** This is best stated quantitatively, ideally with a metric or two you are optimizing for.

* Are there any shortcuts you can take? Ways you can get by with less data? Data preprocessing, heuristics?

> Important to distinguish between making the network’s job easier by giving it a cleaner dataset (good) vs. manual feature engineering (bad, this somewhat defeats the purpose of using a neural network)

![](https://cdn-images-1.medium.com/max/1600/1*UtGFbmJ6h75pfkyXZbEiEg.png)

#### Phase 2: Prepare your data

* Where’s your data coming from? How much do you have? What level of confidence do you have it?

> I wish transfer learning were given more of an emphasis here, rather than just being mentioned in passing. From my experience in the fast.ai course, transfer learning is often the option that best balances time/effort and performance.

* Does your data cover the entire problem space?

> If you’re trying to predict digits 0–9, make sure you have at least one of each class! Might seem obvious but when you get into large classifiers it can be overlooked.

* **Is the class distribution of your data representative of the problem space?**

> Sometimes this isn’t ideal, e.g. if you are trying to predict the presence of a cancer that appears in 0.05% of tests in the real world, you would want a higher percentage of your training set to be positive results. Just be aware of it and adjust accordingly.

* **Normalize and zero mean** your data. Pretty much no reason not to

* Can you use techniques to lower your data’s dimensionality using PCA or SVD without losing too much information?

> For novel deep learning applications, this is likely the most important phase. I think the importance of data is still not emphasized enough as compared to more exciting areas of research like neural net architectures. Hopefully the money talks and that changes.

![](https://cdn-images-1.medium.com/max/1600/1*mNdWU4z2Gw9mt1NCz5oelQ.png)

#### Phase 3: Find an analogy between your application and the closest deep learning applications

* Most low-hanging applications of deep learning have already been tackled by researchers, and chances are that yours can take inspiration from something that already exists. The author suggests this breakdown of categories:

* Image classification: CNNs

* Sequential/temporal data: RNNs (LSTM/GRU)

* Decisionmaking: deep reinforcement learning

* Use code or takeaways from this step to inform your approach to the next one

> Again, the author is getting almost all the way to transfer learning here; I wonder why they don’t take one more step and suggest not only finding code, but finding weights and biases that you can start your own model with.

#### Phase 4: Create a simple baseline model

* **Make a network that is as small as possible**, with a minimum number of layers and parameters, using a common loss function and safe hyperparameters.

> Throughout his fast.ai course (highly recommend btw, Paper Club has gone through both parts together), Jeremy Howard approached problems by building a network with just an input layer, one hidden layer, and an output layer. This might sound useless at first, but time after time it was enough to start to see convergence. I have personally found this approach quite profitable.

* Train on a subset of your training data to short the feedback cycle.

> As long as there is enough data for the model to converge, it’s enough for this step.

* Unit test your network

> I have literally never seen a unit tested neural network, and this makes me wonder a) why not and b) what tools even exist to do so.

* Take advantage of a framework like TensorFlow or PyTorch

![](https://cdn-images-1.medium.com/max/1600/1*OPNjgGFHVzYd55hycpvlVg.png)

#### Phase 5: Create visualization and debugging tools

* “Code once, measure twice”.

> From phases 1 and 2, you should have a very solid idea of the types of results you expect to be getting. Now it’s time to be meticulous about comparing these to your actuals. Check that your architecture, weights, biases, etc. conform to your expectations. I personally find myself printing out the numpy shapes of almost every important tensor in my network at some point or another.

* **High bias** vs. **high variance**: high bias means that your network is converging to the wrong minima or a local minima, and can be addressed with a larger network. High variance means that your network is not converging at all, and suggests that your training data is inadequate.

* **Error analysis** vs **ablative analysis**: error analysis is your classic loss, representing the difference between your network’s performance and a perfect performance. Ablative analysis is the process of setting up a baseline, changing things one at a time in your network, and noting how the altered network compares to the baseline. Both are useful.

* TensorBoard, which is bundled with TensorFlow, is a recommended tool to use

#### Phase 6: Fine tune your model

* **Change everything!** You should tinker with every aspect of your network and record how it affects performance: the architecture design, depth, width, pathways, weight initialization, learning rate, etc.

> You will likely find yourself faced with the problem of overfitting at some point during this process, when your training accuracy keeps improving but your validation accuracy plateaus or goes down. This is a classic machine learning problem, and the first thing to do when you see it is pat yourself on the back because you have a network that is learning! Then, just follow the established playbook for mitigating it:

> - Add more data

> - Use data augmentation

> - Use architectures that generalize well

> - Add regularization

> - Reduce architecture complexity.

* The **loss function** is the most important component of your network, so make sure you think deeply about what type of loss function is appropriate and test several that might fit your criteria.

* This phase is completely open-ended, and without some measure of self-control it might end up being just that. Be sure to constantly refer back to your stated goals from phase 1, and force yourself to stop when you reach them, quieting the voice in your head nagging you about that new loss function that you haven’t tried yet.

![](https://cdn-images-1.medium.com/max/1600/1*BFG_B7UpS3AvJxE6lH0lug.gif)

#### Phase 7: End to end training, ensembles, and other complexities

* There’s still more! The author notes that it’s totally unnecessary to use these if you are satisfied with your network performance, but suggests two that might be most rewarding: end to end training and ensembling. End to end training means trying to consolidate a system with multiple parts into just one neural network that takes the original input and produces the desired final output, even if they are of different formats (e.g. speech to text). Ensembling is the process of training multiple models with different random weight initializations and averaging or weighting their results in order to mitigate any biases or strangeness about the individual models.

And that’s that! Just follow these 7 simple, straightforward, easy, clearly-defined, no-confusion-at-all, steps and you’ll be on top of the world 😉

