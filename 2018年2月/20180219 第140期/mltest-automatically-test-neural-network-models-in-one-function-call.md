# mltest: Automatically test neural network models in one function call

![](https://cdn-images-1.medium.com/max/1600/1*nwfCyEi6d3Jt2CfcC-2_WA.png)

So I got a lot of positive feedback on my last post on [how to unit test machine learning code](https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765). A few people actually messaged me directly saying they caught a bug in their own code with the recommended tests, which is awesome! But these issues are still too common, and it is just as easy to forget to write a test as it is to write the bug in the first place. We need a better, more automated solution.

That is why we are introducing [mltest: Automated ML testing in one function call.](https://github.com/Thenerdstation/mltest)

Check it out!



Done. With incredibly little setup, we now are testing against several different common machine learning issues.

To install it, just run:



The function call mltest.test_suite(…) is the main powerhouse of this library. It runs several tests including:

### 1. Variables change

The test from [my first post](https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765) that helped people the most was the variables change test. Basically, you run the train_op and make sure that all of the variables within a list or scope are modified.

### 2. Variables DON’T change

It is also possible to make sure that only variables within a scope or a list are the ones that change, and that the rest do not change. This is super useful for GAN training, as the generator and discriminator usually have different training ops.

### 3. Logits Range

One common mistake that I would make is adding a non-linearity to my logits output. This would usually cause lots of problems once it hits softmax. Our test_suite() automatically checks that the logits layer has values that are above and below 0. This perhaps isn’t the best way of detecting it, but it helps narrow down the problem.

You can also set the range to check to be whatever you want. Let’s say you expect to have a tanh on the logits, you can set a check to make sure all of the values of the logits are in (-1, 1).

### 4. Input dependencies

One common issue is that sometimes people forget to connect the branches of their network together. Whether you forget to add two tensors together or forget to call the function call for a certain branch. A network can still train and converge poorly with only partial input, so it is import to make sure all of your input values are dependents of the training op.

Of course, any of these tests can be turned off manually with flags in test_suite(). See the code for documentation on how to do this.

### mltest setup

Another useful feature is mltest.setup().



This call will automatically reset the default tensorflow graph and set tensorflow’s, numpy’s, and python’s random seeds. It’s very easy to forget to seed your random values, and can cause a massive headache when trying to recreate bugs.

This suite is still in beta, so if you have any requests or notice any bugs, please add an issue on Github! Also, I accept pull requests. ;)

