# How to Build A Neural Network Framework Like Tensorflow in C++: Part 1

Alright, so as I discussed in [part 0](https://medium.com/p/56f54c672852/), In this set of tutorials, I’ll be showing how you can build a Neural Network Framework similar to Tensorflow, but in C++.

The tutorials will be divided into coding a minimum viable product with:

1. The cost function

2. The minimizer that will minimize the cost function

3. The Neural Network module and its back propagation

In this tutorial, we will talk about the cost function.

First, let’s see the entire class’s header as it is now, and explain it line by line:



Lot’s of stuff to cover, so let’s get started.

Lines 1–19 are just the regular includes in a library like Shogun. It includes Stan’s headers(that we discussed in Part 0 of the tutorials) as well as Eigen (which is a linear algebra library in C++)

Lines 21–28 define some types to make it easier to reference later on. For example, StanVector is an Eigen Matrix that has a bunch of stan variables inside them (See Part 0 for more details).

Then, we have the main class: StanFirstOrderSAGCostFunction, which provides an interface for defining a stochastic average gradient cost function. It also provides the function get_gradient as well as get_average_gradient which is where most of the work goes.

But before I explain how this works, we need to look at the members of StanFirstOrderSAGCostFunction.

The first member is m_X, m_y which are basically the training data and labels of the cost function.

m_trainable_parameters are as the name suggests a bunch of stan variables that are the parameters of the cost function.

m_cost_for_ith_point and m_total_cost are again as the name suggests functions which evaluate the error of the ith point with respect to the trainable parameters, as well as the total cost with respect to all the costs of the ith datapoints.

So, what happens when get_gradient() is called is that we evaluate the errors with respect to the current parameters using the definitions of m_total_cost, then use stan to get the gradient of this error cost function with respect to each of the trainable parameters, and that’s the power of stan! For implementation details, check out this link where I implemented the class:



With this class done, we can now define any arbitrary cost function in terms of stan, and calculate the gradient of it with respect to all parameters using stan. For an example of how to use the class, checkout an example of it being used here, where I implemented mean squared error:



### A Word on Shogun

In case you’re interested in joining Open Source, Shogun is a great place to start. They have a super supportive community, and they welcome new comers to Opensource, so swing by and see if you can help with some of the issues labelled “beginner friendly” on their github!

