# ICML 2018 notes

Some notes I took on ICML 2018

#### Sanjeev Arora: Toward Theoretical Understanding of Deep Learning

[http://unsupervised.cs.princeton.edu/deeplearningtutorial.html](http://unsupervised.cs.princeton.edu/deeplearningtutorial.html)

Sanjeev Arora has a knack of presenting intuition for the results while being rigorous. Here’s one example, in what way is hyper-cube in many dimensions like a “spiky sphere?

![](https://cdn-images-1.medium.com/max/1600/1*ImMv2JSgB6S_bBCNGRci-g.png)

In this tutorial he summarized the current (mostly barren) landscape of mathematical results in deep learning.

Some results:

* perturbed SGD can escape saddle points

* perturbed SGD can escape shallow local minima

* some special net architectures can be optimized to global minimum despite many local minima

* GANs are impossible to train in worst case

Later he went to cover his work on quantifying generalization capacity of Neural Networks. How can 1M parameter neural net generalize after getting trained to convergence on 100k examples? This is not possible in classical statistics.

His group results show are able to bound capacity of neural nets to that below of number of parameters based on “noise stability” of neural net. Basically trained neural net will “reject” Gaussian noise injected at intermediate layers — a few layers down the road, injected noise is attenuated and original activations are mostly unchanged.

![](https://cdn-images-1.medium.com/max/1600/1*8PAB9-3o8NAGC7Su4cQOOw.png)

You can then use this “noise stability” to prove that neural network is compressible — throwing out a lot of the weights is possible without changing numerics much. Compressibility implies generalization since there are few compressible networks.

Note that while noise stability in trained network implies generalization, you want the opposite of noise stability in your untrained network. All perturbations should propagate without attenuation. Initializing neural nets in this way, using orthogonal matrices for fully connected layers, and “delta-orthogonal” for conv layers, enabled training 10k layer tanh network in “[Dynamical Isometry and a Mean Field Theory of CNNs](https://arxiv.org/abs/1806.05393)”.

![](https://cdn-images-1.medium.com/max/1600/1*LQ2YeDK4XhXw07dXo6tf7Q.png)

Later he talked about “role of depth”. Counter-intuitively, you can accelerate optimization by making network deeper, and there’s a simple synthetic optimization problem where this was demonstrated. The task is L4 regression. Note that if you replace it with L2 regression, the extra hidden layer no longer helps.

![](https://cdn-images-1.medium.com/max/1600/1*h2a8Wmwwe9J-j8q3_pWcPA.png)

### Dissecting Adam

[**[1705.07774] Dissecting Adam: The Sign, Magnitude and Variance of Stochastic Gradients**
Abstract: The ADAM optimizer is exceedingly popular in the deep learning community. Often it works very well, sometimes…arxiv.org](https://arxiv.org/abs/1705.07774)[](https://arxiv.org/abs/1705.07774)

Gives a nice to understand version of Adam in terms of “noise-to-signal ratio”. Define “noise to signal ratio” nu as a ratio of gradient variance to gradient magnitude. In the extreme case of perfect gradients, this ratio is 0.

![](https://cdn-images-1.medium.com/max/1600/1*OHXTunnwI8pIilm0OcA0Mw.png)

In terms of this ratio, Adam coordinate scaling rule looks as follows

![](https://cdn-images-1.medium.com/max/1600/1*afx3dkBQyZwZZ8NbvvoRXw.png)

What’s interesting about this, is that it’s connected to the “optimal” per-coordinate scaling rate.

![](https://cdn-images-1.medium.com/max/1600/1*sNuC7CITwcMvp4ZOUSBj9A.png)

Curiously, similar expression was also derived in 3.2 of “No more pesky learning rates.”

![](https://cdn-images-1.medium.com/max/1600/1*Bu2lVTzanrEE8Rljpyr79g.png)

The numerator is gradient norm squared for case of quadratic, so divide by it to get the same form as in Adam.

Note that Adam adds square root to the expression. The square root seems to come out of worst-case regret analysis of AdaGrad which Adam is based on. The variation without the square root is known in online learning as the Online Newton Method.

“Square root free” version of Adam is called “Stochastic Variance Adapted Gradient” and they show it helps in some problems. In the graph below, green is Adam and red is SVAG.

![](https://cdn-images-1.medium.com/max/1600/1*OnhE-r46K98OU0C2qX3yRA.png)

### Decoupled Parallel Backpropagation with Convergence Guarantee

![](https://cdn-images-1.medium.com/max/1600/1*S1ic-T81cSZ3OoMJMZS5vg.png)

[https://icml.cc/Conferences/2018/Schedule?showEvent=2106](https://icml.cc/Conferences/2018/Schedule?showEvent=2106)
Parallelize across GPUs by placing different layers on different GPUs. To avoid locking, use pipelining — ie, while layer n+1 is computing inputs into layer n, do the computation in layer n using inputs from previous step. They see 2x speed-up by going 2 GPUS->4 GPUS with no loss in accuracy. This gives an alternative way of parallelization — instead of parallelizing along batch dimension, you parallelize across layers. I expect this form of parallelism to have similar impact of model quality as increased batch size, because both forms can be viewed as a kind of gradient staleness.

### Error Compensated Quantized SGD and its Applications to Large-scale Distributed Optimization

[**[1806.08054] Error Compensated Quantized SGD and its Applications to Large-scale Distributed…**
Abstract: Large-scale distributed optimization is of great importance in various applications. For data-parallel based…arxiv.org](https://arxiv.org/abs/1806.08054)[](https://arxiv.org/abs/1806.08054)

The notable part is the compensation approach they used to correct for quantization error — store errors locally and fold them into next step’s gradients

![](https://cdn-images-1.medium.com/max/1600/1*aqy2a3qAkUdUqOowQIAOfg.png)

This compensation approach what was also used earlier by NVidia in “deep gradient compression, although in case of NVidia”, they claim 1000x compression — by sending only the most significant gradient entries.

### signSGD: Compressed Optimisation for Non-Convex Problems

[**[1802.04434] signSGD: Compressed Optimisation for Non-Convex Problems**
Abstract: Training large neural networks requires distributing learning across multiple workers, where the cost of…arxiv.org](https://arxiv.org/abs/1802.04434)[](https://arxiv.org/abs/1802.04434)

Instead of sending gradient, binarize coordinates to +1 or -1. For multiple workers, workers send 1/-1 and do majority vote to decide on the aggregated value. The crazy thing is that this doesn’t appear to lose accuracy compared to Adam.

![](https://cdn-images-1.medium.com/max/1600/1*sR0sl5PtSft4OgbKJg9tng.png)

### A Progressive Batching l-BFGS

[**[1802.05374] A Progressive Batching L-BFGS Method for Machine Learning**
Abstract: The standard L-BFGS method relies on gradient approximations that are not dominated by noise, so that search…arxiv.org](https://arxiv.org/abs/1802.05374)[](https://arxiv.org/abs/1802.05374)

The main contribution of this paper seems to be the test on how to increase sample size to make l-BFGS step reliable. The basic intuition is to choose batch size large enough so that gradients are aligned with full-batch gradient sufficiently often. The extension is then to make sure that their estimated stochastic Quasi-Newton direction is aligned with true Quasi-Newton direction sufficiently often. They also replace “line search” component of Quasi-Newton method with a “stochastic line-search”. The algorithm seemed quite complex to describe, but I think the two contributions seem useful on their own:

* expression for how to increase batch size in response to gradient noise

* extension of deterministic line search for mini-batch gradients

### Distributed Asynchronous Optimization with Unbounded Delays: How Slow Can You Go?

Stale gradients are bad. However, they are not so bad if staleness is bounded, as proved in Stale Synchronous Parameter Server [paper](http://www.cs.cmu.edu/~seunghak/SSPTable_NIPS2013.pdf). Also, they are not so bad if they grow sublinearly with time, shown in Bertesekas [book](http://www.mit.edu/~jnt/parallel.html). In this paper they show that they can even grow grow polynomially, and you’d still converge to global minimum for a class of convex problems.

![](https://cdn-images-1.medium.com/max/1600/1*JkMgm2F9ZDHlmW2Q6SkLkA.png)

Larger delays means step sizes must shrink to zero more aggressively in order to aggregate over larger window of gradients. With linear delays, step sizes (alpha’s) must go down slightly faster than geometric rate. Surprisingly, polynomially growing delays don’t require shrinking much faster than for linearly growing delays, and there’s no dependence degree of the polynomial.

### A Delay-tolerant Proximal-Gradient Algorithm for Distributed Learning

The key idea is that instead of parameter server averaging gradients, it averages parameter values. This idea should make sense — a delayed worker would send a gradient computed at a point far away from current parameter server estimate. It would be a bad step for parameter server estimate, but a good step for worker estimate. Therefore let the worker take the step and average parameter values. Basically, combining iterates is more stable than combining gradients. This scheme lets enables convergence without requiring aggressively shrinking step sizes, like in the previous paper.

![](https://cdn-images-1.medium.com/max/1600/1*N7FUdB19N7cOi3rJMzvcgw.png)

#### Asynchronous Decentralized Parallel Stochastic Gradient Descent

Same idea — average iterates, not gradients comes up [here](https://arxiv.org/abs/1710.06952). In addition, their architecture is decentralized: each worker averages its parameters with a random neighbor at each step.

![](https://cdn-images-1.medium.com/max/1600/1*CxkqrXspcVIbIAWiZKmm7w.png)

They compared against AllReduce and Elastic Averaging SGD for 16 workers and found better scaling in training throughput. The scaling difference was especially dramatic when there’s a slow worker or a slowdown in network links.

![](https://cdn-images-1.medium.com/max/1600/1*wJXyF3Gky3fkmK1L5muSKw.png)

Also, unexpectedly, decentralized version of the algorithm (AD-PSGD) achieved higher accuracy than centralized version (D-PSGD) after same number of epochs, similar to AllReduce.

![](https://cdn-images-1.medium.com/max/1600/1*zhMLqtu6kkmyYCMVDBUr9Q.png)

#### Adaptive Regularization Strikes Back

[https://arxiv.org/abs/1806.02958](https://arxiv.org/abs/1806.02958)

Original derivation of AdaGrad requires inverting historical gradient covariance matrix. If your parameter size is large (ie, 1M) and gradient history small (ie, 200), the shape of the matrix looks as follows. Each column of G is a gradient vector.

![](https://cdn-images-1.medium.com/max/1600/1*nIIe5ISmIR5ONuAave-nQw.png)

Note that this matrix has low rank, and this inverse can be computed more efficiently in terms of the following matrix.

![](https://cdn-images-1.medium.com/max/1600/1*-Rp7FHUL40e1vlQlsJwaiw.png)

Instead of inverting 1Mx1M matrix, you are inverting 200x200 matrix.

They got significantly better results on RNN tasks like PTB, outperforming baselines in wall-clock time. The improvement seems more marginal for CIFAR-10, they hypothesize that RNN task was “more nonlinear” so had more to gain from second order preconditioning.

This trick also means you can compute eigenvalues of (low-rank) gradient covariance matrix efficiently. Plotting eigenvalues, they show that RNN task gradients are much worse conditioned, which would lead one to expect more benefit from second order method.

![](https://cdn-images-1.medium.com/max/1600/1*sfxT7EzmU9nMJ1m5lZaP_g.png)

To me this plot suggests that RNN optimization starts in a narrow valley and ends up in a bowl, whereas CNN optimization starts in a bowl, and ends up in a narrow valley.

#### Shampoo: Preconditioned Stochastic Tensor Optimization

Another approach to make full-matrix AdaGrad efficient. The idea is that gradients are shaped as matrices (fully connected layer) or 4D tensors (convolution), and preconditioning along individual dimensions is much cheaper than per-element. Similar idea to [KFAC](https://medium.com/@yaroslavvb/optimizing-deeper-networks-with-kfac-in-pytorch-4004adcba1b0), where the gradient matrix is multiplied on the left (preconditioning the columns) and on the right (preconditioning the rows).

However, KFAC requires access to individual activation and backprop values, whereas here they are able to construct the normalizing factors directly from gradients. The example below, L and R are the column and row normalizing factors respectively.

![](https://cdn-images-1.medium.com/max/1600/1*1PBcem1Qn2lA5lmqFjuvkQ.png)

The impressive result is that using this optimizer they are able to do away with batch-norm layers, and optimize to 5% error on CIFAR using just convolutional layers.

![](https://cdn-images-1.medium.com/max/1600/1*XnLlaB5UeD7sifRjEn7ObA.png)

