# Explaining Maximum Likelihood, Maximum a Posteriori, and Bayesian Parameter Estimation



You have a coin. You flip it 3 times, it comes up heads all 3 times. What’s the probability of the coin coming up heads for the next flip? This is a foundational machine learning problem of parameter estimation from data. In this case, we want to estimate the probability of heads `h`, from data `D` .

#### Maximum Likelihood

One way to do this is to find the value of the parameter `h` that maximizes the likelihood of the observed data, ie. `P(D;h)` . Here, we use `;` to denote that `h` is a parameter of the distribution `P` , meaning `h` defines `P` but `P` only specifies how likely the observed data `D` is.

![](https://cdn-images-1.medium.com/max/1600/1*9yIHxO_J0M3qlP8uYa3rsA.png)

This is a **frequentist**approach to parameter estimation known as **maximum likelihood** estimation. Under this method, we would estimate that `h = 1.0` .

But our intuition tells us this is not probably not true. We know that for most coins, there is some probability that we can get tails as well, and usually we expect something like `h = 0.5` .

#### Priors and Posteriors

How do we encode this intuition mathematically? We can specify a joint distribution over data and our parameters: `p(D, h) = p(D|h)p(h)` . We specify a **prior distribution**`p(h)` that encodes our intuition about what value `h` should have, and a conditional distribution given that value of `h` , `p(D|h)` .

How do we estimate `h` from `D` now? We need the **posterior distribution** `p(h|D)` but we only have `P(D|h)` and `p(h)` . Bayes rule to the rescue!

![](https://cdn-images-1.medium.com/max/1600/1*c-jk-eUWePUGkoxWOkYCcQ.png)

But the denominator here is a problem:

![](https://cdn-images-1.medium.com/max/1600/1*AaQCoMKsWjAHXDKf1kQuaA.png)

It is not possible to compute this integral in general. For this coin example, if you use very specific distributions known as [conjugates](https://en.wikipedia.org/wiki/Conjugate_prior) you can side-step this issue, but we’ll cover that in some other post.

#### Maximum a Posteriori

But wait, we can say something intelligent about `p(h|D)` without the normalization constant `P(D)`! Namely, the normalizing constant doesn’t change the relative magnitudes of the distribution, we can find the mode without doing the integral:

![](https://cdn-images-1.medium.com/max/1600/1*Ey3n3SifJluazEnwCXalIQ.png)

This is known as **maximum a posteriori** (MAP). There are many ways you can find this particular value of `h` , for example using [conjugate gradient descent](https://en.wikipedia.org/wiki/Conjugate_gradient_method).

#### Bayesian Parameter Estimation

With MAP, we incorporated our intuition via priors, and ignored the normalizing integral to obtained a point estimate of `h` at the mode of the posterior distribution.

But what if we instead tried to use approximation methods for the integral? If we make the usual [iid](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables) assumption, we can use the fact that future data `x` is conditionally independent of observed data `D` given `h` :

![](https://cdn-images-1.medium.com/max/1600/1*yq0LVl30nYfi2xPuFdn5MQ.png)

Rather than using the single value of `h` that corresponds to the mode of the posterior `p(h|D)` to compute `P(x|h)` , this is a even more “rigorous” way for us to take into account all possible posterior values of `h` known as **Bayesian parameter estimation.**

Note that this is really quite beautiful. There are two things about a probability distribution you may care about:

* **Inference:** Given the joint distribution with known parameters, estimate the distribution over a subset of variables by marginalizing and conditioning other variables.

* **Parameter Estimation:** Estimate the unknown parameters of a probability distribution from data.

Bayesian parameter estimation frames these two tasks as two sides of the same coin:

> Estimating parameters of a distribution over a set of variables is inference of a larger meta-distribution over the original variables and the parameters.

Of course, actually doing this requires us to compute these difficult integrals, which we’ll have to do with approximate methods such as MCMC or variational inference. I’m planning to start a series on variational inference soon, so keep an eye out for that!

