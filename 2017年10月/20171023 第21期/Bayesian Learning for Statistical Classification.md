# Bayesian Learning for Statistical Classification

原文链接：[Bayesian Learning for Statistical Classification](https://blog.statsbot.co/bayesian-learning-for-statistical-classification-f2362d620428?from=hackcv&hmsr=hackcv.com&utm_medium=hackcv.com&utm_source=hackcv.com)

## A Bayesian introduction to statistical classification problems

*Probability theory is one of the most fundamental tools we have for describing the universe. It is especially relevant to statistical classification and can be used to derive a multitude of important results and to inform our understanding.*

*The* [*Statsbot*](http://statsbot.co/?utm_source=blog&utm_medium=article&utm_campaign=bayesian_learning) *team asked Peter Mills to help you understand probability theory and Bayes’ theorem and how they apply to statistical classification. It will allow you to derive non-obvious results that can vastly improve and simplify your classification models.*



![img](https://cdn-images-1.medium.com/max/2000/0*-fuTiaivZo-YDfu2.)

This introduction to Bayesian learning for statistical classification will provide several examples of the use of Bayes’ theorem and probability in statistical classification. It will also go beyond basic probability to cover other important areas of the field, including calibration and validation.

> *Note that this article, while intended for beginners, nonetheless assumes knowledge of first- and some second-year university-level mathematics, especially linear algebra, but also some single and multi-variate calculus. If the equations seem confusing at times, try to focus instead on solving real problems.*

You will learn a whole lot more about probability and statistical classification by working through some examples than just by reading about it or browsing equations. For this reason, we have prepared a set of problems at the end of the article.

### Review of basic probability

Suppose we roll a die. There will be six possibilities, each of which (in a fairly loaded die) will have a probability of 1/6. We can write this:



![img](https://cdn-images-1.medium.com/max/750/0*V56SwYrOQqjCLGXK.)

where *i* is the number on the top side of the die. Since at least one side will have to come up, we can also write:



![img](https://cdn-images-1.medium.com/max/750/0*BCCAd_9goIZjXPsI.)

(1)

where *n*=6 is the total number of possibilities.

Now suppose we roll two dice. The joint probability of getting one of 36 pairs of numbers is given:



![img](https://cdn-images-1.medium.com/max/750/0*RSmw76M3Fkkrkr1I.)

where *i* is the number on the first die and *j* that on the second.

If we ignore the number on the second die, the probability of getting a certain number (a 6, say) on the first die is given:



![img](https://cdn-images-1.medium.com/max/1000/0*UJnfvNjFTjPWJwJD.)

(2)

This is known as the *prior probability*.

Here’s where things start getting more complicated. What is the probability of getting a certain number on one die given that a certain number on the other die has come up? In this case, the two events are uncorrelated, thus the value, at 1/6, will always be the same, but this need not be the case.

Consider a game of Blackjack. What is the probability that the next card drawn is worth ten points (is a ten or a face card) given that the previous card was also worth ten points?

Suppose there were 7 ten-point cards out of a deck of 34 remaining before the last draw. Now the probabilities are different depending upon the outcome of the previous event. If the previous card was worth ten, there is a 6/33=2/11 chance of getting a card worth ten, otherwise the probability is 7/33.

Since the probability that the previous card was worth ten is 7/34, the joint probability, or the probability of both events occurring is:



![img](https://cdn-images-1.medium.com/max/1000/1*cwQtStIQuK2g8uW71N7ZPw.gif)

where *Pi* is the probability that the previous card was worth ten and *P(j | i)* is the conditional probability that the next card will be worth ten, given that the previous card was also worth ten.

With prior, joint, and conditional probabilities defined, we are set to write down Bayes’ theorem.

*Note that these definitions are symmetric in i and j, thus:*



![img](https://cdn-images-1.medium.com/max/1000/0*Zc-hC_xQ2ADD7tgQ.)

(3)

which is the symmetric form of Bayes’ Theorem.

### Continuous probabilities

The extension to continuous probabilities or probability densities is straightforward. Imagine we have a continuous random variable, *x*, governed by a probability distribution, *P*(*x*). The probability that *x* takes on a value between *x*ₒ and *x*ₒ+d*x* is given:



![img](https://cdn-images-1.medium.com/max/1000/0*O6gp9mk0whRBd_GC.)

(4)

When working with continuous random variables, summations become integrals so that Equation (2) becomes:



![img](https://cdn-images-1.medium.com/max/1000/1*Df3Lx7yq70igJhOtvVHzSA.gif)

(5)

where *P*(*x*, *y*) is the joint probability of both *x* and *y* and the integral is over all of *x*.

In statistical classification, we are dealing with probabilities having a very specific form. One of the variables is scalar and discrete, while the other is vector and continuous:



![img](https://cdn-images-1.medium.com/max/750/0*lUjNbGGm6QjaIufF.)

(6)

Where *i* is the *class* or *class label*and **x** is a vector of *attributes* or *features*.

Typically, the goal of Bayesian-based statistical classification is to estimate either the joint probability, *P*(**x**, *i*), or the conditional probability, *P*(*i* | **x**). Classifications are normally done on the basis of *maximum likelihood*:



![img](https://cdn-images-1.medium.com/max/750/0*UNqIFPn-qEcvFCQt.)

(7)

where *c* is the most likely estimate for the class, that is the index of the largest value of the conditional probability.

> *Note that because* P*(***x***) is the same for a given test point, using either the joint or the conditional probability will produce the same result. The conditional probabilities of the feature space,* P*(***x** *|* i*), are important also as these describe the distributions of each isolated class: that is, if you remove all other class labels leaving only* i*, this is the distribution you are left with.*

We can use the definition of probability density in (4) to derive one of the oldest and most sophisticated statistical classification techniques by simply removing the limit sign. Consider picking a radius from the *test point*, **x**, then counting the number of training samples of one class or another within that distance.

The problem with this is that sometimes the enclosed volume will contain no samples, while other times it may contain a great many. So rather than distance, we instead fix the number of samples and implicitly choose the distance on this basis. This is what is known as a *k-nearest-neighbors* (KNN) classifier, where *k* is the number of neighboring samples used in each classification.

### Binary classifiers

A binary classifier is special because you can, in many cases, draw a single hyperplane in the feature space that separates the two classes. A hyperplane is a subspace having dimension one less than the embedding dimension. So for a two-dimensional feature space the boundary would be a line, while in three-dimensions, a plane.

Most binary classifiers return not an integer having only two values, but a continuous, *decision function*. A convenient form of the decision function would be the difference in conditional probabilities:



![img](https://cdn-images-1.medium.com/max/750/0*ZVuFhaG70Zxf52jP.)

(8)

where, for convenience, we have chosen the class values as -1 and +1.

Unfortunately, most statistical classifiers do not return a decision function that estimates this quantity well, so a significant chunk of this article will be dedicated towards describing methods of calibrating it so that it does.

Consider a pair of equally-sized, one-dimensional Gaussian functions of equal width, *h*, and spaced an equal distance, *b*, from the origin. The difference in

conditional probabilities is given:



![img](https://cdn-images-1.medium.com/max/1000/0*xsgapfsWISh985Wc.)

which, with some manipulation, works out to:



![img](https://cdn-images-1.medium.com/max/1000/0*ia-KlCXEqb_kIorT.)

(9)

In other words, for a pair of equal-size Gaussians, the decision function in one dimension is a hyperbolic tangent.

This may seem like a trivial example, however, the `tanh` function is found throughout the field of machine learning. In statistical classification, it is often used to correct the decision function to better estimate the conditional probabilities.

This is applied in the [*LIBSVM*](http://www.csie.ntu.edu.tw/~cjlin/libsvm/) library, for instance, as well as in my own machine learning library, [*libAGF*](https://github.com/Peteysoft/libmsci). The example illustrates why: the difference in conditional probabilities, *R*, is, more often than not, sigmoidal close to the class borders.

Consider *logistic regression*. In logistic regression, we use the following decision function:





(10)

Where **v** is a vector and *a* is a constant.

The function parameters are fitting by *minimizing a cost function*, for instance a *least squares*:



![img](https://cdn-images-1.medium.com/max/1000/0*1VObaJ7l4U8xO4KR.)

(11)

To fit or “train” the thing, we need some *training data*. This comprises a set of ordered pairs of a vector in the feature space mapping onto its corresponding class value: {**x**ᵢ : *y*ᵢ}. Here, *y*ᵢ takes on one of two values: -1 or +1, that is, *y*ᵢ ∈ {-1, +1}.

The training data represents the “ground truth’’ and could be obtained in a variety of ways. Consider a land classification problem: a satellite instrument measures upwelling electromagnetic radiation in several bands and on this basis, we are interested in classifying the surface type, whether field, forest, city, or water, for instance.

The data could have been painstakingly measured by hand: an aircraft carried a terrestrial version of the instrument aloft and measured radiances, while observers in the craft noted the type of land they were flying over.



![img](https://cdn-images-1.medium.com/max/1000/0*VDjBovxnDbH64umq.)

It could have been modelled: perhaps we have an algorithm that we trust that returns modelled radiances depending on different parameters describing the land surface. In this case, the resulting training data is potentially infinite, although not necessarily all that accurate.

Or perhaps it was measured by the actual instrument but classified by hand. You have a simple app that brings up an image and each pixel can be classified with a mouse click on the basis of color.

Equations (10) and (11) provide a succinct illustration of the entire process of statistical classification. There is a training phase, given by (11), in which a model is derived. In this case the model is defined by a small set of function parameters which makes this an exercise in *parametric statistics*.

Contrast this with a *non-parametric* statistical model, such as KNN, which uses all of the training data for each classification. For the logistic classifier, the fitting will be nonlinear, another common technique in machine learning.

Nonlinear optimization is normally performed with an iterative, numerical algorithm, assuming the problem cannot be reduced to a closed-form, analytic solution. It is in itself a broad and diverse field, so we won’t go into the exact details. See the problem set for more info.

The model is then applied to classify a series of test points using Equation (10).

### Calibration

The nice thing about using a continuous decision function for binary classification is that it allows some degree of calibration. Consider the following classifier:



![img](https://cdn-images-1.medium.com/max/1000/0*4KBNuCKecghWSEcW.)

Varying the classification threshold, *f*₀, allows us to adjust the sensitivity of the classifier.

This is particularly useful in medical diagnostics.

*Note that the case f=f*₀ *is left undefined. To remove bias, a random value should be returned when this occurs in numerical computations.*

Suppose the classifier is trained using data with a prior class distribution of *P*’(*i*) while the population distribution is actually *P*(*i*). Assuming that *f*accurately estimates *R*, we want to find the value for *f*₀ such that the sample statistics are corrected to those of the population:



![img](https://cdn-images-1.medium.com/max/1000/0*40fsJSET6DuN02Jr.)

(12)

To make this more detailed, consider the confusion matrix. The element of the confusion matrix in the *i*th row and *j*th column tells us: for all of the test data, how many test samples had the *i*th class but the classifier returned the *j*th class?

By dividing by the number of test samples, the confusion matrix can be expressed as an approximate joint probability. Consider the confusion matrix for a binary classifier:



![img](https://cdn-images-1.medium.com/max/1000/0*LdTa96LTbvcE6AuR.)

where:

- *n*t=*n*TN+*n*FP+*n*FN+*n*TP is the total number of test samples
- *n*TN is the number of true negatives
- *n*FP is the number of false positives
- *n*FN is the number of false negatives
- *n*TP is the number of true positives

A perfect classifier would return a diagonal matrix: an element is non-zero only when *i*=*j*.



![img](https://cdn-images-1.medium.com/max/1000/0*-UxBjRMgU991bqKn.)

From these five parameters, you can write down all possible skill scores for a simple binary classifier. The receiver operating characteristic (ROC) curve is produced by plotting two such skill scores against one another while varying the classification threshold. These are the hit rate:



![img](https://cdn-images-1.medium.com/max/1000/0*Gg7m2gZAD80mvse-.)

and the false alarm rate:



![img](https://cdn-images-1.medium.com/max/1000/0*DbuKd_BRYq8pNsE2.)

The figure plots the ROC curve for the one-dimensional logistic classifier in (9) for *h*=1 and for different values of *b*. The classifier is assumed to be a perfect estimator for the conditional probabilities.

A more sophisticated calibration exercise would transform the decision function such that it accurately represents the difference in conditional probabilities. Consider the following equation, derived strictly from the background material presented in the first two sections:



![img](https://cdn-images-1.medium.com/max/1000/0*THHc-KiY5vEn6j6g.)

(13)

Where δ is the Dirac delta function:



![img](https://cdn-images-1.medium.com/max/1000/0*e7FoZFDYR13ck-bQ.)

A well-calibrated estimator for the conditional probabilities should obey this equation.

### Validation

Once we have derived a statistical classifier, we need to validate it on some *test data*. This data should be different from that used to train the classifier, otherwise skill scores will be unduly optimistic. This is known as *cross-validation*.

The confusion matrix expresses everything about the accuracy of a discrete classifier over a given database and you can use it to compose any possible skill score. Here, we are going to cover two that are rarely seen in the literature, but are nonetheless important for reasons that will become clear.

The most basic skill score is accuracy:



![img](https://cdn-images-1.medium.com/max/1000/0*Mj4huUbGPnOMnPv3.)

With a maximum-likelihood classification algorithm, accuracy will be maximized. Accuracy, however, has several limitations which can be mitigated by using the following, alternative measures.

The first is the *uncertainty coefficient*. This measure is based on Shannon’s channel capacity and requires, first, a definition of the information entropy. For a discrete probability, this is:



![img](https://cdn-images-1.medium.com/max/1000/0*dEqkVkxBHtEidRlk.)

and tells us how many bits we need to represent *i,* given that its prior distribution is *P*ᵢ. The measure can be extended to multivariate distributions. The conditional entropy is given:



![img](https://cdn-images-1.medium.com/max/1000/1*kErMK_mStZZAAJM1dCJeNg.jpeg)

Once we have these two definitions out of the way, we can write down the uncertainty coefficient:



![img](https://cdn-images-1.medium.com/max/1000/0*IxugDlMyF94bwvxk.)

(14)

which tells us how many bits of information a single classification result in *j*gives us of the true class value, *i*. This makes it a good skill score since the lowest possible value is 0, meaning the classifier provides, on average, no information on the true class values, while the highest is 1, meaning the classifier provides full information.

For binary classifiers, I also recommend the *Pearson correlation coefficient*:



![img](https://cdn-images-1.medium.com/max/1250/0*bRX-sPaanw7iM6c0.)

(15)

Finally, for binary classifiers that return a continuum decision function rather than a discrete, binary value, we can use the ROC curve to measure the average skill for all possible thresholds by calculating the area under the curve.

For a perfect discriminator, the ROC curve will follow the unit square, rising to *H*=1 at *F*=0 and staying there for the duration, thus the area will be 1. An area of 0 is also a perfect classifier, but the sign is reversed, while a classifier with no discrimination value will follow the diagonal with an area of 0.5.

*Note for instance, how the area under the example curves gets larger as the separation between the classes increases.*

### Multi-class classification

We have spent a considerable amount of time discussing binary classifiers. Assuming the only suitable statistical classifier we have at our disposal is a binary classifier, how do we generalize it to classification problems with more than two classes, that is, multi-class classifiers? We can use probability theory to derive an answer.

Suppose we design a set of binary classifiers by multiply partitioning the classes into two sets. A coding matrix, *A*, describes how this partitioning is done: the *i*th row of the matrix describes the partitioning of the *i*th binary classifier with a -1/+1 in the *j*th column, meaning that the *j*th class label was transformed to a -1/+1 for the training and a 0, meaning it was excluded entirely.

The conditional probabilities of the multi-class problem are related to those of the binary classifiers as follows:



![img](https://cdn-images-1.medium.com/max/1000/0*B2GmII8WpDgCTsYP.)

(16)

With some rearrangement, we can transform this into a linear system:



![img](https://cdn-images-1.medium.com/max/1000/0*1GUitiM8eTOFnsWc.)

(17)

where *Ri* is the difference in conditional probabilities for the *i*th binary classifier.

As an example, consider the “one-versus-the-rest” approach to multi-class classification. Here, we compare each class with all the others. The coding matrix is given (similar to the Dirac delta function):



![img](https://cdn-images-1.medium.com/max/1000/0*hVr9y9K9CAtR9kUy.)

(18)

The preceding assumes that the conditional probabilities for the binary classifiers are estimated correctly. Otherwise we need to constrain the resulting multi-class probabilities. Neglecting the second argument, a conditional probability has the same properties as a univariate probability. First, they all ought to sum to one:



![img](https://cdn-images-1.medium.com/max/1000/0*dUNmtZum4tHgDS4F.)

(19)

Second, they should all be positive:



![img](https://cdn-images-1.medium.com/max/1000/0*llDjeIW_L-K3ZT3M.)

The normalization constraint in (18), being an equality constraint, is the easiest to enforce.

One way is to introduce a “slack” variable:



![img](https://cdn-images-1.medium.com/max/750/0*uM3B5omjnvxKNB7L.)

(20)

where *Q***p** = **b** is the linear system for the unconstrained problem and 𝜆 is the slack variable.

For the “one-versus-one’’ method of multi-class classification, where we compare each class with each of the others in turn, this is all we need. It turns out that once the normalization constraint is enforced, all the others fall into place and the solution has only positive elements.

> *Note that because the system of equations is overdetermined, it will need to be solved as a least-squares problem and there is one other caveat: the normalization must be done separately from the least squares minimization.*

In other words, we form the normal equations from (17) first, then plug these into (20). To learn about the normal equations, please see my upcoming article, “Mastering Singular Value Decomposition.”

### Problems

This list of problems is provided to help you with Bayesian learning and probability theory and derive useful formulas related to statistical classification. They will also get you thinking about some of the fundamental issues in the field.

1. Why does the fitting for the logistic classifier in (10) have to be nonlinear? What advantage does this have?
2. Do some research online to find nonlinear optimization algorithms to fit the logistic classifier.
3. Derive Equation (12). (It’s surprisingly difficult.) How important do you think it is, on average, to correct for the class distribution? Explain.
4. How would you calculate the ROC curves shown in the figure? Fill in the missing steps going from Equation (8) to (9) and then to calculation of the ROC curves.
5. Derive Equation (13).
6. List the advantages of the uncertainty coefficient and the correlation coefficient (for binary classifiers) as a measure of classification skill. What happens when a) the class labels are rearranged and b) the distribution of the class labels is changed in the test data? How does this affect the outcome?
7. From the general formula for Pearson correlation, derive Equation (15). Note: this is not trivial.
8. Correlation is normally not appropriate for multi-class classification problems. Why not? What types of problems would be the exceptions?
9. Derive Equation (17) from Equation (16). Hint: what property of P(i|x̄) do you need to complete the derivation?
10. The one-versus-the-rest coding matrix in (18) can use a simplified version of Equation (17). Explain.
11. Write down the coding matrix for the one-versus-one multi-class classifier.
12. Find some statistical classification data online or create some on your own, e.g., by classifying pixels in an image. Perform statistical classifications by fitting a multi-dimensional Gaussian to each of the classes:



![img](https://cdn-images-1.medium.com/max/1000/1*C6aPZ22_rFNDKszgcLaNMg.gif)

Where Σ is the covariance matrix, **𝜇** is the arithmetic mean, and *D* is the dimension of the features data. Measure the accuracy of your results. Don’t forget to divide the data into a test set and a training set.

### Conclusion

I hope you got the idea of bayesian learning for statistical classification. Mathematical models are not closed systems. They can be expanded, re-purposed, and recombined.

The applications of probability and Bayes’ theorem and the problems we can put them to are limited only by the imagination. Here we have presented just a few of the ways to use these tools to help gain a foothold in the complex world of computational learning algorithms.