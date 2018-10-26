# ML算法附录：被动攻击算法

原文链接：[ML Algorithms addendum: Passive Aggressive Algorithms](https://www.bonaccorso.eu/2017/10/06/ml-algorithms-addendum-passive-aggressive-algorithms/?from=hackcv&hmsr=hackcv.com&utm_medium=hackcv.com&utm_source=hackcv.com)

被动攻击（Passive Aggressive ）算法是Crammer在al提出的一系列在线学习算法（用于分类和回归）。这个想法非常简单，并且它们的性能已被证明优于许多其他替代方法，如[Online Perceptron](https://en.wikipedia.org/wiki/Perceptron)和[MIRA](https://en.wikipedia.org/wiki/Margin-infused_relaxed_algorithm)（参见参考部分的原始论文）。

## 分类

我们假设有一个数据集：

[![IMG](https://camo.githubusercontent.com/22404c05ffb42cc77ccb998d2f6f4844dac25e24/68747470733a2f2f7777772e626f6e6163636f72736f2e65752f77702d636f6e74656e742f75706c6f6164732f323031372f31302f6d6c615f7061615f312e706e67)](https://camo.githubusercontent.com/22404c05ffb42cc77ccb998d2f6f4844dac25e24/68747470733a2f2f7777772e626f6e6163636f72736f2e65752f77702d636f6e74656e742f75706c6f6164732f323031372f31302f6d6c615f7061615f312e706e67)

选择索引t来标记时间维度。事实上，在这种情况下，样本可以无限期地继续到达。当然，如果它们来自相同的数据生成分布，算法将继续学习（可能没有大的参数修改），但如果它们是从完全不同的分布中提取的，权重将慢慢地*忘记*前一个并学习新的分布。为简单起见，我们还假设我们正在使用基于双极标签的二进制分类。

给定权重向量w，预测简单地获得为：

[![IMG](https://camo.githubusercontent.com/e9278420d1a8e6646f306c4538e9d2b17cf233ab/68747470733a2f2f7777772e626f6e6163636f72736f2e65752f77702d636f6e74656e742f75706c6f6164732f323031372f31302f6d6c615f7061615f322e706e67)](https://camo.githubusercontent.com/e9278420d1a8e6646f306c4538e9d2b17cf233ab/68747470733a2f2f7777772e626f6e6163636f72736f2e65752f77702d636f6e74656e742f75706c6f6164732f323031372f31302f6d6c615f7061615f322e706e67)



所有这些算法都基于铰链损失（Hinge loss）函数（SVM使用的相同）：

[![IMG](https://camo.githubusercontent.com/f2069a354e9e1c2c9d9814e2a4679af938b08c15/68747470733a2f2f7777772e626f6e6163636f72736f2e65752f77702d636f6e74656e742f75706c6f6164732f323031372f31302f6d6c615f7061615f31312d65313530373338343535323531372e706e67)](https://camo.githubusercontent.com/f2069a354e9e1c2c9d9814e2a4679af938b08c15/68747470733a2f2f7777772e626f6e6163636f72736f2e65752f77702d636f6e74656e742f75706c6f6164732f323031372f31302f6d6c615f7061615f31312d65313530373338343535323531372e706e67)

L的值在0（意味着完全匹配）和K之间取决于f（x（t），θ），其中K> 0（完全错误的预测）。Passive-Aggressive算法通常适用于此更新规则：

[![IMG](https://camo.githubusercontent.com/5051ec50b94a1257a9c92440365a224d3420bf9e/68747470733a2f2f7777772e626f6e6163636f72736f2e65752f77702d636f6e74656e742f75706c6f6164732f323031372f31302f6d6c615f7061615f342e706e67)](https://camo.githubusercontent.com/5051ec50b94a1257a9c92440365a224d3420bf9e/68747470733a2f2f7777772e626f6e6163636f72736f2e65752f77702d636f6e74656e742f75706c6f6164732f323031372f31302f6d6c615f7061615f342e706e67)

为了理解这个规则，让我们假设松弛变量ξ= 0（并且L约束为0）。如果呈现样本x（t），则分类器使用当前权重向量来确定符号。如果符号正确，则丢失函数为0，argmin为w（t）。这意味着当正确分类发生时，算法是**被动的**。我们现在假设发生错误分类：

[![IMG](https://camo.githubusercontent.com/7780f4e087671bb952bdc6c0e7ea879e6460b9d5/68747470733a2f2f7777772e626f6e6163636f72736f2e65752f77702d636f6e74656e742f75706c6f6164732f323031372f31302f6d6c615f7061615f352d373638783438332e706e67)](https://camo.githubusercontent.com/7780f4e087671bb952bdc6c0e7ea879e6460b9d5/68747470733a2f2f7777772e626f6e6163636f72736f2e65752f77702d636f6e74656e742f75706c6f6164732f323031372f31302f6d6c615f7061615f352d373638783438332e706e67)

角度θ> 90°，因此，点积为负，样品分类为-1，但其标签为+1。在这种情况下，更新规则变得非常**激进**，因为它寻找一个必须尽可能接近的新w（否则现有知识会立即丢失），但它必须满足L = 0（换句话说，分类必须正确）。

松弛变量的引入允许具有软边缘（如在SVM中）和由参数C控制的容差度。特别地，损失函数必须是L <=ξ，允许更大的误差。较高的C值产生较强的侵略性（随之而来的是存在噪声时不稳定的较高风险），而较低的值允许更好的适应性。实际上，这种算法在线工作时必须处理存在噪声样本（标签错误）。良好的稳健性是必要的，否则，太快的变化会产生更高的错误分类率。

解决了两个更新条件后，我们得到了封闭形式的更新规则：

[![IMG](https://camo.githubusercontent.com/b691b558f459b2e6503250f26681897b8be61b9e/68747470733a2f2f7777772e626f6e6163636f72736f2e65752f77702d636f6e74656e742f75706c6f6164732f323031372f31302f6d6c615f7061615f362e706e67)](https://camo.githubusercontent.com/b691b558f459b2e6503250f26681897b8be61b9e/68747470733a2f2f7777772e626f6e6163636f72736f2e65752f77702d636f6e74656e742f75706c6f6164732f323031372f31302f6d6c615f7061615f362e706e67)

该规则证实了我们的期望：权重向量用一个因子更新，该因子的符号由y（t）确定，其大小与误差成正比。注意，如果没有错误分类，则分子变为0，因此w（t + 1）= w（t），而在错误分类的情况下，w将朝x（t）旋转并且以L <=ξ的损失停止。在下图中，效果已标记为显示旋转，但是，它通常尽可能地最小：

[![IMG](https://camo.githubusercontent.com/9683f6a00e42d5fc61ed2ecff2914c5b469fdff5/68747470733a2f2f7777772e626f6e6163636f72736f2e65752f77702d636f6e74656e742f75706c6f6164732f323031372f31302f6d6c615f7061615f372d373638783438372e706e67)](https://camo.githubusercontent.com/9683f6a00e42d5fc61ed2ecff2914c5b469fdff5/68747470733a2f2f7777772e626f6e6163636f72736f2e65752f77702d636f6e74656e742f75706c6f6164732f323031372f31302f6d6c615f7061615f372d373638783438372e706e67)

旋转后，θ<90°，点积变为负值，因此样品被正确分类为+1。Scikit-Learn实现了Passive Aggressive算法，但我更喜欢实现代码，只是为了表明它们有多简单。在下一个片段（也在此[GIST中](https://gist.github.com/giuseppebonaccorso/d700d7bd48b1865990d2f226759686b1)可用）中，我首先创建一个数据集，然后使用Logistic回归计算得分，最后应用PA并测量测试集上的最终得分：

```python
import numpy as np


from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# Set random seed (for reproducibility)
np.random.seed(1000)


nb_samples = 5000
nb_features = 4


# Create the dataset
X, Y = make_classification(n_samples=nb_samples, 
                           n_features=nb_features, 
                           n_informative=nb_features - 2, 
                           n_redundant=0, 
                           n_repeated=0, 
                           n_classes=2, 
                           n_clusters_per_class=2)


# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.35, random_state=1000)


# Perform a logistic regression
lr = LogisticRegression()
lr.fit(X_train, Y_train)
print('Logistic Regression score: {}'.format(lr.score(X_test, Y_test)))


# Set the y=0 labels to -1
Y_train[Y_train==0] = -1
Y_test[Y_test==0] = -1


C = 0.01
w = np.zeros((nb_features, 1))


# Implement a Passive Aggressive Classification
for i in range(X_train.shape[0]):
    xi = X_train[i].reshape((nb_features, 1))
    
    loss = max(0, 1 - (Y_train[i] * np.dot(w.T, xi)))
    tau = loss / (np.power(np.linalg.norm(xi, ord=2), 2) + (1 / (2*C)))
    
    coeff = tau * Y_train[i]
    w += coeff * xi
    
# Compute accuracy
Y_pred = np.sign(np.dot(w.T, X_test.T))
c = np.count_nonzero(Y_pred - Y_test)


print('PA accuracy: {}'.format(1 - float(c) / X_test.shape[0]))
```



## 回归

对于回归，算法非常相似，但它现在基于稍微不同的铰链损失函数（称为ε不敏感）：

[![IMG](https://camo.githubusercontent.com/a4d111f9bdc1021f51aed7cf37af7789dedbbadf/68747470733a2f2f7777772e626f6e6163636f72736f2e65752f77702d636f6e74656e742f75706c6f6164732f323031372f31302f6d6c615f7061615f382e706e67)](https://camo.githubusercontent.com/a4d111f9bdc1021f51aed7cf37af7789dedbbadf/68747470733a2f2f7777772e626f6e6163636f72736f2e65752f77702d636f6e74656e742f75706c6f6164732f323031372f31302f6d6c615f7061615f382e706e67)

参数ε确定预测误差的容差。更新条件与分类问题相同，生成的更新规则为：

[![IMG](https://camo.githubusercontent.com/88188a029ae2cf2dc8640c0d363a64497e3b42d1/68747470733a2f2f7777772e626f6e6163636f72736f2e65752f77702d636f6e74656e742f75706c6f6164732f323031372f31302f6d6c615f7061615f392d373638783134312e706e67)](https://camo.githubusercontent.com/88188a029ae2cf2dc8640c0d363a64497e3b42d1/68747470733a2f2f7777772e626f6e6163636f72736f2e65752f77702d636f6e74656e742f75706c6f6164732f323031372f31302f6d6c615f7061615f392d373638783134312e706e67)

就像分类一样，Scikit-Learn也实现了回归，但是，在下一个片段（也可以在这个[GIST中使用](https://gist.github.com/giuseppebonaccorso/d459e15308b4faeb3a63bbbf8a6c9462)）中，有一个自定义实现：

```python
import matplotlib.pyplot as plt
import numpy as np


from sklearn.datasets import make_regression


# Set random seed (for reproducibility)
np.random.seed(1000)


nb_samples = 500
nb_features = 4


# Create the dataset
X, Y = make_regression(n_samples=nb_samples, 
                       n_features=nb_features)


# Implement a Passive Aggressive Regression
C = 0.01
eps = 0.1
w = np.zeros((X.shape[1], 1))
errors = []


for i in range(X.shape[0]):
    xi = X[i].reshape((X.shape[1], 1))
    yi = np.dot(w.T, xi)
    
    loss = max(0, np.abs(yi - Y[i]) - eps)
    
    tau = loss / (np.power(np.linalg.norm(xi, ord=2), 2) + (1 / (2*C)))
    
    coeff = tau * np.sign(Y[i] - yi)
    errors.append(np.abs(Y[i] - yi)[0, 0])
    
    w += coeff * xi
    
# Show the error plot
fig, ax = plt.subplots(figsize=(16, 8))


ax.plot(errors)
ax.set_xlabel('Time')
ax.set_ylabel('Error')
ax.set_title('Passive Aggressive Regression Absolute Error')
ax.grid()


plt.show()
```

错误图如下图所示：

[![IMG](https://camo.githubusercontent.com/ac2c15be938dd3d51019f96a737fb8464b92e1a6/68747470733a2f2f7777772e626f6e6163636f72736f2e65752f77702d636f6e74656e742f75706c6f6164732f323031372f31302f6d6c615f7061615f31302d373638783339382e706e67)](https://camo.githubusercontent.com/ac2c15be938dd3d51019f96a737fb8464b92e1a6/68747470733a2f2f7777772e626f6e6163636f72736f2e65752f77702d636f6e74656e742f75706c6f6164732f323031372f31302f6d6c615f7061615f31302d373638783339382e706e67)

可以通过选择更好的C和ε值来控制回归的质量（特别是，当误差高时的瞬态周期的长度）。特别是，我建议检查C的不同范围中心（100,10,1,0.1,0.01），以确定是否更高的侵略性。

参考文献：

- Crammer K.，Dekel O.，Keshet J.，Shalev-Shwartz S.，Singer Y.，[Online Passive-Aggressive Algorithms](http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf)，Journal of Machine Learning Research 7（2006）551-585

也可以看看：

<https://www.bonaccorso.eu/2017/08/29/ml-algorithms-addendum-instance-based-learning/>