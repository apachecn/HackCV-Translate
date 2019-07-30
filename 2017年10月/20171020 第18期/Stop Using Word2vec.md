# Stop Using word2vec

原文链接：[Stop Using word2vec](https://multithreaded.stitchfix.com/blog/2017/10/18/stop-using-word2vec/?from=hackcv&hmsr=hackcv.com&utm_medium=hackcv.com&utm_source=hackcv.com)

四年前，当我开始使用word2vec时，我需要（幸运的是）十分长的时间来计算。但是由于我们对word2vec的理解有所进步，现在只需要15分钟就能在一台普通的计算机上使用标准数值库[1](https://multithreaded.stitchfix.com/blog/2017/10/18/stop-using-word2vec/?from=hackcv&hmsr=hackcv.com&utm_medium=hackcv.com&utm_source=hackcv.com#1)计算单词向量。单词向量是[awesome](http://multithreaded.stitchfix.com/blog/2015/03/11/word-is-worth-a-thousand-vectors/)但你不需要神经网络 - 并且肯定不需要深入学习 - 找到它们[2](http://multithreaded.stitchfix.com/blog/2015/03/11/word-is-worth-a-thousand-vectors/) 。因此，如果您正在使用单词向量并且没有针对最新技术或纸质出版物进行贡献，那么*停止使用word2vec。*

当我们完成后，你将衡量单词的相似性：

```
facebook ~ twitter, google, ...
```

… 和经典的单词矢量操作： `zuckerberg - facebook + microsoft ~  nadella`

…但你会主要通过计算单词和分割来做到这一点，在构造中梯度没有起任何作用！

## 秘籍

让我们假设您已经掌握了[hacker-news语料库](https://cloud.google.com/bigquery/public-data/hacker-news)并清理并标记了它（或下载了预处理版本[此处](https://zenodo.org/record/49899)）。步骤如下：

1. **Unigram概率**。`word1`和`word2`的频率是？

![Algo visualizing unigram counts](https://multithreaded.stitchfix.com/assets/posts/2017-10-18-stop-using-word2vec/fig_001.gif)

*示例*：这只是一个填充`unigram_counts`数组的简单词。然后将`unigram_counts`数组除以它们之和得到概率`p`并得到如下数字：`p('facebook')`是0.001％而`p('lambda')`是0.000001％

2. **Skipgram概率**。在`word2`附近看到`word1`的频率？这些被称为'skipgrams'，因为我们可以“跳过”`word1`和`word2`之间的几个单词。[3](https://multithreaded.stitchfix.com/blog/2017/10/18/stop-using-word2vec/?from=hackcv&hmsr=hackcv.com&utm_medium=hackcv.com&utm_source=hackcv.com#3)

![Algo visualizing skipgram counts](https://multithreaded.stitchfix.com/assets/posts/2017-10-18-stop-using-word2vec/fig_002.gif)

*示例*：这里我们计算相互附近的单词对，但不一定是在相互邻近的单词。规范化`skipgram_count`数组后，您将得到像`p('facebook'，'twitter')`这样的word-near-word概率。如果`p('facebook'，'twitter')`的值是10^-9那么在10亿个skipgram元组中你通常会看到'facebook'和'twitter'一次。对于另一个像`p('morning'，'facebook')`这样的单词对，这个分数可能会大得多，比如10 ^ -5，因为单词'morning`是一个常用词。

3. **标准化的Skipgram概率（或PMI）**。skipgram频率是否高于或低于我们对单字频率的预期？有些词是非常常见的，有些是非常罕见的，所以将skipgram频率除以两个单字组频率。如果结果大于1.0，则该skipgram发生的频率高于两个输入词的单字组概率，我们将这两个输入字称为“关联”。比率越大，关联性越强。对于小于1.0的比率，“反关联”越多。如果我们记录这个数字的日志，它被称为单词X和单词Y的[逐点互信息（PMI）](https://en.wikipedia.org/wiki/Pointwise_mutual_information)。这是一个很好理解的测量信息理论社区并代表X和Y多次“相互”（或联合）而不是独立地。

![Algo visualizing unigram counts](https://multithreaded.stitchfix.com/assets/posts/2017-10-18-stop-using-word2vec/fig_004.gif)

*示例1*：如果我们看看`facebook`和`twitter`之间的关联，我们会看到它高于1.0：`p('facebook'，'twitter')/ p('facebook')/ p(' twitter')= 1000`。所以`facebook`和`twitter`异常高度共存，我们推断它们必须是相关或类似的。请注意，除了计算和分割之外，我们还没有做任何神经网络的东西或数学计算，但我们已经可以测量两个单词的相关性。稍后我们将根据这些数据计算单词向量，并且这些向量将被约束以重现这些单词到单词的关系。 [4](https://multithreaded.stitchfix.com/blog/2017/10/18/stop-using-word2vec/?from=hackcv&hmsr=hackcv.com&utm_medium=hackcv.com&utm_source=hackcv.com#4)

*示例2*：对于`facebook`和`okra`，PMI(`p('facebook'，'okra')/ p('facebook')/ p('okra')`)接近1.0，所以`facebook`和`okra`在数据中没有依赖关系。稍后我们将形成重建和遵循这种关系的向量，因此几乎没有重叠。但当然，单词数量依旧含有噪声，这种噪声会导致字之间的虚假关联。

4. **PMI矩阵**。制作一个大矩阵，其中每一行代表单词“X”，每列代表单词“Y”，每个值是我们在步骤3中计算的“PMI”：`PMI(X，Y)= log(p(x，y) / p(x)/ p(y))`。因为我们拥有与单词一样多的行和列，所以矩阵的大小是(`n_vocabulary`，`n_vocabulary`)。因为`n_vocabulary`通常是10k-100k，这是一个有很多零的大矩阵，所以最好使用稀疏数组数据结构来表示PMI矩阵。 

![PMI矩阵](https://multithreaded.stitchfix.com/assets/posts/2017-10-18-stop-using-word2vec/fig_005.001.jpeg)

5. **SVD**。现在我们减少该矩阵的维数。这有效地将我们的巨型矩阵压缩成两个较小的矩阵。这些较小的矩阵中的每一个都形成一组具有大小的单词向量(`n_vocabulary`，`n_dim`)。[5](https://multithreaded.stitchfix.com/blog/2017/10/18/stop-using-word2vec/?from=hackcv&hmsr=hackcv.com&utm_medium=hackcv.com&utm_source=hackcv.com#5)一个矩阵的每一行代表一个单词矢量。这是任何线性代数库中的直接操作，在Python中它看起来像：`U, S, V = scipy.sparse.linalg.svds(PMI, k=256)`

![SVD of PMI Matrix](https://multithreaded.stitchfix.com/assets/posts/2017-10-18-stop-using-word2vec/fig_005.003.jpeg)

*Example* SVD是您可以在机器学习中使用的最基本和最棒的工具之一，它正在做大部分的魔法。在[Jeremy Kun](https://twitter.com/jeremyjkun?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor) 很棒的 [系列](https://jeremykun.com/2016/04/18/singular-value-decomposition-part-1-perspectives-on-linear-algebra/)。在这里，我们可以将其视为压缩原始输入矩阵（计数零）接近~100亿个条目（100k x 100k），缩减为两个矩阵，总共约5000万个条目（100k x 256 x 2），空间减少200倍。 SVD输出一个正交的空间，这是我们获得“线性规律性”的地方，也是我们能够有意义地添加和减去单词向量的地方。

6. **搜索**。 SVD输出矩阵的每一行是字向量。一旦你有了这些单词的奇异向量，你就可以搜索最接近`consolas`的标记，这是一种在编程中很流行的字体。


```
# In psuedocode: 
# Get the row vector corresponding to the word 'consolas'
vector_consolas = U['consolas']
# Get how similar it is to all other words
similarities = dot(U, vector_consolas)
# Sort by similarity, and pick the most similar
most_similar = tokens[argmax(similarities)]
most_similar
```

所以搜索`consolas`会产生`verdana`和`inconsolata`最相似 - 这是有意义的，因为这些是其他字体。搜索“功能编程”产生`FP`（首字母缩略词），`Haskell`（一种流行的函数式语言）和`OOP`（代表面向对象编程，是函数式编程的替代）。此外，添加和减去这些向量，然后搜索获得word2vec的标志性特征：在计算类比`Mark Zuckerberg - Facebook + Amazon`时，我们可以将Facebook的首席执行官与亚马逊的首席执行官联系起来，后者适当地评估`Jeff Bezos`词汇。

这是一个更直观，更容易计算跳过的大坑，除了单词计数以获得两个单词的“关联”和SVD结果，而不是理解最简单的神经网络正在做什么。

所以停止使用神经网络，但依旧有趣地构造单词向量！

### 脚注

1. 这里概述的方法并不完全等效，但它的表现与word2vec skipgram负采样大致相同 [SGNS](https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization). 事实证明，word2vec不是[与矩阵分解完全相同](http://building-babylon.net/2016/05/12/skipgram-isnt-matrix-factorisation/)， 但是对于应用目的（例如工业上），SVD技术足够好。如果你关心Word2vec之间的差异， [Glove](https://nlp.stanford.edu/projects/glove/) 和 [Swivel](https://arxiv.org/abs/1602.02215) -- 并且在工业方面，我很少这样做 -- 然后你会关心word2vec SGNS vs这个SVD公式。此外，SVD方法巧妙地融入了一系列基于计数的词袋技术，如TF-IDF，LSI和LDA：

![img](https://multithreaded.stitchfix.com/assets/posts/2017-10-18-stop-using-word2vec/fig_006.png)

TF-IDF将术语频率与文档中的术语频率进行比较，LSI使用SVD对该TF-IDF矩阵进行低秩分解。 LDA还对文档中的术语频率进行计数，但不是SVD，而是使用术语和文档向量中的稀疏性先验导致的结果。 除此之外，之前的那些使它们具有可解释性。 同样，此处简要的表述计算了术语与术语的关联（无文档），SVD将这些关联分解为低秩表示。
 [↩](https://multithreaded.stitchfix.com/blog/2017/10/18/stop-using-word2vec/?from=hackcv&hmsr=hackcv.com&utm_medium=hackcv.com&utm_source=hackcv.com#back-1)

2. 一些警告：如果您正在进行学术研究并剥离自己的嵌入系统（例如在lda2vec中） [[code\]](https://github.com/cemoody/lda2vec), [[paper\]](https://arxiv.org/abs/1605.02019) [[blog\]](http://multithreaded.stitchfix.com/blog/2016/05/27/lda2vec/#topic=38&lambda=1&term=)), 调整神经网络方法可能很有用。此外，SVD时间复杂度为O（N^3），因此它不是最好的。当N >> 100k。在这种情况下，SGD很适合在线问题。 [↩](https://multithreaded.stitchfix.com/blog/2017/10/18/stop-using-word2vec/?from=hackcv&hmsr=hackcv.com&utm_medium=hackcv.com&utm_source=hackcv.com#back-2)

3. 在这个最简单的情况下，我们不会模拟距离加权的跳跃图的效果。我们只考虑在每个单词周围的固定大小的移动窗口内的skipgrams。还要注意，我们没有规范我们的模型，这可以从平滑或形成先验中受益。惩罚复杂性特别有助于小数据情况，但根据我的经验，这对于使用这些方法在广泛的真实语料库中获得适当结果不是必要的。 [↩](https://multithreaded.stitchfix.com/blog/2017/10/18/stop-using-word2vec/?from=hackcv&hmsr=hackcv.com&utm_medium=hackcv.com&utm_source=hackcv.com#back-3)

4. 在一些方法中，特别是在截短的PMI中，一个重要的成分是阈值`k`（通常是25.0的值），它可以抑制低PMI词的影响，表面上是为了处理噪音。我将`k`常数解释为正则化的一种形式，尽管我不清楚这是一个规范的先验。你可以查阅 [gauss2vec](https://arxiv.org/abs/1412.6623) 得到更详细地证明, 并且这篇文章 [paper](https://arxiv.org/abs/1402.3722) 对参数的进行了实际的探索。 [↩](https://multithreaded.stitchfix.com/blog/2017/10/18/stop-using-word2vec/?from=hackcv&hmsr=hackcv.com&utm_medium=hackcv.com&utm_source=hackcv.com#back-4)

5. 解释是这些单词向量解释了PMI矩阵内的协方差 - 本质上是一种低级别的方式，说单词X和Y是相关联的，因为它们共同发生超出其基本速率流行度。因为缩小空间中的轴并且是正交的，这也解释了为什么人们可以找到使原始word2ve着名的“线性规则”。例如，“国王”和“女王”可能会沿着“性别”方向分开，但“伦敦”和“柏林”可能会在“首都”方向上分享一个位置。这些特征向量有效地在输入空间中找到一小组方向，仍然可以最大限度地重建原始的大输入矩阵。 [↩](https://multithreaded.stitchfix.com/blog/2017/10/18/stop-using-word2vec/?from=hackcv&hmsr=hackcv.com&utm_medium=hackcv.com&utm_source=hackcv.com#back-5)