# MapReduce的模式、算法和用例

原文链接：[MAPREDUCE PATTERNS, ALGORITHMS, AND USE CASES](https://highlyscalable.wordpress.com/2012/02/01/mapreduce-patterns/?from=hackcv&hmsr=hackcv.com&utm_medium=hackcv.com&utm_source=hackcv.com)

在这篇文章中，我总结了数种可以在网上/科研文章中找到的MapReduce的模式和算法，并系统化的解释了这些技术的不同之处。另外还提供了几个实际案例。所有相关描述和代码都使用了标准hadoop的MapReduce模型，包括Mappers, Reduces, Combiners, Partitioners,和 sorting。如下图所示。

[![map-reduce](https://highlyscalable.files.wordpress.com/2012/02/map-reduce.png?w=805)](https://highlyscalable.files.wordpress.com/2012/02/map-reduce.png)MapReduce Framework

# 基本MapReduce模式
## 计数与求和

**问题陈述：** 现有很多文档，每个文档都是有一些字段（terms）组成。现要求计算在所有文档中每个字段出现的总次数。或者，他也可以是其他任意一个关于字段的函数（这个要求可以是针对字段的操作）。例如有一个包含相应时间的日志文件，现要求计算平均相应时间。

**解决方案：**

让我们以一些简单的例子开始吧。在下面的代码中，Mapper每遇到一次特定字段就把频次记为“1”，Reducer对这些字段的list进行遍历，并把他们的频次相加：

```python
class Mapper:
    method Map(docid id, doc d)      
    for all term t in doc d do         
        Emit(term t, count 1)
class Reducer
    method Reduce(term t, counts [c1, c2,...])
    sum = 0
    for all count c in [c1, c2,...] do
        sum = sum + c
        Emit(term t, count sum)
```

这样做有一个明显的缺点就是Mapper会使用大量的虚拟计数器。Mapper可以通过先对每个文档进行频次统计从而减少使用虚拟计数器：

```python
class Mapper
    method Map(docid id, doc d)
    H = new AssociativeArray
    for all term t in doc d do
        H{t} = H{t} + 1
        for all term t in H do
            Emit(term t, count H{t})
```

为了使一个Mapper节点同时累加单个文档和全部文档，我们可以利用组合器（Combiner）：

```python
class Mapper
    method Map(docid id, doc d)
    for all term t in doc d do
        Emit(term t, count 1)
        
class Combiner
    method Combine(term t, [c1, c2,...])
    sum = 0
    for all count c in [c1, c2,...] do
        sum = sum + c
        Emit(term t, count sum)
        
class Reducer
    method Reduce(term t, counts [c1, c2,...])
    sum = 0
    for all count c in [c1, c2,...] do
        sum = sum + c
        Emit(term t, count sum)`
```

### 应用：

日志文件分析，数据查询


##

## 整理归类

**问题陈述：** 有一系列条目（items）以及关于某个字段的函数。现要求将具有相同函数值的条目保存在同一个文件下，或将全部条目中关于那个特定字段的内容组成一组，执行一些其他的的计算。最典型应用是倒排索引

**解决方案：**

解决方案很直接。Mapper计算每个条目中的函数，并将函数的值作为key和条目本身一同发出。Reducer获取以函数的key分组的所有条目并进行处理或保存。在倒排索引中，条目是一些字段（字），而函数是找到术语的文档ID。

### 应用：

倒排索引，ETL

## 过滤（“Grepping”）,解析，和校验

**问题陈述：** 有多条记录（records）。现要求找出满足特定条件的记录或将每条记录转化为另外一种形式（各条记录转换相互独立）。包括诸如文本解析和值提取，从一种格式到另一种格式的转换之类的任务都属于后者的用例。

**解决方案：** 解决方案也很直接 - Mappers对记录逐个操作并发出可接受的条目或其他转化形式。

### 应用：

日志文件分析，数据查询，ETL,数据校验

## 分布式任务执行

**问题陈述：** 有一个大型计算问题，他可以分为多个部分分别进行求解，最后合并各个计算结果以得到最终结果（分而治之）。

**解决方案：** 将问题分解为一个个固定大小的小问题，然后将这些小问题作为Mapper的输入数据。每个Mapper解决一个小问题，发出相应计算和结果。最后Reducer合并全部从Mapper中发出的结果从而得到最终结果。

### 案例研究：数字通信系统仿真

有一种类似于WiMAX的数字通信系统仿真软件，他可以通过系统模型传递一定数量的随机数据并计算传输中出错的概率。每个Mapper为样本1/N的数据进行仿真，并发送其错误率。Reducer则计算出平均错误率。

### 应用：

物理和工程仿真，数据分析，性能测试

## 排序

**问题陈述：** 有多条记录，现需要根据某些规则或以特定的顺序对这些记录排序。

**解决方案：** 简单是排序是很直接的 - Mappers只需要将条目与其排序key写成一个条目函数，然后发出全部的条目。然而，实际生活中排序常常使用一些棘手的方法，这就是为什么它被认为是MapReduce(Hadoop)的核心。尤其是使用复合keys来实现二次排序和分组是十分常见的。

在MapReduce中排序是最初是根据key对发出的键值对排序，但是现有一种技术可以通过利用Hadoop实现特定的按值排序，具体可以看下面的博客：[blog](http://www.riccomini.name/Topics/DistributedComputing/Hadoop/SortByValue/)

值得注意的是，MapReduce常用于对原始数据排序（而不是中间数据排序），而BigTable的概念则适用于维护数据的有序状态。换句话说，在插入期间对数据进行排序比对每个MapReduce查询再排序更有效。

### 应用：

ETL,数据分析

# 非基本MapReduce模型
## 迭代消息传递（图处理）

**问题陈述：** 有一个实体的关系网络。现要求根据邻居中其他实体的属性来计算每个实体的状态。这个状态可以表示到其他节点的距离，也可以表示是否存在特定属性的邻居，也可以表示邻居的密度等等。

**解决方案：** 以一组节点和每个节点的相邻节点ID的列表构成网络存储。MapReduce是以迭代的方式工作，每次迭代时每个节点都发送信息给他的邻居。每个邻居根据接受的消息更新他们的状态。迭代将会依据特定的条件中止，例如确定的最大迭代数（网络直径），或连续两次迭代结果可以忽略不计。从技术的角度来看，Mapper发送消息给每个节点，使用相邻节点ID作为每个节点的key。因此，所有消息都以分组的形式传入节点，reducer能够重现计算状态并使用新的状态重新写入节点。该算法如下：

```python
class Mapper
    method Map(id n, object N)
    Emit(id n, object N
    for all id m in N.OutgoingRelations do
        Emit(id m, message getMessage(N))
    
class Reducer
    method Reduce(id m, [s1, s2,...])
    M = null
    messages = []
    for all s in [s1, s2,...] do
        if IsObject(s) then
            M = s
        else s is a message
            messages.add(s)
            M.State = calculateState(messages)
    Emit(id m, item M)
```

值得强调的是，一个节点的状态在整个网络中传播并不稀疏，因为被这个状态所“感染”的所有节点开始“感染”其邻居。这个过程如下图所示：

[![Iterative Message Passing](https://highlyscalable.files.wordpress.com/2012/01/graph-propagation-3.png?w=805)](https://highlyscalable.files.wordpress.com/2012/01/graph-propagation-3.png)

### 案例分析：沿分类树的有效性传递

**问题陈述：** 这个问题的灵感来自现实生活中的电子商务任务。有一种类别的树从大类（如男装，女装，童装）分支到较小的类别（如男装牛仔裤或女装），最终分为不可再分类别（如男装蓝色牛仔裤）。这些不可再分类可以是有效（包含产品）或已无效（没有属于这个类别的产品）。如果一个子类中至少有一个可用的不可再分类，那么可以认为这个子类有效。我们的目的是在已知不可分类的可用性的情况下计算所有类别的有效性。

**解决方案：** 可以使用上一节中描述的框架来解决此问题。 我们定义getMessage和calculateState方法如下：

```python
class N
    State in {True = 2, False = 1, null = 0}, initialized 1 or 2 
    for end-of-line categories, 0 otherwise
        method getMessage(object N)
        return N.State
    method calculateState(state s, data [d1, d2,...])
    return max( [d1, d2,...] )
```

### 案例分析：广度优先搜索

**问题陈述：** 有一个图表，需要计算从一个源节点到图中所有其他节点的距离（跳数）。

**解决方案：** 源节点向其所有邻居发出0，邻节点把接受的信号再转发给其他节点，每次转发就对信号加1：

```python
class N
    State is distance, initialized 0 
    for source node, INFINITY 
        for all other nodes 
            method getMessage(N)
    return N.State + 1
    method calculateState(state s, data [d1, d2,...])
    min( [d1, d2,...] )
```

### 案例分析：PageRank和Mapper-Side数据聚合

这个算法由google提出，然后根据PageRank的这个方程来计算网页相关性。具体的算法是十分复杂的，但是其核心思想就是节点之间的权重传播，其中每个节点自身的权重是通过计算各接如节点权重的平均值得到的：
（PageRank的大致思想是：如果一个网页被很多其他网页链接到的话说明这个网页比较重要（权威），也就是PageRank值会相对较高，同时如果一个PageRank值很高的网页链接到一个其他的网页，那么被链接到的网页的PageRank值会相应地因此而提高）

```python
class N
    State is PageRank 
    method getMessage(object N)
    return N.State / N.OutgoingRelations.size() 
    method calculateState(state s, data [d1, d2,...])
    return ( sum([d1, d2,...]) )
```

值得一提的是，我们使用的是通用的模式，这个模式没有利用状态（state）是数值这一条件。在大多数实际案例中，基于事实我们可以在Mapper端进行聚合从而求值。下面的代码片段展示了（针对于 PageRank 算法）的优化：

```python
class Mapper
    method Initialize
    H = new AssociativeArray
    method Map(id n, object N)
    p = N.PageRank  / N.OutgoingRelations.size()
    Emit(id n, object N)
    for all id m in N.OutgoingRelations do
        H{m} = H{m} + p
        method Close
        for all id n in H do
            Emit(id n, value H{n}) 
            
class Reducer
    method Reduce(id m, [s1, s2,...])
    M = null
    p = 0
    for all s in [s1, s2,...] do
        if IsObject(s) then
            M = s
        else
            p = p + s
            M.PageRank = p
            Emit(id m, item M)
```

### 应用：

图分析，网页索引

## 数值去重 （对唯一项计数）

**问题陈述：** 有一组记录，其包含字段F和G。计算具有相同G的每个记录子集下（按G分组）的字段F的数目。

这个问题可以推广应用于分面搜索，并在分面搜索（faceted search）：

**问题陈述：** 有一组记录，每条记录包括F和任意数量的属性标签G={G1，G2，...}。计算按G标签分组的字段F的数目。
例如：

```
Record 1: F=1, G={a, b}
Record 2: F=2, G={a, d, e}
Record 3: F=1, G={b}
Record 4: F=3, G={a, b} 
Result: a -> 3   
// F=1, F=2, F=3 b -> 2   
// F=1, F=3 d -> 1   
// F=2 e -> 1   
// F=2
```

**解决方案I：**

第一种方法是分两个阶段解决问题。 在第一阶段，Mapper中每对F和G组成复合对，并发出虚拟计数器; Reducer计算每个复合对的总出现次数。该阶段的主要目标是保证F值的唯一性。在第二阶段，对按G分组，并计算每组中的条目总数。

Phase I:

```
class Mapper
    method Map(null, record [value f, categories [g1, g2,...]])
    for all category g in [g1, g2,...]
        Emit(record [g, f], count 1) 
        
class Reducer
    method Reduce(record [g, f], counts [n1, n2, ...])
    Emit(record [g, f], null )
```

Phase II:

```
class Mapper
    method Map(record [f, g], null)
    Emit(value g, count 1) 

class Reducer
    method Reduce(value g, counts [n1, n2,...])
    Emit(value g, sum( [n1, n2,...] ) )
```

**解决方案II：**

第二个解决方案只需要一个MapReduce就可以实现，但它的可扩展性不强，且适用性有限。算法很简单 - Mapper输出值和类别，Reducer从每个值的类别列表中排除重复项，并为每个类别计数加1。 最后一步是对Reducer发出的所有计数器求和。 如这种方法用于只有有限个分类，而且拥有相同F值的记录不是很多的情况。例如Web日志的处理和用户的分类 - 用户总数很高，但一个用户的事件数量有限，以此分类得到的类别也是有限的。值得注意的是，在此模式下，可以在将数据传输到Reducer之前使用Combiners从类别列表中排除重复项。

```
class Mapper
    method Map(null, record [value f, categories [g1, g2,...] )
        for all category g in [g1, g2,...]
            Emit(value f, category g) 
            
class Reducer
    method Initialize
    H = new AssociativeArray : category -> count
    method Reduce(value f, categories [g1, g2,...])
    [g1', g2',..] = ExcludeDuplicates( [g1, g2,..] )
    for all category g in [g1', g2',...]
        H{g} = H{g} + 1
        method Close
    for all category g in H do
        Emit(category g, count H{g})
```

### 应用：

日志分析，用户计数


## 互相关

**问题陈述：** 有一组条目元组。 对于每个可能的条目对，计算项两两共同出现于一个元组的数量。 如果条目总数为N，则应报告N*N值。

这个问题出现在文本分析中（例如，条目是单词和元组是句子），市场分析（购买 *此物* 的客户还可能购买 *那物* ）。 如果N*N非常小以至于这样的矩阵可以直接存储在单个机器中，那么实现起来就比较简单了。

**配对法(Pairs Approach)**

第一种方法是从Mappers中发出所有pairs和虚拟计数器，并在Reducer上对相同条目进行计数器求和。 缺点是：

- 无法有效利用Combiner，因为所有配对都都是不一样的
- 不能有效利用内存

```
class Mapper
    method Map(null, items [i1, i2,...] )
    for all item i in [i1, i2,...]
        for all item j in [i1, i2,...]
            Emit(pair [i j], count 1) 
            
class Reducer
    method Reduce(pair [i j], counts [c1, c2,...])
    s = sum([c1, c2,...])
    Emit(pair[i j], count s)
```

**条纹法(Stripes Approach)**

第二种方法是根据pairs中的第一个条目进行分组，并维护一个关联数组，数组中存储的是所有关联项的计数。 Reducer接收全部第一个条目为i的Stripes，然后合并它们，产生的结果与配对法相同。

- 中间产生的key的数量相对较少，因此该框架的排序较少。
- 可以有效利用Combiner。
- 可在内存执行。不过如果没有正确执行依然会出问题。
- 实现比较复杂。
- 一般来说，“Stripes”比“pairs”更快

```
class Mapper
    method Map(null, items [i1, i2,...] )
    for all item i in [i1, i2,...]
        H = new AssociativeArray : item -> counter
        for all item j in [i1, i2,...]
            H{j} = H{j} + 1
            Emit(item i, stripe H) 
            
class Reducer
    method Reduce(item i, stripes [H1, H2,...])
    H = new AssociativeArray : item -> counter
    H = merge-sum( [H1, H2,...] )
    for all item j in H.keys()
        Emit(pair [i j], H{j})
```

### 应用：

文本分析，市场分析

### 参考文献:

1. Lin J. Dyer C. Hirst G. [Data Intensive Processing MapReduce](http://www.amazon.com/Data-Intensive-Processing-MapReduce-Synthesis-Technologies/dp/1608453421/)

# 用MapReduce 表达关系模式

这一节我们将讲解主要关系操作并讨论如何在MapReduce中使用这些操作。

## 筛选(Selection)

```
class Mapper
    method Map(rowkey key, tuple t)
    if t satisfies the predicate
        Emit(tuple t, null)
```

## 投影(Projection)

投影只比筛选稍微复杂一点，但在这种情况下，我们应该用Reducer来消除可能的重复值。

```
class Mapper
    method Map(rowkey key, tuple t)
    tuple g = project(t)  // extract required fields to tuple g
    Emit(tuple g, null) 
    
class Reducer
    method Reduce(tuple t, array n)   // n is an array of nulls
    Emit(tuple t, null)
```

## 合并(Union)

Mappers包括两个数据集中的全部记录。Reducer是用来消除重复值

```
class Mapper
    method Map(rowkey key, tuple t)
    Emit(tuple t, null) 
    
class Reducer
    method Reduce(tuple t, array n)   // n is an array of one or two nulls
    Emit(tuple t, null)
```

## 交集(Intersection)

Mappers包含两个数据集中的交集部分。Reducer只输出出现了两次以上的记录。因为每条记录都包含主键，而他在一个数据集中只会出现一次，所有当在每个数据集中都包含这条记录时，这个方法是可行的。

```
class Mapper
    method Map(rowkey key, tuple t)
    Emit(tuple t, null) 
    
class Reducer 
    method Reduce(tuple t, array n)   // n is an array of one or two nulls
    if n.size() = 2
        Emit(tuple t, null)
```

## 差异(Difference)

假设我们有两个记录的数据集 -R 和 S。我们想找出两个数据集的不同，即计算 R-S 。Mapper将所有的元组做上标记，表明他们来自于R还是S，Reducer只输出那些存在于R中而不在S中的记录。

```
class Mapper 
    method Map(rowkey key, tuple t)
    Emit(tuple t, string t.SetName)    // t.SetName is either 'R' or 'S' 
    
class Reducer
    method Reduce(tuple t, array n) // array n can be ['R'], ['S'], ['R' 'S'], or ['S', 'R']
    if n.size() = 1 and n[1] = 'R'
        Emit(tuple t, null)
```

## 分组和聚合(GroupBy and Aggregation)

分组和聚合可以使用一个MapReduce按一下步骤完成。Mapper从元组中抽取数据，将分组聚合并发送。Reducer接收已经聚合的分组，然后计算聚合函数（再次聚合）。通常像max和sum这样的聚合函数可以通过流计算的方式来求解，因此并不需要同时保持所有值。然而，在另一些情况下可能就很需要MapReduce的两个阶段 ————看个
每个元组的Mapper提取值将分组并聚合并发出它们。 Reducer接收已经聚合的聚合值并计算聚合函数。 可以以流方式计算诸如sum或max的典型聚合函数，因此不需要同时处理所有值。看看 **（Distinct Values）** 模式的这个例子。

```
class Mapper 
    method Map(null, tuple [value GroupBy, value AggregateBy, value ...])
    Emit(value GroupBy, value AggregateBy)
    
class Reducer
    method Reduce(value GroupBy, [v1, v2,...])
    Emit(value GroupBy, aggregate( [v1, v2,...] ) )  // aggregate() : sum(), max(),...`
```

## 连接（Joining）

在MapReduce框架中完全可以实现连接，但是在面对不同的效率和数据量要求下还是存在很多的方法。 在本节中，我们将研究一些基本方法。 参考部分包含对连接技术详细研究的链接。

### 再分配连接(Reduce端连接, 合并排序式连接)

这个算法按照键值k来连接R和L这两组数据集。Mapper先遍历R和L中的所有元组并从用键值k对元组进行标记，确定其属于R还是L，然后将带有键值k元组发出。Reducer接受全部元组并将它们放入分别对应R和L的两个容器里。当两个容器满了以后，Reducers嵌套循环遍历两个容器中的数据以得到交集，最后输出的每一条结果都包含了R中的数据、L中的数据和K。这种方法有以下缺点：
该算法在一些密钥k上连接两组R和L. Mapper遍历R和L中的所有元组，从元组中提取密钥k，用标记表示元组，该标记表示该元组来自（'R'或'L'），并使用k作为密钥发出标记元组。 Reducer接收特定密钥k的所有元组并将它们放入两个桶中 - 对于R和L.当两个桶被填充时，Reducer在它们上面运行嵌套循环并发出桶的交叉连接。 每个发出的元组是串联R元组，L元组和密钥k。 这种方法有以下缺点：

- Mapper必须发送全部的数据，即使一些key只会在一个集合中出现。
- Reducer应该保持内存中一个key的全部的数据。如果数据无法存入内存（内存已满），Reducers就应该把数据转存到硬盘中。

然而，再分配连接（Repartition Join）是一种最通用的方法，特别是在其他优化技术都不适用的时候。

```
class Mapper 
    method Map(null, tuple [join_key k, value v1, value v2,...])
    Emit(join_key k, tagged_tuple [set_name tag, values [v1, v2, ...] ] ) 
    
class Reducer
    method Reduce(join_key k, tagged_tuples [t1, t2,...])
    H = new AssociativeArray : set_name -> values
    for all tagged_tuple t in [t1, t2,...]     // separate values into 2 arrays
        H{t.tag}.add(t.values)
        for all values r in H{'R'}  // produce a cross-join of the two arrays
            for all values l in H{'L'}
                Emit(null, [k r l] )`
```

### 复制链接(Map端连接, 哈希连接)

在实践中，通常将小数据集与大数据集（例如，具有日志记录列表的用户列表）连接起来。假设我们谅解两个数据集————R和L，其中R相对较小。如此，R可以分布在所有Mapper中并且每个Mapper可以加载它并通过连接键值来索引其数据。这里最常见和最有效的索引技术是哈希表。之后，Mapper遍历L，并将其与存储在哈希表中的R中的相应记录连接，。这种方法非常高效，因为不需要对L中的数据排序，也不需要通过网络传送L中的数据，但是R必须足够小到能够分发给所有的Mapper。

```
class Mapper 
    method Initialize
    H = new AssociativeArray : join_key -> tuple from R
    R = loadR()
    for all [ join_key k, tuple [r1, r2,...] ] in R
        H{k} = H{k}.append( [r1, r2,...] )
        
    method Map(join_key k, tuple l)
    for all tuple r in H{k}
        Emit(null, tuple [k r l] )
```

### 参考文献:

1. [Join Algorithms using Map/Reduce](http://www.inf.ed.ac.uk/publications/thesis/online/IM100859.pdf)
2. [Optimizing Joins in a MapReduce Environment](http://infolab.stanford.edu/~ullman/pub/join-mr.pdf)

# 机器学习和数学方面的MapReduce算法

- C. T. Chu *et al* provides an excellent description of  machine learning algorithms for MapReduce in the article [Map-Reduce for Machine Learning on Multicore](http://www.cs.stanford.edu/people/ang//papers/nips06-mapreducemulticore.pdf).
- FFT using MapReduce: <http://www.slideshare.net/hortonworks/large-scale-math-with-hadoop-mapreduce>
- MapReduce for integer factorization: <http://www.javiertordable.com/files/MapreduceForIntegerFactorization.pdf>
- Matrix multiplication with MapReduce: <http://csl.skku.edu/papers/CS-TR-2010-330.pdf> and <http://www.norstad.org/matrix-multiply/index.html>
