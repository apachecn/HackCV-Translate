---
title: Numpy的70个操作
date: 2018‎-03‎-0‎2‎ 20:37:39
tags: [numpy, python, 数据分析]
categories: [数据]
---
这篇文章主要介绍了Numpy从入门到进阶的操作, 翻译文章，便于自己使用时查阅

<!-- more -->

[原文链接](https://www.machinelearningplus.com/101-numpy-exercises-python/)

## 1-10

```python
import numpy as np

# 查看numpy的版本
>>>np.__version__
'1.13.3

# 创建一个一维数据
>>>arr = np.arange(10)
>>>arr
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# 创建一个布尔型数组
# 方法一：
>>>np.full((3, 3), True, dtype=bool)
# 方法二：
>>>np.ones((3, 3), dtype=bool)
array([[ True,  True,  True],
       [ True,  True,  True],
       [ True,  True,  True]], dtype=bool)

# 提取数组中为奇数的数字
>>>arr = np.arange(10) # 原来的数组
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>>arr[arr % 2 == 1]
array([1, 3, 5, 7, 9])

# 替换数组中为奇数的数字为-1
>>>arr = np.arange(10) # 原来的数组
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# 方法一：
>>>arr[arr % 2 == 1] = -1
>>>arr
# 方法二, 更加安全的操作方法（这里算新的一个操作）：
>>>out = np.where(arr % 2 == 1, -1, arr)
>>>out
array([ 0, -1,  2, -1,  4, -1,  6, -1,  8, -1])

# 数组维度变换
>>>arr = np.arange(10) # 原来的数组
>>>arr.reshape(2, -1)
>>>arr
array([[0, 1, 2, 3, 4],
       [5, 6, 7, 8, 9]])

# 纵向堆叠两个数组
>>>a = np.arange(10).reshape(2, -1) # 原来的数组a
>>>b = np.repeat(1, 10).reshape(2, -1) # 原来的数组b
# 方法一：
>>>np.concatenate([a, b], axis=0)
# 方法二:
>>>np.vstack([a, b])
# 方法三：
>>>np.r_[a, b]
# 三种方法的输出
array([[0, 1, 2, 3, 4],
       [5, 6, 7, 8, 9],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1]])

# 横向堆叠两个数组
>>>a = np.arange(10).reshape(2, -1) # 原来的数组a
>>>b = np.repeat(1, 10).reshape(2, -1) # 原来的数组b
# 方法一：
>>>np.concatenate([a, b], axis=1)
# 方法二:
>>>np.hstack([a, b])
# 方法三：
>>>np.c_[a, b]
# 三种方法的输出
array([[0, 1, 2, 3, 4, 1, 1, 1, 1, 1],
       [5, 6, 7, 8, 9, 1, 1, 1, 1, 1]])

# 根据原有的数组，自由拓展
>>>a = np.array([1, 2, 3]) # 原来的数组
>>>np.r_[np.repeat(a, 3), np.tile(a, 3)]
array([1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])
```

## 11-20

```python
import numpy as np

# 获取两个数组中公共的元素
>>>a = np.array([1, 2, 3, 2, 3, 4, 3, 4, 5, 6])  # 原来的数组a
>>>b = np.array([7, 2, 10, 2, 7, 4, 9, 4, 9, 8]) # 原来的数组b
>>>np.intersect1d(a, b)
array([2, 4])

# 在a数组中移去a、b数组中公共的元素
>>>a = np.array([1, 2, 3, 4, 5])	# 原来的数组a
>>>b = np.array([5, 6, 7, 8, 9])	# 原来的数组b
>>>np.setdiff1d(a, b)
array([1, 2, 3, 4])

# 获取两个数组中公共元素（索引相同）的索引
>>>a = np.array([1, 2, 3, 2, 3, 4, 3, 4, 5, 6])  # 原来的数组a
>>>b = np.array([7, 2, 10, 2, 7, 4, 9, 4, 9, 8]) # 原来的数组b
>>>np.where(a == b)
(array([1, 3, 5, 7], dtype=int64),)

# 获取指定数值区间的元素
>>>a = np.arange(15) # 原来的数组
# 方法一
>>>index = np.where((a >= 5) & (a <= 10))
>>>a[index]
# 方法二
>>>index = np.where(np.logical_and(a>=5, a<=10))
>>>a[index]
# 方法三
>>>a[(a>= 5) & (a <= 10)]
array([ 5,  6,  7,  8,  9, 10])

# numpy运用函数
>>>def maxx(x, y):		# 需要使用的函数
>>>    if x >= y:
>>>        return x
>>>    else:
>>>        return y
>>>
>>>a = np.array([5, 7, 9, 8, 6, 4, 5]) # 原来的数组a
>>>b = np.array([6, 3, 4, 8, 9, 7, 1]) # 原来的数组b
>>>pair_max = np.vectorize(maxx, dtypes=[float])
>>>pair_max(a, b)
array([ 6.,  7.,  9.,  8.,  9.,  7.,  5.])

# 二维数组交换两行
>>>arr = np.arange(9).reshape(3,3)	# 原来的数组
>>>arr[:, [1, 0, 2]]
array([[1, 0, 2],
       [4, 3, 5],
       [7, 6, 8]])

# 二维数组交换两列
>>>arr = np.arange(9).reshape(3,3)	# 原来的数组
>>>arr[[1, 0, 2], :]
array([[3, 4, 5],
       [0, 1, 2],
       [6, 7, 8]])

# 二维数组每行逆序输出
>>>arr = np.arange(9).reshape(3,3)  # 原来的数组
>>>arr[::-1]
array([[6, 7, 8],
       [3, 4, 5],
       [0, 1, 2]])

# 二维数组每列逆序输出
>>>arr = np.arange(9).reshape(3,3)  # 原来的数组
>>>arr[::-1]
array([[2, 1, 0],
       [5, 4, 3],
       [8, 7, 6]])

# 创建一个二维数组，数值区间为5-10
# 方法一
>>>rand_arr = np.random.randint(low=5, high=10, size=(5,3)) + np.random.random((5,3))
# 方法二
>>>rand_arr = np.random.uniform(5,10, size=(5,3))
>>>rand_arr
array([[ 6.75709583,  7.4662972 ,  7.2395679 ],
       [ 9.7802112 ,  6.70830431,  6.76776651],
       [ 6.35564151,  7.8632512 ,  8.53040631],
       [ 7.99457763,  7.85526309,  8.96584465],
       [ 6.18621991,  8.14214732,  7.31162037]])
```

## 21-30

```python
import numpy as np

# 输出数组时，控制输出精度
>>>rand_arr = np.random.random([3,3]) / 1e3  # 原始数组
>>>rand_arr
array([[  5.43404942e-04,   2.78369385e-04,   4.24517591e-04],
       [  8.44776132e-04,   4.71885619e-06,   1.21569121e-04],
       [  6.70749085e-04,   8.25852755e-04,   1.36706590e-04]])
# 第一个操作，指定小数点后3位数
>>>np.set_printoptions(precision=3)
>>>rand_arr
array([[  5.434e-04,   2.784e-04,   4.245e-04],
       [  8.448e-04,   4.719e-06,   1.216e-04],
       [  6.707e-04,   8.259e-04,   1.367e-04]])
# 第二个操作，避免输出e-04，并指定小数点后6位数
>>>np.set_printoptions(suppress=True, precision=6)
array([[ 0.000543,  0.000278,  0.000425],
       [ 0.000845,  0.000005,  0.000122],
       [ 0.000671,  0.000826,  0.000137]])

# 含有很多元素的数组
>>>a = np.arange(15)	# 原始数组
>>>a
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])
# 设定显示元素个数的阈值
>>>np.set_printoptions(threshold=6)
>>>a
array([ 0,  1,  2, ..., 12, 13, 14])
# 设定不隐藏元素输出
>>>np.set_printoptions(threshold=np.nan)
>>>a
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])

# 导入网站数据
>>>url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
>>>iris = np.genfromtxt(url, delimiter=',', dtype='object')
>>>names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
>>>iris[:3]
array([[b'5.1', b'3.5', b'1.4', b'0.2', b'Iris-setosa'],
       [b'4.9', b'3.0', b'1.4', b'0.2', b'Iris-setosa'],
       [b'4.7', b'3.2', b'1.3', b'0.2', b'Iris-setosa']], dtype=object)

# 提取某一列
>>>species = np.array([row[4] for row in iris])
>>>species
array([b'Iris-setosa', b'Iris-setosa', b'Iris-setosa', b'Iris-setosa',
       b'Iris-setosa'],
      dtype='|S15')

# 提取多列
# 方法一，在导入的时候指定列
>>>iris = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0, 1, 2, 3])
# 方法二，使用上面的方法
>>>iris = np.array([row.tolist()[:4] for row in iris], dtype='float')
array([[ 5.1,  3.5,  1.4,  0.2],
       [ 4.9,  3. ,  1.4,  0.2],
       [ 4.7,  3.2,  1.3,  0.2],
       [ 4.6,  3.1,  1.5,  0.2]])

# 计算平均值、方差、中位数
>>>sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])
>>>mu, med, sd = np.mean(sepallength), np.median(sepallength), np.std(sepallength)
5.84333333333 5.8 0.825301291785

# 0-1化数组
>>>sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])
>>>Smax, Smin = sepallength.max(), sepallength.min()
# 方法一：
>>>S = (sepallength - Smin)/(Smax - Smin)
# 方法二：
>>>S = (sepallength - Smin)/sepallength.ptp()
array([ 0.222222,  0.166667,  0.111111, ...,  0.611111,  0.527778,
        0.444444])

# softmax数组
>>>sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])
>>>def softmax(x):
>>>    e_x = np.exp(x - np.max(x))
>>>    return e_x / e_x.sum(axis=0)
>>>
>>>softmax(sepallength)
array([ 0.00222 ,  0.001817,  0.001488, ...,  0.009001,  0.006668,  0.00494 ])
```

## 31-40

```python
import numpy as np

>>>url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# 提取数据中百分位数
>>>sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])
# 提取数据中5%, 95%的数
>>>np.percentile(sepallength, q=[5, 95])
array([ 4.6  ,  7.255])

# 将数据中的缺失值替换为随机数
>>>url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
>>>iris_2d = np.genfromtxt(url, delimiter=',', dtype='object')
# 方法一
>>>i, j = np.where(iris_2d)
>>>np.random.seed(100)
>>>iris_2d[np.random.choice((i), 20), np.random.choice((j), 20)] = np.nan
# 方法二
>>>np.random.seed(100)
>>>iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan
>>>iris_2d[:10]
array([[b'5.1', b'3.5', b'1.4', b'0.2', b'Iris-setosa'],
       [b'4.9', b'3.0', b'1.4', b'0.2', b'Iris-setosa'],
       [b'4.7', b'3.2', b'1.3', b'0.2', b'Iris-setosa'],
       [b'4.6', b'3.1', b'1.5', b'0.2', b'Iris-setosa'],
       [b'5.0', b'3.6', b'1.4', b'0.2', b'Iris-setosa'],
       [b'5.4', b'3.9', b'1.7', b'0.4', b'Iris-setosa'],
       [b'4.6', b'3.4', b'1.4', b'0.3', b'Iris-setosa'],
       [b'5.0', b'3.4', b'1.5', b'0.2', b'Iris-setosa'],
       [b'4.4', nan, b'1.4', b'0.2', b'Iris-setosa'],
       [b'4.9', b'3.1', b'1.5', b'0.1', b'Iris-setosa']], dtype=object)

# 统计数据中缺失值位置
>>>url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
>>>iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
>>>iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan
# 统计缺失值个数
>>>np.isnan(iris_2d[:, 0]).sum()
5
# 统计缺失值位置
>>>np.where(np.isnan(iris_2d[:, 0]))
(array([ 38,  80, 106, 113, 121], dtype=int64),)

# 二维数组多重条件筛选
>>>condition = (iris_2d[:, 2] > 1.5) & (iris_2d[:, 0] < 5.0)
>>>iris_2d[condition]
array([[ 4.8,  3.4,  1.6,  0.2],
       [ 4.8,  3.4,  1.9,  0.2],
       [ 4.7,  3.2,  1.6,  0.2],
       [ 4.8,  3.1,  1.6,  nan],
       [ 4.9,  2.4,  3.3,  1. ]])

# 删除有缺失值的行，注意~
>>>any_nan_in_row = np.array([~np.any(np.isnan(row)) for row in iris_2d])
# 未删除前
>>>iris_2d.shape
(150, 4)
# 删除后
>>>iris_2d[any_nan_in_row].shape()
(130, 4)

# 计算数组间关联度
>>>iris = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
>>>np.corrcoef(iris[:, 0], iris[:, 2])[0, 1]
0.87175415730487182

# 检查数组中是否含有缺失值
>>>np.isnan(iris_2d).any()
True

# 替换数组中的缺失值为0
>>>iris_2d[np.isnan(iris_2d)] = 0
# 替换前
>>>np.isnan(iris_2d).any()
True
# 替换后
>>>np.isnan(iris_2d).any()
False

# 统计数组中非重复数值的情况
>>>np.unique(iris, return_counts=True)
# 上面的数组的表示数字，下面的数组表示出现次数
(array([ 0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  1. ,  1.1,  1.2,  1.3,  1.4,
         1.5,  1.6,  1.7,  1.8,  1.9,  2. ,  2.1,  2.2,  2.3,  2.4,  2.5,
         2.6,  2.7,  2.8,  2.9,  3. ,  3.1,  3.2,  3.3,  3.4,  3.5,  3.6,
         3.7,  3.8,  3.9,  4. ,  4.1,  4.2,  4.3,  4.4,  4.5,  4.6,  4.7,
         4.8,  4.9,  5. ,  5.1,  5.2,  5.3,  5.4,  5.5,  5.6,  5.7,  5.8,
         5.9,  6. ,  6.1,  6.2,  6.3,  6.4,  6.5,  6.6,  6.7,  6.8,  6.9,
         7. ,  7.1,  7.2,  7.3,  7.4,  7.6,  7.7,  7.9]),
 array([ 6, 28,  7,  7,  1,  1,  8,  4,  7, 20, 20, 26, 11,  6, 12,  7,  7,
         6,  6, 12,  6, 11,  5,  9, 14, 10, 27, 12, 13,  8, 12,  8,  4,  4,
         7,  5,  6,  4,  5,  3,  8,  9,  7,  7,  9, 11, 14, 17,  6,  3,  8,
        10, 12, 11, 10,  5,  8,  9,  4, 10,  8,  5,  3, 10,  3,  5,  1,  1,
         3,  1,  1,  1,  4,  1], dtype=int64))

# 将数组中处于相应区间的数换位文本
# 将数组中数组位于[0, 3, 5, 10]区间内的转换为对应的索引
>>>petal_length_bin = np.digitize(iris[:, 2].astype('float'), [0, 3, 5, 10])
>>>label_map = {1: 'small', 2: 'medium', 3: 'large', 4: np.nan}
>>>petal_length_cat = [label_map[x] for x in petal_length_bin]
>>>petal_length_cat[:4]
['small', 'small', 'small', 'small']
```

## 41-50

```python
import numpy as np

>>>url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# 在已经存在的数组中，多加一列
>>>iris_2d = np.genfromtxt(url, delimiter=',', dtype='object')
# 计算
>>>sepallength = iris_2d[:, 0].astype('float')
>>>petallength = iris_2d[:, 2].astype('float')
>>>volume = (np.pi * petallength * (sepallength**2))/3
>>>volume = volume[:, np.newaxis]
# 横向堆叠
>>>out = np.hstack([iris_2d, volume])
>>>out[:4]
array([[b'5.1', b'3.5', b'1.4', b'0.2', b'Iris-setosa', 38.13265162927291],
       [b'4.9', b'3.0', b'1.4', b'0.2', b'Iris-setosa', 35.200498485922445],
       [b'4.7', b'3.2', b'1.3', b'0.2', b'Iris-setosa', 30.0723720777127],
       [b'4.6', b'3.1', b'1.5', b'0.2', b'Iris-setosa', 33.238050274980004]], dtype=object)

# 对数组进行概率抽样
>>>iris = np.genfromtxt(url, delimiter=',', dtype='object')
>>>species = iris[:, 4]
# 方法一：生成概率抽样
>>>np.random.seed(100)
>>>a = np.array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
>>>species_out = np.random.choice(a, 150, p=[0.5, 0.25, 0.25])
# 方法二：直接概率抽样
>>>np.random.seed(100)
>>>probs = np.r_[np.linspace(0, 0.500, num=50), np.linspace(0.501, .750, num=50), np.linspace(.751, 1.0, num=50)]
>>>index = np.searchsorted(probs, np.random.random(150))
>>>species_out = species[index]
>>>np.unique(species_out, return_counts=True)
(array([b'Iris-setosa', b'Iris-versicolor', b'Iris-virginica'], dtype=object), array([77, 37, 36], dtype=int64))

# 对筛选后的数组，取出第二大的数
>>>petal_len_setosa = iris[iris[:, 4] == b'Iris-setosa', [2]].astype('float')
>>>np.unique(np.sort(petal_len_setosa))[-2]
1.7

# 对二维数组按照其中某一列进行排序
# 按照第一列升序
>>>iris[iris[:,0].argsort()][:20]
array([[b'4.3', b'3.0', b'1.1', b'0.1', b'Iris-setosa'],
       [b'4.4', b'3.2', b'1.3', b'0.2', b'Iris-setosa'],
       [b'4.4', b'3.0', b'1.3', b'0.2', b'Iris-setosa'],
       [b'4.4', b'2.9', b'1.4', b'0.2', b'Iris-setosa'],
       [b'4.5', b'2.3', b'1.3', b'0.3', b'Iris-setosa'],
       [b'4.6', b'3.6', b'1.0', b'0.2', b'Iris-setosa'],
       [b'4.6', b'3.1', b'1.5', b'0.2', b'Iris-setosa'],
       [b'4.6', b'3.4', b'1.4', b'0.3', b'Iris-setosa'],
       [b'4.6', b'3.2', b'1.4', b'0.2', b'Iris-setosa'],
       [b'4.7', b'3.2', b'1.3', b'0.2', b'Iris-setosa'],
       [b'4.7', b'3.2', b'1.6', b'0.2', b'Iris-setosa'],
       [b'4.8', b'3.0', b'1.4', b'0.1', b'Iris-setosa'],
       [b'4.8', b'3.0', b'1.4', b'0.3', b'Iris-setosa'],
       [b'4.8', b'3.4', b'1.9', b'0.2', b'Iris-setosa'],
       [b'4.8', b'3.4', b'1.6', b'0.2', b'Iris-setosa'],
       [b'4.8', b'3.1', b'1.6', b'0.2', b'Iris-setosa'],
       [b'4.9', b'2.4', b'3.3', b'1.0', b'Iris-versicolor'],
       [b'4.9', b'2.5', b'4.5', b'1.7', b'Iris-virginica'],
       [b'4.9', b'3.1', b'1.5', b'0.1', b'Iris-setosa'],
       [b'4.9', b'3.1', b'1.5', b'0.1', b'Iris-setosa']], dtype=object)

# 统计出现次数最多的数
>>>vals, counts = np.unique(iris[:, 3], return_counts=True)
>>>vals[np.argmax(counts)]
b'0.2'

# 找出数组中第一次出现比给定数字大的值的缩影
# 这里设定要找比1.0大的值
>>>np.argmax(iris[:, 3].astype('float') > 1.0)
50

# 给定一个区间，如果数字不在这个区间内，就替换为区间的最小/最大值
>>>a = np.random.uniform(1,50, 20)  # 原始数组
array([ 22.153505  ,  47.06146116,  41.06481956,  17.46948556,
         9.59511223,  19.26877027,   1.27873686,  13.36889132,
        39.98746292,   1.74749359,  30.34332547,  30.58642241,
         6.15223659,  19.7152288 ,   2.78732677,  44.63016661,
        49.06512199,   3.93715745,  44.63675129,  29.26817347])
# 方法一：
>>>np.clip(a, a_min=10, a_max=30)
# 方法二：
>>>np.where(a < 10, 10, np.where(a > 30, 30, a))
array([ 22.153505  ,  30.        ,  30.        ,  17.46948556,
        10.        ,  19.26877027,  10.        ,  13.36889132,
        30.        ,  10.        ,  30.        ,  30.        ,
        10.        ,  19.7152288 ,  10.        ,  30.        ,
        30.        ,  10.        ,  30.        ,  29.26817347])

# 统计前n大的元素位置
>>>a = np.random.uniform(1,50, 20)  # 原始数组
>>>a
array([ 22.153505  ,  47.06146116,  41.06481956,  17.46948556,
         9.59511223,  19.26877027,   1.27873686,  13.36889132,
        39.98746292,   1.74749359,  30.34332547,  30.58642241,
         6.15223659,  19.7152288 ,   2.78732677,  44.63016661,
        49.06512199,   3.93715745,  44.63675129,  29.26817347])
# 方法一：
>>>a.argsort()[-5:]
>>>array([ 2, 15, 18,  1, 16], dtype=int64)
# 方法二：
>>>np.argpartition(-a, 5)[:5]
>>>array([16,  1, 18, 15,  2], dtype=int64)
# 获取前n大的元素
>>>a[a.argsort()][-5:]
# 方法1:
>>>a[a.argsort()][-5:]
# 方法2:
>>>np.sort(a)[-5:]
# 方法3:
>>>np.partition(a, kth=-5)[-5:]
# 前三个输出都是
>>>array([ 41.06481956,  44.63016661,  47.06146116,  44.63675129,  49.06512199])
# 方法4:
>>>a[np.argpartition(-a, 5)][:5]
>>>array([ 49.06512199,  47.06146116,  44.63675129,  44.63016661,  41.06481956])


# 计算有唯一值的行数,没看懂
>>>arr = np.random.randint(1,11,size=(6, 10))
>>>arr
array([[ 2,  6,  5,  3,  9,  4,  6,  1, 10,  4],
       [ 7,  4,  5,  8,  7,  4, 10,  1,  5,  5],
       [ 6,  8,  7,  7,  3,  5,  3,  8,  2,  7],
       [ 7,  1,  8,  3,  4,  6,  5,  3,  5,  4],
       [ 8, 10,  1,  1,  6, 10,  7,  7,  6,  7],
       [ 5,  8,  4, 10,  3,  4,  9,  8,  2,  6]])
>>>def counts_of_all_values_rowwise(arr2d):
>>>    num_counts_array = [np.unique(row, return_counts=True) for row in arr2d]
>>>    return [[int(b[a==i]) if i in a else 0 for i in np.unique(arr2d)] for a, b in num_counts_array]
>>>counts_of_all_values_rowwise(arr)
[[1, 1, 1, 2, 1, 2, 0, 0, 1, 1],
 [1, 0, 0, 2, 3, 0, 2, 1, 0, 1],
 [0, 1, 2, 0, 1, 1, 3, 2, 0, 0],
 [1, 0, 2, 2, 2, 1, 1, 1, 0, 0],
 [2, 0, 0, 0, 0, 2, 3, 1, 0, 2],
 [0, 1, 1, 2, 1, 1, 0, 2, 1, 1]]

# 将多维数组转换为一维数组
>>>arr1 = np.arange(3)
>>>arr2 = np.arange(3,7)
>>>arr3 = np.arange(7,10)
>>>array_of_arrays = np.array([arr1, arr2, arr3])
>>>array_of_arrays      # 原始数组
array([array([0, 1, 2]), array([3, 4, 5, 6]), array([7, 8, 9])], dtype=object)
>>>arr_2d = np.concatenate(array_of_arrays)
>>>arr_2d
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

## 51-60

```python
import numpy as np

# 唯一编码位置
>>>np.random.seed(101)
>>>arr = np.random.randint(1,4, size=6)
>>>arr
array([2, 3, 2, 2, 2, 1])
# 返回数组是3x6,因为数组长度为6，最大数为1。1对应的是[1,0,0,],2对应的是[0,1,0],3对应的是[0,0,1]
# 方法一
>>>def one_hot_encodings(arr):
>>>    uniqs = np.unique(arr)
>>>    out = np.zeros((arr.shape[0], uniqs.shape[0]))
>>>    for i, k in enumerate(arr):
>>>        out[i, k-1] = 1
>>>    return out
>>>one_hot_encodings(arr)
array([[ 0.,  1.,  0.],
       [ 0.,  0.,  1.],
       [ 0.,  1.,  0.],
       [ 0.,  1.,  0.],
       [ 0.,  1.,  0.],
       [ 1.,  0.,  0.]])
# 方法二
>>>(arr[:, None] == np.unique(arr)).view(np.int8)
array([[0, 1, 0],
       [0, 0, 1],
       [0, 1, 0],
       [0, 1, 0],
       [0, 1, 0],
       [1, 0, 0]], dtype=int8)

# 获取由变量名分组的行号
>>>url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
>>>species = np.genfromtxt(url, delimiter=',', dtype='str', usecols=4)
>>>np.random.seed(100)
>>>species_small = np.sort(np.random.choice(species, size=20))
>>>species_small
array(['Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa',
       'Iris-setosa', 'Iris-versicolor', 'Iris-versicolor',
       'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor',
       'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor',
       'Iris-versicolor', 'Iris-virginica', 'Iris-virginica',
       'Iris-virginica', 'Iris-virginica', 'Iris-virginica',
       'Iris-virginica'],
      dtype='<U15')

>>>[i for val in np.unique(species_small) for i, grp in enumerate(species_small[species_small==val])]
[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5]

# 根据已知分类组名创建索引
>>>np.random.seed(100)
>>>species_small = np.sort(np.random.choice(species, size=20))
>>>species_small
array(['Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa',
       'Iris-setosa', 'Iris-setosa', 'Iris-versicolor', 'Iris-versicolor',
       'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor',
       'Iris-versicolor', 'Iris-virginica', 'Iris-virginica',
       'Iris-virginica', 'Iris-virginica', 'Iris-virginica',
       'Iris-virginica', 'Iris-virginica', 'Iris-virginica'],
      dtype='<U15')
# 方法一，用列表解析式
>>>output = [np.argwhere(np.unique(species_small) == s).tolist()[0][0] for val in np.unique(species_small) for s in species_small[species_small==val]]
# 方法二，不用列表解析式直接循环
>>>output = []
>>>uniqs = np.unique(species_small)
>>>for val in uniqs:  # uniq values in group
>>>    for s in species_small[species_small==val]:  # each element in group
>>>        groupid = np.argwhere(uniqs == s).tolist()[0][0]  # groupid
>>>        output.append(groupid)
>>>output
[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]

# 数值排序，输出对应的索引值
>>>np.random.seed(10)
>>>a = np.random.randint(20, size=10)
>>>a
[ 9  4 15  0 17 16 17  8  9  0]
>>>a.argsort().argsort()
[4 2 6 0 8 7 9 3 5 1]

# 多维数组排序，输出对应的索引值
>>>np.random.seed(10)
>>>a = np.random.randint(20, size=[2,5])
>>>a
[[ 9  4 15  0 17]
 [16 17  8  9  0]]
>>>a.ravel().argsort().argsort().reshape(a.shape)
[[ 9  4 15  0 17]
 [16 17  8  9  0]]
[[4 2 6 0 8]
 [7 9 3 5 1]]

# 输出多维数组每一行的最大值
>>>np.random.seed(100)
>>>a = np.random.randint(1,10, [5,3])
>>>a
array([[9, 9, 4],
       [8, 8, 1],
       [5, 3, 6],
       [3, 3, 3],
       [2, 1, 9]])
# 方法一
>>>np.amax(a, axis=1)
# 方法二
>>>np.apply_along_axis(np.max, arr=a, axis=1)
array([9, 8, 6, 3, 9])

# 输出多维数组每行最小值比最大值
>>>np.random.seed(100)
>>>a = np.random.randint(1,10, [5,3])
>>>a
array([[9, 9, 4],
       [8, 8, 1],
       [5, 3, 6],
       [3, 3, 3],
       [2, 1, 9]])
>>>np.apply_along_axis(lambda x: np.min(x)/np.max(x), arr=a, axis=1)
array([ 0.44444444,  0.125     ,  0.5       ,  1.        ,  0.11111111])

# 找出数组中重复的数值，输出布尔型
>>>np.random.seed(100)
>>>a = np.random.randint(0, 5, 10)
>>>a
[0 0 3 0 2 4 2 2 2 2]
>>>out = np.full(a.shape[0], True)
>>>unique_positions = np.unique(a, return_index=True)[1]
>>>out[unique_positions] = False
>>>out
[False  True False  True False False  True  True  True  True]

# 输出多维数组每一分类的平均值
>>>iris = np.genfromtxt(url, delimiter=',', dtype='object')
>>>names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
>>>numeric_column = iris[:, 1].astype('float')  # sepalwidth
>>>grouping_column = iris[:, 4]  # species

# 方法一：列表解析
>>>output = [[group_val, numeric_column[grouping_column==group_val].mean()] for group_val in np.unique(grouping_column)]
# 方法二：不用列表解析直接循环
>>>output = []
>>>for group_val in np.unique(grouping_column):
>>>    output.append([group_val, numeric_column[grouping_column==group_val].mean()])
>>>output
[[b'Iris-setosa', 3.418],
 [b'Iris-versicolor', 2.770],
 [b'Iris-virginica', 2.974]]

# 将图像转换为数组存储，并再转换出图像显示
>>>from io import BytesIO
>>>from PIL import Image
>>>import PIL, requests

>>>URL = 'https://upload.wikimedia.org/wikipedia/commons/8/8b/Denali_Mt_McKinley.jpg'
>>>response = requests.get(URL)
>>>I = Image.open(BytesIO(response.content))
>>>I = I.resize([150,150])
>>>arr = np.asarray(I)
>>>im = PIL.Image.fromarray(np.uint8(arr))
>>>Image.Image.show(im)
```

## 61-70

```python
import numpy as np

# 去除数组中所有的缺失值
>>>a = np.array([1,2,3,np.nan,5,6,7,np.nan])
>>>a[~np.isnan(a)]
array([ 1.,  2.,  3.,  5.,  6.,  7.])

# 计算两个数组间的欧几里得
>>>a = np.array([1,2,3,4,5])
>>>b = np.array([4,5,6,7,8])
>>>dist = np.linalg.norm(a-b)
>>>dist
6.7082039324993694

# 找出数组中最大的两个数
>>>a = np.array([1, 3, 7, 1, 2, 6, 0, 1])
>>>doublediff = np.diff(np.sign(np.diff(a)))
>>>peak_locations = np.where(doublediff == -2)[0] + 1
>>>peak_locations
array([2, 5])

# 二维数组数组数值每项减去一维数组对应的行号
>>>a_2d = np.array([[3,3,3],[4,4,4],[5,5,5]])
>>>b_1d = np.array([1,2,3])
>>>a_2d - b_1d[:,None]
[[2 2 2]
 [2 2 2]
 [2 2 2]]

# 数组中重复的第n次的数的索引
>>>x = np.array([1, 2, 1, 1, 3, 4, 3, 1, 1, 2, 1, 1, 2])
>>>n = 5
# 方法一：列表解析
>>>[i for i, v in enumerate(x) if v == 1][n-1]
# 方法二：numpy方法
>>>np.where(x == 1)[0][n-1]
8

# 将`datetime64`格式转换为`datetime`
>>>dt64 = np.datetime64('2018-02-25 22:10:10')
# 方法一：利用datetime模块
>>>from datetime import datetime
>>>dt64.tolist()
# 方法二：numpy直接转换
>>>dt64.astype(datetime)
datetime.datetime(2018, 2, 25, 22, 10, 10)

# 这个没看懂。。
# 如何计算numpy数组的移动平均值(累加和)
>>>np.random.seed(100)
>>>Z = np.random.randint(10, size=10)
>>>Z
[8 8 3 7 7 0 4 2 5 2]
# 方法一: np.cumsum
>>>def moving_average(a, n=3) :
>>>    ret = np.cumsum(a, dtype=float)
>>>    ret[n:] = ret[n:] - ret[:-n]
>>>    return ret[n - 1:] / n
>>>moving_average(Z, n=3).round(2)
# 方法二: np.convolve
>>>np.convolve(Z, np.ones(3)/3, mode='valid')
[ 6.33  6.    5.67  4.67  3.67  2.    3.67  3.  ]

# 创建一个数组，只给出起始值、步长、数组长度
>>>length = 10
>>>start = 5
>>>step = 3
>>>def seq(start, length, step):
>>>    end = start + (step*length)
>>>    return np.arange(start, end, step)
>>>seq(start, length, step)
array([ 5,  8, 11, 14, 17, 20, 23, 26, 29, 32])

# numpy数组填补不完整的日期
>>>dates = np.arange(np.datetime64('2018-02-01'), np.datetime64('2018-02-25'), 2)
>>>dates
['2018-02-01' '2018-02-03' '2018-02-05' '2018-02-07' '2018-02-09'
 '2018-02-11' '2018-02-13' '2018-02-15' '2018-02-17' '2018-02-19'
 '2018-02-21' '2018-02-23']
# 方法一: 列表解析
>>>filled_in = np.array([np.arange(date, (date+d)) for date, d in zip(dates, np.diff(dates))]).reshape(-1)
>>>output = np.hstack([filled_in, dates[-1]])
# 方法二: 不用列表解析循环
>>>out = []
>>>for date, d in zip(dates, np.diff(dates)):
>>>    out.append(np.arange(date, (date+d)))
>>>filled_in = np.array(out).reshape(-1)
>>>output = np.hstack([filled_in, dates[-1]])
>>>output
array(['2018-02-01', '2018-02-02', '2018-02-03', '2018-02-04',
       '2018-02-05', '2018-02-06', '2018-02-07', '2018-02-08',
       '2018-02-09', '2018-02-10', '2018-02-11', '2018-02-12',
       '2018-02-13', '2018-02-14', '2018-02-15', '2018-02-16',
       '2018-02-17', '2018-02-18', '2018-02-19', '2018-02-20',
       '2018-02-21', '2018-02-22', '2018-02-23'], dtype='datetime64[D]')

# 给定一维数组，变形为二维数组（给定每维长度）
>>>arr = np.arange(15)
>>>arr
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])
>>>def gen_strides(arr, stride_len=5, window_len=5):
>>>    n_strides = (a.size-window_len)//stride_len + 1
>>>    return np.array([a[s:(s+window_len)] for s in np.arange(0, n_strides*stride_len, stride_len)])
>>>gen_strides(np.arange(15), stride_len=2, window_len=4)
[[ 0  1  2  3]
 [ 2  3  4  5]
 [ 4  5  6  7]
 [ 6  7  8  9]
 [ 8  9 10 11]
 [10 11 12 13]]
```

