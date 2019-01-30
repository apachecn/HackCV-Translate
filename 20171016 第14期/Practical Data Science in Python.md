# Python中的实用数据科学

原文链接：[Practical Data Science in Python](https://radimrehurek.com/data_science_python/?from=hackcv&hmsr=hackcv.com&utm_medium=hackcv.com&utm_source=hackcv.com)

这本笔记本伴随着我在2014年12月在布拉格的[经济大学](https://www.vse.cz/english/)上的“数据科学与Python”的讨论。欢迎提出问题和评论[@RadimRehurek](https：//twitter.com/radimrehurek)。

本演讲的目的是展示（文本）机器学习背后的一些高级的介绍性概念。 这些概念通过这个笔记本中的具体代码示例来演示，您可以在自己的计算机上自行运行（在安装IPython之后，见下文）。

我们假设听众（读者）将拥有一些基本的编程知识（虽然不一定是Python）和一些基本的入门数据挖掘背景。 对于机器学习专家来说，这不是一个“高级谈话”。

代码示例构建了一个可工作的可执行原型：一个应用程序，用于将英语中的电话SMS消息（以及英语的“SMS类型”）分类为“垃圾邮件”或“火腿”（=非垃圾邮件）。


[![img](https://radimrehurek.com/data_science_python/python.png)](https://xkcd.com/353/)

整个过程中使用的语言将是[Python](https://www.python.org/)，这是一种通用语言，有助于管道的所有部分：I/O，数据清洗和预处理，模型训练和评估。虽然Python绝不是唯一的选择，但由于其成熟的科学计算生态系统，它提供了灵活性，易开发性和性能的独特组合。其庞大的开源生态系统还避免了任何单个特定框架或库的锁定（以及相关的bitrot）。

Python（及其大多数库）也是独立于平台的，因此您可以在Windows，Linux或OS X上运行此笔记本而无需更改。

其中一个Python工具，IPython notebook =以HTML呈现的交互式Python，你现在正在观看。我们将介绍下面广泛用于数据科学行业的其他实用工具。



想以交互方式运行以下示例吗？ （可选的）

1.安装（免费）[Anaconda](https://store.continuum.io/cshop/anaconda/)Python发行版，包括Python本身。
2.安装“自然语言处理”TextBlob库：[此处说明](https://textblob.readthedocs.org/en/dev/install.html)。
3.将此笔记本的源代码下载到您的计算机：[http://radimrehurek.com/data_science_python/data_science_python.ipynb](https://radimrehurek.com/data_science_python/data_science_python.ipynb)并运行它：
   `$ ipython notebook data_science_python.ipynb`
4.观看[IPython教程视频](https://www.youtube.com/watch?v=H6dLGQw9yFQ)了解笔记本导航基础知识。
5.运行下面的第一个代码单元格;如果它没有错误地执行，你将可以顺利的进入下一步！


# 端到端示例：自动垃圾邮件过滤

In [1]:

```python
%matplotlib inline
import matplotlib.pyplot as plt
import csv
from textblob import TextBlob
import pandas
import sklearn
import cPickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.learning_curve import learning_curve
```



## 第1步：加载数据，总览全局

跳过*真正的*第一步（充实规范，找出我们想要做的事情 - 在实践中经常非常重要！），让我们下载我们将在本演示中使用的数据集。 转到https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection并下载zip文件。 在`data`子目录下解压缩它。 你应该看到一个名为`SMSSpamCollection`的文件，大小约为0.5MB：

```
$ ls -l data
total 1352
-rw-r--r--@ 1 kofola  staff  477907 Mar 15  2011 SMSSpamCollection
-rw-r--r--@ 1 kofola  staff    5868 Apr 18  2011 readme
-rw-r-----@ 1 kofola  staff  203415 Dec  1 15:30 smsspamcollection.zip
```

此文件包含**超过5千条短信电话消息的集合**（有关详细信息，请参阅`readme`文件）：

In [2]:

```python
messages = [line.rstrip() for line in open('./data/SMSSpamCollection')]
print len(messages)
```

```
5574
```

文本集合有时也称为“语料库”。 让我们输出这个SMS语料库中的前十条消息：

In [3]:

```python
for message_no, message in enumerate(messages[:10]):
    print message_no, message
```



```
0 ham	Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...
1 ham	Ok lar... Joking wif u oni...
2 spam	Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's
3 ham	U dun say so early hor... U c already then say...
4 ham	Nah I don't think he goes to usf, he lives around here though
5 spam	FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, £1.50 to rcv
6 ham	Even my brother is not like to speak with me. They treat me like aids patent.
7 ham	As per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set as your callertune for all Callers. Press *9 to copy your friends Callertune
8 spam	WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.
9 spam	Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030
```

我们看到这是一个[TSV](https://en.wikipedia.org/wiki/Tab-separated_values)（“制表符分隔值”）文件，其中第一列是一个标签，说明给定的消息是否正常 消息（“火腿”）或“垃圾邮件”。 第二列是消息本身。

这个语料库将是我们标记的训练集。 使用这些火腿/垃圾邮件示例，我们将**训练机器学习模型，以学习自动区分火腿/垃圾邮件**。 然后，通过训练有素的模型，我们将能够将任意未标记的消息**分类为火腿或垃圾邮件**。

[![img](https://radimrehurek.com/data_science_python/plot_ML_flow_chart_11.png)](http://www.astroml.org/sklearn_tutorial/general_concepts.html#supervised-learning-model-fit-x-y)

我们可以使用Python的`pandas`库为我们完成工作，而不是手工解析TSV（或CSV或Excel ...）文件：

In [4]:

```python
messages = pandas.read_csv('./data/SMSSpamCollection', sep='\t', quoting=csv.QUOTE_NONE,
                           names=["label", "message"])
print messages
```



```
     label                                            message
0      ham  Go until jurong point, crazy.. Available only ...
1      ham                      Ok lar... Joking wif u oni...
2     spam  Free entry in 2 a wkly comp to win FA Cup fina...
3      ham  U dun say so early hor... U c already then say...
4      ham  Nah I don't think he goes to usf, he lives aro...
5     spam  FreeMsg Hey there darling it's been 3 week's n...
6      ham  Even my brother is not like to speak with me. ...
7      ham  As per your request 'Melle Melle (Oru Minnamin...
8     spam  WINNER!! As a valued network customer you have...
9     spam  Had your mobile 11 months or more? U R entitle...
10     ham  I'm gonna be home soon and i don't want to tal...
11    spam  SIX chances to win CASH! From 100 to 20,000 po...
12    spam  URGENT! You have won a 1 week FREE membership ...
13     ham  I've been searching for the right words to tha...
14     ham                I HAVE A DATE ON SUNDAY WITH WILL!!
15    spam  XXXMobileMovieClub: To use your credit, click ...
16     ham                         Oh k...i'm watching here:)
17     ham  Eh u remember how 2 spell his name... Yes i di...
18     ham  Fine if thats the way u feel. Thats the way ...
19    spam  England v Macedonia - dont miss the goals/team...
20     ham          Is that seriously how you spell his name?
21     ham    I‘m going to try for 2 months ha ha only joking
22     ham  So ü pay first lar... Then when is da stock co...
23     ham  Aft i finish my lunch then i go str down lor. ...
24     ham  Ffffffffff. Alright no way I can meet up with ...
25     ham  Just forced myself to eat a slice. I'm really ...
26     ham                     Lol your always so convincing.
27     ham  Did you catch the bus ? Are you frying an egg ...
28     ham  I'm back &amp; we're packing the car now, I'll...
29     ham  Ahhh. Work. I vaguely remember that! What does...
...    ...                                                ...
5544   ham           Armand says get your ass over to epsilon
5545   ham             U still havent got urself a jacket ah?
5546   ham  I'm taking derek &amp; taylor to walmart, if I...
5547   ham      Hi its in durban are you still on this number
5548   ham         Ic. There are a lotta childporn cars then.
5549  spam  Had your contract mobile 11 Mnths? Latest Moto...
5550   ham                 No, I was trying it all weekend ;V
5551   ham  You know, wot people wear. T shirts, jumpers, ...
5552   ham        Cool, what time you think you can get here?
5553   ham  Wen did you get so spiritual and deep. That's ...
5554   ham  Have a safe trip to Nigeria. Wish you happines...
5555   ham                        Hahaha..use your brain dear
5556   ham  Well keep in mind I've only got enough gas for...
5557   ham  Yeh. Indians was nice. Tho it did kane me off ...
5558   ham  Yes i have. So that's why u texted. Pshew...mi...
5559   ham  No. I meant the calculation is the same. That ...
5560   ham                             Sorry, I'll call later
5561   ham  if you aren't here in the next  &lt;#&gt;  hou...
5562   ham                  Anything lor. Juz both of us lor.
5563   ham  Get me out of this dump heap. My mom decided t...
5564   ham  Ok lor... Sony ericsson salesman... I ask shuh...
5565   ham                                Ard 6 like dat lor.
5566   ham  Why don't you wait 'til at least wednesday to ...
5567   ham                                       Huh y lei...
5568  spam  REMINDER FROM O2: To get 2.50 pounds free call...
5569  spam  This is the 2nd time we have tried 2 contact u...
5570   ham               Will ü b going to esplanade fr home?
5571   ham  Pity, * was in mood for that. So...any other s...
5572   ham  The guy did some bitching but I acted like i'd...
5573   ham                         Rofl. Its true to its name

[5574 rows x 2 columns]
```


使用`pandas`，我们还可以轻松查看汇总统计信息：

In [5]:

```
messages.groupby('label').describe()
```

Out[5]:

|        |                                                   | message |
| ------ | ------------------------------------------------- | ------- |
| label  |                                                   |         |
| ham    | count                                             | 4827    |
| unique | 4518                                              |         |
| top    | Sorry, I'll call later                            |         |
| freq   | 30                                                |         |
| spam   | count                                             | 747     |
| unique | 653                                               |         |
| top    | Please call our customer service representativ... |         |
| freq   | 4                                                 |         |



消息有多长？

In [6]:

```python
messages['length'] = messages['message'].map(lambda text: len(text))
print messages.head()
```



```
  label                                            message  length
0   ham  Go until jurong point, crazy.. Available only ...     111
1   ham                      Ok lar... Joking wif u oni...      29
2  spam  Free entry in 2 a wkly comp to win FA Cup fina...     155
3   ham  U dun say so early hor... U c already then say...      49
4   ham  Nah I don't think he goes to usf, he lives aro...      61
```

In [7]:

```
messages.length.plot(bins=20, kind='hist')
```

Out[7]:

```
<matplotlib.axes._subplots.AxesSubplot at 0x1174a54d0>
```



![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAZQAAAEACAYAAACUMoD1AAAABHNCSVQICAgIfAhkiAAAAAlwSFlz%0AAAALEgAACxIB0t1+/AAAF79JREFUeJzt3X+w3XWd3/HnWwi7IIyRug2/0rmpGyrprg2LS9x1d7jT%0AWjZ2doTWjmCrVZexdtAVbKdK3E7BztTCzrgG7MDORiRgFyrVlsI2ItHmtrY7S9Y10WiMkDVpTTDB%0AqgjsDEvAd/84n8vnGG/IuTff7z3f+z3Px8yZfL+fc773fM4Lct/5ft7ne05kJpIknaiXjHsCkqR+%0AsKBIkhphQZEkNcKCIklqhAVFktQIC4okqRGtFZSIWBkR2yLiGxHx9Yh4Xxm/ISIORMSOcnvD0DEb%0AIuLRiNgTEZcOjV8UEbvKfTe3NWdJ0sJFW9ehRMRZwFmZuTMiTgf+DLgceDPwVGb+3lGPXwPcDfwy%0AcC7wBWB1ZmZEbAfem5nbI2ILcEtmPtjKxCVJC9LaGUpmHsrMnWX7aeCbDAoFQMxxyGXAPZl5JDP3%0AA3uBdRFxNnBGZm4vj7uLQWGSJHXIovRQImIKuBD4kzL02xHx1Yi4PSKWl7FzgANDhx1gUICOHj9I%0ALUySpI5ovaCU5a7PANeUM5XbgFXAWuC7wEfbnoMkqX0nt/nDI2IZ8FngP2TmfQCZ+fjQ/Z8AHii7%0AB4GVQ4efx+DM5GDZHh4/OMdz+aFkkrQAmTlXG2Le2nyXVwC3A7szc+PQ+NlDD/v7wK6yfT9wZUSc%0AEhGrgNXA9sw8BDwZEevKz3wbcN9cz5mZ3jK5/vrrxz6HrtzMwizM4sVvTWrzDOV1wFuBr0XEjjL2%0AIeAtEbEWSGAf8G6AzNwdEfcCu4HngKuzvtqrgc3AqcCW9B1eL2r//v3jnkJnmEVlFpVZtKO1gpKZ%0A/4u5z4A+9yLHfAT4yBzjfwb8YnOzkyQ1zSvle+gd73jHuKfQGWZRmUVlFu1o7cLGxRYR2ZfXIkmL%0AJSLIrjflNT4zMzPjnkJnmEVlFpVZtMOCIklqhEtekjTBXPKSJHWOBaWHXB+uzKIyi8os2mFBkSQ1%0Awh6KJE0weyiSpM6xoPSQ68OVWVRmUZlFOywokqRG2EORpAlmD0WS1DkWlB5yfbgyi8osKrNohwVF%0AktQIeyiSNMHsoUiSOseC0kOuD1dmUZlFZRbtsKBIkhphD0WSJpg9FElS51hQesj14cosKrOozKId%0AJ497Ak361re+teBjly9fzooVKxqcjSRNll71UH72Z1ewbNnL5n3ss8/+kHe+80puu+2WFmYmSd3V%0AZA+lV2cozzxzE8888/YFHHkLzz+/t/H5SNIksYfSQ64PV2ZRmUVlFu2woEiSGmFB6aHp6elxT6Ez%0AzKIyi8os2mFBkSQ1woLSQ64PV2ZRmUVlFu2woEiSGmFB6SHXhyuzqMyiMot2WFAkSY2woPSQ68OV%0AWVRmUZlFOywokqRGtFZQImJlRGyLiG9ExNcj4n1l/MyI2BoRj0TEQxGxfOiYDRHxaETsiYhLh8Yv%0Aiohd5b6b25pzX7g+XJlFZRaVWbSjzTOUI8D7M/NvAq8F3hMRFwDXAVsz83zgi2WfiFgDXAGsAdYD%0At0bE7AeW3QZclZmrgdURsb7FeUuSFqC1gpKZhzJzZ9l+GvgmcC7wRuDO8rA7gcvL9mXAPZl5JDP3%0AA3uBdRFxNnBGZm4vj7tr6BjNwfXhyiwqs6jMoh2L0kOJiCngQuBhYEVmHi53HQZmv4TkHODA0GEH%0AGBSgo8cPlnFJUoe0/vH1EXE68Fngmsx8qq5iQWZmRDT4hSybgH1lezmwFpgu+zPlz7n3H3vsADMz%0AMy+src7+C2Yp7k9PT3dqPu53Z39WV+Yzrv3Zsa7MZzH3Z2Zm2Lx5MwBTU1M0qdUv2IqIZcAfAZ/L%0AzI1lbA8wnZmHynLWtsx8VURcB5CZN5bHPQhcD/yf8pgLyvhbgEsy858d9VwJm4GFfR/Ku961lz/4%0AA79gS9JkafILttp8l1cAtwO7Z4tJcT/1t/7bgfuGxq+MiFMiYhWwGtiemYeAJyNiXfmZbxs6RnM4%0A+l+jk8wsKrOozKIdbS55vQ54K/C1iNhRxjYANwL3RsRVwH7gzQCZuTsi7gV2A88BV2c9fbqawenH%0AqcCWzHywxXlLkhagV98p75KXJM3PkljykiRNFgtKD7k+XJlFZRaVWbTDgiJJaoQFpYeG32s/6cyi%0AMovKLNphQZEkNcKC0kOuD1dmUZlFZRbtsKBIkhphQekh14crs6jMojKLdlhQJEmNsKD0kOvDlVlU%0AZlGZRTssKJKkRlhQesj14cosKrOozKIdFhRJUiMsKD3k+nBlFpVZVGbRDguKJKkRFpQecn24MovK%0ALCqzaIcFRZLUCAtKD7k+XJlFZRaVWbTDgiJJaoQFpYdcH67MojKLyizaYUGRJDXCgtJDrg9XZlGZ%0ARWUW7bCgSJIaYUHpIdeHK7OozKIyi3ZYUCRJjbCg9JDrw5VZVGZRmUU7LCiSpEZYUHrI9eHKLCqz%0AqMyiHRYUSVIjLCg95PpwZRaVWVRm0Q4LiiSpERaUHnJ9uDKLyiwqs2iHBUWS1AgLSg+5PlyZRWUW%0AlVm0w4IiSWpEqwUlIj4ZEYcjYtfQ2A0RcSAidpTbG4bu2xARj0bEnoi4dGj8oojYVe67uc0594Hr%0Aw5VZVGZRmUU72j5DuQNYf9RYAr+XmReW2+cAImINcAWwphxza0REOeY24KrMXA2sjoijf6Ykacxa%0ALSiZ+SXgh3PcFXOMXQbck5lHMnM/sBdYFxFnA2dk5vbyuLuAy9uYb1+4PlyZRWUWlVm0Y1w9lN+O%0AiK9GxO0RsbyMnQMcGHrMAeDcOcYPlnFJUoeMo6DcBqwC1gLfBT46hjn0muvDlVlUZlGZRTtOXuwn%0AzMzHZ7cj4hPAA2X3ILBy6KHnMTgzOVi2h8cPzv3TNwH7yvZyBjVruuzPlD/n3t+06eNs2vTxEV/F%0AT9u2bdvgp5X/UWdPqd133333u7Q/MzPD5s2bAZiamqJRmdnqDZgCdg3tnz20/X7g7rK9BtgJnMLg%0ADObPgSj3PQysY9B72QKsn+N5EjYn5AJuN+fg+IUcmzmIsTu2bds27il0hllUZlGZRVV+fzXy+77V%0AM5SIuAe4BHhFRHwHuB6Yjoi1g1/g7APeXQrb7oi4F9gNPAdcXV4swNXAZuBUYEtmPtjmvCVJ8xf1%0Ad/bSFhE5qDlvX8DRtwDXMKhxC3p2+pKjpMkSEWTmXO+8nTevlJckNcKC0kOzDTiZxTCzqMyiHcct%0AKBFxVrle5MGyvyYirmp/apKkpeS4PZRSSO4AficzXx0Ry4AdmfkLizHBUdlDkaT5W+weyisy89PA%0A8wCZeYTBu7AkSXrBKAXl6Yj4K7M7EfFa4EftTUknyvXhyiwqs6jMoh2jXIfyLxhczf7XI+KPgZ8D%0A/mGrs5IkLTkjXYcSEScDf4PBGc2esuzVKfZQJGn+FrWHEhEvBTYA12bmLmAqIn6ziSeXJPXHKD2U%0AO4BngV8t+48B/7a1GemEuT5cmUVlFpVZtGOUgvLKzLyJQVEhM/+i3SlJkpaiUQrKX0bEqbM7EfFK%0A4C/bm5JO1OxHVssshplFZRbtGOVdXjcADwLnRcTdwOuAd7Q4J0nSEvSiZygR8RLg5cCbgHcCdwOv%0AycxtizA3LZDrw5VZVGZRmUU7XvQMJTN/HBEfKFfK/9EizUmStASN8lleNwL/D/g08EJDPjN/0O7U%0A5sfrUCRp/pq8DmWUHsqVDH7Tvueo8VVNTECS1A/HfZdXZk5l5qqjb4sxOS2M68OVWVRmUZlFO457%0AhhIRb+Kn14J+BOzKzMdbmZUkackZpYfy34BfAbYBAVwCfIXBkte/ycy72p7kKOyhSNL8LXYPZRlw%0AQWYeLk++AvgUsA74n0AnCookabxGuVJ+5WwxKR4vY9+nfByLusX14cosKrOozKIdo5yhbCvLXvcy%0AWPJ6EzBTPoX4iTYnJ0laOkbpobwE+AcMPnIF4H8Dn82ONQ3soUjS/C1qD6VcLf9l4EeZuTUiTgNO%0AB55qYgKSpH4Y5Qu2/inwn4DfL0PnAfe1OSmdGNeHK7OozKIyi3aM0pR/D/BrwJMAmfkI8FfbnJQk%0AaekZpYeyPTMvjogdmXlh+X75r2TmqxdniqOxhyJJ87eo3ykP/I+I+B3gtIj4uwyWvx5o4sklSf0x%0ASkG5DvgesAt4N7AF+FdtTkonxvXhyiwqs6jMoh2jvMvr+Yi4D7jPz+6SJB3LMXsoERHA9cB7gZPK%0A8PPAxxl8hlenmgb2UCRp/harh/J+Bhcz/nJmvjwzXw5cXMbe38STS5L648UKyj8B/lFm7psdyMxv%0AA/+43KeOcn24MovKLCqzaMeLFZSTM/N7Rw+WsVE+A0ySNEFerIeyIzMvnO9942IPRZLmb7F6KK+O%0AiKfmugG/OOJEPxkRhyNi19DYmRGxNSIeiYiHImL50H0bIuLRiNgTEZcOjV8UEbvKfTcv5IVKktp1%0AzIKSmSdl5hnHuI265HUHsP6oseuArZl5PvDFsk9ErAGuANaUY24t7zQDuA24KjNXA6sj4uifqSGu%0AD1dmUZlFZRbtGOXCxgXLzC8BPzxq+I3AnWX7TuDysn0ZcE9mHsnM/cBeYF1EnA2ckZnby+PuGjpG%0AktQRrRaUY1gx9A2Qh4EVZfsc4MDQ4w4A584xfrCM6ximp6fHPYXOMIvKLCqzaMc4CsoLysWRdrMl%0AqQfG8fbfwxFxVmYeKstZsx/nchBYOfS48xicmRws28PjB+f+0ZuA2ctmlgNrgemyP1P+PNb+7Nio%0Aj//J/dk12dl/+Yxzf3h9uAvzGef+7FhX5jPO/Z07d3Lttdd2Zj7j3N+4cSNr167tzHwWc39mZobN%0AmzcDMDU1RaMys9UbMAXsGtr/XeCDZfs64MayvQbYCZwCrAL+nPq25oeBdQy+034LsH6O50nYnJAL%0AuN1czpQWcuzgLKtLtm3bNu4pdIZZVGZRmUVVfn818vv+uN+HciIi4h7gEuAVDPol/xr4r8C9wF8D%0A9gNvzswnyuM/BPwW8BxwTWZ+voxfxOAik1OBLZn5vjmey+tQJGmemrwOpdWCspiWakGp74xeuL78%0AN5S0+Bb7C7bUujyB208b7h9MOrOozKIyi3ZYUCRJjXDJCxj/kteJ/DewfyNp4VzykiR1jgWlh1wf%0ArsyiMovKLNphQZEkNcIeCmAPRdKksociSeocC0oPuT5cmUVlFpVZtMOCIklqhD0UwB6KpEllD0WS%0A1DkWlB5yfbgyi8osKrNoxzi+YKuXmvjUYElayuyhAE30UMZz7OD4vvw3lLT47KFIkjrHgtJDrg9X%0AZlGZRWUW7bCgSJIaYQ8FsIciaVLZQ5EkdY4FpYdcH67MojKLyizaYUGRJDXCHgpgD0XSpLKHIknq%0AHAtKD7k+XJlFZRaVWbTDgiJJaoQ9FMAeiqRJZQ9FktQ5FpQecn24MovKLCqzaIcFRZLUCHsogD0U%0ASZPKHookqXMsKD3k+nBlFpVZVGbRDguKJKkR9lAAeyiSJpU9FElS54ytoETE/oj4WkTsiIjtZezM%0AiNgaEY9ExEMRsXzo8Rsi4tGI2BMRl45r3kuB68OVWVRmUZlFO8Z5hpLAdGZemJkXl7HrgK2ZeT7w%0AxbJPRKwBrgDWAOuBWyPCsytJ6pCx9VAiYh/wmsz8/tDYHuCSzDwcEWcBM5n5qojYAPw4M28qj3sQ%0AuCEz/2ToWHsokjRPfemhJPCFiPhyRLyrjK3IzMNl+zCwomyfAxwYOvYAcO7iTFOSNIqTx/jcr8vM%0A70bEzwFby9nJCzIzB2cdxzTHfZuAfWV7ObAWmC77M+XPY+3Pjo36+Kb2Oc79ox0/uyY8PT39E+vD%0A09PTP3X/JO3PjnVlPuPc37lzJ9dee21n5jPO/Y0bN7J27drOzGcx92dmZti8eTMAU1NTNKkTbxuO%0AiOuBp4F3MeirHIqIs4FtZcnrOoDMvLE8/kHg+sx8eOhnuORVzMzMvPA/0qQzi8osKrOomlzyGktB%0AiYjTgJMy86mIeCnwEPBh4PXA9zPzplJElmfmdaUpfzdwMYOlri8AP59Dk7egSNL8NVlQxrXktQL4%0ALxExO4c/zMyHIuLLwL0RcRWwH3gzQGbujoh7gd3Ac8DV6W9RSeqUsTTlM3NfZq4tt1/IzH9Xxn+Q%0Ama/PzPMz89LMfGLomI9k5s9n5qsy8/PjmPdSMdw/mHRmUZlFZRbt8FoOSVIjOtGUb4I9FEmav75c%0AhyJJ6hELSg+5PlyZRWUWlVm0w4IiSWqEPRTAHoqkSWUPRZLUORaUHnJ9uDKLyiwqs2iHBUWS1Ah7%0AKIA9FEmTyh6KJKlzLCg95PpwZRaVWVRm0Q4LiiSpEfZQAHsokiaVPRRJUudYUHrI9eHKLCqzqMyi%0AHRYUSVIj7KEA9lAkTSp7KJKkzrGg9JDrw5VZVGZRmUU7LCiSpEbYQwHsoUiaVPZQJEmdY0HpIdeH%0AK7OozKIyi3ZYUCRJjbCHAthDkTSp7KFIkjrHgtJDrg9XZlGZRWUW7bCgSJIaYQ8FsIciaVLZQ5Ek%0AdY4FpYdcH67MojKLyizaYUGRJDXCHgpgD0XSpLKHIknqnCVTUCJifUTsiYhHI+KD455Pl0TEgm99%0A51p5ZRaVWbRjSRSUiDgJ+PfAemAN8JaIuGC8s+qSPOr2sTnG5rr1386dO8c9hc4wi8os2nHyuCcw%0AoouBvZm5HyAi/iNwGfDNcU6qu54Y+ZEncpayFHo3TzwxehZ9ZxaVWbRjqRSUc4HvDO0fANaNaS49%0As/A3E/S9GEman6VSUEb67XPqqR9j2bLPzPuHP/vst3nmmXkf1mH7F+l5xlOM5uvDH/5wYz9rKRfC%0A/fv3j3sKnWEW7VgSbxuOiNcCN2Tm+rK/AfhxZt409JjuvxBJ6qCm3ja8VArKycC3gL8DPAZsB96S%0AmfZQJKkjlsSSV2Y+FxHvBT4PnATcbjGRpG5ZEmcokqTuWxLXoRzPJF30GBErI2JbRHwjIr4eEe8r%0A42dGxNaIeCQiHoqI5UPHbCjZ7ImIS8c3+3ZExEkRsSMiHij7E5lFRCyPiM9ExDcjYndErJvgLDaU%0AvyO7IuLuiPiZSckiIj4ZEYcjYtfQ2Lxfe0RcVPJ7NCJuHunJM3NJ3xgsge0FpoBlwE7ggnHPq8XX%0AexawtmyfzqC3dAHwu8AHyvgHgRvL9pqSybKS0V7gJeN+HQ1n8s+BPwTuL/sTmQVwJ/BbZftk4GWT%0AmEV5Pd8Gfqbsf5rBh/xNRBbArwMXAruGxubz2mdXrrYDF5ftLcD64z13H85QXrjoMTOPALMXPfZS%0AZh7KzJ1l+2kGF3eeC7yRwS8Uyp+Xl+3LgHsy80gOLgzdyyCzXoiI84C/B3yCwSdtwgRmEREvA349%0AMz8Jg75jZv6ICcwCeBI4ApxW3tBzGoM380xEFpn5JeCHRw3P57Wvi4izgTMyc3t53F1DxxxTHwrK%0AXBc9njumuSyqiJhi8C+Rh4EVmXm43HUYWFG2z2GQyay+5fMx4F8CPx4am8QsVgHfi4g7IuIrEbEp%0AIl7KBGaRmT8APgr8XwaF5InM3MoEZjFkvq/96PGDjJBJHwrKRL6rICJOBz4LXJOZTw3fl4Nz1BfL%0ApReZRcRvAo9n5g7q2clPmJQsGCxx/RJwa2b+EvAXwHXDD5iULCLilcC1DJZwzgFOj4i3Dj9mUrKY%0AywivfcH6UFAOAiuH9lfyk5W1dyJiGYNi8qnMvK8MH46Is8r9ZwOPl/Gj8zmvjPXBrwJvjIh9wD3A%0A346ITzGZWRwADmTmn5b9zzAoMIcmMIvXAH+cmd/PzOeA/wz8CpOZxaz5/J04UMbPO2r8uJn0oaB8%0AGVgdEVMRcQpwBXD/mOfUmhh8ZsntwO7M3Dh01/3Ubxd7O3Df0PiVEXFKRKwCVjNoti15mfmhzFyZ%0AmauAK4H/nplvYzKzOAR8JyLOL0OvB74BPMCEZQHsAV4bEaeWvy+vB3YzmVnMmtffifL/05PlnYIB%0AvG3omGMb9zsSGnpXwxsYvNtpL7Bh3PNp+bX+GoN+wU5gR7mtB84EvgA8AjwELB865kMlmz3Ab4z7%0ANbSUyyXUd3lNZBbA3wL+FPgqg3+Vv2yCs/gAg4K6i0ETetmkZMHgbP0x4FkG/eV3LuS1AxeV/PYC%0At4zy3F7YKElqRB+WvCRJHWBBkSQ1woIiSWqEBUWS1AgLiiSpERYUSVIjLCiSpEZYUCRJjfj/+71V%0Av4dJRyEAAAAASUVORK5CYII=)

In [8]:

```python
messages.length.describe()
```

Out[8]:

```python
count    5574.000000
mean       80.604593
std        59.919970
min         2.000000
25%        36.000000
50%        62.000000
75%       122.000000
max       910.000000
Name: length, dtype: float64
```

什么是超长消息？

In [9]:

```python
print list(messages.message[messages.length > 900])
```



```
["For me the love should start with attraction.i should feel that I need her every time around me.she should be the first thing which comes in my thoughts.I would start the day and end it with her.she should be there every time I dream.love will be then when my every breath has her name.my life should happen around her.my life will be named to her.I would cry for her.will give all my happiness and take all her sorrows.I will be ready to fight with anyone for her.I will be in love when I will be doing the craziest things for her.love will be when I don't have to proove anyone that my girl is the most beautiful lady on the whole planet.I will always be singing praises for her.love will be when I start up making chicken curry and end up makiing sambar.life will be the most beautiful then.will get every morning and thank god for the day because she is with me.I would like to say a lot..will tell later.."]
```

垃圾邮件和火腿之间的邮件长度有什么不同吗？

In [10]:

```python
messages.hist(column='length', by='label', bins=50)
```

Out[10]:

```python
array([<matplotlib.axes._subplots.AxesSubplot object at 0x1174ade10>,
       <matplotlib.axes._subplots.AxesSubplot object at 0x11757c9d0>], dtype=object)
```



![img](data:image/png;base64,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%0ALMkJwszMkv4/0VSQ6FMw/FIAAAAASUVORK5CYII=)

很有趣，但我们如何让计算机自己理解纯文本消息呢？ 或者可以在这种畸形的胡言乱语之下呢？

## 步骤2：数据预处理

在本节中，我们将按原始消息（字符序列）按向矢量（数字序列）。

映射不是1比1; 我们将使用[bag-of-words](https://en.wikipedia.org/wiki/Bag-of-words_model)方法，其中文本中的每个唯一单词将由一个数字表示。

作为第一步，让我们编写一个将消息拆分为单个单词的函数：

In [11]:

```python
def split_into_tokens(message):
    message = unicode(message, 'utf8')  # convert bytes into proper unicode
    return TextBlob(message).words
```

这里再次输出原始文本：

In [12]:

```python
messages.message.head()
```

Out[12]:

```
0    Go until jurong point, crazy.. Available only ...
1                        Ok lar... Joking wif u oni...
2    Free entry in 2 a wkly comp to win FA Cup fina...
3    U dun say so early hor... U c already then say...
4    Nah I don't think he goes to usf, he lives aro...
Name: message, dtype: object
```

...以下是相同信息，分词：

...and here are the same messages, tokenized:

In [13]:

```python
messages.message.head().apply(split_into_tokens)
```

Out[13]:

```
0    [Go, until, jurong, point, crazy, Available, o...
1                       [Ok, lar, Joking, wif, u, oni]
2    [Free, entry, in, 2, a, wkly, comp, to, win, F...
3    [U, dun, say, so, early, hor, U, c, already, t...
4    [Nah, I, do, n't, think, he, goes, to, usf, he...
Name: message, dtype: object
```

NLP问题：

1. 大写字母是否蕴含信息？
2. 区分变形形式（“去”与“去”）是否蕴含信息？
3. 有感叹词，是否决定蕴含信息吗？

换句话说，我们希望更好地“标准化”文本。

使用textblob，我们会检测[词性（POS）](http://www.ling.upenn.edu/courses/Fall_2007/ling001/penn_treebank_pos.html)标签：

In [14]:

```python
TextBlob("Hello world, how is it going?").tags  # list of (word, POS) pairs
```

Out[14]:

```
[(u'Hello', u'UH'),
 (u'world', u'NN'),
 (u'how', u'WRB'),
 (u'is', u'VBZ'),
 (u'it', u'PRP'),
 (u'going', u'VBG')]
```

并将单词标准化为基本形式（[lemmas](https://en.wikipedia.org/wiki/Lemmatisation)）：

In [15]:

```python
def split_into_lemmas(message):
    message = unicode(message, 'utf8').lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma 
    return [word.lemma for word in words]

messages.message.head().apply(split_into_lemmas)
```

Out[15]:

```
0    [go, until, jurong, point, crazy, available, o...
1                       [ok, lar, joking, wif, u, oni]
2    [free, entry, in, 2, a, wkly, comp, to, win, f...
3    [u, dun, say, so, early, hor, u, c, already, t...
4    [nah, i, do, n't, think, he, go, to, usf, he, ...
Name: message, dtype: object
```

现在更好了。 您可以想到更多改进预处理的方法：解码HTML实体（我们在上面看到的那些`＆amp;`和`＆lt;`）; 过滤掉停用词（代词等）; 添加更多功能，例如所有大写字母指示符等。

## 步骤3：向量的数据

现在我们将每个消息（表示为上面的标记（lemmas）列表）转换为机器学习模型可以理解的向量。

这样做基本上需要三个步骤，在词袋模型中：

1.计算每个消息中出现一个单词的次数（术语频率）
2.加权计数，使频繁的令牌获得较低的权重（逆文档频率）
3.将向量归一化为单位长度，从原始文本长度（L2范数）中抽象出来

每个向量的维度与SMS语料库中的唯一单词一样多：

In [16]:

```python
bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(messages['message'])
print len(bow_transformer.vocabulary_)
```

```
8874
```

在这里，我们使用了`scikit-learn`（`sklearn`），这是一个功能强大的Python库，用于使用机器学习。 它包含多种方法和选项。

让我们拿一条短信，把它的字袋计数作为一个向量，使用我们新的`bow_transformer`：

In [17]:

```python
message4 = messages['message'][3]
print message4
```

```
U dun say so early hor... U c already then say...
```

In [18]:

```python
bow4 = bow_transformer.transform([message4])
print bow4
print bow4.shape
```

```
  (0, 1158)	1
  (0, 1899)	1
  (0, 2897)	1
  (0, 2927)	1
  (0, 4021)	1
  (0, 6736)	2
  (0, 7111)	1
  (0, 7698)	1
  (0, 8013)	2
(1, 8874)
```

所以，消息nr中有九个独特的单词。 4，其中两个出现两次，其余只出现一次。 理智检查：是什么单词出现了两次？

In [19]:

```python
print bow_transformer.get_feature_names()[6736]
print bow_transformer.get_feature_names()[8013]
```

```
say
u
```

整个SMS语料库的词袋计数是一个庞大的稀疏矩阵：

In [20]:

```python
messages_bow = bow_transformer.transform(messages['message'])
print 'sparse matrix shape:', messages_bow.shape
print 'number of non-zeros:', messages_bow.nnz
print 'sparsity: %.2f%%' % (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
```

```
sparse matrix shape: (5574, 8874)
number of non-zeros: 80272
sparsity: 0.16%
```

最后，在计数之后，可以使用[TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)，使用scikit-learn的`TfidfTransformer`来完成术语加权和归一化。：

In [21]:

```python
tfidf_transformer = TfidfTransformer().fit(messages_bow)
tfidf4 = tfidf_transformer.transform(bow4)
print tfidf4
```

```
  (0, 8013)	0.305114653686
  (0, 7698)	0.225299911221
  (0, 7111)	0.191390347987
  (0, 6736)	0.523371210191
  (0, 4021)	0.456354991921
  (0, 2927)	0.32967579251
  (0, 2897)	0.303693312742
  (0, 1899)	0.24664322833
  (0, 1158)	0.274934159477
```

什么是“u”字的IDF（逆文档频率）是多少？ "university"这个词？

In [22]:

```python
print tfidf_transformer.idf_[bow_transformer.vocabulary_['u']]
print tfidf_transformer.idf_[bow_transformer.vocabulary_['university']]
```

```
2.85068150539
8.23975323521
```

将整个词袋语料库立即转换为TF-IDF语料库：

In [23]:

```python
messages_tfidf = tfidf_transformer.transform(messages_bow)
print messages_tfidf.shape
```

```
(5574, 8874)
```

有多种方法可以对数据进行预处理和向量化。 这两个步骤，也称为“特征工程”，通常是构建预测管道的最耗时和“不合时宜”的部分，但它们非常重要并且需要一些经验。 诀窍是不断评估：分析模型的错误，改进数据清理和预处理，为新功能进行头脑风暴，评估......

## 步骤4：训练模型，检测垃圾邮件

将消息表示为向量，我们最终可以训练我们的垃圾邮件/火腿分类器。 这部分非常简单，有许多库可以实现训练算法。

我们将在这里使用scikit-learn，一开始选择[Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)分类器：

In [24]:

```python
%time spam_detector = MultinomialNB().fit(messages_tfidf, messages['label'])
```

```python
CPU times: user 3.16 ms, sys: 699 µs, total: 3.86 ms
Wall time: 3.33 ms
```

让我们尝试对单个随机消息进行分类：

In [25]:

```python
print 'predicted:', spam_detector.predict(tfidf4)[0]
print 'expected:', messages.label[3]
```

```
predicted: ham
expected: ham
```

万岁！ 您也可以使用自己的文本进行尝试。

自然而然的有一个问题是，我们总共正确分类了多少条消息？

In [26]:

```python
all_predictions = spam_detector.predict(messages_tfidf)
print all_predictions
```

```
['ham' 'ham' 'spam' ..., 'ham' 'ham' 'ham']
```

In [27]:

```python
print 'accuracy', accuracy_score(messages['label'], all_predictions)
print 'confusion matrix\n', confusion_matrix(messages['label'], all_predictions)
print '(row=expected, col=predicted)'
```

```
accuracy 0.969501255831
confusion matrix
[[4827    0]
 [ 170  577]]
(row=expected, col=predicted)
```

In [28]:

```python
plt.matshow(confusion_matrix(messages['label'], all_predictions), cmap=plt.cm.binary, interpolation='nearest')
plt.title('confusion matrix')
plt.colorbar()
plt.ylabel('expected label')
plt.xlabel('predicted label')
```

Out[28]:

```
<matplotlib.text.Text at 0x11b311910>
```

![img](data:image/png;base64,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%0AQmCC)

从这个混淆矩阵，我们可以计算准确率和召回旅，或他们的组合（调和平均值）F1：

In [29]:

```python
print classification_report(messages['label'], all_predictions)
```

```
             precision    recall  f1-score   support

        ham       0.97      1.00      0.98      4827
       spam       1.00      0.77      0.87       747

avg / total       0.97      0.97      0.97      5574
```

评估模型性能有很多可选择的指标。哪一个最合适取决于任务。例如，错误预测“垃圾邮件”为“火腿”的成本可能远低于错误预测“火腿”为“垃圾邮件”。

## 步骤5：如何进行实验？

在上面的“评价”中，我们犯了一个严重的问题。为了简化演示，我们评估了用于训练的相同数据的准确性。 **永远不要评估您训练的同一数据集！坏！乱伦！**

这样的评估没有告诉我们模型真正的预测能力。如果我们只是在训练期间记住每个例子，那么即使我们无法对任何新消息进行分类，训练数据的准确性也只有100％。

一种正确的方法是将数据拆分为训练/测试集，其中模型仅在其模型拟合和参数调整期间看到**训练数据**。 **测试数据**从未以任何方式使用 - 由于这个过程，我们确保我们不是“作弊”，并且我们对测试数据的最终评估代表了真正的预测性能。

In [30]:

```python
msg_train, msg_test, label_train, label_test = \
    train_test_split(messages['message'], messages['label'], test_size=0.2)

print len(msg_train), len(msg_test), len(msg_train) + len(msg_test)
```

```
4459 1115 5574
```

因此，根据要求，测试大小是整个数据集的20％（总共5574个中的1115个消息），其余的训练（5574个中有4459个）。

让我们回顾整个流程到目前为止，将步骤明确地放入scikit-learn的“Pipeline”中：

In [31]:

```python
pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=split_into_lemmas)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])
```

通常的做法是将训练集再次划分为较小的子集; 例如，5个大小相等的子集。 然后我们在四个部分上训练模型，并在最后一部分计算精度（称为“验证集”）。 重复五次（每次评估不同的部分），我们得到模型“稳定性”的感觉。 如果模型为不同的子集提供了截然不同的分数，则表明存在错误（不良数据或模型差异）的迹象。 返回，分析错误，重新检查输入数据是否有垃圾，重新检查数据清理。

在我们的例子中，一切顺利但是：

In [32]:

```python
scores = cross_val_score(pipeline,  # steps to convert raw messages into models
                         msg_train,  # training data
                         label_train,  # training labels
                         cv=10,  # split data randomly into 10 parts: 9 for training, 1 for scoring
                         scoring='accuracy',  # which scoring metric?
                         n_jobs=-1,  # -1 = use all cores = faster
                         )
print scores
```

```
[ 0.93736018  0.96420582  0.94854586  0.94183445  0.96412556  0.94382022
  0.94606742  0.96404494  0.94831461  0.94606742]
```

分数确实比我们训练整个数据集时差一点（5574训练样例，准确度0.97）。 它们相当稳定：

In [33]:

```python
print scores.mean(), scores.std()
```

```
0.9504386476 0.00947200821389
```

一个自然的问题是，我们如何改进这个模型？ 这里的分数已经很高了，但我们如何改进模型呢？

朴素贝叶斯是[高偏差 - 低差异](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)分类器（又简单且稳定，不易过度拟合）的一个例子。 来自频谱另一侧的示例将是最近邻（kNN）分类器或决策树，其具有低偏差但高方差（易于过度拟合）。 套袋（随机森林）作为降低方差的一种方法，通过训练许多（高方差）模型和平均。

[![img](https://radimrehurek.com/data_science_python/plot_bias_variance_examples_2.png)](http://www.astroml.org/sklearn_tutorial/practical.html#bias-variance-over-fitting-and-under-fitting)

换一种说法：

-  **高偏差** =分类是固执己见的。 用数据改变主意的空间不大，它有自己的想法。 另一方面，它没有那么多空间可以欺骗自己过度拟合（左图）。
-  **低偏差** =分类器更听话，但也更神经质。 将完全按照你的要求去做，众所周知，这可能是一个真正的麻烦（右图）。


In [34]:

```python
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
```

In [35]:

```python
%time plot_learning_curve(pipeline, "accuracy vs. training set size", msg_train, label_train, cv=5)
```

```
CPU times: user 382 ms, sys: 83.1 ms, total: 465 ms
Wall time: 28.5 s
```

Out[35]:

```
<module 'matplotlib.pyplot' from '/Volumes/work/workspace/vew/sklearn_intro/lib/python2.7/site-packages/matplotlib/pyplot.pyc'>
```

![img](data:image/png;base64,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%0A/fff88EHHzBo0CDnvQYNGkTfvn05evQop06dYujQoeX6BR0dHV0ita7rfn5+Pg8++CDPPvssx44d%0AIzs7m549ezp7G1ebq4yJieHw4cPOY1UlLS2NmJiYMm1aFpGRkbzxxhukp6ezYMECnnrqKfbv3098%0AfDwHDhxwe427FL+NGzcu837x8fF89tlnJVL8njt3jujo6Ct+Tk9hnEgF8fPxq3A+9NoQGReMzopQ%0A3DPtcbgHXQ52ocfhHswdPrdCPyyutY3Q0FBeeuklhg0bxieffMK5c+e4cOECa9asYdy4ccDlQzlx%0AcXEkJyczfvx48vPz2blzJ2+99RaPPPIIAO+//z5ZWVnO9kUEHx8f1q9fz65duygqKiI4OBh/f398%0Afd2H8/H39+ehhx5i7NixZGdn061bN+e53NxcHA4HAQEBbN26laVLl5brhZSHH36YV199lfT0dLKz%0As5kxY4bzXEFBAQUFBc6hvTVr1rB27aWhwsjISE6cOMGZM2fctv3QQw+RmprKunXruHDhArNnz6Zu%0A3brOoajSuBseK+bDDz/k6FErokVYWBgigq+vL7179yYzM5O5c+eSn59PTk4OW7duBWDgwIFMnTqV%0A48ePc/z4cV566SW3a2OKGTp0KBMmTHA60qysrDLfxqsWqmpypSY2qnFi3fDbxdu/Z0uWLNHExESt%0AX7++RkVFae/evXXLli2qak0gDxkypET9o0ePau/evbVBgwaakJCgCxYscJ575JFHtFGjRhoUFKQ3%0A33yzc0J+2bJl2rJlS61fv75GRkbqqFGjrjihv2nTJhURHT58eInyFStWaJMmTTQ4OFh79+6tI0aM%0AcOo7ePBgicly14n1wsJCHT16tIaHh2vz5s113rx5JerOmzdPIyMjNSwsTIcMGaIDBw50Tk6rqj7+%0A+OMaHh6uDodDMzIyLrPLRx99pK1bt9bQ0FDt2rWr/vjjj85zTZs21a+++sp57M6mxTz77LMaExOj%0AQUFBmpCQoAsXLnSe++GHH/Suu+5Sh8OhUVFROnPmTFVVPX/+vI4cOVKjo6M1OjpaR40apfn5+apq%0ATazHxcWVuMfFixd1zpw52rJlSw0ODtaEhAR9/vnn3eop67tLFU6smxXr1YDJ01G1mHwiBkP5qI4V%0A62Y4y2AwGAyVxvREDIarYHoihtqK6YkYDAaDwasxTqQa8IZ1DeXB6DQYDBXFOBGDwWAwVBozJ2Iw%0AXAUzJ2KorZh8IgaDl2AiNBsM7vHocJaI3CMiP4vIPhEZ5+Z8hIh8JiI7ROQHEXnU5VyYiKwQkZ9E%0A5EcRae9JrZ6ktozhG53uqewirPXr19f4gtzrRWdt0OitOj2Nx5yIiPgCfwfuAVoDA0WkValqw4Hv%0AVPVWoCswW0SKe0dzgdWq2gr4L+AnT2n1NGVF/PQ2jM6qxeisOmqDRqg9OqsST/ZEkoBfVPWQql4A%0APgDuL1UnEwix90OAE6paKCKhQCdVfQtAVQtV9bQHtXqUU6dO1bSEcmF0Vi1GZ9VRGzRC7dFZlXjS%0AicQAaS7HR+0yVxYCN4lIBvA9MMoubwZkichiEdkuIgtFpJ4HtRoMBoOhEnjSiZRnMG4CsENVGwO3%0AAvNEJBhrwv/3wGuq+nvgLPCcx5R6GNcwz96M0Vm1GJ1VR23QCLVHZ5Xiwcmc9sBnLsfjgXGl6qwG%0AOrocfwUkAlHAQZfy24FVbu6hZjOb2cxmtopvVfWs9+QrvtuAG0SkKZABDAAGlqrzM3A38LWIRAIt%0AgQOqelJE0kTkRlXda9fZXfoGWkXvORsMBoOhcnjMidgT5MOBzwFfYJGq/iQiT9rnFwDTgMUi8j3W%0A0NqzqnrSbmIEsEREAoD9wGOe0mowGAyGylGrV6wbDAaDoWaptbGzrraQsZq1HBKRnSLynYhstcsa%0AiMgXIrJXRNaKSJhL/fG27p9FpLsHdb0lIr+KyC6XsgrrEpH/FpFd9rm51aRzsogctW36nYjc6wU6%0A40RkvYjsthfHjrTLvcqmV9DpNTYVkboi8q290PhHEZlul3ubLcvS6TW2LKXX19bzqX3seXvW9GrK%0ASk7a+wK/AE0Bf2AH0KoG9RwEGpQqm4U1PAcwDphh77e29frb+n8BfDykqxPQFthVSV3FPdWtQJK9%0Avxq4pxp0TgKecVO3JnVGAbfa+0HAHqCVt9n0Cjq9yqZAPftfP+AbrBdovMqWV9DpVbZ0uf8zwBJg%0ApX3scXvW1p5IeRYyVjelJ/n7AO/Y++8Afe39+4FlqnpBVQ9h/ecleUKQqm4Csq9BVzsRiQaCVXWr%0AXe9dl2s8qRMut2lN6/yPqu6w93OxoijE4GU2vYJO8CKbquo5ezcA64dhNl5myyvoBC+yJYCIxAI9%0AgTddtHncnrXViZRnIWN1osCXIrJNRP7HLotU1V/t/V+BSHu/MZbeYqpbe0V1lS5Pp/r0jhCR70Vk%0AkUs33Ct0ivXWYVvgW7zYpi46v7GLvMamIuIjIjuwbLZeVXfjhbYsQyd4kS1tXgb+F7joUuZxe9ZW%0AJ+JtbwN0VNW2wL3AMBHp5HpSrX7hlTTXyOcph66a5HWsyAW3YoXHmV2zci4hIkHAP4FRqprjes6b%0AbGrrXIGlMxcvs6mqXlQrbl4s0FlE7ih13its6UZnV7zMliLSGzimqt/hvofkMXvWVieSDsS5HMdR%0A0ntWK6qaaf+bBXyENTz1q4hEAdhdxGN29dLaY+2y6qIiuo7a5bGlyj2uV1WPqQ1W97x4yK9GdYqI%0AP5YDeU9VP7aLvc6mLjrfL9bprTZVKy5eKvDfeKEt3ehM9EJbJgN9ROQgsAy4U0TeoxrsWVudiHMh%0Ao1jrSAYAK2tCiIjUEytUCyJSH+gO7LL1/NGu9keg+IGzEkgRkQARaQbcgDWRVV1USJeq/gc4IyLt%0ARESAIS7XeAz7C19MPyyb1qhOu91FwI+q+orLKa+yaVk6vcmmYqWBCLP3A4FuwHd4ny3d6ix+MNvU%0A+PdTVSeoapyqNgNSgHWqOoTqsGdl3wKo6Q1r6GgP1oTQ+BrU0QzrLYcdwA/FWoAGwJfAXmAtEOZy%0AzQRb989ADw9qW4YVLaAAaw7pscrowvqFuMs+92o16Hwca0JvJ1Zgzo+xxnZrWuftWOPNO7AeeN9h%0ApTrwKpuWofNeb7IpcAuw3da4E/jfyv7deNiWZen0Glu60dyFS29nedyeZrGhwWAwGCpNbR3OMhgM%0ABoMXYJyIwWAwGCqNcSIGg8FgqDTGiRgMBoOh0hgnYjAYDIZKY5yIwWAwGCqNcSIGr0dEwl1CbmfK%0ApRDc20XkionV7LDWVw1nLSJfV53imkdEHhWRv9W0DsP1jyfT4xoMVYKqnsAKIoiITAJyVHVO8XkR%0A8VXVojKu/Tfw73Lco2MVyfUWzAIwQ7VgeiKG2oiIyNsiMl9EvgFmishtIrLZ7p18LSI32hW7yqUE%0APZPFSoC1XkT2i8gIlwZzXepvEJEPReQnEXnfpU5Pu2ybiLxa3G4pYb4i8hcR2WpHeH3CLh8tIovs%0A/VvESvpTV0SSytD9qIh8LFYioYMiMlxExtr1toiIw663QUResXtmu0TkNjeaGorIClvTVhFJtsu7%0AuPTwtosVsNFgqBCmJ2KorShW2OoOqqpixS/rpKpFInI3MA3o7+a6G4E7gBBgj4i8ZvdiXH+534qV%0AtCcT+Np+6G4H5tv3OCwiS3H/a/9PwClVTRKROsD/FZHPgVeADSLSDyvcxBOqel5EfrqC7ptsl+4W%0APwAAAm5JREFULYHAfqyQG78XkTnAH4C5toZAVW0rVvTot7BCdbhGcp0LvKyqX4tIPPCZ/fnGAE+p%0A6hYRqQfkX8XmBsNlGCdiqM18qJfi9oQB74pIC6wHq7+b+gqkqpXI7ISIHMPKr5BRqt5WVc0AECuP%0ARDPgHHBAVQ/bdZYBT7i5R3fgFhEpdgQhwA2243kUKybR66q6pQzdrn+T61X1LHBWRE4BxT2fXcB/%0AudRbBlZyLxEJEZHQUpruBlpZ8fQACBYrWOjXwMsisgT4l6pWZzRpw3WCcSKG2sw5l/3/A3ylqv1E%0ApAmwoYxrClz2i3D/N5Dvpk7pXofbnA02w1X1CzflNwI5lEzycyXdrjouuhxfLEO3a93SWtupakGp%0A8pkisgrohdXj6qGqe67QrsFwGWZOxHC9EMKlHsVjZdS50oP/SihWxOjm9oMerPQD7oazPgeeKn5r%0ATERuFCtdQCjWsFInIFxEHqyA7tJIqf0B9r1uxxpKyylVfy0w0nmByK32vwmqultVZwH/D2hZzvsb%0ADE6MEzHUZlwf4rOA6SKyHSsPtrqpd6XMbu7qXypQPQ88BXwmItuAM/ZWmjeBH4HtIrILKwOeHzAH%0A+Luq/oI1bzJDRCKuoLu01tL7rvXO29e/Zrddus5IINGe6N/NpWG4UfZk/PdYPbQ1bi1jMFwBEwre%0AYCgnIlLfnqNAROYBe1X1qmtQPKxpPTBGVbfXpA7DbxfTEzEYys//2K/D7sYahlpQ04IMhprG9EQM%0ABoPBUGlMT8RgMBgMlcY4EYPBYDBUGuNEDAaDwVBpjBMxGAwGQ6UxTsRgMBgMlcY4EYPBYDBUmv8P%0AasrGYHAu0scAAAAASUVORK5CYII=)

（我们对所有可用数据的64％进行了有效训练：我们为上述测试集保留了20％，并且5折交叉验证为验证集保留了另外20％=>`0.8 * 0.8 * 5574 = 3567`训练左边的例子。）

由于性能在训练和交叉验证分数方面都在不断增长，我们认为我们的模型不够复杂/灵活，无法在很少数据的情况下捕获所有细微差别。在这种特殊情况下，它并不是非常明显，因为无论如何精度都很高。

此时，我们有两个选择：

1. 使用更多的训练数据，克服低模型的复杂性
2. 使用更复杂（更低偏差）的模型开始，从现有数据中获取更多信息

在过去几年中，随着大量训练数据收集变得更加可用，并且随着机器变得更快，方法1.变得越来越流行（更简单的算法，更多数据）。简单易懂的算法，如朴素贝叶斯，也具有更易于理解的额外好处（与一些更复杂的黑盒模型，如神经网络相比）。

知道如何正确评估模型，我们现在可以探索不同的参数如何影响性能。

##步骤6：如何调整参数？

到目前为止我们所看到的仅仅是冰山一角：还有许多其他参数需要调整。一个例子是用于训练的算法。


[![img](https://radimrehurek.com/data_science_python/drop_shadows_background.png)](https://peekaboo-vision.blogspot.cz/2013/01/machine-learning-cheat-sheet-for-scikit.html)

我们可以问：IDF加权对准确性的影响是什么？ 词形还原的额外处理成本（仅与普通词汇相比）真的有帮助吗？

我们来看看：

In [37]:

```python
params = {
    'tfidf__use_idf': (True, False),
    'bow__analyzer': (split_into_lemmas, split_into_tokens),
}

grid = GridSearchCV(
    pipeline,  # pipeline from above
    params,  # parameters to tune via cross validation
    refit=True,  # fit using all available data at the end, on the best found param combination
    n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
    scoring='accuracy',  # what score are we optimizing?
    cv=StratifiedKFold(label_train, n_folds=5),  # what type of cross validation to use
)
```

In [38]:

```python
%time nb_detector = grid.fit(msg_train, label_train)
print nb_detector.grid_scores_
```



```
CPU times: user 4.09 s, sys: 291 ms, total: 4.38 s
Wall time: 20.2 s
[mean: 0.94752, std: 0.00357, params: {'tfidf__use_idf': True, 'bow__analyzer': <function split_into_lemmas at 0x1131e8668>}, mean: 0.92958, std: 0.00390, params: {'tfidf__use_idf': False, 'bow__analyzer': <function split_into_lemmas at 0x1131e8668>}, mean: 0.94528, std: 0.00259, params: {'tfidf__use_idf': True, 'bow__analyzer': <function split_into_tokens at 0x11270b7d0>}, mean: 0.92868, std: 0.00240, params: {'tfidf__use_idf': False, 'bow__analyzer': <function split_into_tokens at 0x11270b7d0>}]
```

（首先显示最佳参数组合：在这种情况下，`use_idf = True`和`analyzer = split_into_lemmas`是最优的）。

快速全面地检查：

In [39]:

```python
print nb_detector.predict_proba(["Hi mom, how are you?"])[0]
print nb_detector.predict_proba(["WINNER! Credit for free!"])[0]
```

```
[ 0.99383955  0.00616045]
[ 0.29663109  0.70336891]
```

`predict_proba`返回每个类（火腿，垃圾邮件）的预测概率。 在第一种情况下，预测消息的概率为> 99％，垃圾邮件<1％。 因此，如果被迫选择，该模型会说“火腿”：

In [40]:

```python
print nb_detector.predict(["Hi mom, how are you?"])[0]
print nb_detector.predict(["WINNER! Credit for free!"])[0]
```

```
ham
spam
```

测试集上的整体分数，我们在训练期间根本没有计算分数：

In [41]:

```python
predictions = nb_detector.predict(msg_test)
print confusion_matrix(label_test, predictions)
print classification_report(label_test, predictions)
```



```
[[973   0]
 [ 46  96]]
             precision    recall  f1-score   support

        ham       0.95      1.00      0.98       973
       spam       1.00      0.68      0.81       142

avg / total       0.96      0.96      0.96      1115
```

这是我们可以从垃圾邮件检测管道中获得的真实预测性能，当使用带有词形化的小写，TF-IDF和Naive Bayes用于分类器时。

让我们尝试使用另一个分类器：[支持向量机（SVM）](https://en.wikipedia.org/wiki/Support_vector_machine)。 在对文本数据进行分类时，SVM是一个很好的起点，可以非常快速地获得最先进的结果，并且调参得很愉快（虽然比Naive Bayes多一点）：

In [42]:

```python
pipeline_svm = Pipeline([
    ('bow', CountVectorizer(analyzer=split_into_lemmas)),
    ('tfidf', TfidfTransformer()),
    ('classifier', SVC()),  # <== change here
])

# pipeline parameters to automatically explore and tune
param_svm = [
  {'classifier__C': [1, 10, 100, 1000], 'classifier__kernel': ['linear']},
  {'classifier__C': [1, 10, 100, 1000], 'classifier__gamma': [0.001, 0.0001], 'classifier__kernel': ['rbf']},
]

grid_svm = GridSearchCV(
    pipeline_svm,  # pipeline from above
    param_grid=param_svm,  # parameters to tune via cross validation
    refit=True,  # fit using all data, on the best detected classifier
    n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
    scoring='accuracy',  # what score are we optimizing?
    cv=StratifiedKFold(label_train, n_folds=5),  # what type of cross validation to use
)
```

In [43]:

```python
%time svm_detector = grid_svm.fit(msg_train, label_train) # find the best combination from param_svm
print svm_detector.grid_scores_
```



```python
CPU times: user 5.24 s, sys: 170 ms, total: 5.41 s
Wall time: 1min 8s
[mean: 0.98677, std: 0.00259, params: {'classifier__kernel': 'linear', 'classifier__C': 1}, mean: 0.98654, std: 0.00100, params: {'classifier__kernel': 'linear', 'classifier__C': 10}, mean: 0.98654, std: 0.00100, params: {'classifier__kernel': 'linear', 'classifier__C': 100}, mean: 0.98654, std: 0.00100, params: {'classifier__kernel': 'linear', 'classifier__C': 1000}, mean: 0.86432, std: 0.00006, params: {'classifier__gamma': 0.001, 'classifier__kernel': 'rbf', 'classifier__C': 1}, mean: 0.86432, std: 0.00006, params: {'classifier__gamma': 0.0001, 'classifier__kernel': 'rbf', 'classifier__C': 1}, mean: 0.86432, std: 0.00006, params: {'classifier__gamma': 0.001, 'classifier__kernel': 'rbf', 'classifier__C': 10}, mean: 0.86432, std: 0.00006, params: {'classifier__gamma': 0.0001, 'classifier__kernel': 'rbf', 'classifier__C': 10}, mean: 0.97040, std: 0.00587, params: {'classifier__gamma': 0.001, 'classifier__kernel': 'rbf', 'classifier__C': 100}, mean: 0.86432, std: 0.00006, params: {'classifier__gamma': 0.0001, 'classifier__kernel': 'rbf', 'classifier__C': 100}, mean: 0.98722, std: 0.00280, params: {'classifier__gamma': 0.001, 'classifier__kernel': 'rbf', 'classifier__C': 1000}, mean: 0.97040, std: 0.00587, params: {'classifier__gamma': 0.0001, 'classifier__kernel': 'rbf', 'classifier__C': 1000}]
```

显然，具有“C = 1”的线性内核是最佳参数组合。

再次进行全面检查：

In [44]:

```python
print svm_detector.predict(["Hi mom, how are you?"])[0]
print svm_detector.predict(["WINNER! Credit for free!"])[0]
```

```
ham
spam
```

In [45]:

```python
print confusion_matrix(label_test, svm_detector.predict(msg_test))
print classification_report(label_test, svm_detector.predict(msg_test))
```



```
[[965   8]
 [ 13 129]]
             precision    recall  f1-score   support

        ham       0.99      0.99      0.99       973
       spam       0.94      0.91      0.92       142

avg / total       0.98      0.98      0.98      1115
```

这是我们在使用SVM时可以从垃圾邮件检测管道中获得的实际预测性能。

## 步骤7：生成预测变量

通过基本分析和调整，真正的工作（工程）开始了。

生产预测器的最后一步是再次对整个数据集进行训练，以充分利用所有可用数据。 当然，我们使用上面通过交叉验证找到的最佳参数。 这与我们在开始时所做的非常相似，但这一次对其行为和稳定性有所了解。 在不同的训练/测试子集分裂上诚实地进行评估。

最终预测器可以序列化到磁盘，以便下次我们想要使用它时，我们可以跳过所有培训并直接使用训练模型：

In [46]:

```python
# store the spam detector to disk after training
with open('sms_spam_detector.pkl', 'wb') as fout:
    cPickle.dump(svm_detector, fout)

# ...and load it back, whenever needed, possibly on a different machine
svm_detector_reloaded = cPickle.load(open('sms_spam_detector.pkl'))
```

加载的结果是一个与原始行为完全相同的对象：

In [47]:

```python
print 'before:', svm_detector.predict([message4])[0]
print 'after:', svm_detector_reloaded.predict([message4])[0]
```

```
before: ham
after: ham
```

生产发布的另一个重要部分是**性能**。在如此处所示的快速，迭代模型调整和参数搜索之后，可以将表现良好的模型翻译成不同的语言并进行优化。交易几个精度点会给我们一个更小，更快的模型吗？是否值得优化内存使用，也许使用`mmap`来跨进程共享内存？

请注意，并不总是需要优化;总是从实际的分析开始。

对于生产管道，此处需要考虑的其他事项包括：**稳健性**（服务故障转移，冗余，负载平衡），**监控**（包括异常情况下的自动警报）和**HR可替代性**（避免关于事情如何完成的“知识孤岛”，神秘/锁定技术，调整结果的黑色艺术。如今，即使是开源世界也可以在所有这些领域提供可行的解决方案。根据OSI批准的开源许可证，今天显示的所有工具都可以免费用于商业用途。

# 其他实用概念

数据稀疏性

在线学习，数据流

`mmap`用于内存共享，系统“冷启动”加载时间

可伸缩性，分布式（集群）处理

# 无监督学习

大多数数据*不是*结构化的。获得洞察力，没有内在的评估可能（或者成为有监督的学习！）。

如何在没有标签的情况下培训*任何东西*？这是什么魔法？

[分布式假设](https://en.wikipedia.org/wiki/Distributional_semantics)：*“在类似情境中出现的词语往往具有相似的含义”*。上下文=句子，文档，滑动窗口......

对于无监督学习，请查看[Google的word2vec的实时演示](https://radimrehurek.com/2014/02/word2vec-tutorial/#app)。简单的模型，大数据（谷歌新闻，1000亿字，没有标签）。

# 下一个？

这个笔记本的静态（非交互版本）在[http://radimrehurek.com/data_science_python](https://radimrehurek.com/data_science_python)上呈现为HTML（你现在可能正在观看它，但只是在案例）。

交互式笔记本电脑源于GitHub：<https://github.com/piskvorky/data_science_python>（有关安装说明，请参见顶部）。

我的公司[RaRe Technologies](http://rare-technologies.com/)，生活在**务实的商业系统建设**和**前沿研究**的令人兴奋的交叉点。对实习/合作感兴趣？ [联系](http://rare-technologies.com/#contactus)。
