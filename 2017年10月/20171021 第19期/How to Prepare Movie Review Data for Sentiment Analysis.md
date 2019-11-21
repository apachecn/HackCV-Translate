# 如何准备电影评论数据以进行情感分析

原文链接：[How to Prepare Movie Review Data for Sentiment Analysis](https://machinelearningmastery.com/prepare-movie-review-data-sentiment-analysis/?from=hackcv&hmsr=hackcv.com&utm_medium=hackcv.com&utm_source=hackcv.com)

每个问题的文本数据准备都不同。

准备工作从简单的步骤开始，例如加载数据，但是对于与您正在使用的数据非常相关的清理任务，很快就会变得困难。 从原始数据到准备好建模的步骤，您需要有关从何处开始以及以什么顺序工作的帮助。

在本教程中，您将逐步了解如何准备电影评论文本数据以进行情感分析。

完成本教程后，您将知道：

- 如何加载和清除文本数据以删除标点符号和其他非单词。
- 如何开发词汇表，定制词汇表并将其保存到文件中。
- 如何使用清洁和预定义的词汇准备电影评论，并将其保存到可供建模的新文件中。

让我们开始吧。

- **2017年10月更新**：修复了跳过不匹配文件时的一个小错误，感谢Jan Zett。

- **2017年12月更新**：修复了完整示例中的小错字，感谢Ray和Zain。

  

如何准备电影评论数据以进行情感分析
[肯尼思·鲁](https://www.flickr.com/photos/toasty/1125019024/)摄，保留一些权利。

## 教程概述

本教程分为5个部分。 他们是：

1. 电影评论数据集
2. 加载文本数据
3. 清除文本数据
4. 训练词汇量
5. 保存准备的数据







### 在文本数据深度学习方面需要帮助吗？

立即参加我的7天免费电子邮件崩溃课程（包含代码）。

单击以注册，并获得该课程的免费PDF电子书版本。

[Start Your FREE Crash-Course Now](https://machinelearningmastery.lpages.co/leadbox/144855173f72a2%3A164f8be4f346dc/5655638436741120/)







## 1. 电影评论数据集

电影评论数据是Bo Pang和Lillian Lee在2000年代初期从imdb.com网站检索的电影评论的集合。 收集了这些评论，并将其作为他们对自然语言处理的研究的一部分提供。

这些评论最初于2002年发布，但于2004年发布了更新和清理的版本，称为“ * v2.0 *”。

该数据集由从[IMDB](http://reviews.imdb.com/Reviews)托管的rec.arts.movies.reviews新闻组的存档中抽取的1,000条正面和负面电影评论组成。 作者将此数据集称为“ *极性数据集*”。

> 我们的数据包含2002年之前撰写的1000条正面评论和1000条负面评论，每个类别每位作者的评论上限为20（总共312作者）。 我们将此语料库称为极性数据集。

— [A Sentimental Education: Sentiment Analysis Using Subjectivity Summarization Based on Minimum Cuts](http://xxx.lanl.gov/abs/cs/0409058), 2004.

数据已经进行了一些清理，例如：

- 数据集仅包含英文评论。
- 所有文字均已转换为小写。
- 标点符号周围有空格，例如句号，逗号和方括号。
- 文本已被分成每行一个句子。

该数据已用于一些相关的自然语言处理任务。 对于分类，经典模型（例如支持向量机）在数据上的性能在高70％至低80％（例如78％至82％）的范围内。

进行10倍交叉验证后，更复杂的数据准备可能会看到高达86％的结果。 如果我们希望在现代方法的实验中使用此数据集，那么这将使我们处于80年代中期到中期的水平。

> …根据下游极性分类器的选择，我们可以实现统计学上显着的改进（从82.8％到86.4％）

— [A Sentimental Education: Sentiment Analysis Using Subjectivity Summarization Based on Minimum Cuts](http://xxx.lanl.gov/abs/cs/0409058), 2004.

您可以从此处下载数据集：

- [Movie Review Polarity Dataset](http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz) (review_polarity.tar.gz, 3MB)

解压缩文件后，您将拥有一个名为“ *txt_sentoken*”的目录，该目录包含两个子目录，分别包含否定和肯定评论的文本“ *neg*”和“ *pos*”。 评论每个文件存储一个，命名规则为*cv000*至*cv999*，分别用于neg和pos。

接下来，让我们看一下加载文本数据。

## 2. 载入文本数据

在本节中，我们将研究加载单个文本文件，然后处理文件目录。

我们将假定已下载审阅数据并在当前工作目录的“ *txt_sentoken*”文件夹中提供该审阅数据。

我们可以通过打开单个文本文件，读取ASCII文本并关闭文件来加载它。 这是标准的文件处理内容。 例如，我们可以按如下方式加载第一个负面评论文件“ *cv000_29416.txt*”：

```python
# load one file
filename = 'txt_sentoken/neg/cv000_29416.txt'
# open the file as read only
file = open(filename, 'r')
# read all text
text = file.read()
# close the file
file.close()
```

这样会将文档加载为ASCII并保留所有空白，例如换行。

我们可以将其转换为一个名为load_doc()的函数，该函数将文档的文件名加载并返回文本。

```python
# load doc into memory
def load_doc(filename):	
    # open the file as read only
    file = open(filename, 'r')
    # read all text	
    text = file.read()
    # close the file
    file.close()
    return text
```

我们有两个目录，每个目录包含1000个文档。 我们可以依次使用以下命令依次处理每个目录： [listdir() function](https://docs.python.org/3/library/os.html#os.listdir), 然后依次加载每个文件。

例如，我们可以使用* load_doc()*函数在负目录中加载每个文档，以进行实际加载。

```python
from os import listdir 
# load doc into memory
def load_doc(filename):	
    # open the file as read only	
    file = open(filename, 'r')	
    # read all text	text = file.read()	
    # close the file	
    file.close()	
    return text 
# specify directory to load
directory = 'txt_sentoken/neg'
# walk through all files in the folder
for filename in listdir(directory):	
    # skip files that do not have the right extension	
    if not filename.endswith(".txt"):		
        continue	
    # create the full path of the file to open	
    path = directory + '/' + filename	
    # load document	
    doc = load_doc(path)	
    print('Loaded %s' % filename) 
```

运行此示例将在加载每个评论后打印其文件名。



```python
...Loaded cv995_23113.txt
Loaded cv996_12447.txt
Loaded cv997_5152.txt
Loaded cv998_15691.txt
Loaded cv999_14636.txt
```

我们也可以将文档的处理转换为一个函数，稍后将其用作模板以开发用于清理文件夹中所有文档的函数。 例如，下面我们定义一个* process_docs()*函数来执行相同的操作。


```python
from os import listdir 
# load doc into memory
def load_doc(filename):	
    # open the file as read only	
    file = open(filename, 'r')	
    # read all text	
    text = file.read()	
    # close the file	
    file.close()	
    return text 
# load all docs in a directory
def process_docs(directory):	
# walk through all files in the folder	
for filename in listdir(directory):		
# skip files that do not have the right extension		
if not filename.endswith(".txt"):			
    continue		
    # create the full path of the file to open		
    path = directory + '/' + filename		
    # load document		
    doc = load_doc(path)		
    print('Loaded %s' % filename) 
    # specify directory to load
    directory = 'txt_sentoken/neg'process_docs(directory)
```

现在，我们知道了如何加载电影评论文本数据，让我们来看看如何清理它。

## 3. 清洗文本数据

In this section, we will look at what data cleaning we might want to do to the movie review data.

We will assume that we will be using a bag-of-words model or perhaps a word embedding that does not require too much preparation.

Split into Tokens

First, let’s load one document and look at the raw tokens split by white space. We will use the *load_doc()* function developed in the previous section. We can use the *split()* function to split the loaded document into tokens separated by white space.



| 12345678910111213141516 | # load doc into memorydef load_doc(filename):	# open the file as read only	file = open(filename, 'r')	# read all text	text = file.read()	# close the file	file.close()	return text # load the documentfilename = 'txt_sentoken/neg/cv000_29416.txt'text = load_doc(filename)# split into tokens by white spacetokens = text.split()print(tokens) |
| ----------------------- | ------------------------------------------------------------ |
|                         |                                                              |

Running the example gives a nice long list of raw tokens from the document.



| 12   | ...'years', 'ago', 'and', 'has', 'been', 'sitting', 'on', 'the', 'shelves', 'ever', 'since', '.', 'whatever', '.', '.', '.', 'skip', 'it', '!', "where's", 'joblo', 'coming', 'from', '?', 'a', 'nightmare', 'of', 'elm', 'street', '3', '(', '7/10', ')', '-', 'blair', 'witch', '2', '(', '7/10', ')', '-', 'the', 'crow', '(', '9/10', ')', '-', 'the', 'crow', ':', 'salvation', '(', '4/10', ')', '-', 'lost', 'highway', '(', '10/10', ')', '-', 'memento', '(', '10/10', ')', '-', 'the', 'others', '(', '9/10', ')', '-', 'stir', 'of', 'echoes', '(', '8/10', ')'] |
| ---- | ------------------------------------------------------------ |
|      |                                                              |

Just looking at the raw tokens can give us a lot of ideas of things to try, such as:

- Remove punctuation from words (e.g. ‘what’s’).
- Removing tokens that are just punctuation (e.g. ‘-‘).
- Removing tokens that contain numbers (e.g. ’10/10′).
- Remove tokens that have one character (e.g. ‘a’).
- Remove tokens that don’t have much meaning (e.g. ‘and’)

Some ideas:

- We can filter out punctuation from tokens using the string *translate()* function.
- We can remove tokens that are just punctuation or contain numbers by using an *isalpha()*check on each token.
- We can remove English stop words using the list loaded using NLTK.
- We can filter out short tokens by checking their length.

Below is an updated version of cleaning this review.



| 1234567891011121314151617181920212223242526272829 | from nltk.corpus import stopwordsimport string # load doc into memorydef load_doc(filename):	# open the file as read only	file = open(filename, 'r')	# read all text	text = file.read()	# close the file	file.close()	return text # load the documentfilename = 'txt_sentoken/neg/cv000_29416.txt'text = load_doc(filename)# split into tokens by white spacetokens = text.split()# remove punctuation from each tokentable = str.maketrans('', '', string.punctuation)tokens = [w.translate(table) for w in tokens]# remove remaining tokens that are not alphabetictokens = [word for word in tokens if word.isalpha()]# filter out stop wordsstop_words = set(stopwords.words('english'))tokens = [w for w in tokens if not w in stop_words]# filter out short tokenstokens = [word for word in tokens if len(word) > 1]print(tokens) |
| ------------------------------------------------- | ------------------------------------------------------------ |
|                                                   |                                                              |

Running the example gives a much cleaner looking list of tokens



| 12   | ...'explanation', 'craziness', 'came', 'oh', 'way', 'horror', 'teen', 'slasher', 'flick', 'packaged', 'look', 'way', 'someone', 'apparently', 'assuming', 'genre', 'still', 'hot', 'kids', 'also', 'wrapped', 'production', 'two', 'years', 'ago', 'sitting', 'shelves', 'ever', 'since', 'whatever', 'skip', 'wheres', 'joblo', 'coming', 'nightmare', 'elm', 'street', 'blair', 'witch', 'crow', 'crow', 'salvation', 'lost', 'highway', 'memento', 'others', 'stir', 'echoes'] |
| ---- | ------------------------------------------------------------ |
|      |                                                              |

We can put this into a function called *clean_doc()* and test it on another review, this time a positive review.



| 12345678910111213141516171819202122232425262728293031323334 | from nltk.corpus import stopwordsimport string # load doc into memorydef load_doc(filename):	# open the file as read only	file = open(filename, 'r')	# read all text	text = file.read()	# close the file	file.close()	return text # turn a doc into clean tokensdef clean_doc(doc):	# split into tokens by white space	tokens = doc.split()	# remove punctuation from each token	table = str.maketrans('', '', string.punctuation)	tokens = [w.translate(table) for w in tokens]	# remove remaining tokens that are not alphabetic	tokens = [word for word in tokens if word.isalpha()]	# filter out stop words	stop_words = set(stopwords.words('english'))	tokens = [w for w in tokens if not w in stop_words]	# filter out short tokens	tokens = [word for word in tokens if len(word) > 1]	return tokens # load the documentfilename = 'txt_sentoken/pos/cv000_29590.txt'text = load_doc(filename)tokens = clean_doc(text)print(tokens) |
| ----------------------------------------------------------- | ------------------------------------------------------------ |
|                                                             |                                                              |

Again, the cleaning procedure seems to produce a good set of tokens, at least as a first cut.



| 12   | ...'comic', 'oscar', 'winner', 'martin', 'childs', 'shakespeare', 'love', 'production', 'design', 'turns', 'original', 'prague', 'surroundings', 'one', 'creepy', 'place', 'even', 'acting', 'hell', 'solid', 'dreamy', 'depp', 'turning', 'typically', 'strong', 'performance', 'deftly', 'handling', 'british', 'accent', 'ians', 'holm', 'joe', 'goulds', 'secret', 'richardson', 'dalmatians', 'log', 'great', 'supporting', 'roles', 'big', 'surprise', 'graham', 'cringed', 'first', 'time', 'opened', 'mouth', 'imagining', 'attempt', 'irish', 'accent', 'actually', 'wasnt', 'half', 'bad', 'film', 'however', 'good', 'strong', 'violencegore', 'sexuality', 'language', 'drug', 'content'] |
| ---- | ------------------------------------------------------------ |
|      |                                                              |

There are many more cleaning steps we could take and I leave them to your imagination.

Next, let’s look at how we can manage a preferred vocabulary of tokens.

## 4. Develop Vocabulary

When working with predictive models of text, like a bag-of-words model, there is a pressure to reduce the size of the vocabulary.

The larger the vocabulary, the more sparse the representation of each word or document.

A part of preparing text for sentiment analysis involves defining and tailoring the vocabulary of words supported by the model.

We can do this by loading all of the documents in the dataset and building a set of words. We may decide to support all of these words, or perhaps discard some. The final chosen vocabulary can then be saved to file for later use, such as filtering words in new documents in the future.

We can keep track of the vocabulary in a [Counter](https://docs.python.org/3/library/collections.html#collections.Counter), which is a dictionary of words and their count with some additional convenience functions.

We need to develop a new function to process a document and add it to the vocabulary. The function needs to load a document by calling the previously developed *load_doc()* function. It needs to clean the loaded document using the previously developed *clean_doc()* function, then it needs to add all the tokens to the Counter, and update counts. We can do this last step by calling the *update()* function on the counter object.

Below is a function called *add_doc_to_vocab()* that takes as arguments a document filename and a Counter vocabulary.



| 12345678 | # load doc and add to vocabdef add_doc_to_vocab(filename, vocab):	# load doc	doc = load_doc(filename)	# clean doc	tokens = clean_doc(doc)	# update counts	vocab.update(tokens) |
| -------- | ------------------------------------------------------------ |
|          |                                                              |

Finally, we can use our template above for processing all documents in a directory called process_docs() and update it to call *add_doc_to_vocab()*.



| 1234567891011 | # load all docs in a directorydef process_docs(directory, vocab):	# walk through all files in the folder	for filename in listdir(directory):		# skip files that do not have the right extension		if not filename.endswith(".txt"):			continue		# create the full path of the file to open		path = directory + '/' + filename		# add doc to vocab		add_doc_to_vocab(path, vocab) |
| ------------- | ------------------------------------------------------------ |
|               |                                                              |

We can put all of this together and develop a full vocabulary from all documents in the dataset.



| 12345678910111213141516171819202122232425262728293031323334353637383940414243444546474849505152535455565758596061 | from string import punctuationfrom os import listdirfrom collections import Counterfrom nltk.corpus import stopwords # load doc into memorydef load_doc(filename):	# open the file as read only	file = open(filename, 'r')	# read all text	text = file.read()	# close the file	file.close()	return text # turn a doc into clean tokensdef clean_doc(doc):	# split into tokens by white space	tokens = doc.split()	# remove punctuation from each token	table = str.maketrans('', '', punctuation)	tokens = [w.translate(table) for w in tokens]	# remove remaining tokens that are not alphabetic	tokens = [word for word in tokens if word.isalpha()]	# filter out stop words	stop_words = set(stopwords.words('english'))	tokens = [w for w in tokens if not w in stop_words]	# filter out short tokens	tokens = [word for word in tokens if len(word) > 1]	return tokens # load doc and add to vocabdef add_doc_to_vocab(filename, vocab):	# load doc	doc = load_doc(filename)	# clean doc	tokens = clean_doc(doc)	# update counts	vocab.update(tokens) # load all docs in a directorydef process_docs(directory, vocab):	# walk through all files in the folder	for filename in listdir(directory):		# skip files that do not have the right extension		if not filename.endswith(".txt"):			continue		# create the full path of the file to open		path = directory + '/' + filename		# add doc to vocab		add_doc_to_vocab(path, vocab) # define vocabvocab = Counter()# add all docs to vocabprocess_docs('txt_sentoken/neg', vocab)process_docs('txt_sentoken/pos', vocab)# print the size of the vocabprint(len(vocab))# print the top words in the vocabprint(vocab.most_common(50)) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
|                                                              |                                                              |

Running the example creates a vocabulary with all documents in the dataset, including positive and negative reviews.

We can see that there are a little over 46,000 unique words across all reviews and the top 3 words are ‘*film*‘, ‘*one*‘, and ‘*movie*‘.



| 12   | 46557[('film', 8860), ('one', 5521), ('movie', 5440), ('like', 3553), ('even', 2555), ('good', 2320), ('time', 2283), ('story', 2118), ('films', 2102), ('would', 2042), ('much', 2024), ('also', 1965), ('characters', 1947), ('get', 1921), ('character', 1906), ('two', 1825), ('first', 1768), ('see', 1730), ('well', 1694), ('way', 1668), ('make', 1590), ('really', 1563), ('little', 1491), ('life', 1472), ('plot', 1451), ('people', 1420), ('movies', 1416), ('could', 1395), ('bad', 1374), ('scene', 1373), ('never', 1364), ('best', 1301), ('new', 1277), ('many', 1268), ('doesnt', 1267), ('man', 1266), ('scenes', 1265), ('dont', 1210), ('know', 1207), ('hes', 1150), ('great', 1141), ('another', 1111), ('love', 1089), ('action', 1078), ('go', 1075), ('us', 1065), ('director', 1056), ('something', 1048), ('end', 1047), ('still', 1038)] |
| ---- | ------------------------------------------------------------ |
|      |                                                              |

Perhaps the least common words, those that only appear once across all reviews, are not predictive. Perhaps some of the most common words are not useful too.

These are good questions and really should be tested with a specific predictive model.

Generally, words that only appear once or a few times across 2,000 reviews are probably not predictive and can be removed from the vocabulary, greatly cutting down on the tokens we need to model.

We can do this by stepping through words and their counts and only keeping those with a count above a chosen threshold. Here we will use 5 occurrences.



| 1234 | # keep tokens with > 5 occurrencemin_occurane = 5tokens = [k for k,c in vocab.items() if c >= min_occurane]print(len(tokens)) |
| ---- | ------------------------------------------------------------ |
|      |                                                              |

This reduces the vocabulary from 46,557 to 14,803 words, a huge drop. Perhaps a minimum of 5 occurrences is too aggressive; you can experiment with different values.

We can then save the chosen vocabulary of words to a new file. I like to save the vocabulary as ASCII with one word per line.

Below defines a function called *save_list()* to save a list of items, in this case, tokens to file, one per line.



| 12345 | def save_list(lines, filename):	data = '\n'.join(lines)	file = open(filename, 'w')	file.write(data)	file.close() |
| ----- | ------------------------------------------------------------ |
|       |                                                              |

The complete example for defining and saving the vocabulary is listed below.



```python
from string import punctuationfrom os import listdirfrom collections import Counterfrom nltk.corpus import stopwords # load doc into memorydef load_doc(filename):	# open the file as read only	file = open(filename, 'r')	# read all text	text = file.read()	# close the file	file.close()	return text # turn a doc into clean tokensdef clean_doc(doc):	# split into tokens by white space	tokens = doc.split()	# remove punctuation from each token	table = str.maketrans('', '', punctuation)	tokens = [w.translate(table) for w in tokens]	# remove remaining tokens that are not alphabetic	tokens = [word for word in tokens if word.isalpha()]	# filter out stop words	stop_words = set(stopwords.words('english'))	tokens = [w for w in tokens if not w in stop_words]	# filter out short tokens	tokens = [word for word in tokens if len(word) > 1]	return tokens # load doc and add to vocabdef add_doc_to_vocab(filename, vocab):	# load doc	doc = load_doc(filename)	# clean doc	tokens = clean_doc(doc)	# update counts	vocab.update(tokens) # load all docs in a directorydef process_docs(directory, vocab):	# walk through all files in the folder	for filename in listdir(directory):		# skip files that do not have the right extension		if not filename.endswith(".txt"):			continue		# create the full path of the file to open		path = directory + '/' + filename		# add doc to vocab		add_doc_to_vocab(path, vocab) # save list to filedef save_list(lines, filename):	data = '\n'.join(lines)	file = open(filename, 'w')	file.write(data)	file.close() # define vocabvocab = Counter()# add all docs to vocabprocess_docs('txt_sentoken/neg', vocab)process_docs('txt_sentoken/pos', vocab)# print the size of the vocabprint(len(vocab))# print the top words in the vocabprint(vocab.most_common(50))# keep tokens with > 5 occurrencemin_occurane = 5tokens = [k for k,c in vocab.items() if c >= min_occurane]print(len(tokens))# save tokens to a vocabulary filesave_list(tokens, 'vocab.txt')
```

Running this final snippet after creating the vocabulary will save the chosen words to file.

It is a good idea to take a look at, and even study, your chosen vocabulary in order to get ideas for better preparing this data, or text data in the future.



```python hasntupdatingfigurativelysymphonyciviliansmightfishermanhokumwitchbuffoons...
```

Next, we can look at using the vocabulary to create a prepared version of the movie review dataset.

## 5. Save Prepared Data

We can use the data cleaning and chosen vocabulary to prepare each movie review and save the prepared versions of the reviews ready for modeling.

This is a good practice as it decouples the data preparation from modeling, allowing you to focus on modeling and circle back to data prep if you have new ideas.

We can start off by loading the vocabulary from ‘*vocab.txt*‘.



```python
# load doc into memory
def load_doc(filename):	
    # open the file as read only	
    file = open(filename, 'r')	
    # read all text	
    text = file.read()	
    # close the file	file.close()	
    return text # load vocabulary
vocab_filename = 'review_polarity/vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
```

Next, we can clean the reviews, use the loaded vocab to filter out unwanted tokens, and save the clean reviews in a new file.

One approach could be to save all the positive reviews in one file and all the negative reviews in another file, with the filtered tokens separated by white space for each review on separate lines.

First, we can define a function to process a document, clean it, filter it, and return it as a single line that could be saved in a file. Below defines the *doc_to_line()* function to do just that, taking a filename and vocabulary (as a set) as arguments.

It calls the previously defined *load_doc()* function to load the document and *clean_doc()* to tokenize the document.


```python
# load doc, clean and return line of tokens
def doc_to_line(filename, vocab):	
    # load the doc	
    doc = load_doc(filename)	
    # clean doc	
    tokens = clean_doc(doc)	
    # filter by vocab	
    tokens = [w for w in tokens if w in vocab]
    return ' '.join(tokens)
```

Next, we can define a new version of *process_docs()* to step through all reviews in a folder and convert them to lines by calling *doc_to_line()* for each document. A list of lines is then returned.



```python
# load all docs in a directory
def process_docs(directory, vocab):	
    lines = list()	
    # walk through all files in the folder	
    for filename in listdir(directory):		
        # skip files that do not have the right extension		
        if not filename.endswith(".txt"):			
        	continue		
        # create the full path of the file to open		
        path = directory + '/' + filename		
        # load and clean the doc		
        line = doc_to_line(path, vocab)		
        # add to list		
        lines.append(line)	
    return lines
```

We can then call *process_docs()* for both the directories of positive and negative reviews, then call *save_list()* from the previous section to save each list of processed reviews to a file.

The complete code listing is provided below.



```python
from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords 
# load doc into memory
def load_doc(filename):	
    # open the file as read only	
    file = open(filename, 'r')	
    # read all text	text = file.read()	
    # close the file	
    file.close()	
    return text 
# turn a doc into clean tokens
def clean_doc(doc):	
    # split into tokens by white space	
    tokens = doc.split()	
    # remove punctuation from each token	
    table = str.maketrans('', '', punctuation)	
    tokens = [w.translate(table) for w in tokens]	
    # remove remaining tokens that are not alphabetic	
    tokens = [word for word in tokens if word.isalpha()]	
    # filter out stop words	
    stop_words = set(stopwords.words('english'))	
    tokens = [w for w in tokens if not w in stop_words]	
    # filter out short tokens	
    tokens = [word for word in tokens if len(word) > 1]	
    return tokens 
# save list to file
def save_list(lines, filename):	
    data = '\n'.join(lines)	
    file = open(filename, 'w')	
    file.write(data)	
    file.close() 
# load doc, clean and return line of tokens
def doc_to_line(filename, vocab):	
    # load the doc	
    doc = load_doc(filename)	
    # clean doc	
    tokens = clean_doc(doc)	
    # filter by vocab	
    tokens = [w for w in tokens if w in vocab]	
    return ' '.join(tokens) 
# load all docs in a directory
def process_docs(directory, vocab):	
    lines = list()	
    # walk through all files in the folder	
    for filename in listdir(directory):		
        # skip files that do not have the right extension		
        if not filename.endswith(".txt"):			
            continue		
        # create the full path of the file to open		path = directory + '/' + filename		
        # load and clean the doc		
        line = doc_to_line(path, vocab)		
        # add to list		
        lines.append(line)	
    return lines 
# load vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
# prepare negative reviews
negative_lines = process_docs('txt_sentoken/neg', vocab)
save_list(negative_lines, 'negative.txt')
# prepare positive reviews
positive_lines = process_docs('txt_sentoken/pos', vocab)
save_list(positive_lines, 'positive.txt')
```

Running the example saves two new files, ‘*negative.txt*‘ and ‘*positive.txt*‘, that contain the prepared negative and positive reviews respectively.

The data is ready for use in a bag-of-words or even word embedding model.

## Extensions

This section lists some extensions that you may wish to explore.

- **Stemming**. We could reduce each word in documents to their stem using a stemming algorithm like the Porter stemmer.
- **N-Grams**. Instead of working with individual words, we could work with a vocabulary of word pairs, called bigrams. We could also investigate the use of larger groups, such as triplets (trigrams) and more (n-grams).
- **Encode Words**. Instead of saving tokens as-is, we could save the integer encoding of the words, where the index of the word in the vocabulary represents a unique integer number for the word. This will make it easier to work with the data when modeling.
- **Encode Documents**. Instead of saving tokens in documents, we could encode the documents using a bag-of-words model and encode each word as a boolean present/absent flag or use more sophisticated scoring, such as TF-IDF.

If you try any of these extensions, I’d love to know.
Share your results in the comments below.

## Further Reading

This section provides more resources on the topic if you are looking go deeper.

### Dataset

- [Movie Review Data](http://www.cs.cornell.edu/people/pabo/movie-review-data/)
- [A Sentimental Education: Sentiment Analysis Using Subjectivity Summarization Based on Minimum Cuts](http://xxx.lanl.gov/abs/cs/0409058), 2004.
- [Movie Review Polarity Dataset](http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz) (.tgz)
- Dataset Readme [v2.0](http://www.cs.cornell.edu/people/pabo/movie-review-data/poldata.README.2.0.txt) and [v1.1](http://www.cs.cornell.edu/people/pabo/movie-review-data/README.1.1).

### APIs

- [nltk.tokenize package API](http://www.nltk.org/api/nltk.tokenize.html)
- [Chapter 2, Accessing Text Corpora and Lexical Resources](http://www.nltk.org/book/ch02.html)
- [os API Miscellaneous operating system interfaces](https://docs.python.org/3/library/os.html)
- [collections API – Container datatypes](https://docs.python.org/3/library/collections.html)

## Summary

In this tutorial, you discovered how to prepare movie review text data for sentiment analysis, step-by-step.

Specifically, you learned:

- How to load text data and clean it to remove punctuation and other non-words.
- How to develop a vocabulary, tailor it, and save it to file.
- How to prepare movie reviews using cleaning and a predefined vocabulary and save them to new files ready for modeling.

Do you have any questions?
Ask your questions in the comments below and I will do my best to answer.