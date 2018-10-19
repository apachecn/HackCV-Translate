# How to Prepare Movie Review Data for Sentiment Analysis

by [Jason Brownlee](https://machinelearningmastery.com/author/jasonb/) on October 16, 2017 in [Deep Learning for Natural Language Processing](https://machinelearningmastery.com/category/natural-language-processing/)









Text data preparation is different for each problem.

Preparation starts with simple steps, like loading data, but quickly gets difficult with cleaning tasks that are very specific to the data you are working with. You need help as to where to begin and what order to work through the steps from raw data to data ready for modeling.

In this tutorial, you will discover how to prepare movie review text data for sentiment analysis, step-by-step.

After completing this tutorial, you will know:

- How to load text data and clean it to remove punctuation and other non-words.
- How to develop a vocabulary, tailor it, and save it to file.
- How to prepare movie reviews using cleaning and a pre-defined vocabulary and save them to new files ready for modeling.

Let’s get started.

- **Update Oct/2017**: Fixed a small bug when skipping non-matching files, thanks Jan Zett.
- **Update Dec/2017**: Fixed a small typo in full example, thanks Ray and Zain.



How to Prepare Movie Review Data for Sentiment Analysis
Photo by [Kenneth Lu](https://www.flickr.com/photos/toasty/1125019024/), some rights reserved.

## Tutorial Overview

This tutorial is divided into 5 parts; they are:

1. Movie Review Dataset
2. Load Text Data
3. Clean Text Data
4. Develop Vocabulary
5. Save Prepared Data







### Need help with Deep Learning for Text Data?

Take my free 7-day email crash course now (with code).

Click to sign-up and also get a free PDF Ebook version of the course.

[Start Your FREE Crash-Course Now](https://machinelearningmastery.lpages.co/leadbox/144855173f72a2%3A164f8be4f346dc/5655638436741120/)







## 1. Movie Review Dataset

The Movie Review Data is a collection of movie reviews retrieved from the imdb.com website in the early 2000s by Bo Pang and Lillian Lee. The reviews were collected and made available as part of their research on natural language processing.

The reviews were originally released in 2002, but an updated and cleaned up version was released in 2004, referred to as “*v2.0*“.

The dataset is comprised of 1,000 positive and 1,000 negative movie reviews drawn from an archive of the rec.arts.movies.reviews newsgroup hosted at [IMDB](http://reviews.imdb.com/Reviews). The authors refer to this dataset as the “*polarity dataset*“.

> Our data contains 1000 positive and 1000 negative reviews all written before 2002, with a cap of 20 reviews per author (312 authors total) per category. We refer to this corpus as the polarity dataset.

— [A Sentimental Education: Sentiment Analysis Using Subjectivity Summarization Based on Minimum Cuts](http://xxx.lanl.gov/abs/cs/0409058), 2004.

The data has been cleaned up somewhat, for example:

- The dataset is comprised of only English reviews.
- All text has been converted to lowercase.
- There is white space around punctuation like periods, commas, and brackets.
- Text has been split into one sentence per line.

The data has been used for a few related natural language processing tasks. For classification, the performance of classical models (such as Support Vector Machines) on the data is in the range of high 70% to low 80% (e.g. 78%-to-82%).

More sophisticated data preparation may see results as high as 86% with 10-fold cross validation. This gives us a ballpark of low-to-mid 80s if we were looking to use this dataset in experiments on modern methods.

> … depending on choice of downstream polarity classifier, we can achieve highly statistically significant improvement (from 82.8% to 86.4%)

— [A Sentimental Education: Sentiment Analysis Using Subjectivity Summarization Based on Minimum Cuts](http://xxx.lanl.gov/abs/cs/0409058), 2004.

You can download the dataset from here:

- [Movie Review Polarity Dataset](http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz) (review_polarity.tar.gz, 3MB)

After unzipping the file, you will have a directory called “*txt_sentoken*” with two sub-directories containing the text “*neg*” and “*pos*” for negative and positive reviews. Reviews are stored one per file with a naming convention *cv000* to *cv999* for each of neg and pos.

Next, let’s look at loading the text data.

## 2. Load Text Data

In this section, we will look at loading individual text files, then processing the directories of files.

We will assume that the review data is downloaded and available in the current working directory in the folder “*txt_sentoken*“.

We can load an individual text file by opening it, reading in the ASCII text, and closing the file. This is standard file handling stuff. For example, we can load the first negative review file “*cv000_29416.txt*” as follows:



| 12345678 | # load one filefilename = 'txt_sentoken/neg/cv000_29416.txt'# open the file as read onlyfile = open(filename, 'r')# read all texttext = file.read()# close the filefile.close() |
| -------- | ------------------------------------------------------------ |
|          |                                                              |

This loads the document as ASCII and preserves any white space, like new lines.

We can turn this into a function called load_doc() that takes a filename of the document to load and returns the text.



| 123456789 | # load doc into memorydef load_doc(filename):	# open the file as read only	file = open(filename, 'r')	# read all text	text = file.read()	# close the file	file.close()	return text |
| --------- | ------------------------------------------------------------ |
|           |                                                              |

We have two directories each with 1,000 documents each. We can process each directory in turn by first getting a list of files in the directory using the [listdir() function](https://docs.python.org/3/library/os.html#os.listdir), then loading each file in turn.

For example, we can load each document in the negative directory using the *load_doc()*function to do the actual loading.



| 123456789101112131415161718192021222324 | from os import listdir # load doc into memorydef load_doc(filename):	# open the file as read only	file = open(filename, 'r')	# read all text	text = file.read()	# close the file	file.close()	return text # specify directory to loaddirectory = 'txt_sentoken/neg'# walk through all files in the folderfor filename in listdir(directory):	# skip files that do not have the right extension	if not filename.endswith(".txt"):		continue	# create the full path of the file to open	path = directory + '/' + filename	# load document	doc = load_doc(path)	print('Loaded %s' % filename) |
| --------------------------------------- | ------------------------------------------------------------ |
|                                         |                                                              |

Running this example prints the filename of each review after it is loaded.



| 123456 | ...Loaded cv995_23113.txtLoaded cv996_12447.txtLoaded cv997_5152.txtLoaded cv998_15691.txtLoaded cv999_14636.txt |
| ------ | ------------------------------------------------------------ |
|        |                                                              |

We can turn the processing of the documents into a function as well and use it as a template later for developing a function to clean all documents in a folder. For example, below we define a *process_docs()* function to do the same thing.



| 12345678910111213141516171819202122232425262728 | from os import listdir # load doc into memorydef load_doc(filename):	# open the file as read only	file = open(filename, 'r')	# read all text	text = file.read()	# close the file	file.close()	return text # load all docs in a directorydef process_docs(directory):	# walk through all files in the folder	for filename in listdir(directory):		# skip files that do not have the right extension		if not filename.endswith(".txt"):			continue		# create the full path of the file to open		path = directory + '/' + filename		# load document		doc = load_doc(path)		print('Loaded %s' % filename) # specify directory to loaddirectory = 'txt_sentoken/neg'process_docs(directory) |
| ----------------------------------------------- | ------------------------------------------------------------ |
|                                                 |                                                              |

Now that we know how to load the movie review text data, let’s look at cleaning it.

## 3. Clean Text Data

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



| 1234567891011121314151617181920212223242526272829303132333435363738394041424344454647484950515253545556575859606162636465666768697071727374 | from string import punctuationfrom os import listdirfrom collections import Counterfrom nltk.corpus import stopwords # load doc into memorydef load_doc(filename):	# open the file as read only	file = open(filename, 'r')	# read all text	text = file.read()	# close the file	file.close()	return text # turn a doc into clean tokensdef clean_doc(doc):	# split into tokens by white space	tokens = doc.split()	# remove punctuation from each token	table = str.maketrans('', '', punctuation)	tokens = [w.translate(table) for w in tokens]	# remove remaining tokens that are not alphabetic	tokens = [word for word in tokens if word.isalpha()]	# filter out stop words	stop_words = set(stopwords.words('english'))	tokens = [w for w in tokens if not w in stop_words]	# filter out short tokens	tokens = [word for word in tokens if len(word) > 1]	return tokens # load doc and add to vocabdef add_doc_to_vocab(filename, vocab):	# load doc	doc = load_doc(filename)	# clean doc	tokens = clean_doc(doc)	# update counts	vocab.update(tokens) # load all docs in a directorydef process_docs(directory, vocab):	# walk through all files in the folder	for filename in listdir(directory):		# skip files that do not have the right extension		if not filename.endswith(".txt"):			continue		# create the full path of the file to open		path = directory + '/' + filename		# add doc to vocab		add_doc_to_vocab(path, vocab) # save list to filedef save_list(lines, filename):	data = '\n'.join(lines)	file = open(filename, 'w')	file.write(data)	file.close() # define vocabvocab = Counter()# add all docs to vocabprocess_docs('txt_sentoken/neg', vocab)process_docs('txt_sentoken/pos', vocab)# print the size of the vocabprint(len(vocab))# print the top words in the vocabprint(vocab.most_common(50))# keep tokens with > 5 occurrencemin_occurane = 5tokens = [k for k,c in vocab.items() if c >= min_occurane]print(len(tokens))# save tokens to a vocabulary filesave_list(tokens, 'vocab.txt') |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
|                                                              |                                                              |

Running this final snippet after creating the vocabulary will save the chosen words to file.

It is a good idea to take a look at, and even study, your chosen vocabulary in order to get ideas for better preparing this data, or text data in the future.



| 1234567891011 | hasntupdatingfigurativelysymphonyciviliansmightfishermanhokumwitchbuffoons... |
| ------------- | ------------------------------------------------------------ |
|               |                                                              |

Next, we can look at using the vocabulary to create a prepared version of the movie review dataset.

## 5. Save Prepared Data

We can use the data cleaning and chosen vocabulary to prepare each movie review and save the prepared versions of the reviews ready for modeling.

This is a good practice as it decouples the data preparation from modeling, allowing you to focus on modeling and circle back to data prep if you have new ideas.

We can start off by loading the vocabulary from ‘*vocab.txt*‘.



| 123456789101112131415 | # load doc into memorydef load_doc(filename):	# open the file as read only	file = open(filename, 'r')	# read all text	text = file.read()	# close the file	file.close()	return text # load vocabularyvocab_filename = 'review_polarity/vocab.txt'vocab = load_doc(vocab_filename)vocab = vocab.split()vocab = set(vocab) |
| --------------------- | ------------------------------------------------------------ |
|                       |                                                              |

Next, we can clean the reviews, use the loaded vocab to filter out unwanted tokens, and save the clean reviews in a new file.

One approach could be to save all the positive reviews in one file and all the negative reviews in another file, with the filtered tokens separated by white space for each review on separate lines.

First, we can define a function to process a document, clean it, filter it, and return it as a single line that could be saved in a file. Below defines the *doc_to_line()* function to do just that, taking a filename and vocabulary (as a set) as arguments.

It calls the previously defined *load_doc()* function to load the document and *clean_doc()* to tokenize the document.



| 123456789 | # load doc, clean and return line of tokensdef doc_to_line(filename, vocab):	# load the doc	doc = load_doc(filename)	# clean doc	tokens = clean_doc(doc)	# filter by vocab	tokens = [w for w in tokens if w in vocab]	return ' '.join(tokens) |
| --------- | ------------------------------------------------------------ |
|           |                                                              |

Next, we can define a new version of *process_docs()* to step through all reviews in a folder and convert them to lines by calling *doc_to_line()* for each document. A list of lines is then returned.



| 123456789101112131415 | # load all docs in a directorydef process_docs(directory, vocab):	lines = list()	# walk through all files in the folder	for filename in listdir(directory):		# skip files that do not have the right extension		if not filename.endswith(".txt"):			continue		# create the full path of the file to open		path = directory + '/' + filename		# load and clean the doc		line = doc_to_line(path, vocab)		# add to list		lines.append(line)	return lines |
| --------------------- | ------------------------------------------------------------ |
|                       |                                                              |

We can then call *process_docs()* for both the directories of positive and negative reviews, then call *save_list()* from the previous section to save each list of processed reviews to a file.

The complete code listing is provided below.



| 123456789101112131415161718192021222324252627282930313233343536373839404142434445464748495051525354555657585960616263646566676869707172737475 | from string import punctuationfrom os import listdirfrom collections import Counterfrom nltk.corpus import stopwords # load doc into memorydef load_doc(filename):	# open the file as read only	file = open(filename, 'r')	# read all text	text = file.read()	# close the file	file.close()	return text # turn a doc into clean tokensdef clean_doc(doc):	# split into tokens by white space	tokens = doc.split()	# remove punctuation from each token	table = str.maketrans('', '', punctuation)	tokens = [w.translate(table) for w in tokens]	# remove remaining tokens that are not alphabetic	tokens = [word for word in tokens if word.isalpha()]	# filter out stop words	stop_words = set(stopwords.words('english'))	tokens = [w for w in tokens if not w in stop_words]	# filter out short tokens	tokens = [word for word in tokens if len(word) > 1]	return tokens # save list to filedef save_list(lines, filename):	data = '\n'.join(lines)	file = open(filename, 'w')	file.write(data)	file.close() # load doc, clean and return line of tokensdef doc_to_line(filename, vocab):	# load the doc	doc = load_doc(filename)	# clean doc	tokens = clean_doc(doc)	# filter by vocab	tokens = [w for w in tokens if w in vocab]	return ' '.join(tokens) # load all docs in a directorydef process_docs(directory, vocab):	lines = list()	# walk through all files in the folder	for filename in listdir(directory):		# skip files that do not have the right extension		if not filename.endswith(".txt"):			continue		# create the full path of the file to open		path = directory + '/' + filename		# load and clean the doc		line = doc_to_line(path, vocab)		# add to list		lines.append(line)	return lines # load vocabularyvocab_filename = 'vocab.txt'vocab = load_doc(vocab_filename)vocab = vocab.split()vocab = set(vocab)# prepare negative reviewsnegative_lines = process_docs('txt_sentoken/neg', vocab)save_list(negative_lines, 'negative.txt')# prepare positive reviewspositive_lines = process_docs('txt_sentoken/pos', vocab)save_list(positive_lines, 'positive.txt') |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
|                                                              |                                                              |

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