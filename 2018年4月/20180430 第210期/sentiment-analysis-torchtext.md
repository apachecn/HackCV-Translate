# Sentiment Analysis — TorchText

![](https://cdn-images-1.medium.com/max/1600/1*0ITnfX60UAdBlZ26DGNUag.jpeg)

This post is the second part of the series. In the [first part](https://medium.com/@sonicboom8/sentiment-analysis-with-variable-length-sequences-in-pytorch-6241635ae130) I built sentiment analysis model in pure pytorch. In this post I do the same task but in[ torchtext](https://github.com/pytorch/text) and demonstrate where torchtext shines and also it reduces a lot of code.

Sentiment analysis is a classification task where each sample is assigned a positive or negative label. You can follow along with the code [here](https://github.com/hpanwar08/sentiment-analysis-torchtext).

Download dataset from [2]

#### Typical components of classification task in NLP

1. Preprocessing and tokenization

2. Generating vocabulary of unique tokens and converting words to indices

3. Loading pretrained vectors e.g. Glove, Word2vec, Fasttext

4. Padding text with zeros in case of variable lengths

5. Dataloading and batching

6. Model creation and training

#### Why use torchtext

Torchtext provides set of classes that are useful in NLP tasks. These classes takes care of first 5 points above with very minimal code. We will look into each of the point in detail.

I have split the data in train and validation set and saved as csv.

> Note: Make sure to remove all the ‘\n’ character before saving the csv as torchtext have trouble handling ‘\n’ character.

#### 1. Define how to process data

![](https://cdn-images-1.medium.com/max/1600/1*adpPfDvsvSrCcH8R9zXnWA.png)

The first step is to declare what attributes (columns) in the dataframe we want to use and how to process them. The dataframe consists of 4 columns (‘ItemID’, ‘Sentiment’, ‘SentimentSource’, ‘SentimentText’) and we want to use only ‘Sentiment’ and ‘SentimentText’.

The label column (‘Sentiment’) is binary and already in numerical form, so there is no need to process it. The tweet column (‘SentimentText’) needs processing and tokenization, so that it can be converted into indices.



Let’s break down the above code. In torchtext, a column can be called as field. The field object takes arguments on how to process (tokenize etc.) the text. This field object will later be attached to dataset.

Line 10 defines the blueprint of how a column or field will be handled when we pass the actual data (tweets) in the future.











Line 15 defines the blueprint of how a column or field will be handled when we pass the actual data (labels) in the future.







In line 20 we define how each column will be processed. The columns with None value will be ignored and will not be loaded.

#### 2. Create torchtext dataset

You may ask why not use default pytorch Dataset. The reason is torchtext provide a set of datasets specifically for NLP tasks. One such dataset is TabularDataset which is specially designed to read csv and tsv files and process them. It is a wrapper around pytorch Dataset with additional features.

So far we have defined the blueprints for processing. Now we will actually load the data for processing.

Line 27 contains TabularDataset.splits() method is used when we want to process multiple files (train, validation, test) in one go that uses same processing.













TabularDataset.splits() will return train dataset and validation dataset.

Let’s look at what this TabularDataset object contains.



TabularDataset is a list which contains Example object. Example object wraps all the columns (text and labels) in single object. These columns can be accessed by column names as written in the above code.

#### 3. Load pretrained word vectors and building vocabulary

Torchtext makes loading of pretrained word vectors very easy. Just mention the name of the pretrained word vector (e.g. glove.6B.50d, fasttext.en.300d, etc.) and torchtext will download that particular vector and then you can use it in embedding layer.

Or if you have already downloaded pretrained vectors then you can specify the path and torchtext will read and load it. I have used downloaded vectors in the code below.



In line 4 and 6 above, torchtext build the vocabulary based on the text that is provided in column “SentimentText”. Vocabulary is built on the text in train dataset and validation dataset and define max number of unique words as 100000. Words which are not in vocabulary will be assigned <unk> token. Pass the pretrained vectors during vocabulary building.

Now when you execute line 4, torchtext creates a dictionary of all unique words and arrange them in decreasing order of their frequency and adds <unk> and <pad> token at the beginning of this dictionary. Next torchtext assign unique integer to each word and keep this mapping in txt_field.vocab.stoi (string to index) and reverse mapping in txt_field.vocab.itos (index to string).

#### 4. Loading the data in batches

For data with variable length sentences torchtext provides BucketIterator() dataloader which is wrapper around pytorch Dataloader. BucketIterator provides some additional benefits like sorting the data according to length of the text and group together similar length text in a batch. This helps reduce the amount of padding required. Note that I have not fixed the text to any particular length. BucketIterator pads the batch according the maximum length sample.



> Note: BucketIterator returns a Batch object instead of text index and labels. Also Batch object is not iterable like pytorch Dataloader. A single Batch object contains the data of one batch .The text and labels can be accessed via column names.

This is one of the small hiccups in torchtext. But this can be easily overcome in two ways. Either write some extra code in the training loop for getting the data out of the Batch object or write a iterable wrapper around Batch Object that returns the desired data. I will take the second approach as this is much cleaner.



With the code above we can directly use it in the training loop just like pytorch Dataloader.

#### 5. Finally Model and training

Below is the code for [ConcatPooling](https://arxiv.org/abs/1801.06146) model along with pretrained embedding. I have excluded the training loop code.

![](https://cdn-images-1.medium.com/max/1600/1*qJggHpIPUkkzG0KQZ-EUcQ.jpeg)



#### Other classes in torchtext

Torchtext also provide classes for loading other type of data like for language modelling, sequence tagging, translation etc.

This wraps up the short discussion on torchtext for sentiment analysis task. It was an overview of torchtext. In the next post I will discuss about implementing Attention for sentence classification.

#### References

[1] [https://arxiv.org/abs/1801.06146](https://arxiv.org/abs/1801.06146)

[2] [http://thinknook.com/wp-content/uploads/2012/09/Sentiment-Analysis-Dataset.zip](http://thinknook.com/wp-content/uploads/2012/09/Sentiment-Analysis-Dataset.zip)

[3] [http://anie.me/On-Torchtext/](http://anie.me/On-Torchtext/)

