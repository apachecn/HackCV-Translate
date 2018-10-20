# Best Practices for Document Classification with Deep Learning

原文链接：[Best Practices for Document Classification with Deep Learning](https://machinelearningmastery.com/best-practices-document-classification-deep-learning/?from=hackcv&hmsr=hackcv.com&utm_medium=hackcv.com&utm_source=hackcv.com)

Text classification describes a general class of problems such as predicting the sentiment of tweets and movie reviews, as well as classifying email as spam or not.

Deep learning methods are proving very good at text classification, achieving state-of-the-art results on a suite of standard academic benchmark problems.

In this post, you will discover some best practices to consider when developing deep learning models for text classification.

After reading this post, you will know:

- The general combination of deep learning methods to consider when starting your text classification problems.
- The first architecture to try with specific advice on how to configure hyperparameters.
- That deeper networks may be the future of the field in terms of flexibility and capability.

Let’s get started.



Best Practices for Document Classification with Deep Learning
Photo by [storebukkebruse](https://www.flickr.com/photos/tusnelda/6140792529/), some rights reserved.

## Overview

This tutorial is divided into 5 parts; they are:

1. Word Embeddings + CNN = Text Classification
2. Use a Single Layer CNN Architecture
3. Dial in CNN Hyperparameters
4. Consider Character-Level CNNs
5. Consider Deeper CNNs for Classification







### Need help with Deep Learning for Text Data?

Take my free 7-day email crash course now (with code).

Click to sign-up and also get a free PDF Ebook version of the course.

[Start Your FREE Crash-Course Now](https://machinelearningmastery.lpages.co/leadbox/144855173f72a2%3A164f8be4f346dc/5655638436741120/)







## 1. Word Embeddings + CNN = Text Classification

The modus operandi for text classification involves the use of a word embedding for representing words and a Convolutional Neural Network (CNN) for learning how to discriminate documents on classification problems.

Yoav Goldberg, in his primer on deep learning for natural language processing, comments that neural networks in general offer better performance than classical linear classifiers, especially when used with pre-trained word embeddings.

> The non-linearity of the network, as well as the ability to easily integrate pre-trained word embeddings, often lead to superior classification accuracy.

— [A Primer on Neural Network Models for Natural Language Processing](https://arxiv.org/abs/1510.00726), 2015.

He also comments that convolutional neural networks are effective at document classification, namely because they are able to pick out salient features (e.g. tokens or sequences of tokens) in a way that is invariant to their position within the input sequences.

> Networks with convolutional and pooling layers are useful for classification tasks in which we expect to find strong local clues regarding class membership, but these clues can appear in different places in the input. […] We would like to learn that certain sequences of words are good indicators of the topic, and do not necessarily care where they appear in the document. Convolutional and pooling layers allow the model to learn to find such local indicators, regardless of their position.

— [A Primer on Neural Network Models for Natural Language Processing](https://arxiv.org/abs/1510.00726), 2015.

The architecture is therefore comprised of three key pieces:

1. **Word Embedding**: A distributed representation of words where different words that have a similar meaning (based on their usage) also have a similar representation.
2. **Convolutional Model**: A feature extraction model that learns to extract salient features from documents represented using a word embedding.
3. **Fully Connected Model**: The interpretation of extracted features in terms of a predictive output.

Yoav Goldberg highlights the CNNs role as a feature extractor model in his book:

> … the CNN is in essence a feature-extracting architecture. It does not constitute a standalone, useful network on its own, but rather is meant to be integrated into a larger network, and to be trained to work in tandem with it in order to produce an end result. The CNNs layer’s responsibility is to extract meaningful sub-structures that are useful for the overall prediction task at hand.

— Page 152, [Neural Network Methods for Natural Language Processing](http://amzn.to/2vRopQz), 2017.

The tying together of these three elements is demonstrated in perhaps one of the most widely cited examples of the combination, described in the next section.

## 2. Use a Single Layer CNN Architecture

You can get good results for document classification with a single layer CNN, perhaps with differently sized kernels across the filters to allow grouping of word representations at different scales.

Yoon Kim in his study of the use of pre-trained word vectors for classification tasks with Convolutional Neural Networks found that using pre-trained static word vectors does very well. He suggests that pre-trained word embeddings that were trained on very large text corpora, such as the freely available word2vec vectors trained on 100 billion tokens from Google news may offer good universal features for use in natural language processing.

> Despite little tuning of hyperparameters, a simple CNN with one layer of convolution performs remarkably well. Our results add to the well-established evidence that unsupervised pre-training of word vectors is an important ingredient in deep learning for NLP

— [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882), 2014.

He also discovered that further task-specific tuning of the word vectors offer a small additional improvement in performance.

Kim describes the general approach of using CNN for natural language processing. Sentences are mapped to embedding vectors and are available as a matrix input to the model. Convolutions are performed across the input word-wise using differently sized kernels, such as 2 or 3 words at a time. The resulting feature maps are then processed using a max pooling layer to condense or summarize the extracted features.

The architecture is based on the approach used by Ronan Collobert, et al. in their paper “[Natural Language Processing (almost) from Scratch](https://arxiv.org/abs/1103.0398)“, 2011. In it, they develop a single end-to-end neural network model with convolutional and pooling layers for use across a range of fundamental natural language processing problems.

Kim provides a diagram that helps to see the sampling of the filters using differently sized kernels as different colors (red and yellow).



An example of a CNN Filter and Polling Architecture for Natural Language Processing.
Taken from “Convolutional Neural Networks for Sentence Classification”, 2014.

Usefully, he reports his chosen model configuration, discovered via grid search and used across a suite of 7 text classification tasks, summarized as follows:

- Transfer function: rectified linear.
- Kernel sizes: 2, 4, 5.
- Number of filters: 100
- Dropout rate: 0.5
- Weight regularization (L2): 3
- Batch Size: 50
- Update Rule: Adadelta

These configurations could be used to inspire a starting point for your own experiments.

## 3. Dial in CNN Hyperparameters

Some hyperparameters matter more than others when tuning a convolutional neural network on your document classification problem.

Ye Zhang and Byron Wallace performed a sensitivity analysis into the hyperparameters needed to configure a single layer convolutional neural network for document classification. The study is motivated by their claim that the models are sensitive to their configuration.

> Unfortunately, a downside to CNN-based models – even simple ones – is that they require practitioners to specify the exact model architecture to be used and to set the accompanying hyperparameters. To the uninitiated, making such decisions can seem like something of a black art because there are many free parameters in the model.

— [A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1510.03820), 2015.

Their aim was to provide general configurations that can be used for configuring CNNs on new text classification tasks.

They provide a nice depiction of the model architecture and the decision points for configuring the model, reproduced below.



Convolutional Neural Network Architecture for Sentence Classification
Taken from “*A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural Networks for Sentence Classification*“, 2015.

The study makes a number of useful findings that could be used as a starting point for configuring shallow CNN models for text classification.

The general findings were as follows:

- The choice of pre-trained word2vec and GloVe embeddings differ from problem to problem, and both performed better than using one-hot encoded word vectors.
- The size of the kernel is important and should be tuned for each problem.
- The number of feature maps is also important and should be tuned.
- The 1-max pooling generally outperformed other types of pooling.
- Dropout has little effect on the model performance.

They go on to provide more specific heuristics, as follows:

- Use word2vec or GloVe word embeddings as a starting point and tune them while fitting the model.
- Grid search across different kernel sizes to find the optimal configuration for your problem, in the range 1-10.
- Search the number of filters from 100-600 and explore a dropout of 0.0-0.5 as part of the same search.
- Explore using tanh, relu, and linear activation functions.

The key caveat is that the findings are based on empirical results on binary text classification problems using single sentences as input.

I recommend reading the full paper to get more details:

- [A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1510.03820), 2015.

## 4. Consider Character-Level CNNs

Text documents can be modeled at the character level using convolutional neural networks that are capable of learning the relevant hierarchical structure of words, sentences, paragraphs, and more.

Xiang Zhang, et al. use a character-based representation of text as input for a convolutional neural network. The promise of the approach is that all of the labor-intensive effort required to clean and prepare text could be overcome if a CNN can learn to abstract the salient details.

> … deep ConvNets do not require the knowledge of words, in addition to the conclusion from previous research that ConvNets do not require the knowledge about the syntactic or semantic structure of a language. This simplification of engineering could be crucial for a single system that can work for different languages, since characters always constitute a necessary construct regardless of whether segmentation into words is possible. Working on only characters also has the advantage that abnormal character combinations such as misspellings and emoticons may be naturally learnt.

— [Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626), 2015.

The model reads in one-hot encoded characters in a fixed-sized alphabet. Encoded characters are read in blocks or sequences of 1,024 characters. A stack of 6 convolutional layers with pooling follows, with 3 fully connected layers at the output end of the network in order to make a prediction.



Character-based Convolutional Neural Network for Text Classification
Taken from “*Character-level Convolutional Networks for Text Classification*“, 2015.

The model achieves some success, performing better on problems that offer a larger corpus of text.

> … analysis shows that character-level ConvNet is an effective method. […] how well our model performs in comparisons depends on many factors, such as dataset size, whether the texts are curated and choice of alphabet.

— [Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626), 2015.

Results using an extended version of this approach were pushed to the state-of-the-art in a follow-up paper covered in the next section.

## 5. Consider Deeper CNNs for Classification

Better performance can be achieved with very deep convolutional neural networks, although standard and reusable architectures have not been adopted for classification tasks, yet.

Alexis Conneau, et al. comment on the relatively shallow networks used for natural language processing and the success of much deeper networks used for computer vision applications. For example, Kim (above) restricted the model to a single convolutional layer.

Other architectures used for natural language reviewed in the paper are limited to 5 and 6 layers. These are contrasted with successful architectures used in computer vision with 19 or even up to 152 layers.

They suggest and demonstrate that there are benefits for hierarchical feature learning with very deep convolutional neural network model, called VDCNN.

> … we propose to use deep architectures of many convolutional layers to approach this goal, using up to 29 layers. The design of our architecture is inspired by recent progress in computer vision […] The proposed deep convolutional network shows significantly better results than previous ConvNets approach.

Key to their approach is an embedding of individual characters, rather than a word embedding.

> We present a new architecture (VDCNN) for text processing which operates directly at the character level and uses only small convolutions and pooling operations.

— [Very Deep Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626), 2016.

Results on a suite of 8 large text classification tasks show better performance than more shallow networks. Specifically, state-of-the-art results on all but two of the datasets tested, at the time of writing.

Generally, they make some key findings from exploring the deeper architectural approach:

- The very deep architecture worked well on small and large datasets.
- Deeper networks decrease classification error.
- Max-pooling achieves better results than other, more sophisticated types of pooling.
- Generally going deeper degrades accuracy; the shortcut connections used in the architecture are important.

> … this is the first time that the “benefit of depths” was shown for convolutional neural networks in NLP.

— [Very Deep Convolutional Networks for Text Classification](https://arxiv.org/abs/1606.01781), 2016.

## Further Reading

This section provides more resources on the topic if you are looking go deeper.

- [A Primer on Neural Network Models for Natural Language Processing](https://arxiv.org/abs/1510.00726), 2015.
- [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1103.0398), 2014.
- [Natural Language Processing (almost) from Scratch](https://arxiv.org/abs/1103.0398), 2011.
- [Very Deep Convolutional Networks for Text Classification](https://arxiv.org/abs/1606.01781), 2016.
- [Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626), 2015.
- [A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1510.03820), 2015.

Have you come across some good resources on deep learning for document classification?
Let me know in the comments below.

## Summary

In this post, you discovered some best practices for developing deep learning models for document classification.

Specifically, you learned:

- That a key approach is to use word embeddings and convolutional neural networks for text classification.
- That a single layer model can do well on moderate-sized problems, and ideas on how to configure it.
- That deeper models that operate directly on text may be the future of natural language processing.

Do you have any questions?
Ask your questions in the comments below and I will do my best to answer.