# First time with Kaggle: A ConvNet to classify toxic comments with Keras

Work has been slow in the first week of the year, so I decided to try my hand at a [Kaggle](https://www.kaggle.com/) competition for the first time (yeah I know I am late to the party). After signing up and looking around, I ended up on the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge). Incase you only browse Medium and have no idea what a toxic comment means, here you go:

![](https://cdn-images-1.medium.com/max/1600/1*wz643XSNzyWT_-BzeKmcCQ.jpeg)

This post describes my (kinda) successful attempt at training a ConvNet to classify a comment into one or more types of toxicity: threat, obscenity, insult, etc. (6 classes in total). Compared to the leader [log-loss](http://www.exegetic.biz/blog/2015/12/making-sense-logarithmic-loss/) of 0.022, my simple model scored ~0.055 — Not amazing, but pretty good for <100 lines of code with Keras! At the end of the post, I will also mention some meta-learnings about my first go at competitive ML :-).

### Preprocessing the Text

The training data was provided as a CSV file with ~100k rows. Each row contained a unique ID, the text, and a 1/0 per class denoting classification.





Being a Deep learning noob, I started writing some kickass preprocessing code in [NLTK](http://www.nltk.org/). Turns out, Keras provides a handy [Tokenizer class](https://keras.io/preprocessing/text/) to deal with all basic tasks such as special-character-removal and conversion to lowercase. So I got lazy and just used that:







### Word Embeddings

For [word embeddings](https://en.wikipedia.org/wiki/Word_embedding), I used the [Glove Twitter](https://nlp.stanford.edu/projects/glove/) vectors with 100 dimensions. Other options were pre-trained vectors from [Word2Vec](https://github.com/mmihaltz/word2vec-GoogleNews-vectors) or [Fasttext](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md). I tried Word2Vec, and like [others](https://arxiv.org/pdf/1703.00993.pdf), Glove worked better for me. I did not get to applying Fasttext, which is a promising prospect — mainly because Fasttext has vectors for ‘fractions’ of words, and that might be useful for misspelt words (or other OOV terms) commonly found in comments.

If you are used to the [Gensim](https://radimrehurek.com/gensim/) Python package like me, you can use [their script](https://radimrehurek.com/gensim/scripts/glove2word2vec.html) to convert the Glove embeddings into word2vec format. Once that is done, the vectors can be loaded pretty easily:





An embedding layer can be defined in Keras as:



Notice the trainable=True part in the above snippet — We could use the embeddings as-they-are, but fine-tuning them during training adjusts their semantic ‘location’ for our particular application. This is basically a form of [Transfer Learning](http://ruder.io/transfer-learning/).

### The CNN

At this point you might be wondering why I used CNNs for a text-understanding task. One, because I had never trained a CNN in Keras (and I wanted to). But well…[this article](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/) gives a good intuition about how 1-dimensional convolutions could be useful for processing text. In 1-D convolutions, you essentially go over patches of words instead of pixels (think about a sliding window of words, a. la. reading). For a visual feel (and to make this post more attractive), I have added this totally-original image:

![](https://cdn-images-1.medium.com/max/1600/1*OkBy2QoL5HkqtLcqBhhE0g.png)

CNNs are not particularly good for most NLP tasks since they lose out on the sequential flow of information. But since the objective here boils down to recognizing ‘blocks’ of sentiments scattered in text, they work decently well!

We use 2 Convolutional+Max-Pooling blocks followed by 3 dense layers:







Sigmoid (and not Softmax) is the more appropriate objective function here, since each sample could belong to multiple classes (A comment could be an insult and obscene at the same time).

I tried using Dropout for regularization, but it did not seem to help with the scores. Therefore, I dropped the idea.

### Training

I found that Adagrad (with its default settings) worked best for this use-case. Keras has support for [various optimizers](https://keras.io/optimizers/), and I did not try tuning parameters such as decay (which could have reduced the error further). For a brief overview of the various optimization techniques out there, look at Ruder’s [awesome blog post](http://ruder.io/optimizing-gradient-descent/).



The binary_crossentropy objective is [Keras’ version](https://keras.io/losses/) of log-loss (so you get the same value). Since I used pre-trained vectors and a dataset of ~85k instances, 2 epochs is enough (based on Keras logs, the loss seems to plateau in the last half of the second epoch itself).

For the sake of brevity, I won’t write out the code I used for inference and building the output file (You could do it easily with model.predict). The overall log-loss computed by Keras turned out to be around 0.055, which is not bad considering this single-model approach.

And now, some random musings on losing my Kaggle virginity:

1. **Complex isn’t always better**: I started off with Adam, but Adagrad turned out to be better. There are multiple StackOverflow discussions on how Adagrad also works better than its extension Adadelta at times. In any case, you shouldn’t be worried about the exact optimizer at the beginning, since a majority of them will usually converge to a good enough value.

2. **Start with the big changes first**: This one did not come to me naturally, and I admit it should have. I have mostly been copy-pasting TensorFlow snippets from other blog posts till now, so I never had to fine-tune my own Neural Network(s). For tuning hyperparameters (when you aren’t using something like [hyperopt](https://github.com/hyperopt/hyperopt)), it is always better to first play with those changes that will have the maximum impact: for example, Number of layers > Momentum value in optimizer. This might not always be true, but a good rule-of-thumb.

3. **Ensembles are king (for better scores)**: Most Kaggle leaders use ensemble frameworks (like XGBoost), or average over outputs from multiple complex models (someone on the discussion boards used an LSTM+CNN).

4. **Kaggle is a good exercise in learning-about-learning:** While there is [valid skepticism](https://www.datascienceweekly.org/articles/5-reasons-kaggle-projects-won-t-help-your-data-science-resume) over how relevant Kaggle experience is to industry-oriented data science, it sure is a good learning experience. Trying out multiple algorithms, reading discussion boards, and simply fine-tuning params (and observing training logs) tells you so much about how deep learning behaves in practice. In fact, with just one go at Kaggle I have learnt quite a few rules-of-thumb that I wouldn’t know otherwise (for example, batch size=32 is [usually a good place to start with](https://arxiv.org/abs/1206.5533). For me, 16 was too slow, and 128 never really converged).

5. **It’s addictive**: Maybe its just me, but I couldn’t stop myself from peeking at the training logs again and again to see how the loss values were behaving for each experiment. This is primarily the reason I won’t do Kaggle during some serious work at the office.

6. **You don’t need theory to do Deep Learning:** My knowledge of deep learning has improved in the past few months, but I still found myself using classes/methods with no clue on how they exactly worked (like the whole 1D Convolutions concept). This is good in a way, since it makes learning accessible to everyone with a good computer and knowledge of Python. That being said, having a knowledge of the theory is good to get started in the right direction, and to be able to apply ML algorithms to non-obvious (read: available on the internet) use cases.

Conventional wisdom says I should make a [Call to Action](https://socialwayne.com/2016/01/25/7-closing-call-to-action-formats-writers-use-on-medium/) right about now. But what bothers me is that there is some possible joke about how bad comments would provide me more data for this model, but I can’t quite figure it out :-(

