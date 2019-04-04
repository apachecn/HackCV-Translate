# Introducing prose v2.0.0: Bringing NLP to Go

A guide to using Go for natural language processing (NLP).

We’re pleased to announce the [v2.0.0 release](https://github.com/jdkato/prose/) of `prose`, a natural language processing (NLP) library for Go.

v2.0.0 represents a major shift in the project’s focus: instead of simply offering an assortment of prose-related utilities, we’re focusing on bringing a more refined NLP experience to Go. This means that the development of v1.0.0’s higher-level features (e.g., the title-case converter) will be moved to other repositories going forward.

> In order to avoid breaking code already importing prose, v2.0.0 will be exposed via gopkg.in/jdkato/prose.v2— allowing github.com/jdkato/prose to still point to v1.0.0.

Among the new features of v2.0.0 is a new, more cohesive API built around `Documents`.



The document-creation process consists of four steps — tokenization, segmentation, POS tagging, and named-entity extraction — which are discussed in more detail below.

### Tokenization

Given a piece of text, tokenization is the task of breaking it up into units referred to as tokens. For example,



And while there’s really no “correct” way to tokenize text, you definitely need to do more than identify word boundaries to be useful. Some examples of non-word tokens that `prose` can identify are given below.



So, for example, a sentence like



becomes



### Segmentation

Text segmentation is the process of dividing text into sentences. This is generally a more challenging task than tokenization due to the ambiguity of sentence boundaries. Fortunately, the developers of the [pragmatic_segmenter](https://github.com/diasks2/pragmatic_segmenter#the-golden-rules) have complied a [test suite of edge-case scenarios](https://github.com/diasks2/pragmatic_segmenter#the-golden-rules) that can be used to evaluate segmenters. Their results are as follows (with prose added):



As you can see, `prose` performed relatively well. Most of its missed cases (Golden Rules 31-39) were list-containing sentences, which seem to be pretty rare.

### Part-of-Speech (POS) Tagging

POS tagging is the process of assigning part-of-speech tags (e.g., `NN` for nouns) to individual tokens. `prose` includes a POS tagger based on Matthew Honnibal’s [Averaged Perceptron implmentation](https://explosion.ai/blog/part-of-speech-pos-tagger-in-python).

To evaluate the performance of our tagger, we used the portion of the University of Pennsylvania’s [Treebank-3 corpus](https://catalog.ldc.upenn.edu/ldc99t42) available through Python’s [NLTK](https://www.nltk.org/) library ([see here](https://github.com/jdkato/prose/blob/v2/scripts/test_model.py) for the test script).



> † Given a list of reference values and a corresponding list of test values, return the fraction of corresponding values that are equal.

> source: NLTK

### Named-Entity Recognition (NER)

NER is the process of assigning labels to particular entities within text (e.g., people, places, organizations, etc.). v2.0.0 includes a much improved version of v1.0.0's `chunk` package, which can identify people (`PERSON`) and geographical/political Entities (`GPE`) by default.



This generally works pretty well. However, instead of focusing on fine-tuning the default model, we’ve put a lot of effort into making it easy to train your own models for specific use cases — for instance, maybe you want to be able to identify all Apple products as `APPLE`.

To train a new model, all you need to do is provide a slice of `LabeledEntities`:



Keep a look out for our next post, which will cover training a new `prose`-compatible NER model using [Prodigy](https://prodi.gy/).

### Going Forward

`prose` started out as simply an assortment of prose-related utilities we needed for [Vale](https://github.com/errata-ai/vale), however, its goals are now more inline with what JavaScript’s [compromise](https://github.com/spencermountain/compromise) has accomplished: being a relatively simple, yet practical NLP library.

The next major step for the project is to add support for text classification, which will allow us to label text as being related to certain topics. If you’d like to get involved, head over to the [GitHub repository](https://github.com/jdkato/prose).

