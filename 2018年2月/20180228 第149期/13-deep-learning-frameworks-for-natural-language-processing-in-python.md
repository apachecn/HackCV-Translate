# 13 Deep Learning Frameworks for Natural Language Processing in Python

by Olga Davydova

In this paper, we discuss the most popular neural network frameworks and libraries that can be utilized for natural language processing (NLP) in the Python programming language. We also look at existing examples of these tools.

A [comparative table](https://docs.google.com/spreadsheets/d/1brGCIAQUl1LKIMQSaYGnLGXmJZX9jt4PrgR8n-MfPDg/edit?usp=sharing) was specially created. Every cell with a plus sign contains a link to a framework usage example in NLP task and network type perspectives.

![](https://cdn-images-1.medium.com/max/1600/0*BIwC9_4tB1qVjMOn.)

### Chainer

[Chainer](https://chainer.org/), developed by the Japanese company [Preferred Networks](https://www.preferred-networks.jp/en/about) founded in 2014, is a powerful, flexible, and intuitive Python-based framework for neural networks that adopts a [“define-by-run”](https://docs.chainer.org/en/stable/tutorial/basic.html#define-by-run) scheme [[1]](https://chainer.org/). It stores the history of computation instead of programming logic. Chainer supports [CUDA](https://en.wikipedia.org/wiki/CUDA) computation and [multi-GPU](http://www.nvidia.com/object/multi-gpu-technology.html). The framework released under the [MIT License](https://github.com/chainer/chainer/blob/master/LICENSE) and is already applied for sentiment analysis, machine translation, speech recognition, question answering, and so on using different types of neural networks like convolutional networks, recurrent networks, and sequence to sequence models [[2]](http://learningsys.org/papers/LearningSys_2015_paper_33.pdf).

![](https://cdn-images-1.medium.com/max/1600/0*ViGe6-ncZpwp9OOo.)

### Deeplearning4j

[Deeplearning4j](https://deeplearning4j.org/) is a deep learning Java programming library, but it also has a Python API, Keras that will be described below. Distributed CPUs and GPUs, parallel training via iterative reduce, and micro-service architecture adaptation are its main features [[3]](https://deeplearning4j.org/). [Vector space modeling](https://en.wikipedia.org/wiki/Vector_space_model) enables the tool to solve text-mining problems. Parts of speech (PoS) tagging, dependency parsing, and word2vec for creating word embedding are discussed in the [documentation](https://deeplearning4j.org/nlp).

### Deepnl

[Deepnl](https://github.com/attardi/deepnl) is another neural network Python library especially created for natural language processing by [Giuseppe Attardi](https://github.com/attardi). It provides tools for part-of-speech tagging, named entity recognition, semantic role labeling (using convolutional neural networks [[4]](http://www.aclweb.org/anthology/W15-1515)), and word embedding creation [[5]](https://github.com/attardi/deepnl).

![](https://cdn-images-1.medium.com/max/1600/0*8TjI2ds3-Uzh4SuA.)

### Dynet

[Dynet](https://github.com/clab/dynet) is a tool developed by [Carnegie Mellon University](http://www.cmu.edu/) and many others. It supports C++ and Python languages, runs on either CPU or GPU [[6]](https://github.com/clab/dynet). Dynet is based on the dynamic declaration of network structure [[7]](https://arxiv.org/pdf/1701.03980.pdf). This tool was used for creating outstanding systems for NLP problems including syntactic parsing, machine translation, morphological inflection, and many others.

![](https://cdn-images-1.medium.com/max/1600/0*BnrEBs-sSp1kAzKO.)

### Keras

[Keras](https://keras.io/) is a high-level neural-network based Python API that runs on CPU or GPU. It supports convolutional and recurrent networks and may run on top of [TensorFlow](https://www.tensorflow.org/), [CNTK](https://www.microsoft.com/en-us/cognitive-toolkit/), or [Theano](https://github.com/Theano/Theano). The main focus is to enable users fast experimentation [[8]](https://keras.io/). There are many examples of Keras usage in the [comparative table](https://docs.google.com/spreadsheets/d/1Wizmq1RTOdFL75Q8eVu9CwrKCvpa42VOqeNoBnLDx9o/edit?usp=sharing): classification, text generation and summarization, tagging, parsing, machine translation, speech recognition, and others.

### Nlpnet

Erick Rocha Fonseca’s [nlpnet](https://github.com/erickrf/nlpnet) is also a Python library for NLP tasks based on neural networks. Convolutional networks enable users to perform part-of-speech tagging, semantic role labeling, and dependency parsing [[9]](https://github.com/erickrf/nlpnet). Most of the architecture is language independent [[10]](http://nilc.icmc.usp.br/nlpnet/).

![](https://cdn-images-1.medium.com/max/1600/0*yGKdkPKpKVg3hvOk.)

### OpenNMT

[OpenNMT](http://opennmt.net/) is a Python machine translation tool that works under the MIT license and relies on the [PyTorch](http://pytorch.org/) library. The system demonstrates efficiency and state-of-the-art translation accuracy and is used by many translation providers [[11]](http://opennmt.net/). It also incorporates text summarization, speech recognition, and image-to-text conversion blocks [[12]](http://opennmt.net/OpenNMT/applications/).

![](https://cdn-images-1.medium.com/max/1600/0*UnXCz1vmSnwXl3Nf.)

### PyTorch

[PyTorch](http://pytorch.org/) is a fast and flexible neural network framework with an imperative paradigm. It builds neural networks on a tape-based autograd system and provides tensor computation with strong GPU acceleration [[13]](http://pytorch.org/about/). Recurrent neural networks are mostly used in PyTorch for machine translation, classification, text generation, tagging, and other NLP tasks.

![](https://cdn-images-1.medium.com/max/1600/0*4VR_lMPSRF_ir-Re.)

### SpaCy

Developers called [spaCy](https://spacy.io/) the fastest system in the world. They also affirm that their tool is the best way to prepare text for deep learning. Spacy works excellent with well-known Python libraries like gensim, Keras, TensorFlow, and scikit-learn. [Matthew Honnibal](https://github.com/honnibal), the author of the library, says that spaCy’s mission is to make cutting-edge NLP practical and commonly available [[14]](https://spacy.io/). Text classification, named entity recognition, part of speech tagging, dependency parsing, and other examples are presented in the comparative table.

![](https://cdn-images-1.medium.com/max/1600/0*TyvkldosSiE8oAoY.)

### Stanford’s CoreNLP

[Stanford’s CoreNLP](https://stanfordnlp.github.io/CoreNLP/) is a flexible, fast, and modern grammatical analysis tool that provides APIs for most common programming languages including Python. It also has an ability to run as a simple web service. As mentioned on the [official website](https://stanfordnlp.github.io/CoreNLP/), the framework has a part-of-speech (POS) tagger, named entity recognizer (NER), parser, coreference resolution system, sentiment analysis, bootstrapped pattern learning, and open information extraction tools [[15]](https://stanfordnlp.github.io/CoreNLP/).

![](https://cdn-images-1.medium.com/max/1600/0*ZWprmP38Yu_Kb_bL.)

### Tensorflow

The [Google Brain](https://en.wikipedia.org/wiki/Google_Brain) Team developed [TensorFlow](https://www.tensorflow.org/) and released it in 2015 for research purposes. Now many companies like Airbus, Intel, IBM, Twitter and others use TensorFlow at production scale. The system architecture is flexible, so it is possible to perform computations on CPUs or GPUs. The main concept is flow graphs usage. Nodes of the graph reflect mathematical operations, while the edges represent multidimensional data arrays (tensors) communicated between them [[16]](https://www.tensorflow.org/). One of the most known of TensorFlow’s NLP application is [Google Translate](https://en.wikipedia.org/wiki/Google_Translate). Other applications are text classification and summarization, speech recognition, tagging, and so on.

### TFLearn

As Tensorflow is a low-level API, many high-level APIs were created to run on top of it to make the user experience faster and more understandable. [TFLearn](http://tflearn.org/) is one of these tools that runs on CPU and GPU. It has a special graph visualization tool with details about weights, gradients, activations, and so on [[17]](http://tflearn.org/). The library is already used for sentiment analysis, text generation, and named entity recognition. It lets users work with convolutional neural networks and recurrent neural networks (LSTM).

![](https://cdn-images-1.medium.com/max/1600/0*M4Pf_BT5O2KI9Ifp.)

### Theano

[Theano](https://github.com/Theano/Theano) is a numerical computation Python library that enables users to create their own machine learning models [[18]](https://github.com/Theano/Theano). Many frameworks like Keras are built on top of Theano. There are tools for machine translation, speech recognition, word embedding, and text classification. Look at Theano’s applications in the [table](https://docs.google.com/spreadsheets/d/1Wizmq1RTOdFL75Q8eVu9CwrKCvpa42VOqeNoBnLDx9o/edit?usp=sharing).

### Summary

In this paper, we described neural network supporting Python tools for natural language processing. These tools are Chainer, Deeplearning4j, Deepnl, Dynet, Keras, Nlpnet, OpenNMT, PyTorch, SpaCy, Stanford’s CoreNLP, TensorFlow, TFLearn, and Theano. A table lets readers easily compare the frameworks discussed above.

### Resources

1. [https://chainer.org/](https://chainer.org/)

2. [http://learningsys.org/papers/LearningSys_2015_paper_33.pdf](http://learningsys.org/papers/LearningSys_2015_paper_33.pdf)

3. [https://deeplearning4j.org/](https://deeplearning4j.org/)

4. [http://www.aclweb.org/anthology/W15-1515](http://www.aclweb.org/anthology/W15-1515)

5. [https://github.com/attardi/deepnl](https://github.com/attardi/deepnl)

6. [https://github.com/clab/dynet](https://github.com/clab/dynet)

7. [https://arxiv.org/pdf/1701.03980.pdf](https://arxiv.org/pdf/1701.03980.pdf)

8. [https://keras.io/](https://keras.io/)

9. [https://github.com/erickrf/nlpnet](https://github.com/erickrf/nlpnet)

10. [http://nilc.icmc.usp.br/nlpnet/](http://nilc.icmc.usp.br/nlpnet/)

11. [http://opennmt.net/](http://opennmt.net/)

12. [http://opennmt.net/OpenNMT/applications/](http://opennmt.net/OpenNMT/applications/)

13. [http://pytorch.org/about/](http://pytorch.org/about/)

14. [https://spacy.io/](https://spacy.io/)

15. [https://stanfordnlp.github.io/CoreNLP/](https://stanfordnlp.github.io/CoreNLP/)

16. [https://www.tensorflow.org/](https://www.tensorflow.org/)

17. [http://tflearn.org/](http://tflearn.org/)

18. [https://github.com/Theano/Theano](https://github.com/Theano/Theano)

19. [https://github.com/odashi/chainer_nmt](https://github.com/odashi/chainer_nmt)

### Additional resources

[https://arxiv.org/pdf/1703.04783.pdf](https://arxiv.org/pdf/1703.04783.pdf)

[https://github.com/chainer/chainer/tree/master/examples/word2vec](https://github.com/chainer/chainer/tree/master/examples/word2vec)

[https://github.com/chainer/chainer/tree/master/examples/sentiment](https://github.com/chainer/chainer/tree/master/examples/sentiment)

[https://github.com/marevol/cnn-text-classification](https://github.com/marevol/cnn-text-classification)

[https://github.com/butsugiri/chainer-rnn-ner](https://github.com/butsugiri/chainer-rnn-ner)

[https://github.com/khanhptnk/seq2seq-chainer](https://github.com/khanhptnk/seq2seq-chainer)

[https://github.com/kenkov/seq2seq](https://github.com/kenkov/seq2seq)

[https://github.com/chainer/chainer/tree/master/examples/ptb](https://github.com/chainer/chainer/tree/master/examples/ptb)

[https://github.com/masashi-y/chainer-parser](https://github.com/masashi-y/chainer-parser)

[https://cs.stanford.edu/~danqi/papers/emnlp2014.pdf](https://cs.stanford.edu/~danqi/papers/emnlp2014.pdf)

[http://learningsys.org/papers/LearningSys_2015_paper_33.pdf](http://learningsys.org/papers/LearningSys_2015_paper_33.pdf)

[https://arxiv.org/pdf/1611.01604.pdf](https://arxiv.org/pdf/1611.01604.pdf)

[https://deeplearning4j.org/nlp](https://deeplearning4j.org/nlp)

[https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/nlp/paragraphvectors/ParagraphVectorsClassifierExample.java](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/nlp/paragraphvectors/ParagraphVectorsClassifierExample.java)

[https://deeplearning4j.org/word2vec](https://deeplearning4j.org/word2vec)

[https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/character/GravesLSTMCharModellingExample.java](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/character/GravesLSTMCharModellingExample.java)

[http://www.aclweb.org/anthology/W15-1515](http://www.aclweb.org/anthology/W15-1515)

[https://github.com/attardi/deepnl](https://github.com/attardi/deepnl)

[https://github.com/attardi/deepnl/blob/master/deepnl/pos_tagger.py](https://github.com/attardi/deepnl/blob/master/deepnl/pos_tagger.py)

[https://github.com/attardi/deepnl/blob/master/deepnl/networkconv.pyx](https://github.com/attardi/deepnl/blob/master/deepnl/networkconv.pyx)

[https://github.com/attardi/deepnl/blob/master/deepnl/ner_tagger.py](https://github.com/attardi/deepnl/blob/master/deepnl/ner_tagger.py)

[https://github.com/attardi/deepnl/blob/master/deepnl/classifier.pyx](https://github.com/attardi/deepnl/blob/master/deepnl/classifier.pyx)

[https://github.com/attardi/deepnl/blob/master/deepnl/sentiwords.pyx](https://github.com/attardi/deepnl/blob/master/deepnl/sentiwords.pyx)

[http://www.aclweb.org/anthology/W15-1515](http://www.aclweb.org/anthology/W15-1515)

[https://github.com/attardi/deepnl/blob/master/deepnl/embeddings.py](https://github.com/attardi/deepnl/blob/master/deepnl/embeddings.py)

[https://github.com/attardi/deepnl/blob/master/deepnl/tagger.pyx](https://github.com/attardi/deepnl/blob/master/deepnl/tagger.pyx)

[https://github.com/attardi/deepnl/blob/master/deepnl/networkseq.pyx](https://github.com/attardi/deepnl/blob/master/deepnl/networkseq.pyx)

[https://github.com/attardi/deepnl/blob/master/deepnl/extractors.pyx](https://github.com/attardi/deepnl/blob/master/deepnl/extractors.pyx)

[https://github.com/clab/dynet](https://github.com/clab/dynet)

[https://arxiv.org/pdf/1701.03980.pdf](https://arxiv.org/pdf/1701.03980.pdf)

[https://github.com/neubig/lamtram](https://github.com/neubig/lamtram)

[https://github.com/bplank/bilstm-aux](https://github.com/bplank/bilstm-aux)

[http://phontron.com/slides/emnlp2016-dynet-tutorial-part1.pdf](http://phontron.com/slides/emnlp2016-dynet-tutorial-part1.pdf)

[https://github.com/toru34/kim_emnlp_2014](https://github.com/toru34/kim_emnlp_2014)

[https://github.com/roeeaharoni/dynmt-py](https://github.com/roeeaharoni/dynmt-py)

[http://phontron.com/slides/emnlp2016-dynet-tutorial-part2.pdf](http://phontron.com/slides/emnlp2016-dynet-tutorial-part2.pdf)

[http://dynet.readthedocs.io/en/latest/tutorials_notebooks/RNNs.html](http://dynet.readthedocs.io/en/latest/tutorials_notebooks/RNNs.html)

[https://github.com/clab/lstm-parser](https://github.com/clab/lstm-parser)

[https://github.com/clab/joint-lstm-parser](https://github.com/clab/joint-lstm-parser)

[https://github.com/neubig/modlm](https://github.com/neubig/modlm)

[https://github.com/odashi/nmtkit](https://github.com/odashi/nmtkit)

[https://github.com/lvapeab/nmt-keras](https://github.com/lvapeab/nmt-keras)

[https://chsasank.github.io/spoken-language-understanding.html](https://chsasank.github.io/spoken-language-understanding.html)

[https://github.com/igormq/asr-study](https://github.com/igormq/asr-study)

[https://github.com/llSourcell/How_to_make_a_text_summarizer](https://github.com/llSourcell/How_to_make_a_text_summarizer)

[https://github.com/fchollet/keras/blob/master/examples/imdb_lstm.py](https://github.com/fchollet/keras/blob/master/examples/imdb_lstm.py)

[https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py](https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py)

[https://github.com/farizrahman4u/seq2seq](https://github.com/farizrahman4u/seq2seq)

[https://github.com/fchollet/keras/blob/master/examples/imdb_cnn.py](https://github.com/fchollet/keras/blob/master/examples/imdb_cnn.py)

[https://github.com/udibr/headlines](https://github.com/udibr/headlines)

[https://github.com/wolet/s2s-dependency-parsers](https://github.com/wolet/s2s-dependency-parsers)

[https://github.com/0xnurl/keras_character_based_ner](https://github.com/0xnurl/keras_character_based_ner)

[https://github.com/fchollet/keras/blob/master/examples/imdb_bidirectional_lstm.py](https://github.com/fchollet/keras/blob/master/examples/imdb_bidirectional_lstm.py)

[https://github.com/fchollet/keras/blob/master/examples/reuters_mlp.py](https://github.com/fchollet/keras/blob/master/examples/reuters_mlp.py)

[https://github.com/codekansas/keras-language-modeling](https://github.com/codekansas/keras-language-modeling)

[https://link.springer.com/content/pdf/10.1186%2Fs13173-014-0020-x.pdf](https://link.springer.com/content/pdf/10.1186/s13173-014-0020-x.pdf)

[http://nilc.icmc.sc.usp.br/nlpnet/models.html#word-embeddings-portuguese](http://nilc.icmc.sc.usp.br/nlpnet/models.html#word-embeddings-portuguese)

[http://nilc.icmc.sc.usp.br/nlpnet/models.html#pos-portuguese](http://nilc.icmc.sc.usp.br/nlpnet/models.html#pos-portuguese)

[http://nilc.icmc.sc.usp.br/nlpnet/models.html#srl-portuguese](http://nilc.icmc.sc.usp.br/nlpnet/models.html#srl-portuguese)

[http://nilc.icmc.sc.usp.br/nlpnet/models.html#dependency-and-pos-english](http://nilc.icmc.sc.usp.br/nlpnet/models.html#dependency-and-pos-english)

[https://github.com/erickrf/nlpnet/blob/master/nlpnet/taggers.py](https://github.com/erickrf/nlpnet/blob/master/nlpnet/taggers.py)

[https://github.com/erickrf/nlpnet/blob/master/nlpnet/networkconv.pyx](https://github.com/erickrf/nlpnet/blob/master/nlpnet/networkconv.pyx)

[https://github.com/erickrf/nlpnet/blob/master/nlpnet/networkdependencyconv.pyx](https://github.com/erickrf/nlpnet/blob/master/nlpnet/networkdependencyconv.pyx)

[http://www.aclweb.org/anthology/W15-1508](http://www.aclweb.org/anthology/W15-1508)

[https://github.com/erickrf/nlpnet](https://github.com/erickrf/nlpnet)

[http://nilc.icmc.usp.br/nlpnet/](http://nilc.icmc.usp.br/nlpnet/)

[http://opennmt.net/OpenNMT/applications/#machine-translation](http://opennmt.net/OpenNMT/applications/#machine-translation)

[http://opennmt.net/OpenNMT/applications/#summarization](http://opennmt.net/OpenNMT/applications/#summarization)

[http://opennmt.net/OpenNMT/applications/#speech-recognition](http://opennmt.net/OpenNMT/applications/#speech-recognition)

[http://opennmt.net/OpenNMT/applications/#sequence-tagging](http://opennmt.net/OpenNMT/applications/#sequence-tagging)

[http://opennmt.net/OpenNMT/applications/#language-modelling](http://opennmt.net/OpenNMT/applications/#language-modelling)

[http://opennmt.net/OpenNMT/training/embeddings/](http://opennmt.net/OpenNMT/training/embeddings/)

[https://arxiv.org/pdf/1701.02810.pdf](https://arxiv.org/pdf/1701.02810.pdf)

[http://opennmt.net/OpenNMT/applications/](http://opennmt.net/OpenNMT/applications/)

[http://pytorch.org/about/](http://pytorch.org/about/)

[http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html](http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)

[http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#](http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)

[http://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html](http://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html)

[http://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html](http://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html)

[https://github.com/spro/practical-pytorch/blob/master/char-rnn-generation/char-rnn-generation.ipynb](https://github.com/spro/practical-pytorch/blob/master/char-rnn-generation/char-rnn-generation.ipynb)

[https://github.com/spro/practical-pytorch/blob/master/conditional-char-rnn/conditional-char-rnn.ipynb](https://github.com/spro/practical-pytorch/blob/master/conditional-char-rnn/conditional-char-rnn.ipynb)

[https://spacy.io/](https://spacy.io/)

[https://spacy.io/docs/usage/pos-tagging](https://spacy.io/docs/usage/pos-tagging)

[https://spacy.io/docs/usage/word-vectors-similarities](https://spacy.io/docs/usage/word-vectors-similarities)

[https://spacy.io/docs/usage/entity-recognition](https://spacy.io/docs/usage/entity-recognition)

[https://spacy.io/docs/usage/dependency-parse](https://spacy.io/docs/usage/dependency-parse)

[https://spacy.io/docs/usage/deep-learning](https://spacy.io/docs/usage/deep-learning)

[https://explosion.ai/blog/spacy-deep-learning-keras](https://explosion.ai/blog/spacy-deep-learning-keras)

[https://stanfordnlp.github.io/CoreNLP/](https://stanfordnlp.github.io/CoreNLP/)

[http://apps.cs.utexas.edu/tech_reports/reports/tr/TR-2222.pdf](http://apps.cs.utexas.edu/tech_reports/reports/tr/TR-2222.pdf)

[https://nlp.stanford.edu/projects/mt.shtml](https://nlp.stanford.edu/projects/mt.shtml)

[https://github.com/Lynten/stanford-corenlp](https://github.com/Lynten/stanford-corenlp)

[https://github.com/stanfordnlp/treelstm](https://github.com/stanfordnlp/treelstm)

[https://arxiv.org/pdf/1609.08409.pdf](https://arxiv.org/pdf/1609.08409.pdf)

[https://nlp.stanford.edu/sentiment/](https://nlp.stanford.edu/sentiment/)

[https://www.tensorflow.org/](https://www.tensorflow.org/)

[https://github.com/tensorflow/nmt](https://github.com/tensorflow/nmt)

[https://arxiv.org/pdf/1609.08144.pdf](https://arxiv.org/pdf/1609.08144.pdf)

[https://github.com/mrahtz/tensorflow-pos-tagger](https://github.com/mrahtz/tensorflow-pos-tagger)

[https://github.com/pannous/tensorflow-speech-recognition](https://github.com/pannous/tensorflow-speech-recognition)

[https://www.tensorflow.org/tutorials/word2vec](https://www.tensorflow.org/tutorials/word2vec)

[https://github.com/monikkinom/ner-lstm](https://github.com/monikkinom/ner-lstm)

[https://github.com/dennybritz/cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf)

[https://research.googleblog.com/2016/08/text-summarization-with-tensorflow.html](https://research.googleblog.com/2016/08/text-summarization-with-tensorflow.html)

[https://github.com/tensorflow/models/tree/master/research/textsum](https://github.com/tensorflow/models/tree/master/research/textsum)

[https://www.tensorflow.org/tutorials/recurrent](https://www.tensorflow.org/tutorials/recurrent)

[http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)

[http://jrmeyer.github.io/tutorial/2016/02/01/TensorFlow-Tutorial.html](http://jrmeyer.github.io/tutorial/2016/02/01/TensorFlow-Tutorial.html)

[http://tflearn.org/](http://tflearn.org/)

[https://github.com/dhwajraj/NER-RNN](https://github.com/dhwajraj/NER-RNN)

[https://github.com/tflearn/tflearn/blob/master/examples/nlp/cnn_sentence_classification.py](https://github.com/tflearn/tflearn/blob/master/examples/nlp/cnn_sentence_classification.py)

[https://github.com/tflearn/tflearn/blob/master/examples/nlp/bidirectional_lstm.py](https://github.com/tflearn/tflearn/blob/master/examples/nlp/bidirectional_lstm.py)

[https://github.com/tflearn/tflearn/blob/master/examples/nlp/lstm.py](https://github.com/tflearn/tflearn/blob/master/examples/nlp/lstm.py)

[https://github.com/tflearn/tflearn/blob/master/examples/nlp/seq2seq_example.py](https://github.com/tflearn/tflearn/blob/master/examples/nlp/seq2seq_example.py)

[https://github.com/tflearn/tflearn/blob/master/examples/nlp/lstm_generator_shakespeare.py](https://github.com/tflearn/tflearn/blob/master/examples/nlp/lstm_generator_shakespeare.py)

[https://github.com/tflearn/tflearn/blob/master/examples/nlp/lstm_generator_cityname.py](https://github.com/tflearn/tflearn/blob/master/examples/nlp/lstm_generator_cityname.py)

[https://github.com/EdinburghNLP/nematus](https://github.com/EdinburghNLP/nematus)

[https://github.com/ZhangAustin/Deep-Speech](https://github.com/ZhangAustin/Deep-Speech)

[https://github.com/llSourcell/How_to_make_a_text_summarizer](https://github.com/llSourcell/How_to_make_a_text_summarizer)

[http://deeplearning.net/tutorial/rnnslu.html](http://deeplearning.net/tutorial/rnnslu.html)

[https://raberrytv.wordpress.com/2016/12/26/efficient-embeddings-with-theano/](https://raberrytv.wordpress.com/2016/12/26/efficient-embeddings-with-theano/)

[https://github.com/yoonkim/CNN_sentence](https://github.com/yoonkim/CNN_sentence)

[https://deeplearning4j.org/convolutionalnets.html](https://deeplearning4j.org/convolutionalnets.html)

[https://deeplearning4j.org/usingrnns](https://deeplearning4j.org/usingrnns)

[https://github.com/neulab/xnmt](https://github.com/neulab/xnmt)

[https://github.com/memeda/sequence-labeling-by-nn](https://github.com/memeda/sequence-labeling-by-nn)

[https://cs.umd.edu/~miyyer/pubs/2017_acl_dynsp.pdf](https://cs.umd.edu/~miyyer/pubs/2017_acl_dynsp.pdf)

[http://ben.bolte.cc/blog/2016/language.html](http://ben.bolte.cc/blog/2016/language.html)

[http://pyvideo.org/pydata-carolinas-2016/deep-language-modeling-for-question-answering-using-keras.html](http://pyvideo.org/pydata-carolinas-2016/deep-language-modeling-for-question-answering-using-keras.html)

[https://github.com/chartbeat-labs/textacy](https://github.com/chartbeat-labs/textacy)

[http://iamaaditya.github.io/2016/04/visual_question_answering_demo_notebook](http://iamaaditya.github.io/2016/04/visual_question_answering_demo_notebook)

[https://github.com/hans/corenlp-summarizer](https://github.com/hans/corenlp-summarizer)

[https://nlp.stanford.edu/software/relationExtractor.html](https://nlp.stanford.edu/software/relationExtractor.html)

[https://github.com/spiglerg/RNN_Text_Generation_Tensorflow](https://github.com/spiglerg/RNN_Text_Generation_Tensorflow)

[https://github.com/paarthneekhara/neural-vqa-tensorflow](https://github.com/paarthneekhara/neural-vqa-tensorflow)

[https://github.com/DeepRNN/visual_question_answering](https://github.com/DeepRNN/visual_question_answering)

[https://github.com/llSourcell/How_to_do_Sentiment_Analysis](https://github.com/llSourcell/How_to_do_Sentiment_Analysis)

[https://github.com/Lasagne/Recipes/blob/master/examples/lstm_text_generation.py](https://github.com/Lasagne/Recipes/blob/master/examples/lstm_text_generation.py)

[https://github.com/glample/tagger](https://github.com/glample/tagger)

[https://github.com/Sentimentron/Dracula](https://github.com/Sentimentron/Dracula)

[http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.646.4491&rep=rep1&type=pdf](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.646.4491&rep=rep1&type=pdf)

[https://github.com/hiroki13/neural-semantic-role-labeler](https://github.com/hiroki13/neural-semantic-role-labeler)

[https://github.com/carpedm20/hali](https://github.com/carpedm20/hali)

[https://github.com/saltypaul/Seq2Seq-Chatbot](https://github.com/saltypaul/Seq2Seq-Chatbot)

