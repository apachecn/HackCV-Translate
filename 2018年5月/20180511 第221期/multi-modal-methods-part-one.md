# Multi-Modal Methods: Visual Speech Recognition (Lip Reading)

**Recent Intersections Between Computer Vision and Natural Language Processing (Part One)**

This is the first instalment of our latest publication series looking at some of the intersections between Computer Vision (CV) and Natural Language Processing (NLP). Readers are encouraged to view the piece through our [website](http://www.themtank.org/) for the best experience: [http://www.themtank.org/multi-modal-methods](http://www.themtank.org/multi-modal-methods)

**Part One: Visual Speech Recognition (Lip Reading)**

Part Two: Image Captioning (From Translation to Attention)

Part Three: Image Captioning (Reinforcement Learning and Beyond)

Feedback and comments are welcomed, either through medium or directly to [info@themtank.com](mailto:info@themtank.com.). Thanks for reading!

### Introduction

In this series of pieces we decided to examine the interplay between Computer Vision (CV) and Natural Language Processing (NLP) — a fitting segue from the previous CV-centric piece: “A Year in Computer Vision” (available****[here](http://www.themtank.org/a-year-in-computer-vision))[[1]](#bb45). While advancements within a singular field are often impressive, knowledge, by its very nature, is additive and combinatorial. These characteristics mean that improvements and breakthroughs in one field may catalyse further progress in other fields. Often two seemingly distinct bodies of knowledge coalesce to push our understanding, technologies and solutions into exciting and unforeseen areas.

> In our previous work, we briefly attempted to outline Computer Vision’s claim on intelligence; building systems that can learn, infer and reason about the world from visual data alone. Here we hope to add to this discussion; what part does language play in the creation and recreation of intelligence?

This topic, when broached, has historically been a source of contention among linguists, neuroscientists and AI researchers. We can at least say that vision and language are inextricably intertwined, from an evolutionary standpoint, with the human experience. An experience that weighs learning heavily. For instance, when the concept of a ‘cat’ is evoked in your mind, there are numerous different associations around the nexus of cat almost instantly. Such as:

* **An image of a generic cat or a specific cat**

* **The feeling of petting a cat’s soft fur**

* **The letters ‘c’, ‘a’ and ‘t’**

* **Sometimes-selfish and largely independent creatures**

These basic experiences of the concept ‘cat’ all inform our understanding of what a cat is and its relationship to us and the world. This knowledge of ‘cat’ is iterable; it may be altered through direct experience, pondering cat-related things or by gaining information through any medium. Although not all experiences require language when recalling, the articulation of the experience or thought to oneself is often through language.

> If Computer Vision recognises patterns, then perhaps the addition of NLP could augment this process. It could enable processes in machines analogous to how people associate many modalities and experiences with the aforementioned concept of ‘cat’. The addition of language may eventually provide a means for machines to group, reason and articulate complex concepts in the future.

Much the same way we iterate, link and update concepts through whatever modality of input our brain takes — multi-modal approaches in deep learning are coming to the fore. Below are just some of the intersections between CV and NLP:

* **Lip Reading** — Input is visual; output is text

* **Image Captioning** — Input is visual; output is text

* **Visual Question Answering** — Input is visual and text; output is text

* **Image Generation from Captions**— Input is text; output is visual

Of these fields, we hope to provide insight into the progression and techniques of Lip Reading and Image Captioning in this series. While Visual Question Answering and Image Generation from Captions may be the subject of some future work.

In Deep Learning: Practice and Trends (NIPS 2017)[[2]](#1afa), prominent researchers offered a simple abstraction — that **virtually** all deep learning approaches can be characterised as either augmenting **architectures or loss functions, or applying the previous to new input/output combinations**. While an oversimplification, the generalisability of current deep learning approaches is impressive. And as we shall see, these general approaches are also circumscribing new territories of competence as they progress.

There are also interesting second-order effects due to the generalisability of these methods and their relatively recent successes across-domains. This is despite the seemingly-troublesome issue of handling completely different inputs and output formats. Researchers can now work in many different areas and apply their techniques to issues across the spectrum, from social sciences to healthcare, and from sports to finance. Regardless of application, the tricks and knowledge gathered on architectures and loss functions may be repurposed and used anew somewhere else.

> This partial disintegration of some research silos, or the encouragement of greater interdisciplinary work using AI-tools and techniques, follows on from our remarks about the combinatorial nature of knowledge. Second-order effects mean that CV researchers often understand NLP techniques, and vice-versa. Introductory courses and books on deep learning cover use cases within NLP, CV, Reinforcement Learning and Generative models.

In some senses, we are getting closer to a generalisable artificial intelligence; knowledge in deep learning is consolidating into a more paradigmatic approach. Such congruency allows researchers from all disciplines to leverage AI in new and exciting ways. Perhaps, a true general intelligence lies ahead, although how many paradigms must be disequilibrated and reinstated anew before such a point is reached is unknown. What we do know is that work in generalisable models continues to captivate us, as we watch techniques perform across multiple tasks, domains and modalities [[3]](#7690)[[4]](#dd31)[[5]](#5421).

> In keeping with the last publication, we aim to be as accessible as possible for our audience, and to provide individuals with the tools to learn about AI at whatever depth they desire. However, in this piece we sacrificed expanse for greater depth into the research areas themselves. We will continue to experiment with scope and timelines, to understand how best to convey topics to the reader. For those lacking technical proficiency there may be short sections which are tedious; but their omission won’t impinge the lay-reader greatly. We hope that one should be able to take something of value away regardless of their skillset.

Further inroads will be made in the coming years into a greater number of fields, with better techniques deployed at an ever-increasing rate. Understanding that our assumptions may be incorrect, about what AI can and can’t do, is an important step for society. Ultimately, these technologies aim to emulate and improve the processes through which we navigate the world around us. To learn their own meta-structures for the world that we deploy everyday, subconsciously.

> If humanity has never accepted limitations to our abilities, why would we assume that mechanised intelligence will be inherently limited in some way? And with new, unforeseen breakthroughs, the assumption that anyone can predict the long-term future of technology is perhaps untenable at best. The best strategy may be to simply stay as informed as we can and actively engage with the advancements on the horizon.

With thanks,

![](https://cdn-images-1.medium.com/max/1200/1*rFnzjQJ9WPDLL3b92OQiHQ.png)

### Part One: Visual Speech Recognition (Lip Reading)

Previous work from the team detailed some of the many advancements within the field of Computer Vision. In practice, research isn’t siloed into isolated fields and, with this in mind, we present a short exploration of an intersection between Computer Vision (CV) and Natural Language Processing (NLP) — namely, Visual Speech Recognition, also more commonly known as lip reading.

> Similar to the advancements seen in Computer Vision, NLP as a field has seen a comparable influx and adoption of deep learning techniques, especially with the development of techniques such as Word Embeddings[6] and Recurrent Neural Networks (RNNs)[7]. Moreover, the drive to tackle complex, cross-domain problems using a combination of inputs has spawned much to be excited about. One source of excitement for us comes from seeing the skill of Lip Reading move from human-dominance to machine-dominance in the accuracy rankings. Another still from the method by which this was accomplished.

It was not so long ago that lip reading was heralded to be a difficult problem, much like the difficulty ascribed to the game of Go; albeit not quite as well-known. In addition to solving this problem, advancements in lip reading may potentially enable several new applications. For instance, dictating messages in a noisy environment, dealing with multiple simultaneous speakers better, and improving the performance of speech recognition systems in general. Conversely, extracting conversations from video alone may be an area of concern in the future.

Our focus on this niche application, one hopes, is both illustrative and informative. A relatively small body of deep learning work on lip reading was enough to upset the traditional primacy of the expertly-trained lip reader. Meanwhile, the combinatorial nature of AI research and the technologies at the centre of these advancements blend the demarcations between fields in a scintillating way. Where, if ever, such advancements plateau is the question on everyone’s lips.

### Framing the problem

> The task of predicting innovations and advancements in technologies is notoriously quite difficult, and best reserved for small wagers between colleagues and friends. Where estimates are made, one usually compares a machine’s performance to tasks that humans are already good at, e.g. walking, writing, playing sports, etc. It surprised us to learn two things with regards to lip reading. Firstly, that machines managed to surpass expert-humans recently, and secondly, that expert-humans weren’t that accurate to begin with.

Irrespective of the bar set by the expert, we think it best to delve into what makes this a tough challenge to master. Visemes, **analogous to the lip-movements that comprise a lip reading alphabet**, pose a clear challenge to those who’ve ever attempted to apply them. Namely, that multiple sounds share the same shape. There exists a level of ambiguity between consonants, which cannot be dispensed with — a problem well documented by Fisher in his extensive study on visemes [[8]](#fb19).

**Figure 1: Viseme Examples**

![](https://cdn-images-1.medium.com/max/1600/0*Qvs1dql-HxhZ054U.)

Since there are only so many shapes that one’s mouth can make in articulation, mapping said shapes accurately to the underlying words is challenging [[10]](#fb19). Especially when much communication relies more on sound than on visual information; vocal communication is sound-dependent. Hence, achieving high accuracy without the context of the speech [[11]](#f803) is extremely difficult — for people and machines.

### Early results

With these limitations it’s not surprising that early studies focused on simplified versions of the problem. Initially, feature engineering produced improvements using facial recognition models which placed bounding boxes around the mouth, and extracted a model of the lips independent from the orientation of the face. Some common features used were the width-height ratio of a bounding box for detecting mouths, the appearance of the tongue (pixel intensity in the red channel of the image) and an approximation of the amount of teeth from the ‘whiteness’ in the image [[12]](#f803).

**Figure 2**:****Extracting Lips as a Feature

![](https://cdn-images-1.medium.com/max/1600/0*EUPTm5FK1IS1QkfF.)

These approaches obtained impressive results (over 70% word accuracy) for tests performed with classifiers trained on **the same speaker they were tested on**. But performance was heavily damaged when trying to lip read from individuals not included in the training set. Lip detection in males with moustaches was also more difficult and, therefore, the performance on such cases was poor. Hence, the feature engineering approaches, while an improvement, ultimately failed to generalise well.

Following this, using different viseme classification methods with defined language models improved state of the art (SOTA) performance.[[14] ](#4a0a)Language models help filter results that are obviously incorrect and improve results by selecting from only plausible options, e.g. ’n’ for the 4th character in “soon” rather than “soow” or “soog”. Greater improvements still were made by “fine-tuning” the viseme classifier for phoneme classification, which enabled them to deal with multiple possible solutions for words containing the same visemes in similar intervals. This improved accuracy and performed comparatively better than previous approaches.

> These early techniques brought performance to roughly 19% accuracy on an unseen test set, an improvement over the prior best of 17% (+/- 12%) accuracy generated by a sample of hearing-impaired lip readers. A sample group which outperforms the general population on average.[15]

McGurk and MacDonald argue in their 1976 paper[[16]](#5f21) that speech is best understood as bimodal, that is taking both visual and audio inputs — and that comprehension in individuals may be compromised if either of these two domains are absent. Intuitively, many of us can recall mishearing speech while on the phone, or the difficulties one has in pairing sound and lips in a noisy environment. The requirement of bimodal inputs, as well as contextual constraints, hampers the ability of people and machines to read lips with accuracy. This pointed to the need for further studies on the use of these combined information sources. A direction which brings us into the most recent epoch of lip reading approaches.

### The arrival of Deep Learning

> It is with this point that we introduce recent work from Assael et al. (2016) — “LipNet: End-to-End Sentence-level Lipreading.”[17] “LipNet” introduces the first approach for an end-to-end lip reading algorithm at sentence level. Earlier work by Wand, Koutník and Schmidhuber[18] applied LSTMs[19] to the task, but only for word classifications. However, their earlier advances, including end-to-end[20] trainability, were undoubtedly valuable to the body of work in the space. For those wishing to know more about LSTMs and their variants, Christopher Olah provides an intuitive and detailed explanation of their use here.[21]

**Figure 3**:****LipNet Example at Sentence Level

![](https://cdn-images-1.medium.com/max/1600/0*vSAekxVqp3RvP8mi.)

On a high level in the architecture, the frames extracted from a video sequence are processed in small sets within a Convolutional Neural Network (CNN),[[23]](#ff73) while an LSTM-variant runs on the CNN output sequentially to generate output characters. More precisely, a 10-frame sequence is grouped together in a block (width x height x 10), sequence length may vary, but the consecutive nature of these frames creates a Spatiotemporal CNN.

Then the output of this LSTM-variant, called a Gated Recurrent Unit (GRU),[[24]](#ab5f) is processed by a multi-layered perceptron (MLP) to output values for the different characters derived from the Spatiotemporal CNN. Lastly, a Connectionist Temporal Classification (CTC) provides final processing on the sequence outputs to make it more intelligible in terms of precise outputs, i.e. words and sentences. This approach allows information to be passed through the time periods comprising both words and, ultimately, sentences, improving the accuracy of network predictions.

> The authors note that ‘LipNet addresses the issues of generalisation across speakers,’ i.e. the variance problems seen in earlier approaches, ‘and the extraction of motion features’, originally classed as open problems in Zhou et al. (2014).[25][26], The approach in LipNet, we feel, is interesting and exciting outside of the narrow confines of accuracy measures alone. The combination of CNNs and RNNs in the network — itself a hark back to our comments around the lego-like approach of deep learning research — is, perhaps, more evidence for the soon-to-be-primacy of differential programming. Deep Learning est en train de mourir. Vive Differentiable Programming![27]

LipNet also makes use of an additional algorithm typically used in speech recognition systems — a Connectionist Temporal Classification (CTC) output. After the classification of framewise characters, which in combination with more characters define an output sequence, CTC can group the probabilities of several sequences (e.g. “c__aa_tt” and “ccaaaa__t”) into the same word candidates (in this case “cat”) for the final sentence prediction. Thus the algorithm is alignment-free. CTC solves the problem of matching sequences where timing is variable.

**Figure 4**: CTC in Action

![](https://cdn-images-1.medium.com/max/1600/0*3SJXkbo9kV7hxXon.)

By predicting the alphabet characters and an additional “_” (space) character, it’s possible to generate a word prediction by removing repeated letters and empty spaces, as can be seen in fig. 5 for the classification of the word “please”. In practical terms this means that elongated pronunciations, variations in emphasis and timings, as well as pauses between syllables and words can still produce consistent predictions using the CTC for outputs.

**Figure 5**: Saliency map of “Please”

![](https://cdn-images-1.medium.com/max/1600/0*nttCKknWvSAg9trG.)

CTC is a function for output alignment and a loss correction function based on that alignment, and is independent of the CNN and LSTM-variants. One can also think of CTC as similar to a softmax due to converting the raw output of a network (e.g. raw class scores or in our case, characters) into the expected output (e.g. a probability distribution or in this case, words and sentences). CTC makes matching a single character output to word level possible. Awni Hannun provides an excellent dynamic publication that explains CTC operation; available [here](https://distill.pub/2017/ctc/).[[29]](#1ba1)

> There is a great video which covers some of LipNet’s functionality, as well as a specific use case — operating within autonomous vehicles. Seeing LipNet in operation ties together much of what we’ve discussed about the system so far.[30]



### Architecture and results

A hallmark of this method is that the output labels are not conditioned on each other. For example, the letter ‘a’ in ‘cat’ is not conditioned on ‘c’ or ‘t’. Instead this relation is extracted by three spatio-temporal convolutions, followed by two GRUs which process a set number of the input images. The output from the GRUs then goes through a MLP to compute CTC loss (see fig. 6).

**Figure 6**: LipNet Architecture

![](https://cdn-images-1.medium.com/max/1600/0*QZDWj-HKAbtsRrwb.)

> The architecture of LipNet was deemed an empirical success, achieving a prediction accuracy of 95.2% on sentences from the GRID dataset, an audiovisual sentence corpus for research purposes.[31] However, literature on deep speech recognition (Amodei et al., 2015)[32] suggested that further performance improvements would inevitably be achieved with more data and larger models. Commentators, reminded of earlier difficulties in generalisability and moustache-handling, expressed concern over the unusual sentences taken from GRID which formed the LipNet example video. The limited nature of GRID produced fears of overfitting; but how would LipNet fare in the real-world?

**Figure 7**: LipNet and other approaches

![](https://cdn-images-1.medium.com/max/1600/0*NrWWrYzQIK881mgi.)

### The arrival of richer data

Not long after LipNet, DeepMind released ‘Lip Reading Sentences in the Wild’, [[33]](#fad6) and addressed some of the concerns around LipNet’s generalisability. Taking inspiration from both CNNs for visual feature extraction[[34]](#51f2) and the use of LSTMs for speech transcription,[[35]](#9662) the authors present an innovative approach to the problem of lip reading. By adding individual attention mechanisms for each of the input types, and combining them afterwards to produce character outputs, improvements in both the accuracy and generalisability of the original LipNet architecture were realised.

> Attention mechanisms, discussed at length in part two of this piece, refer to a technique for focusing on specific parts of the input or previous layer(s) within neural networks. A somewhat-recent technique, taking inspiration from earlier work but popularised by Alex Graves’s in 2013/2014, it has grown in use partially from his memory-related work: the now-famous sequence generation paper[36] along with his work on neural turing machines.[37]

Attention mechanisms have been an enabler of some the recent success within deep learning; due to more efficient and clever processing of data. It also allows these models to have more interpretability, i.e. if asking why a network thinks a certain image is a dog it is often hard to look at and understand the internals of the network to find out why. Attention allows the network to highlight the salient parts of the image used in its prediction, e.g. a snout and pointed ears. Attention has become such a common technique that it spawned papers like “attention is all you need”, which foregoes convolution and recurrence techniques entirely for the problem of machine translation.


eturning to “Lip Reading Sentences in the Wild”, Chung et al. (2017) present their WLAS Network. Composed of three main submodules (watch, listen spell) — with attention sprinkled into the spell module. The system is as follows:

* **Watch** (image encoder): Takes images and encodes them into a deep representation to be processed by further modules.

* **Listen** (audio encoder): Allows the system to take in audio format as optional help to lip reading. This directly processes 13-dimensional MFCC features (see next section).

* **Spell** (character decoder): This module incorporates the information from all previous modules. Each encoder above transforms their respective input sequence into a fixed-dimensional state vector and sequences of encoder outputs. The character decoder, which is an LSTM transducer, then reads the fixed state and attention vectors from both encoders and produces a probability distribution over the output character sequence. Finally, the attention vectors are fused with the output states to produce the context vectors that contain the information required to produce the next step output.

* **Attend** (independent regulation of audio and video attention mechanisms): Attend to what is important in each specific input signal/stream, i.e. audio or video. Without attention the model gets word error rates of over 100% and seems to forget the input signal. This shows that the dual-attention mechanism truly allowed this technique to work end-to-end. It also allows the network to handle out of sync audio/video (different sampling rates), including an absent stream.

### WLAS functionality; greater details from more data

Watch is a VGG-M[[38] ](#b77a)that extracts a framewise feature representation to be consumed by an LSTM, which generates a state vector and an output signal. The Watch module looks at each frame in the video and extracts the relevant features that the module has learned to look for, i.e. certain lip movements/positions. This is done by a regular VGG-M CNN which outputs a feature representation for each frame.

This sequence of feature representations are then fed into a regular LSTM which generates a state vector (or cell state) and an output signal. With LSTMs and GRUs there’s an output and a “state” input to the next LSTM cell. The output is a character prediction (or a probability distribution of predicted character), while state is what encodes “the past”, i.e. what an LSTM has computed/stored of the past which is used to predict the next output.

**Figure 8**: Watch, Listen, Attend and Spell architecture

![](https://cdn-images-1.medium.com/max/1600/0*A0BTu7EM43dfvSOn.)

The Listen module uses the Mel-frequency cepstral coefficients (MFCCs)[[39]](#ef1f) as its input. These parameters define a representation of the short-term power spectrum of a sound based on signal transformations. MFCCs ensure transformations are scaled to a frequency which simulates the human hearing range. Following this, independent attention mechanisms in the Attend module for each of the audio and video inputs are combined. These are then in turn passed through the Spell module. With a multi-layered perceptron (MLP) at each time step, the output from the LSTM ends up in a softmax to define the probabilities of the output characters.

> With this, we return to similar themes of progress alluded to in our previous work: data availability and network stack-ability. Neural network-based approaches are typically characterised by heavy data demands. Concomitant to the progress in lip reading is the creation of a unique dataset for training and testing the network. Previously, research in lip reading was hampered by the available datasets and their small vocabularies. One only has to look at the desirable characteristics of Chung et al.’s (2016/2017) datasets, the LRW and the LRS, as expressed by Stafylakis and Tzimiropoulos (2017, p. 2), to understand the value of such data in improving research efforts:

> “We chose to experiment with the LRW database, since it combines many attractive characteristics, such as large size (∼500K clips), high variability in speakers, pose and illumination, non-laboratory in-the-wild conditions, and target-words as part of whole utterances rather than isolated.”[40]

Chung et al. (2017) created a pipeline to automatically generate the dataset(s)[[41]](#b8f6) from BBC recordings as well as from the contained closed captions, which enabled progress in a data-intensive research area. Their creation is a ‘Lip Reading Sentences’ (LRS) dataset for visual speech recognition, consisting of over 100,000 natural sentences from British television.’

**Figure 9**: Pipeline to generate LRW/LRS dataset

![](https://cdn-images-1.medium.com/max/1600/0*R5SjYrGKIrKfjG5r.)

The authors also corrupt said datum with storm noises (i.e. weather storms[[42]](#9203)), demonstrating the network’s ability to use distorted and low volume data, or to discard audio completely for prediction purposes. Determining whether there’s value to the prediction in listening or not. For those wishing to see more, Joon Son Chung presents a fantastic overview of the authors’ work at CVPR.[[43]](#4c47)



> Although movements towards lower data requirements are pressing-on, this paradigm has yet to shift; and it’s likely that it shall remain this way for some time to come. As for stackability, the very nature of the LipNet and Lip Reading in the Wild architectures illustrate the lego-like nature of neural nets — e.g. CNNs plugged into RNNs with attention techniques.[44] While it’s true that this is a gross oversimplification, as a heuristic we find it increasingly useful in interpreting and understanding the rapid advancements across a lot of existing, and new, AI research.

> Here this last point extends outside of the architecture itself, inscribing the potential stacking of inputs into our heuristic also. A great contribution of these works is the creation of an end-to-end architecture capable of using audio, video, or combinations of both as inputs to generate a text prediction as output: creating a truly multimodal model. Multiple input sequences resulting in a singular output sequence. Solving this multi-modal problem, and others like it, potentially opens new paths to explore in connecting video, audio and language systems.

### New paths of exploration


urious as to what would follow the approaches detailed previously, we turn our attention to some of the most recent work in this space. Although not exhaustive, here’s a smattering of the best improvements we came across in this domain:

* **Combining Residual Networks with LSTMs for Lipreading**[[45]](#e1b4): Improves on the original LRW paper by using a spatiotemporal convolutional network, a residual network and stacked bidirectional Long Short-Term Memory networks (BiLSTMs). The latter of which processes the input features in forward and reverse order like the BiGRU mentioned earlier. Their approach improves from 76.2% to 83% word accuracy[[46]](#1270) on the LRW paper, and also improves the accuracy on the GRID dataset. They do not extend, at present, to sentences on the LRS dataset.

* **Improving Speaker-Independent Lipreading with Domain-Adversarial Training**[[47]](#e3f1): Helps improve the performance on a target speaker with only a small amount of data. It isn’t as effective in instances where the model is trained on a lot of data. Hence, we would be interested to see its performance on LRS which has a 1000+ speakers.

* **End-to-End Multi-View Lipreading**[[48]](#990f): Achieves classification of non-frontal lip views, also utilising Bidirectional Long-Short Memory (BiLSTM). For each viewpoint the authors create an identical encoding MLP architecture (a stream of video processing), which enables the network to train with multiple views simultaneously (see fig. 10).

**Figure 10**: Architecture

![](https://cdn-images-1.medium.com/max/1600/0*V6Fauh409vsLqkSr.)

* **Visual Speech Enhancement using Noise-Invariant Training**[[50]](#3892): Tackles a somewhat related problem by providing a method for enhancing the voice of visible speakers in noisy environments. The approach uses the audio-visual inputs seen previously to disentangle the voice from background noise by matching lip movements. Although it differs from the other approaches, the idea itself is novel and, frankly, pretty cool. Especially since it makes use of a lipreading dataset for this task.

> “Visual speech enhancement is used on videos shot in noisy environments to enhance the voice of a visible speaker and to reduce background noise. While most existing methods use audio-only inputs, we propose an audio-visual neural network model for this purpose. The visible mouth movements are used to separate the speaker’s voice from the background sounds”

### References

1. [^](#4a99) The M Tank. (2017). A Year in Computer Vision. [Online] TheMTank.com. Available: http://www.themtank.org/a-year-in-computer-vision

2. [^](#215e) Oriol Vinyals and Scott Reed (2017). Deep Learning: Practice and Trends (NIPS 2017 Tutorial, parts I & II). [Online Video] Steven Van Vaerenbergh (www.youtube.com). Available: [https://www.youtube.com/watch?v=YJnddoa8sHk](https://www.youtube.com/watch?v=YJnddoa8sHk)

3. [^](#5571) Kaiser et al. (2017). One Model To Learn Them All. [Online] arXiv: 1706.05137. Available: [https://arxiv.org/abs/1706.05137v1](https://arxiv.org/abs/1706.05137v1)

4. [^](#5571) Hashimoto et al. (2017). A Joint Many-Task Model: Growing a Neural Network for Multiple NLP Tasks. [Online] arXiv: 1611.01587. Available: [arXiv:1611.01587v5](https://arxiv.org/abs/1611.01587v5)

5. [^](#5571) Zamir et al. (2018). Taskonomy. [Website] Taskonomy/Stanford. Available: [https://taskonomy.vision/](https://taskonomy.vision/)

6. [^](#abc1) Ruder, S. (2016). On word embeddings — Part 1. [Online] Sebastian Ruder Blog (ruder.io). Available: [http://ruder.io/word-embeddings-1/](http://ruder.io/word-embeddings-1/)

7. [^](#abc1) Karpathy, A. (2015). The Unreasonable Effectiveness of Recurrent Neural Networks. [Online] Andrej Karpathy Blog ([http://karpathy.github.io/](http://karpathy.github.io/)). Available: [http://karpathy.github.io/2015/05/21/rnn-effectiveness/](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

8. [^](#9685) Fisher, C.G. (1968). Confusions Among Visually Perceived Consonants. Journal of Speech, Language, and Hearing Research, December, Vol. 11, p. 796–804.

9. [^](#b677) [http://www.lipread.com.au/](http://www.lipread.com.au/)

10. [^](#40b8) Motion, pose (full-frontal view (0◦), angled view (45◦), and side view), Multiple people, video conditions/resolution/lighting, speech methods (accents, styles and rates of speech). Taken from: Bear, H.L. (2017, p. 25). Decoding visemes: Improving machine lip-reading. [Online] arXiv: 1710.01288. Available: [arXiv:1710.01288v1](https://arxiv.org/abs/1710.01288v1). (Originally available 2016 from IEEE conference proceedings).

11. [^](#40b8) Contexts of speech: subject, sound, time, place.

12. [^](#1796) Hassanat, A.B.A. (2011). Visual Speech Recognition. [Online] arXiv: 1409.1411. Available: [arXiv:1409.1411v1](https://arxiv.org/abs/1409.1411v1)**.**

13. [^](#1796) ibid

14. [^](#e3aa) Bear, H.L. (2017). Decoding visemes: Improving machine lip-reading. [Online] arXiv: 1710.01288. Available: [arXiv:1710.01288v1](https://arxiv.org/abs/1710.01288v1). (Originally available 2016 from IEEE conference proceedings).

15. [^](#0a54) Easton, R.D., Basala, M. (1982). Perceptual dominance during lipreading. Perception & Psychophysics, 32(6), p. 562–570.

16. [^](#64bd) MacDonald, J., McGurk, H. (1976). Hearing Lips and Seeing Voices, Nature, volume 264, December, p. 746–748. Available: [http://usd-apps.usd.edu/coglab/schieber/psyc707/pdf/McGurk1976.pdf](http://usd-apps.usd.edu/coglab/schieber/psyc707/pdf/McGurk1976.pdf)

17. [^](#86d8) Assael et al. (2016). Lipnet: End-to-End Sentence-Level Lipreading. [Online] arXiv: 1611.01599. Available: [arXiv:1611.01599v2](https://arxiv.org/abs/1611.01599v2)

18. [^](#86d8) Wand et al. (2016). Lipreading with Long Short-Term Memory. [Online] arXiv: 1601.08188. Available: [arXiv:1601.08188v1](https://arxiv.org/abs/1601.08188v1)

19. [^](#86d8) Long short-term memory (LSTM) is a type of unit within a Recurrent Neural Network (RNN) which is responsible for remembering and passing values over time periods. See: Hochreiter, S., Schmidhuber, J. (1997). Long Short Term Memory. Neural Computation, Vol. 9(8), p. 1735–1780. Available: [http://www.bioinf.jku.at/publications/older/2604.pdf](http://www.bioinf.jku.at/publications/older/2604.pdf)

20. [^](#86d8) End-to-end, we define as, the joint training of all parameters within all parts of a neural network or system. In contrast to training them separately and in isolation.

21. [^](#86d8) Olah, C. (2015). Understanding LSTM Networks. [Online] Colah’s Blog ([http://colah.github.io/](http://colah.github.io/)). Available: [http://colah.github.io/posts/2015-08-Understanding-LSTMs/](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

22. [^](#554e) Assael, Y. (2016). LipNet: How easy do you think lipreading is? [Online Video] Yannis Assael ([www.youtube.com](http://www.youtube.com)). Available: [https://www.youtube.com/watch?v=fa5QGremQf8](https://www.youtube.com/watch?v=fa5QGremQf8)

23. [^](#cad0) Karn, U. (2016). An Intuitive Explanation of Convolutional Neural Networks. [Online] Ujjwal Karn Blog ([https://ujjwalkarn.me/author/ujwlkarn/](https://ujjwalkarn.me/author/ujwlkarn/)). Available: [https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/)

24. [^](#1edf) Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. [Online] arXiv: 1406.1078. Available: [arXiv:1406.1078v3](https://arxiv.org/abs/1406.1078v3)

25. [^](#9f6c) See Zhou et al. (2014). A review of recent advances in visual speech decoding. Image and Vision Computing, Vol. 32(9), p. 590–605.

26. [^](#9f6c) Quotes/reference taken from: Assael et al. (2016). Lipnet: End-to-End Sentence-Level Lipreading. [Online] arXiv: 1611.01599. Available: [arXiv:1611.01599v2](https://arxiv.org/abs/1611.01599v2)

27. [^](#9f6c) French comes courtesy of Google’s neural machine translation services. See Yann LeCun’s post: [https://www.facebook.com/yann.lecun/posts/10155003011462143](https://www.facebook.com/yann.lecun/posts/10155003011462143)

28. [^](#9f01) Brueckner, R. (2016). Accelerating Machine Learning with Open Source Warp-CTC. [Online] Inside HPC(insidehpc.com). Available: [https://insidehpc.com/2016/01/warp-ctc/](https://insidehpc.com/2016/01/warp-ctc/)

29. [^](#b279) Hannun, A. (2017). Sequence Modeling With CTC. [Online] Distill ([https://distill.pub/](https://distill.pub/)). Available: [https://distill.pub/2017/ctc/](https://distill.pub/2017/ctc/)

30. [^](#bbb5) Assael, Y. (2016). LipNet in Autonomous Vehicles | CES 2017. [Online Video] Yannis Assael ([www.youtube.com](http://www.youtube.com)). Available: [https://www.youtube.com/watch?v=YTkqA189pzQ](https://www.youtube.com/watch?v=YTkqA189pzQ)

31. [^](#75aa) Barker et al. (2018). The GRID audiovisual sentence corpus. [Online] University of Sheffield. Available: [http://spandh.dcs.shef.ac.uk/gridcorpus/](http://spandh.dcs.shef.ac.uk/gridcorpus/) (last update, 18/03/2013).

32. [^](#75aa) Amodei et al. (2015). Deep Speech 2: End-to-End Speech Recognition in English and Mandarin. [Online] arXiv: 1512.02595. Available: [arXiv:1512.02595v1](https://arxiv.org/abs/1512.02595v1)

33. [^](#9239) Chung et al. (2017). Lip Reading Sentences in the Wild. [Online] arXiv: 1611.05358. Available: [arXiv:1611.05358v2](https://arxiv.org/abs/1611.05358v2)

34. [^](#9239) Simonyan, K., Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition. [Online] arXiv: 1409.1556. Available: [arXiv:1409.1556v6](https://arxiv.org/abs/1409.1556v6)

35. [^](#9239) Graves, A., Jaitly, N. (2014). Towards End-to-End Speech Recognition with Recurrent Neural Networks, Proceedings of the 31st International Conference on Machine Learning, Beijing, China, JMLR: W&CP volume 32. Available: [http://proceedings.mlr.press/v32/graves14.pdf](http://proceedings.mlr.press/v32/graves14.pdf)

36. [^](#3dc7) Graves, A. (2014). Generating Sequences With Recurrent Neural Networks. [Online] arXiv: 1308.0850. Available: [arXiv:1308.0850v5](https://arxiv.org/abs/1308.0850v5)

37. [^](#3dc7) Graves et al. (2014). Neural Turing Machines. [Online] arXiv: 1410.5401. Available: [arXiv:1410.5401v2](https://arxiv.org/abs/1410.5401v2)

38. [^](#aa0c) Chatfield et al. (2014). Return of the Devil in the Details: Delving Deep into Convolutional Nets. [Online] arXiv: 1405.3531. Available: [arXiv:1405.3531v4](https://arxiv.org/abs/1405.3531v4)

39. [^](#c750) Practical Cryptography. (2018). Mel Frequency Cepstral Coefficient (MFCC) tutorial. [Online] Practical Cryptography ([http://www.practicalcryptography.com/](http://www.practicalcryptography.com/)). Available: [http://www.practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/](http://www.practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/) (accessed: 20/02/2018).

40. [^](#ea01) Stafylakis, T., Tzimiropoulos, G. (2017). Combining Residual Networks with LSTMs for Lipreading. [Online] arXiv: 1703.04105v4. Available: [arXiv:1703.04105v4](https://arxiv.org/abs/1703.04105v4)

41. [^](#6289) Lip Reading Words (LRW) and Lip Reading Sentences (LRS)

42. [^](#b6f9) Multi-modal training. The audio stream is the easiest to learn from. During training they randomly select combinations of training with video, audio, or both. They also add noise to the audio to stop it from dominating the learning process, since Lip Reading is a much harder problem.

43. [^](#9b46) Chung, J.S. (2017). Lip Reading Sentences in the Wild (Lip Reading Sentences Dataset), CVPR 2017. [Online Video] Preserve Knowledge ([www.youtube.com](http://www.youtube.com)). Available: [https://www.youtube.com/watch?v=103CXDFhpcc](https://www.youtube.com/watch?v=103CXDFhpcc)

44. [^](#22eb) Another trick this paper used was curriculum learning, which involves showing the network progressively more difficult data to learn during the training process. This can greatly speed up the the training process and decrease overfitting. In this case, the model was first trained with single words, then 2-words, 3-words and, eventually, full sentences.

45. [^](#cb19) Stafylakis, T., Tzimiropoulos, G. (2017). Combining Residual Networks with LSTMs for Lipreading. [Online] arXiv: 1703.04105v4. Available: [arXiv:1703.04105v4](https://arxiv.org/abs/1703.04105v4)

46. [^](#1ddb) Word Accuracy = 1 − Word error rate

47. [^](#1ddb) Wand, M., Schmidhuber, J. (2017). Improving Speaker-Independent Lipreading with Domain-Adversarial Training. [Online] arXiv: 1708.01565. Available: [arXiv:1708.01565v1](https://arxiv.org/abs/1708.01565v1)

48. [^](#0eeb) Petridis et al. (2017). End-to-End Multi-View Lipreading. [Online] arXiv: 1709.00443. Available: **arXiv:1709.00443v1**

49. [^](#7eac) The idea behind it is that the derivatives of image features are associated with feature extractors (e.g. HoG, Sobel filter, etc.) https://www.learnopencv.com/histogram-of-oriented-gradients/

50. [^](#7eac) Gabbay et al. (2017). Visual Speech Enhancement using Noise-Invariant Training. [Online] arXiv: 1711.08789. Available: [arXiv:1711.08789v2](https://arxiv.org/abs/1711.08789v2)

