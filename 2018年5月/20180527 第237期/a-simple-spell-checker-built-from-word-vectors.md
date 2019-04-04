# A simple spell checker built from word vectors

If you have heard of word vectors, you will probably know they enable a type of algebra on words. So, for example,



gives an answer which is closest to the word `Germany`.

What is less well known¹ is that you can use the same word vector algebra to fix spelling mistakes. This post describes how to build a lookup table containing the most common spelling mistakes and their corrections using the pre-trained [GloVe vectors from Stanford](https://nlp.stanford.edu/projects/glove/).

To accompany the post there is a [jupyter notebook on github](https://github.com/er214/spellchecker) which walks through the code. You will also find a version of the lookup table to download.

**Word vectors**

Computers only deal with numbers, and so to get a computer to analyse text data — for example, to find topics, to translate, to summarise, etc — you must first convert the data into numbers. A ‘word vector’ is simply a set of numbers which represent a word: the computer’s internal representation of that word.

If we train a computer to predict the missing word from a sentence, giving it millions of examples to learn from, and we allow the computer to improve its predictions by changing the numbers allocated to each word, we find that synonyms end up being allocated numbers that are close to one another.

There are lots of blog posts and tutorials out there which explain the mechanics behind the word vector training process. My aim below is to give an understanding of why the words end up in the places they do — of why synonyms end up close together.

Consider the problem of predicting the word missing from the following sentence:

> “I picked up the _____ and started to write.”

What can we say about the missing word? It has to be a noun; it’s probably a thing (although it could be a metaphorical thing); but it’s most likely something to write with, or something to write on.

Assume that the computer’s representation for words consists of 2 numbers — i.e. the word vector has 2 dimensions. As such, we can view each vector as the coordinates for a point a world map — each word is located at a distinct point on this world (word?) map. At the start of the training process, the word allocation is random, with words spread uniformly across the world.

Next, imagine that the computer makes a prediction by throwing a hypothetical dart at the map. Its prediction will be the word closest to the point where the dart lands.

Our hypothetical darts playing computer has a rather shaky aim. Faced with the sentence above, it thinks ‘pen’ is the most likely word. Actually, it thinks of a a set of coordinates — the word vector — which happens to be a point located over Barbados. So, the computer aims for Barbados, but the dart lands on St Lucia some way to the east. Given that the initial allocation of words is random, it ends up predicting a word that has nothing to do with writing.

How can the computer improve its predictions? Clearly, if all writing implement words were clustered around the West Indies then we might at least hit ‘pencil’ when we take aim for ‘pen’, and this is probably the right answer in at least a few cases where ‘pen’ is appropriate.

So far, so good. But what about the following (somewhat contrived) example:

> “I shall _____ a letter to him forthwith.”

Both ‘write’ and ‘pen’ look like good options, but this is going to cause us problems. If both writing implement nouns and writing verbs are clustered together, how can we avoid predicting ‘pencil’ or ‘biro’ in this example; or verbs such as ‘write’ in the first example?

The answer is to give the computer more numbers per word — a higher dimensional word vector. Although stretching an already sketchy analogy, imagine the computer now throws two darts: one at the map, and another at a height chart with a scale stretching from the bottom of the sea bed up to the stratosphere. This gives us the space to allow ‘things to do with writing’ to continue to be located over the West Indies. Noun-like words might found near sea level, and verbs somewhere up in the sky; with ‘pen’ sitting between the two, perhaps being closer to the ground since it’s most commonly used as a noun.

The most widely used word vectors have 300 dimensions. That is, we allocate 300 numbers for each word. This provides an enormous space to store information about all kinds of different aspects of words — whether it is a noun, verb, adjective, etc; the tense of a verb; whether nouns are plural; various different aspects relating to the meaning of the word; and, as we shall see below, whether it is spelled correctly.

Before turning to spelling mistakes, note again that the computer doesn’t know **anything** about words beyond the contexts in which they are found. In particular, it doesn’t know which letters are used. It has no way of knowing that ‘write and ‘writing’ share a common root. The two words end up in a similar part of the vector space because the contexts in which they are found overlap. Furthermore, some of the differences between the two word vectors will display a specific pattern because ‘write’ is the infinitive of the verb and ‘writing’ is the present participle; and this difference will be similar across all verbs because the contexts in which infinitives and present participles are used also overlap.

**Word vectors and spelling mistakes**

One version of the GloVe vectors was trained on 840 billion words of data crawled from the web, which results in vectors for 2.2 million ‘words’.

That’s a lot of words. A lot more than the [228,132 entries](https://en.oxforddictionaries.com/explore/how-many-words-are-there-in-the-english-language/) in the Oxford English Dictionary. A large part of the difference consists of proper nouns — names of people, places and so on that will not appear in a dictionary. Then there are numbers, plurals, different verb endings, punctuation, words in CAPITALS, and so on.

However, there are also lots of vectors for words containing spelling mistakes. This gave me an idea for an answer to a question posted on the [fast.ai](http://forums.fast.ai/t/nlp-any-libraries-dictionaries-out-there-for-fixing-common-spelling-errors/16411/40) forums which asked for ways to identify and correct common misspellings. Given that a misspelled word is used in the same context as its correct counterpart, then perhaps we can find the correct spelling by looking at the neighbours of the incorrectly spelled word in vector space?

It turns out this doesn’t work. Far from it. But the reality is much more interesting.

**A paralel univers of speling msitakes**

The following list shows you the 10 words closest to ‘reliable’ in vector space:



Now, consider the nearest neighbours of the misspelling ‘relieable’:



We’ve got 7 additional ways of spelling reliable wrong, followed by misspelled synonyms of reliable. The correct spelling ‘reliable’ is nowhere to be seen.

This is a strange result. Surely, `relieable` is a synonym for `reliable`? They will be found in the very similar contexts — possibly in otherwise identical sentences. So, why don’t these misspellings appear in the same place in the vector space?

I’ll come back to that question below. For the time being, we note that there appears to be a very distinct part of the vector space associated with incorrect spellings. This begs the question: is there a way to fix spellings by transforming the incorrectly spelled word vector to shift its position within the vector space into the parallel universe of correct spellings?

The answer is that you can, and it’s very simple. It works in exactly the same way as the concept algebra mentioned above. Instead of:



You have something like:



To generalise this approach (make it less reliant on reliable…), we can build a spelling transformation vector by taking the average difference between a set of pairs of correct and incorrectly spelled words. We can then fix a spelling mistake by subtracting this spelling transformation vector from the incorrectly spelled word vector and finding the word closest to where we end up.

**The spelling transformation vector**

The following charts plot the values of the difference between the vectors of correctly and incorrectly spelled words across the 300 dimensions for a set of commonly misspelled words. (Note, there is no sequential relationship between each of the dimensions. I have drawn a line chart to make the patterns more visible).

There appears to be a characteristic pattern with 2 downward spikes towards the left hand side of the plots.

It also looks like more extreme spelling errors produce longer spikes, or to put it another way, very common spelling mistakes produce shorter spikes. For example, `recieve — receive` and `calender — calendar` both have very short spikes. Whereas we couldn’t find `reliable` in the close neighbours of `relieable`, `receive` is the second closest neighbour of `recieve` (after `recive`).

So, it is perhaps better to think of a misspelling direction in word vector space rather than a specific area.

What is clear from the charts is that the difference between incorrect and correctly spelled word vectors various quite a lot between word pairs. Therefore, to capture the essence of the transformation from bad to good spellings we need to take an average across a large set of word pairs.

Unfortunately, I don’t have such a data set. As a starting point, I’ve used the [this list](https://en.oxforddictionaries.com/spelling/common-misspellings), which contains just over 100 examples. Setting aside 15% of this small set for testing, I have built an initial spelling transformation vector equal to the average difference between the vectors for incorrect and correct words.

Applying this to our test set of 17 words we manage to fix 15 (88%). The only two it struggles with are `pharaoh` and `Fahrenheit`. As with `calendar` and `receive`, it is likely that these are spelled wrong so often that there is very little difference between the respective correct and incorrect word vectors.

**Building a better spelling transformation vector**

88% is good, but we can do better. We can build a much bigger set of examples to ‘train’ on by applying our initial transformation in reverse to locate discover more spelling mistakes. For example, applied to `because` we get:



That looks like a fairly comprehensive list of different ways to spell `because`, but we also get an `unfortunatly` into the bargain. Unfortunately, this might be because the two words are often co-located — but it might also have something to do with the fact that ‘unfortunately’ appeared in the set of words the transformation vector was built from.

Applied to `and` we get:



There are a couple points to note about this list.

* All the examples are typos — you can’t really spell `and` wrong. As such they are clearly influenced by the QWERTY keyboard layout (s and f sit either side of d, and b is next to n), but the list is missing `amd`. The reason is that `amd` is more commonly found referring to the computer chip company, and so it’s seen as a correct spelling in close proximity to `athlon`, `pentium`, `intel`, and `cpu`.

* It still contains `and`. We could fix this by moving further in the misspelling direction — by adding a multiple of the spelling transformation vector.

In contrast to `and`, none of the candidates for `because` are fat-finger typos. This points to one of the limitations in this approach to spelling correction — we can only deal with cases that appear in the word vector vocabulary.

**A spelling correctness score**

With a little care, we now have a way of generating common spelling mistakes for a given correct spelling. Now all we need is a long list of correctly spelled words.

I don’t have this either, but the word vectors are stored in descending order of the frequency in which they occur. So, we could just take the 10,000 most frequently occurring words, and assume that the vast majority of them will be spelled correctly.

This would probably be fine, but we can do a little bit better. We can use our initial spelling transformation vector to gauge the likelihood that a word is misspelled. Remember that the transformation vector points in the direction of misspellings. If we project a word vector onto the transformation vector we can measure how far it points in that direction. That is, we take the dot product of the two vectors and call it a ‘spelling score’.

Here are some examples of the highest and lowest scoring words according to this metric. (To exaggerate the score, the following is based on the dot product between un-normalised word vectors and our transformation vector).





Our approach is now:

* take only the best scoring words from the 50,000 most common words (with a few other filters) as our list of correct spellings.

* apply the reverse transformation and find nearest neighbours, filtering these candidates by excluding options that are clearly different words, to build a large set of examples of correct — incorrect spelling pairs.

* calculate a revised transformation vector as the average difference of the word vectors for these correct — incorrect pairs.

**Comparison of the initial and spelling vector**

The new vector is essentially a more extreme version of the original. Correlation between the 2 vectors is over 90%, but the variance of the new vector is much higher. The following chart plots the new vector in blue and the original in orange:

![](https://cdn-images-1.medium.com/max/1600/1*z5urnX2tOW2LnVXPr6Qfiw.png)

Applied to our original test set of 17 examples, it now gets them all right — even ‘pharaoh’!

Applied to the 14,000 or so pairs that we built the transformation vector from, and with a few extra hacks and tweaks, we end up getting over 90% right. Looking through some examples of the errors, it’s clear that many of these represent mistakes in the training set — a consequence of the automated manner in which we built the dataset.

**Why do spelling mistakes cluster in vector space?**

We are now in a better position to explain why all the spelling mistakes have been shunted into a parallel space? The most obvious clue lies in lists of best and worst spelled words presented above. The ‘best’ spelled words are those you might find in a serious news article; the ‘worst’ spelled words are mainly examples of informal language.

You find two quite distinct groups of text content on the web:

1. Content that has been carefully proofed, edited and/or spell checked before publication.

2. Unfiltered user-generated content from forums, emails, twitter, etc.

The second set of texts are far more likely to contain spelling mistakes. Seeing words like `plz` or `ROFL` is going to be a very strong indicator you are looking at user-generated content. As such, you are then more likely to see typos and spelling mistakes. Equally, when we start reading `percentage`, `economics`, `Government`, etc, it’s reasonable to guess the text came from a news source and therefore will be spelled correctly.

Another piece of supporting evidence is that if we try to use the GloVe word vectors trained on a Wikipedia dump and the Gigaword news corpus — data that will include a lot less user generated content — the method doesn’t work at all well. There is still an area/direction for spelling mistakes, but it is much less well defined, with fewer examples of misspellings.

If the argument set out above is correct, then our ‘spelling score’ is better viewed as a ‘formality’ score, indicating how likely a word is to have come from a published source as opposed to a user.

To test this idea — and to see if this new metric might be of any practical use — I built a set of data consisting of BBC business news stories from [this source](http://mlg.ucd.ie/datasets/bbc.html), and compared it to a set of samples from the [IMDB movie reviews dataset](http://ai.stanford.edu/~amaas/data/sentiment/). Each sentence or review gets a score based on the average ‘formality score’ of the words within that text.

The results aren’t great, but the method does manage to separate at least some of the texts, as can be seen in the following plot. The BBC news story scores form the blue distribution, with the IMDB reviews in orange.

![](https://cdn-images-1.medium.com/max/1600/1*NkgyTjp7OSD1kdPO5G0G3g.png)

**Final thoughts**

Aside from being an interesting feature of word vectors, the parallel world of spelling mistakes / informal language, allows us to create a simple lookup table to correct common misspellings. This ought to have many applications, but should be especially useful as a very simple first step in cleaning up raw text data.

However, it is worth acknowledging its limitations:

* Even if it worked perfectly, it could only ever correct misspellings that appear in the pre-trained word vector vocabulary.

* It cannot correct typos that occur `partw ay` through a word. It can only operate on a single word token.

* Similarly, it cannot correct errors where `twowords` are missing a space.

* Although the word vectors are built on context, we are trying to correct words in isolation. As such, the method cannot tackle the more difficult cases where a typo or misspelling creates a different (but correctly spelled) word, as in our `amd` example above.

* As noted above, we tend not to have examples of typos for longer words, presumably because they don’t occur often enough. As such, we can correct a `misstake` but not a `misdtake.`

There are lots of ways you might go about building on the basic method outlined here to cater for these limitations. For example:

* Using a character-rnn to learn patterns from the examples of common misspellings we have identified in the process above, which could then be used for out of vocabulary words. An interesting thought here is that you might want to split the vocabulary by word length, and then train on the short word examples to learn the patterns for QWERTY based typos.

* Using a language model to build new word vectors to expand vocabulary.

* Using more traditional spell checking techniques to see if you can split a words that have poor spelling/formality scores into two or more separate correctly spelled words. Even here the structure of the word vector space can help: for example, the nearest neighbours of `downloadand` are `theand`, `yourand`, `toand`,`ofand`.

The final very word is one of thanks — to Rachel Thomas and Jeremy Howard for creating [fast.ai](https://fast.ai). The lectures are a fantastic resource, and it now has a growing, enthusiastic and supportive community. If, like me, you are starting out in machine learning and for some reason haven’t already come across their course, I can’t recommend it highly enough.

|¹ Although no-one to date appears to have taken the step to turn this feature into a practical dictionary / spell checker, at least some in the machine learning / NLP community were aware of the relationship within the word vector space. For example, Jake Mannix mentions it in [this presentation](https://www.youtube.com/watch?v=IwLK2A4eFIs).

#### More where this came from

This story is published in [Noteworthy](http://blog.usejournal.com), where thousands come every day to learn about the people & ideas shaping the products we love.

Follow our publication to see more product & design stories featured by the [Journal](https://usejournal.com/?/utm_source=usejournal.com&utm_medium=blog&utm_campaign=guest_post) team.

