# What was AI in 2017?

When I started grad school in 2005, AI was still a taboo word, an anachronism used only by centuries-old emereti and some of the more colorful cranks hanging around the edges of the academic system (the proper nomenclature was “machine learning”). I was told by folks on the admissions committee in no uncertain terms that Systems was much more important — UT was on a hiring tear in Systems at the time — and that if I wanted to be competitive on the job market I should study Systems, and stay away from AI.

![](https://cdn-images-1.medium.com/max/1200/1*roIBTjEF5JLaifkCbp2EOQ.png)

Fast forward to 2018, where just last month Jeff Dean at Google Brain [released](http://learningsys.org/nips17/assets/slides/dean-nips17.pdf) mind-blowing work on learned index structures — that is, Machine Learning replacing core heuristics used in computer systems design. The tables have turned, and now AI is eating software.

The set of algorithms and technologies that we colloquially call “AI” are rapidly changing what is possible with software. This is already profoundly affecting people’s lives in unforeseen ways as such algorithms become incorporated into core social institutions, like healthcare, criminal justice and education (as Kate Crawford notes in her [NIPS 2017 keynote](https://www.youtube.com/watch?v=fMym_BKWQzk)). Looking forward, as AI technology continues to move ahead, it is clear that it will leave large swaths of people, organizations and institutions behind (Eric Brynjolfsson, [“The Second Machine Age”](http://secondmachineage.com/)). This has lead to blunt calls for [regulation of AI](https://techcrunch.com/2017/07/19/elon-musk-clarifies-that-ai-regulation-should-follow-observation-and-insight/) by Elon Musk, [MIRI](https://intelligence.org/summary/) and [others](https://www.nytimes.com/2017/09/01/opinion/artificial-intelligence-regulations-rules.html).

Last year, Ruchi Sanghvi (founder of [South Park Commons](http://southparkcommons.com/)) and I wanted to better understand **how** the specific technologies that are driving the AI revolution worked and precisely **what** changes might be coming in the future. In fall 2017, we produced a technical speaker series charting some of its breadth and impact. The series sought to answer three questions:

* What fields and disciplines will be the most critically impacted in the next 5–10 years?

* What are the specific technical advances that will drive this transformation?

* What ethical challenges and risk factors must be addressed?

### The 2017 AI Series at South Park Commons

Links to all recorded talks and videos:

* [“Building creative tools with machine learning”](https://vimeo.com/251555563) (David Ha and Adam Roberts @ **Google Brain**)

* “[AI + Quantitative Finance](https://vimeo.com/237834525)” (Justin Nelson @ **Crabel Capital Management**)

* “[Distributed Tensorflow](https://vimeo.com/234601530)” (Jonathan Hseu @ **Google Brain**)

* “[Interactive and Interpretable Machine Learning Models for Human Machine Collaboration](https://vimeo.com/234601515)” (Been Kim @ **Google Brain**)

* “[The Future of AI Hardware](https://vimeo.com/238818665)” (Dave Patterson @ **Google Brain, UC**Berkeley, Bryan Catanzaro @ **Applied Deep Learning Research at NVIDIA**and Andrew Feldman @ **Cerebras Systems**, Cade Metz @ **The New York Times**)

* “[AI Risk & Safety](https://vimeo.com/234601511)” (Daniel Dewey & Dario Amodei @ **OpenAI**)

* “[AI & Agricultural Robotics](https://vimeo.com/230531571)” (Lewis Anderson @ **Traptic**)

* [“AI in Information Security](https://vimeo.com/230502013)” (Clarence Chio @ **Shape Security**)

* “A perspective on TensorFlow, Cloud TPUs, and machine learning progress” (Zak Stone @ **Google Brain**)

Below are a few highlights from what we learned.

### The second golden age of computer architecture

Dave Patterson believes that we are at the beginning of a second “golden age of computer architecture”, with a Cambrian explosion of new and old companies focused on making matrix multiplication primitives incredibly fast. Hardware titans like NVIDIA and Intel, new entrants like Google TPU, and smaller startups like Cerebras are driving investment in a variety of new hardware models: GPUs, ASICs, TPUs, and more. Hardware capabilities are rapidly co-evolving with software needs and its uncertain what architecture(s) will ultimately win out.

Watch the full panel here:



### AI is eating software

With the rise of machine learning frameworks, clean abstractions and modular design patterns inherent to the practice of software engineering are being replaced by high-dimensional floating-point tensors and efficient matrix multiplication. As this trend continues, it will necessitate new entirely new engineering paradigms:

* [Andrej Karpathy’s Software 2.0](https://medium.com/@karpathy/software-2-0-a64152b37c35): “Neural networks are not just another classifier, they represent the beginning of a fundamental shift in how we write software. They are Software 2.0.”

* [Pete Warden’s Deep Learning is Eating Software](https://petewarden.com/2017/11/13/deep-learning-is-eating-software/): “In ten years I predict most software jobs won’t involve programming. “

At SPC, we welcomed [Zak Stone](https://research.googleblog.com/2017/05/introducing-tensorflow-research-cloud.html), the PM for Tensorflow and Cloud TPUs at Google Brain, to give a talk at where he wove together a several challenging threads that are shaping future machine learning progress, including the rise of scalable frameworks for differentiable programming such as Tensorflow, and the race to build cross-platform “linear algebra” compiler to keep up with the proliferation of new hardware (read more [here](https://medium.com/south-park-commons/matmul-is-eating-software-afebccda1745)).

### Unlocking human creativity

Finally, [@hardmaru](http://blog.otoro.net/) and [Adam Roberts](https://research.google.com/pubs/104881.html) joined us to discuss how AI is making strides in augmenting human creative work, highlighting work from the Google Magenta project on building ML-powered creative tools to help artists and musicians. They covered recent advances in the space including vector drawing generation with [Sketch-RNN](https://magenta.tensorflow.org/sketch-rnn-demo) and music generation with Google’s Magenta platform, which won [Best Demo](https://nips.cc/Conferences/2017/Schedule?showEvent=9762) for the second year in a row at NIPS (check out their in-browser demo [here](https://magenta.tensorflow.org/performance-rnn-browser)).



AI has the potential to unlock true creative collaboration, “understanding” the visual or musical world not just in terms of pixels or notes, but in terms of the underlying concepts and themes. That is, AI can be seen as a bridge between the powerful, but rote automation inherent to classical computing, and the softer, more intuitive world of human concept understanding.

### What’s next for AI?

This year saw tremendous advances in AI, but as Josh Tenenbaum put it in his [CCN talk](https://www.youtube.com/watch?v=Z3mFBEOH2y4) this year: “All of these ‘AI’ systems we see, none of them is ‘real’ AI”. That is, we’re able to interpolate pretty well between known examples, but we’re still nowhere close to having algorithms that can learn novel concepts from scratch. Furthermore, our algorithms learn nowhere **nearly** as efficiently as animals or humans (for example: requiring hundreds of thousands or millions of examples of cats in order to learn to classify cats).

I’m personally excited by research in transfer-learning and one-shot concept learning that aims to reduce the amount of training data required. Not only will this make our models more efficient, it will effectively democratize models training — you’ll no longer need the data and infrastructure scale of a Google or Facebook to learn effectively (Yann LeCun recently [addressed this](https://www.youtube.com/watch?v=uYwH4TSdVYs&feature=youtu.be)).

### Looking forward to 2018

In 2018, we’re excited to continue the series and we’re going to hone in our focus on two core themes:

* The transformative effects AI will have on **digital media**, and

* The **ethical challenges** of this and other new developments in AI.

Note that these two areas have non-trivial overlap! For example, AI coupled with the rendering and special effects pipelines of Pixar or ILM has profound implications on what constitutes evidence in criminal cases. There is incredible potential for abuse for systems that, e.g., can[ synthesize photo-realistic video of Obama](https://www.youtube.com/watch?v=9Yq67CjDqvw) synced up to any audio clip. What will the next 5 to 10 years look like as these technologies mature? How will our existing institutions have to change to cope with this?



To stay in the loop on the discussion in 2018, join the South Park Commons [email newsletter](https://mailchi.mp/116e4aebefbc/southparkcommons).

