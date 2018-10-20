## Intel® Nervana™ Neural Network Processors (NNP) Redefine AI Silicon

原文链接：[Intel® Nervana™ Neural Network Processors (NNP) Redefine AI Silicon)](http://iamtrask.github.io/2015/07/12/basic-python-network/?from=hackcv&hmsr=hackcv.com&utm_medium=hackcv.com&utm_source=hackcv.com)

As our Intel CEO Brian Krzanich discussed earlier today at Wall Street Journal’s [D.Live event](https://dlive.wsj.com/), Intel will soon be shipping the world’s first family of processors designed from the ground up for artificial intelligence (AI): the [Intel® Nervana™ Neural Network Processor](https://newsroom.intel.com/editorials/intel-pioneers-new-technologies-advance-artificial-intelligence/) family (formerly known as “Lake Crest”). This family of processors is over 3 years in the making, and on behalf of the team building it, I’d like to share a bit more insight on the motivation and design behind the world’s first neural network processor.

Machine Learning and Deep Learning are quickly emerging as the most important computational workloads of our time. These methods allow us extract meaningful insights from data. We’ve been listening to our customers and applying changes to Intel’s silicon portfolio to deliver superior Machine Learning performance. [Intel® Xeon® Scalable Processor](https://newsroom.intel.com/press-kits/next-generation-xeon-processor-family/)[s](https://newsroom.intel.com/press-kits/next-generation-xeon-processor-family/) and [Intel data center ](https://www.intel.com/content/www/us/en/servers/accelerators/accelerators.html)[accelerators](https://www.intel.com/content/www/us/en/servers/accelerators/accelerators.html) are powering the vast majority of general purpose Machine Learning and inference workloads for businesses today. We continue to optimize these product lines to support our customers’ evolving data processing needs. The computational needs of Deep Learning have uncovered the need for new thinking around the hardware required to support AI computations. We have responded to this by listening to the silicon and designing a new chip for Deep Learning called the Intel® Nervana™ Neural Network Processor (Intel® Nervana™ NNP).

The Intel Nervana NNP is a purpose built architecture for deep learning. The goal of this new architecture is to provide the needed flexibility to support all deep learning primitives while making core hardware components as efficient as possible.

We designed the Intel Nervana NNP to free us from the limitations imposed by existing hardware, which wasn’t explicitly designed for AI.

 

### **New memory architecture designed for maximizing utilization of silicon computation**

Matrix multiplication and convolutions are a couple of the important primitives at the heart of Deep Learning. These computations are different from general purpose workloads since the operations and data movements are largely known *a priori*.  For this reason, the Intel Nervana NNP does not have a standard cache hierarchy and on-chip memory is managed by software directly. Better memory management enables the chip to achieve high levels of utilization of the massive amount of compute on each die. This translates to achieving faster training time for Deep Learning models.

 

### **Achieve new level of scalability AI models**

Designed with high speed on- and off-chip interconnects, the Intel Nervana NNP enables massive bi-directional data transfer.  A stated design goal was to achieve true model parallelism where neural network parameters are distributed across multiple chips.  This makes multiple chips act as one large virtual chip that can accommodate larger models, allowing customers to capture more insight from their data.

 

### **High degree of numerical parallelism: Flexpoint**

Neural network computations on a single chip are largely constrained by power and memory bandwidth.  To achieve higher degrees of throughput for neural network workloads, in addition to the above memory innovations, we have invented a new numeric format called Flexpoint.  Flexpoint allows scalar computations to be implemented as fixed-point multiplications and additions while allowing for large dynamic range using a shared exponent.  Since each circuit is smaller, this results in a vast increase in parallelism on a die while simultaneously decreasing power per computation.

 

### **Meaningful performance**

The current AI revolution is actually a computational evolution. Intel has been at the heart of advancing the limits of computation since the invention of the integrated circuit. We have early partners in industry and research who are walking with us on this journey to make the first commercially neural network processor impactful for every industry. We have a product roadmap that puts us on track to exceed the goal we set last year to achieve [a 100x increase in deep learning training performance](https://newsroom.intel.com/news-releases/intel-ai-day-news-release/) by 2020.

In designing the Intel Nervana NNP family, Intel is once again listening to the silicon for cues on how to make it best for our customers’ newest challenges. Additionally, we are thrilled to have Facebook* in close collaboration sharing their technical insights as we bring this new generation of AI hardware to market. Our hope is to open up the possibilities for a new class of AI applications that are limited only by our imaginations.

We hope you’ll join us on this exciting journey to build the future of Artificial Intelligence.

 