# How To Create A Machine Learning Framework From Scratch In 491 steps



**Note**: We already posted a a short post-mortem of this project [on reddit](https://www.reddit.com/r/MachineLearning/comments/71ji5v/p_sigma_creating_a_machine_learning_framework/) about 4 months ago. This is a significantly expanded, more detailed and polished version with detailed insights, design choices and code examples.

#### All of Machine Learning in One Sentence

Alright, we got you covered: here is the entirety of machine learning, this article and our framework in a single sentence, from bits on one end to the bits on the other end that output your funny cat images:

> From images, text files, or your cat videos, bits are fed to the data pipeline that transforms them into usable data chunks and in turn to data sets,which are then fed in small pieces to a trainer that manages all the training and passes it right on to the underlying neural network,which consists of many underlying neural network layers connected through an arbitrarily linear or funky architecture,which consist of many underlying neurons that form the smallest computational unit and are nudged in the right direction according to the trainer’s optimiser,which takes the network and the transient training data in the shape of layer buffers, marks the parameters it can improve, runs every layer, and calculates a “how well did we do” score based on the calculated and correct answers from the supplied small pieces of the given dataset according to the optimiser’s settings, which computes the gradient of every parameter with respect to the score and then nudges the individual neurons correspondingly,which then is run again and again until the optimiser reports results that are good enough as set in a rich criteria and hook system,which is based on global and local nested parameter-identifier-registries that contain the shared parameters and distribute them safely to all workerswhich are the actual workhorses of the training process that do as their operator says using individual and separate mathematical backends, which use the layer-defined placeholder computation graphs and put in the raw data and then execute it on their computational backend,which are all also managed by the operator that distributes the worker’s work as needed and configured and also functions as a coordinator to the owning trainer,which connects the network, the optimiser, the operator, the initialisers, which tell the trainer with which distribution to initialise what parameters, which work similar to hooks that act as a bridge between them all and communicate with external things using the Sigma environment,which is the container and laid-back manager to everything that also supplies and runs these external things called monitors, which can be truly anything that makes us of the training data andwhich finally display the learned funny cat image… from the hooks from the workers from their operator from its assigned network from its dozens of layers from its millions of individual neurons derived from some data records from data chunks from data sets from data extractors.

On second thought, you may want to read the entire article instead.

### Abstract

2.5 years ago we asked the machine learning subreddit for advice on a machine learning topic for our high school thesis. We ended up researching and writing a machine learning framework from scratch. It can’t do as much as all the others, isn’t as fast or as pretty, but we still think it’s kind of cool.

We briefly outline the project history, then describe our research and planning process up to the final design choices, technical implementation details. Mathematical and programming expertise are not required but helpful for the more technical explanations.

Up front, o[ur github repository](https://github.com/ThinkingTransistor/Sigma) and a brief demo of our graphical client:



### The Spark

This is the story of how we ended up creating a machine learning framework for our high school thesis. It’s an epic tale of dragon-sized bugs, elves and the amazing journey of our two tragic heroes that save the galaxy with their incredible machine learning framework machinery — well, scratch all the cool bits (except for the magic) and your expectations will be right on. So, if you’re looking for a proper tutorial on designing and creating your own machine learning framework, we can’t deliver and we don’t think you will ever get one. However, there are many interesting lessons in what we learned, in how we went about creating something of this scale from scratch, and in the many pains and the few insights we experienced during development.

Once upon a time, about 3 years ago, it all started: we saw a video of [MarI/O](https://www.youtube.com/watch?v=qv6UVOQ0F44), a Super Mario AI that could learn to play Super Mario levels. The AI was rather primitive, building a map from current surrounding blocks to button presses through many hours of trial and error. We thought that was about the coolest thing of all time and wanted to do something similar for our high school thesis. But.. what, exactly?

#### Aspiring Ideas: Another Arcade AI Agent

Many moons ago we asked for advice on what kind of machine learning project we could feasibly do for our senior year high school thesis (see [original thread on reddit](https://www.reddit.com/r/MachineLearning/comments/3jn5xf/question_what_interesting_machine_learning/)). In an ambitious decision to do something cool, we proposed a time investment of about 1000 hours total (as in 500 hours each over the course of 8 months) — for what project, we didn’t know yet, we just knew we wanted to do something. And something cool at that, of course. We were met with a lot of generous help: advice ranged from reproducing papers over implementing specific real-world solutions to understanding the theory and then doing whatever particularly peaked our interest.

Our own ideas wandered from general purpose image recognition to trying to “simply” improving existing models. Both of which were already well explored and researched topics that, while there certainly is still a lot to of room for improvement, lack the novelty and immediate excitement factor we sought. After many more weeks of consideration, we set our mind on implementing something along the lines of DeepMinds arcade game AI, like this:

![](https://cdn-images-1.medium.com/max/1600/1*EM8x5jAL-SeUUG7b4anCQg.gif)

Our “plan” was to implement a similar arcade-game-playing agent in an existing framework and then generalise it. Visionary as we were, we even fantasised about a program you could drag over games and have it figure everything out organically. We meant to achieve this gradually by e.g. adding more ways to supply the agent with scoring, such as from text recognition (i.e. we wanted it to be able to figure out how well it’s doing just by looking at the game instead of us telling it so for every move). For some inexplicable reason, we thought that would be a reasonable undertaking for two high school seniors. Mainly because we vastly underestimated how difficult it is to even come up with the right model, never mind feeding it with the right data in the right training environment in an entirely new framework.

Soon after getting more serious about a detailed plan we realised that

1. we had no idea what we were doing and

2. it would be a shame to do all that work from scratch and have it be so “arbitrarily specific”*.

*=overly specific according to our thinking back then. Looking back, just re-implementing a proper arcade game AI would have been a great project. Oh well. We just had really no idea of the scale of anything similar.

#### Pivoting and Generalising (a.k.a. “Scope Creep”)

Before the proper programming could commence, we had to study proper machine learning and get to know what the heck we were doing. We figured this might take a while and allocated over 3 months to research and planning. As this was a rather untypically long for just the research part of a high school thesis we also used that time to write the theoretical part of our thesis so we would have at least something to show others for our efforts. Also, writing everything down in an understandable and structured manner was helpful for our own understanding — two birds, kind of.

A few weeks into our senior year we had to hand in an official target definition for our thesis. Because we still had no idea what we were actually going to do we set the most generic “goals” we could get away with — targets like “writing a machine learning API” and “persistent storage”. To give you an idea of how broad everything was: for a concerning number of months our project was officially titled “Software framework for diverse machine learning tasks” with an even longer and even more obtuse subtitle.

During study and halfway into our first attempt to draft a proper, usable target definition, our plans gradually shifted from the original machine learning model for playing specific types of games to an “any kind of visual input” learning framework and then finally to an “anything” machine learning framework — because why not, it seemed like an interesting challenge and we figured “screw it, let’s start, we’ll see how far we get”. And thus it began.

### Research

Alright, so we had decided to write a machine learning framework. Starting with nothing, an empty project file, staring back at us with accusing emptiness (and a dragon-bug lurking in the distance). Now, how exactly does one write an entire machine learning framework? It takes many weeks to become decent in using a given framework, and that with proper tutorials and forums for help and guidance. Creating a machine learning framework is a whole other story, an utterly different beast of a task, with no 12 step guidelines and skillfully narrated animations to follow.

During the many months between the decision to make the framework and the first line of code we had no idea what we actually had to implement. No idea how it could work. No idea where to start looking even. How does such a framework work? Where does the data go in, where does the magic happen, how does the data get back out? How do you make it employ GPUs or even multiple CPU cores simultaneously? How can you let the user set the floating point precision level? … there’s much more, but you get the point: it’s very confusing for someone who hasn’t studied AI programming before.

Researching machine learning as an aspiring beginner is hard. There is no shortage of great tutorials on introductory topics but the articles quickly dry up at an intermediate level. No proper top-level overviews of inner workings. A constant wave of new and conflicting terms, definitions and not-so-obvious-“but the actual code / conclusion is obvious and left as an exercise”-articles is frustrating. And of course, no explanation anywhere on how to write your own framework, but that’s besides the point and to be expected.

After a few weeks of extensive internet research we got a basic grasp of how this machine learning thing worked: something with mathematical functions, certain points, some calculus to make some metric go down. For a clear and common understanding of the subject matter we summarised everything we found out in the theoretical part of our thesis. The resulting 21-page theoretical summary would serve as a written overview of the most popular machine learning terms and concepts. The idea was that we would later come back to revise and update it — we ended up doing neither out of impracticality (and laziness), but it was still helpful during the initial learning period.

#### Our Machine Learning

The world of machine learning is vast, incredible and incredibly confusing at times, all at the same time. We aren’t anywhere close to being knowledgeable experts but have gained a passable level of understanding through authoring this framework. But beware, the following explanations are simplified and tailored to our very specific use case (i.e. training of neural networks with a known correct result). This is only an insight into our understanding of machine learning which translates into how we think a machine learning framework could work, in turn defining how our framework actually works.

**Note**: If you already understand machine learning with neural networks well or just want to skim the architecture you might want to skip to the next part, “The Architecture of Our Machine Learning Framework”.

#### Our Neural Networks

Any kind of neural network is a specific chain of mathematical operations — no matter what kind of layer types and network architecture. Imagine your regular linear function y(x) = x*k+d: Two parameters, k and d, which affect how the function y(x) behaves; d changes the offset, k changes the steepness. The values of these parameters doesn’t affect the type of function (i.e. “layer”), you can just change them to adjust the function behaviour.

Artificial neural networks consist of many of these functions, albeit “larger” ones to enable more sophisticated ways of combining inputs and parameters, but same principle. Artificial neural networks consist of hundreds to many billions of these little functions, which happen to be called “neurons”. These neurons are grouped into layers, and each layer takes inputs from previous functions in the previous layer and passes the results on to the next layer. Occasionally layers also apply what’s called an “activation function”, which just means putting the result of a layer through another function to make sure it remains in a certain range (often [-1, 1] or [0, 1] because it’s convenient and works well). Inputs, like the colour values of an image, are fed to the first layer and then the final output, like what kind of animal is in that image, is taken from the last layer, often in the form of x values representing the likelihood of the x different animal types (e.g. bear, ape, dragon-bug).

For convenience (and because it makes parallelisation-hungry GPUs happy) neural networks are most commonly expressed as matrices; that is, inputs, intermediate results (i.e. values at each neuron) and parameters are all matrices, which is just a bunch of values grouped together. There is no magic, there just was — and still is — a bunch of trial and error to figure out what kind of layer configurations work best for what kind of problem and data. That’s it.

#### Our “Training”

There is no actual “training” either, not of the kind the word would imply. Neural networks’s don’t get smarter by themselves, nor do they ever actually get smart. It’s up to us to make them appear smart, and we do that by fitting a large function (the neural network) to a hopefully very large set of input-expected-output pairs as good as we can. The key here is that the resulting function should be able to work well with new, previously unseen inputs. The more expected input-output pairs we use, the better the function can generalise and “recognise a pattern” (i.e. work with new data).

So, no fairy-tale “training”, but instead fitting a mathematical function to data pairs (e.g. x/y-pairs). And how does “fitting a function” work? Take your basic linear least-squares fitting algorithm, which finds a linear function that minimises the vertical distance to all data points. It essentially attempts to plot the line that best describes the x / y relationship in some data. Like this:

![](https://cdn-images-1.medium.com/max/1600/1*BDIA6zfqhqd5iaM4i6RxdA.png)

We can all agree that the above red line approximates the location of the blue points very well; in other words, we have fit the function well to the data. This kind of fitting is simple and only takes one step to do, i.e. the line won’t fit any better if we try to fit it again the same way with the same data.

Typically, neural networks are used to describe more complex relationships than can be represented with a simple line. The idea is the same, but it takes more than one step to get to the best fit (i.e. “line”). Because unlike with lines, it’s often not possible or feasible to calculate a perfect solution by solving a few simple equations due to the complexity and amount of parameters involved in the “function” of a neural network. Imagine millions of individual lines in conjunction — it is impractical to calculate the perfect solution.

Instead of solving the entire neural network function, these kind of many-parametered functions are fitted step-by-step by individually adjusting their parameters in the direction we think is right (i.e. increase or decrease). And to know which direction seems right, we first have to quantify what right means. In the least squares example above, the quantitative metric for right fit is based on how far away the dots are from the line. In mathematical terms, that is the squared distance the blue points vary from the red line on average — fittingly called variance. The closer blue dots are to the red line, the less variance; the less such error the red line has, the better the fit.

When fitting neural network “functions”, the error metric usually represents how well the function does by comparing its output to the expected output. This can be accomplished by

* simply computing the absolute difference between what our function did and what we want it to do (this is called squared error) or

* by considering how much better or worse it did in the past in addition to how well it did this time (like in Adagrad, Adadelta or ADAM) or

* by using a million different variations, all of which perform better for some and worse for other neural networks and data types.

So, the neural network is supplied with inputs, we look at its outputs and compare that to what it should be. Using that difference, we calculate how wrong it was with an error metric. Now we go back — propagate backwards, so to speak — and analyse how every single one of the function parameters influenced the error metric. Luckily, we don’t have to to this manually but can instead employ calculus, more specifically, partial derivatives.

The value of the partial derivative of a single parameter with respect to the error metric tells us how much it influenced the error metric (absolute value) and in what way (positive or negative). It represents a kind of impact score, which we then add to and multiply by the current parameter value and a few training parameters and update it to the new one. One of these training parameters is almost always a value used to change how much the parameter should change with a single update (the stepping rate or learning rate).

A useful visualisation of neural network performance with certain parameters as starting conditions is a simple graph — each axis is a parameter, the illustrated value is the error metric at that point. The generated error surface is used to visualise the performance of a particular network at specific parameter points. And more commonly, the performance and behaviour of certain optimisers is compared by plotting them like this:

![](https://cdn-images-1.medium.com/max/1600/1*Y2KPVGrVX9MQkeI8Yjy59Q.gif)

Optimisers are supposed to avoid local minima (like the one in the centre) and head for the global minimum as fast as possible. Different optimisers perform better for different surfaces, but there are a few that have proven useful for common use cases. For our framework, you should have heard of:

* **Gradient Descent**: The basis for pretty much all other optimisation algorithms. As the name suggests, it is the most basic version of a gradient-based optimiser, naively using the same learning rate to update all parameters based on their gradients from the last step.

* **Stochastic Gradient Descent**: Same as gradient descent but with only 1 sample per iteration (i.e. the so called minibatch size is fixed at 1, meaning that only a single data record is used per training pass).

* **AdaGrad**: Considers a history of past gradients to adjust the learning rate for every parameter (e.g. larger gradients result in smaller learning rate).

* **AdaDelta**: Also considers a history of past gradients to adjust the learning rate for every parameter without any requirement for preset parameters from the user (such as an initial learning rate). Conceptually similar to AdaGrad, best to just try both and see what works best.

Neural networks typically don’t just have 2 parameters, but rather hundreds to billions, rendering their error surfaces many-dimensional and complex (and thereby unfeasible to visualise directly). In real world use, the error surface isn’t as flat as the one for the previous line example: we can’t just go all in on whatever direction we currently think is best. It’s most effective to constantly adjust and advance parameters by a small constant to find the best combination for your function and data. That’s it, one iteration done, rinse and repeat until satisfied with the neural network performance. If not satisfied and no improvement in sight, reset and choose different parameters, another network layout, different data, or different optimisers. Remember that being “satisfied with performance” does not require and even excludes perfect accuracy — the idea being that a properly trained network (or human) still makes mistakes, and if there are none, it’s a good sign it just memorised the answers to our sample test and is useless in the real world.

### The Architecture of A Machine Learning Framework

As soon as we knew a bit about the art of machine learning we eagerly advanced to the creating-the-actual-framework part. Because there are no guides for that, we resorted to reading the source code of established frameworks — all for us relevant parts, many times, until we understood internal structure and control flow. There is no special ingredient here, all it took was time and electricity. In the meantime, we had decided to use C# as our primary language — mostly because we were already very familiar with it and didn’t want to also have to learn a new language, but officially also because there were no proper neural network frameworks for .NET.

Alongside reading the source code of machine learning libraries (mainly Deeplearning4J, Brainstorm and Tensorflow) we sketched out how we wanted our own framework to be used. We felt like there was some unnecessary confusion in getting to know machine learning frameworks as an outsider and we set out to design our API to avoid that. Note that because our design makes sense to us doesn’t mean that it makes more sense than the existing ones to other people, nor do we recommend everyone wishing to use machine learning to write their own framework, just to spare their own sanity.

#### How to Talk Machine Learning to a Framework

How do you make any framework do what you want it to do? How do you get it to train a specific model from some specific data using a specific optimiser on some specific hardware while visualising the outputs in some specific configuration? There are a great number of things a machine learning framework should be able to do, and all of them should be easily usable, configurable, interchangeable, and readable. This is not a problem unique to machine learning frameworks; all kinds of programming frameworks are supposed to be used in some specific way. Because everything depends on this user-facing side, it’s usually considered first, so that’s what we did too.

Many of the well established machine learning frameworks support the general workflow of defining either the computation graph directly or the model structure using layers (as with neural networks). We thought the latter was easier for newcomers because you wouldn’t even have to know what a computation graph is and adopted that for our design. Our envisioned workflow was inspired by our mostly object-oriented programming experience, as is evident from our first “official”code example **draft**:

* **Create a Sigma environment** to contain and manage everything else



* **Optionally add “monitors”**to monitor the enviroment (e.g. in a GUI)



* **Tell the monitors to get ready** before adding trainers



* **Define a dataset** to use with our data processing pipeline (ETL style)



* **Define a network architecture** using neural network layers



* **Create a trainer**within the previously created enviroment



* **Assign structural parameters** to the trainer (network, initialiser, data)



* **Assign behavioural parameters** to the trainer (optimisers, hooks)



* **Configure optional settings**for monitors or other systems



* **Start the environment** (that starts the trainers that start the operators that start the workers that start the actual training)



All in all, it was intended to look and feel more like a smart configuration file than actual programming as we thought that would be the easiest to read, understand and write. Our naïve ideas on how a machine learning should look like were inspired by our C#/Java based programming experience.
 
It should be noted that the final framework is very similar to what we envisioned early on with these code examples: adjusting a few syntax tidbits and interchanging with the exact names, the above example from about a year ago can be used 1:1 in our current framework. The jury is still out on whether that’s a sign of good or really bad design. Also note the python-style variable keywords notation for layer constructor arguments, which was soon discarded in favour of something that actually compiles in C#.

#### Core Components of Our Machine Learning Framework

After all this research, the in-depth code examples, and the structure sketches we thought our framework needed, we finally arrived at the principle architecture for what we call “Sigma.Core”. Our overall architecture is divided into core components which represent individual namespaces (logically separate groups of functionality and code). Core components interact with each other using exposed interfaces and the lifecycle. While our lifecycle was designed upfront, most of the interfaces were defined and changed as needed.

#### Utils: Common Helpers, Observers, Exceptions.. and Registries

Utils contains mostly boring and standard, well, utility stuff. But also registries, which represent an enhanced key-value store and are a key part of our architecture. Registries enable us to keep a global access-protected and type-protected data store across multiple threads and even processes. This originated from our desire to analyse and visualise everything in any way, for which we required a global way to access everything by identifier — a registry.

Our registry implementation is a classic key-value table (i.e. a “dictionary”) with a string key and a value of any type, which in itself may contain more registries. The type of value may be restricted using a special data type table, which protects it from nasty errors (e.g. when changing a value modifier from 0.4 to “banana”). Nested registries are resolved using registry resolvers in dot notation, like “network.layers.1-input”. Nested identifiers may also include fancy wildcards and type tags in angel brackets (e.g. “network.layers.d*<fc>.weights” for all layers tagged as “fc” that start with “d”).









Besides registries, the Utils component also defines time dependent variables and constants. These constants are used for timekeeping to communicate about certain events happening, such as an optimisation iteration, or a pause in execution, or a complete reset. All these events are what we call time scales — abstract units of a certain occurrence that we might want to time with. And that timing is done through time steps, which are countdowns of a certain time scale event happening a certain number of times. This is particularly convenient for executing specific code when e.g. the optimisation algorithm has completed 10 iterations or the trainer was halted again.

#### Data: Datasets, Data Processing, Data Extraction, Data Sources

The data component is — very surprisingly — everything data. It contains

* the actual datasets,

* the data record blocks that make up datasets in various formats,

* the data records that make up data blocks in various formats,

* the data buffers that make up data records in various formats, and

* the pipeline to load, extract, prepare and cache data blocks from disk, web, or wherever else, and make them available to datasets.

We support two kinds of datasets: extracted and raw. In contrast to extracted datasets, which are extracted from an external source, raw datasets are “manually” populated from code (useful for debugging and experimentation). Data record blocks are parts of a dataset, consist of many individual records, each representing one data row. To avoid loading the entirety of a potentially very large dataset into memory at once, we employ partial data record blocks which are then further split up by data iterators before being fed to the model.

In practice, the code for reading even moderately complex data streams into compliant record blocks turned out rather long and verbose. To balance out the need for detailed configuration in complex cases we added simplified templates as well as ready-to-use datasets. For example, this is the full code for the processing pipeline of the popular MNIST images (28x28 fields monochrome digits for classification):

















#### Architecture: Abstract Model Layout Definitions

Abstract definitions for machine learning models made of layer constructs. Constructs are lightweight placeholder layers defining what a layer will look like before its fully instantiated; only behaviour and parameters without the heavy memory footprint of a full layer. These layers may be in any order (though it’s advisable to put inputs first and outputs last) and connected with however many other layers they would like.







In the above example, input and output constructs are defined and linked manually. Manual linkage and configuration are supported to facilitate arbitrarily linked network architectures beyond linear models. In contrast to these point-to-point models, linear models may be defined through a more intuitive, simplified “stack-via-plus” notation:



#### Layers: Neural Network Layer Implementations

“Layers” is an unfortunate misnomer since the “Layers” component design includes all types of layered structures and not only neural network layers. We started out with just neural networks, but later expanded our architecture to all kinds of machine learning structures that can be divided into “layers”. Nevertheless, a layer in our implementation is for all intents and purposes a neural network layer. Analogue to neural network layers in theory, “our” layers are defined by their

* size (in all dimensions),

* other meta parameters (e.g. name, activation function)

* trainable parameters (e.g. weights, biases)

* behaviour (in code, inferred by their instantiation type)

Note that the split into meta-parameters and trainable parameters is a cosmetic one and not strictly necessary, implemented for usability. The layer-type-specific behaviour is implemented in each layer’s ILayer.Run function, which is called every iteration of the optimisation algorithm by the owning trainer. Precisely, the to-us-mystical layer function is defined in code as:



The layer buffer interface bundles all relevant transient parameters required for a single invocation of the run function; that is, all parameters, inputs from the previous layer and outputs to the next layer. It represents a data container without any special behaviour, merely used to reduce clutter when using the function. As the name IComputationHandler suggests, the computation handler is used to define computations on the parameters in the buffer. The less exciting “is training pass” flag is used to disable training features (such as randomisation) in production mode.

#### Math: Low-level Mathematical Variables and Relations

The math component is exactly what you would expect (or maybe not, our models can’t predict your expectations yet): mathematical and low-level computational definitions, i.e. mathematical variables and their relations. All mathematical variables are programming objects and define interfaces for other variables to interact with by means of operations in the computation handler. These objects can either be scalars (represented as INumbers) or n-dimensional arrays (e.g. vectors, matrices, all represented as INDArrays).

For further abstraction, the user is never presented with the live data but rather with these abstract representations. And even when requested, a copy is returned — the only way to modify the live data is through the given computation handler. This hassle with forcing every data manipulation through the computation handler is highly useful for asynchronous processing. The requested computations can be executed separately without having to synchronise data with the main thread all the time (enormously useful for multi-threading and GPU support). Also, the component is cleaner by separating the concerns of “what to do” and “how to do it” clearly.





Besides, the heavy abstraction of mathematical objects neatly serves the ability to swap and interchange mathematical processing backends without disturbing the end user or the model developer. Want to use your single CPU-core with 32-bit precision for development but then deploy to your magic high-end multi-GPU server farm with 64-bit precision for optimal results? No problem, just change a line in the configuration (i.e. trainer definition) and all your custom layers and models work exactly the same.

#### Training: Detailed Training Process Configuration

The largest component with many sub-components, all concerned with the actual training process. A training process is defined in a “trainer”, which is a container object that may specify the following components:

**Initialisers** define how model parameters are initialised, which can be configured with registry identifiers. For example,



would initialise all parameters named “biases” with a Gaussian distribution scaled by 0.1 (mean 0). Similarly, weights and other parameters can be initialised to random (or other) distributions or custom constants.

**Modifiers** modify registry identifiable parameters according to specific rules at runtime, for example to clip weights to a certain range. Modifiers are a feature we observed in another machine learning framework and deemed convenient for quick prototyping. As such, modifiers were intended to be the simplest way of specifying rules for parameters. As we however invested a lot of time into improving the usability of the substantially more powerful hook system with similar templates, the modifier system became obsolete.

**Optimisers** define how a model learns (e.g. gradient descent). Because we mainly considered neural networks, we only implemented gradient based optimisers. Because there are no algorithmic constraints for the optimiser, the interface theoretically supports any kind of optimisation algorithm, even randomised or genetic ones. For reference, the concerned method from the API which defines a single optimisation step (i.e. iteration):



**Hooks** “hook” into the training process at certain time steps and execute arbitrary code. Communication — albeit only rudimentarily — between hooks is realised using a shared global registry. And using additional helper logic, hooks can be applied conditionally when certain criteria are met, e.g. if the parameter “error” hasn’t decreased for over 5 iterations.

Often, the kind of logic you would want to implement as a hook is very similar to a basic “if this, then that” system — if a new top score has been reached, print all metrics and sound a notification. Or if 1000 iterations are completed and the score hasn’t increased for 5 iterations, stop the training process and store the current network on disk.

The “if this” part is accomplished using the aforementioned criteria, which are used to form conditional pseudo-statements like



Such statements may also include a repeat specifier if the condition has to remain true for a certain number of time steps before the criteria is met (e.g. score has to decrease 5 times). Multiple criteria may also be combined into a new criteria using classic Boolean operators (AND, OR, NOT).

The “do that” part can truly be anything, but there are a few common themes:

* loading network state or parameters (mostly custom / inline),

* storing network state or parameters (Saviors),

* computing metrics based on network state and parameters (Processors),

* scoring network performance using validation datasets (Scorers),

* printing anything to console, file or network (Reporters),

for each of which there are multiple templates and base classes to use or expand if insufficient. Of course, with multiple hooks and multiple worker threads there quickly arises a problem: how to resolve dependencies? What happens when one hook requires the result of another hook?

Hook dependency management to the rescue! This unsuspecting sub-component turned out to be tricky due to a few unforeseen difficulties. The main reason for supporting managed dependencies was to move the burden of ensuring properly ordered execution of all hooks from the user to the framework. Thus, now our system has to figure out which hooks resolve to which dependencies, what to do with cyclic dependencies (hint: ban them) and so on. This part of the problem can be solved fairly easily using a dependency graph and by ordering priorities in certain hooks (i.e. first hooks that get data, then hooks that process data, then hooks that print data).

For some reason we did not anticipate that just ordering the hooks correctly didn’t help the actual execution part when multiple threads are involved, which is always the case in our multithreaded operator / worker architecture. Firstly, the worker thread shouldn’t be “distracted” from its actual job (i.e. doing optimisation) for too long executing these hooks. This can be countered by setting a limit on the amount of time a hook may take and offloading “slow” hooks to a separate worker thread. Naturally, this creates another set of difficulties, namely that this separate thread may not access the original data directly. We can’t just copy everything either, as that’s very slow, so we have to first figure out what part is actually needed, only copy that and then dispatch.

Secondly, and much more painfully, there may be cross-region hooks and therefore cross-thread dependencies. As every person who has ever tried to do multiple difficult things at the same time knows, multithreading isn’t easy. It becomes even harder in performance-critical applications that need to exchange information (i.e. the parameters) and then execute conditional code on shared data based on that information. After lots of trial-and-error, our final solution was satisfyingly simple: do the same thing we did for the first problem, just with more hooks bundled together. We figure out which hooks need to be in such a “bucket execution thread” together by analysing their dependencies, owners and thread ids, which is a basic sorting problem. Tada.

#### Operators: Training Management and Work Delegation

Operators operate the training process. They delegate work to workers and then combine their results according to user configured parameters. Further, operators are an essential design point enabling the simple deployment of multi-core, multi-GPU or even multi/cross-device processing during training. Key to this design is our separation of “global” and “local” processing: The global scope is the most recent global and public version of all data in the operator while the local scope is individual to each worker.

The global state is fetched by workers to their respective local scope. The workers then proceed to duly do their work within their scope, handling events on their own, and report back with their results when they’re done with an iteration. A global timestep event is ejected when all local workers have submitted their work for that timestep (e.g. iteration), facilitating fine control in distributed learning (e.g. notification when everyone is done).

#### Handlers: Low-level Mathematical Processing

The direct low-level processing of mathematical operations is done in the Handlers component. Our backend handlers are specialised mathematical processors that execute mathematical operations for a certain system or device using a certain data type and precision. They apply the operations defined in the Math component using placeholders of n-dimensional arrays and scalars to raw data. There currently is no limit on the accuracy of mathematical operations, giving programmers and maintainers the freedom to favour speed over accuracy when implementing optimised routines.

Backend handlers may implement their processing in whatever way they like, if they correspond to only two important restrictions:

* May not complain or otherwise act up when multiple threads simultaneously request the same operations on different data.

* When the underlying data of a variable is requested, all operations concerning it must be finished when returned.

This may sound trivial, but in fact it requires the backend handler to keep tab on all ongoing operations across all variables and tidy up quickly when someone needs to peek under the hood. For reference, our CUDA (GPU) processor accomplishes this by duplicating all host operations and variables to the GPU and keeping the host memory version as a “shallow copy”. After initial synchronisation, transfers are only done when a result is requested.

#### Sigma: Global Environments For Trainers

The main component that can create and manage Sigma environments. Sigma environments are containers and laid-back managers to all the action — they loosely connect trainers, monitors and environments and enable them to pass messages. A Sigma environment may contain multiple trainers, each of which may be attached to multiple independent monitors simultaneously. The only requirement the Sigma environment has is that all components’ lifecycles must end before itself can shutdown gracefully (it runs in its own thread).

#### Monitors: Talking to The Outside

Because monitors were meant to be separately usable components, they reside outside the core project. Nevertheless, monitors are important components that, when attached to a corresponding Sigma environment and trainer, can provide managed external access to the training process. Essentially, they are how you would typically interact with a Sigma trainer when you’re not a framework programmer — for example with graphical applications, monitoring websites, external logging and so on.

Monitors can fetch any kind of information from the global training data registry, e.g. for visualisation or logging. Special behaviour like shutdown can be injected using commands, a special form of hooks that are only invoked once. Due to their logical separation from other Sigma components, monitors can be used (almost) independently and can also be pretty much anything.

### Implementing It All

We spent about 2 months meticulously researching and planning our framework and had the framework architecture and code examples to show for it. And that’s exactly what we implemented, feature by feature, bug by bug. In the beginning, we mainly worked on Sigma.Core, which is the name of the core component containing most of the Sigma logic. Simultaneously we developed our visualisation interface (the Windows GUI) — two very different parts as we wanted something to demo people as fast as possible and one of us working on the core and the other on the graphics in parallel seemed best.

This way, we were working, programming and testing separately to a common interface for several months until we could finally run first tests on both of our parts combined. Miraculously, it worked! Something was displaying on the cool live graphical interface! We were gods! And then it crashed hard. That’s the entire development of our framework in a nutshell — there are so many individual components, complicated on their own, that have to work together perfectly for many days straight. It was challenging and very frustrating at times, but an ultimately very rewarding experience (see Conclusion).

The specifics of the implementation process are rather dull to an outsider— after all, the more exciting part about creating a framework is the design process, not the implementation, which is comparatively rather mundane. Most of the time things didn’t work and then when we fixed something, we moved on to the next thing that didn’t do as it should. Rinse and repeat until framework is done or maximum insanity is reached (may not be exclusive).

#### Bottom up: Low-level Data and Mathematical Processing

The first goal was getting the data “ETL” (extract transform load, a classic in data processing) pipeline up and running. It is divided in fetching data from a variety of sources, loading them into a dataset, and extracting them as blocks. We then focused on the mathematical processing part — everything that had to do with using math and calculating derivatives in our framework. We based our automatic differentiation component, aptly named “SigmaDiff”, on an F# library for automatic differentiation named “DiffSharp”, which we modified significantly to support any-dimensional arrays, multiple backends, variable data types. Of course, also for performance improvements and bug fixes.

As previously explained, the specifics of making data processing work aren’t very interesting — lots of glue code, refactoring and late-night dragon-bug-chasing because the backpropagation didn’t work as it should with some specific combination of operations, data or solar flares. To illustrate, one of these “fun” bugs was that the derivation (which works backwards from the operation result) wouldn’t work when ending a chain of operations with a specific matrix/matrix multiply operation. After about 3 days of intense debugging it turned out that the original developer of the library we adapted had accidentally flagged the wrong computation component as “constant”, removing all derivatives for that part of the chain.

#### Performance Considerations

Compared to all the backend work, the “middleware” made of layers and optimisers was trivial to implement. There is a myriad of tutorials and papers on the inner workings of neural network layers and optimisers. All it took was translating those formulas and code examples to our own framework, which went smoothly (seriously, for once, things actually worked first try).

Really, that part shouldn’t have taken nearly as long as it did, but there were “unexpected” issues (as unexpected as problems can be during software development). Dozens of bugs, stability and usability issues uncovered during real world testing that our — admittedly quite lackluster after a while — unit testing didn’t catch, which of course had to be taken care of. “Every ”single fix and patch are not particularly exciting to an outsider. However, the issues stemming from design faults rather than from programming inaptitude are very interesting. Leaving out those uninteresting classic programmer-at-fault bugs and fixes, it should be noted that at this point performance of the framework was quite bad. We’re talking 300ms/iteration of 100 MNIST records with just a few dense layers on a high-end computer bad — in other words, the immediately disqualifying, unusable kind of “bad”.

This kind of performance not only increased training time but also slowed the actual development down by quite a lot, hiding critical bugs and never letting us test the entire framework in a real-world use case within a reasonable time. As is evident by the revelation of the aforementioned stability issues, this is bad. You might wonder why we didn’t just fix the performance from the get-go — we wanted to make the training work first somehow so we would have something to show for our thesis. In hindsight not the ideal choice, but it still worked out quite well and otherwise we wouldn’t have been able to demonstrate our project adequately in time for the final presentation.

#### Memory Issues All Over: Performance & Places

It took about 5 months before we got to seriously addressing the performance issues and, as usual, there was no single fix in sight. Instead, a combination of dozens of small, mostly unconnected and sometimes painstakingly chased improvements that slowly but surely made Sigma respectably fast.

A major issue was the way our SigmaDiff mathematical processor handles mathematical operations: for every operation, for every result, new memory is allocated to keep the original variables untouched (in contrast to in-place operations). That added up, and the heap and general performance suffered accordingly. The copying was necessary because our backpropagation implementation requires all intermediate values for all operations, so we couldn’t just not make copies. We couldn’t create all required buffers in a static (and therefore faster) manner ahead of time either because there is no way to traverse the operations that will be executed for a network before it is executed — the computation graph is constructed anew every time, and we can’t rely on it to remain constant due to each layers’ dynamic “run” method (e.g. a layer may change with the current mode, training or deployed).

As it turned out, what we could do was to skip copying some things. And here is where some painful manual computation and data flow analysis were done: by figuring out which intermediate values were actually required we could mark and skip those that were always left untouched (i.e. not needed again). While this may sound quite basic in theory, it was time consuming to find a way to algorithmically track down the unnecessary stale buffers. In the end, we additionally used manual analysis (read: debugger) to figure out which mathematical operations required intermediate values later on during backpropagation. We then manually disabled copies for the unnecessary ones and used in-place operations instead (without intermediate variables).

With the problem of unnecessary temporary copies solved, we were left to improve the performance of necessary copies. To that end, we added sessions: A session in Sigma is a set of operations that would be repeated many times, essentially the mostly static part of a dynamic computation graph. When a session is started we store all allocated buffer memory and then when a buffer of the same size is requested in the next session we return the one from last session — all without allocating any new memory. If more memory is required than last time, we could allocate it regardless, if less was used, we could discard it for the next session, rendering this neatly self-adjusting and quite fast (for small changes in data flow). Essentially a neat scoped cache.

Now, there was the new problem of newly created variables whose value was needed in the next session, as they would be simply be marked as reusable and overwritten like all the other “dynamic” buffers. To not overwrite data that was created within a session but was needed for the next one (e.g. parameters) we added a “limbo” buffer, basically just a flag that could be set at runtime for a certain buffer that marked it as “do not reuse” until that flag was reset and it could be cleared again. For static computation graphs this automatically sets up a kind of inverse “backbuffer” when using buffers in Sigma, similar to that used in OpenGL — draw to the back, swap, repeat without any transition artefacts.

#### SIMD: Single Instruction, Multiple Data

Another significant performance improvement was implementing SIMD instructions. SIMD (single instruction, multiple data) instructions describe instructions that support processing of multiple values (typically 8 on modern CPUs) at the same time for simple CPU-bound arithmetic operations. For example, subtracting a vector B from a fixed scalar A is not a standard BLAS operation and thus cannot be accelerated on CPU using our standard OpenBLAS library. Instead, we used the C# inbuilt SIMD instructions. The following example is a fully optimised version of scalar A minus vector B:





#### Accelerating Slow Mathematical Operations With Approximation

The vast majority of machine learning models doesn’t rely on perfect precision of floating point data or the employed mathematical functions. Exploting that “good enough” mindest for our advantage we can optimise heavy CPU and GPU functions with much faster and slightly less accurate versions. For example, most values in machine learning are small (typically in the [-1, 1] range). A naively accelerated version of the exponent function would be:



This works because the exponential function e^x can be considered as the limit of (1 + x/n)^n for n -> infinity. For any reasonably value (< ~7) the error is extremely small and the speedup with SIMD is significant.

#### Final CPU Performance

By analysing profilers to death, we eventually got the iteration time for the MNIST sample down to an acceptable 18ms in release configuration (total speedup of ~17x). Incidentally, the core was now so fast that our visualiser sometimes crashed because it couldn’t keep up with all the incoming data.

#### The Monitoring System

When developing Sigma, we not only focused on the “mathematical” backend but also implemented a feature rich monitoring system which allows any* application to be built on top of Sigma (or better said Sigma.Core). Every parameter can be observed, every change hooked, every parameter managed. With this monitoring system, we built a monitor (i.e. application) that can be used to learn Sigma and study machine learning in general.

*=theoretically, of course. And of course, technically it’s true.

#### The “Windows GUI”-Monitor

Users should be able to not only let Sigma learn, but also learn with Sigma. That was the idea anyway. To fit that bold claim, we built a feature-rich GUI (with WPF) that lets users interact with Sigma, plot learning graphs, manage parameters and control AI like controlling a music player. All components were designed with re-usability in mind, which allows users to build their own complex application on top of the default graphical monitor. But why describe a graphical user interface with words? See it for yourself, here is the UI (and Sigma) in action (example builds of Sigma can be downloaded [on GitHub](https://github.com/ThinkingTransistor/Sigma/tree/master/Sigma.Samples)).







#### Finishing Touches: GPU and CUDA Support

Only 1 month shy of the deadline we started finalising and polishing our framework: adding CUDA support, fixing leftover stability issues and rounding off annoying rough spots — in other words, we started making our framework actually usable a few weeks before we had to present it.

The CUDA support part was particularly tricky as we could only use CuBLAS, not CuDNN, because our backend doesn’t understand individual layers but just raw computation graphs (by design). A problematic side effect of the previously described session-logic is that there is no guarantee when buffers would be freed from host memory, as that was the job of the (to the user) indeterministic GC. To not constantly leak CUDA device memory we added our own bare-bones reference counter to the device memory allocator, which is notified on buffer allocations and frees. This works surprisingly well, considering it was implemented as a “temporary” hack in about an hour.

#### Final GPU Performance

With CuBLAS, a few dozen custom optimised kernels and many nights with little sleep and many kernel recompilations we achieved around 5ms/iteration for the same sample on a single GTX 1080 (4ms in 2x GTX 1080 SLI), which we deemed acceptable for our target use cases.

### Conclusion

Approximately 3000 combined hours, tens of thousands of lines of code and many long nights later we are proud to present something we deem reasonably usable for what it is: [Sigma](https://github.com/ThinkingTransistor/Sigma), a machine learning framework that might help you understand machine learning and frameworks a bit more.

As of now, we probably won’t be adding many new features to Sigma, mainly because we have accomplished what we set out to do and are now focusing on new things. Even though our frameworks lacks plenty of convenient features (most importantly the host of default layer types other frameworks offer), we’re quite happy with how far we got with our project and hope that it’s an adequate update to our original question some years ago. We would be glad if some of you could check it out, give feedback or even contribute.

#### Final Feature Overview

In the end, this is what our machine learning framework can do:

* **Layers**: Dense, Dropout, Recurrent, SoftmaxCE / SquaredDiff cost

* **Networks**: Linear and non-linear architecture (acyclic)

* **Optimisers**: Gradient descent, Momentum, Adadelta, Adagrad

* **Analysis & Training**: Hooks for storing / restoring checkpoints, timekeeping, stopping (or doing other things) on certain criteria, computing and reporting runtime metrics like standard deviation of parameter updates (i.e. update rate)

* **Performance:**Distributed multi- and single- CPU and GPU (CUDA), CPU using optimised SIMD instructions and CUDA using specialised kernels

* **GUI**: Native graphical interface for Windows, parameters can be interacted with and monitored in real-time (e.g. fancy charts, see above)

* **Usability**: Functional automatic differentiation, only forward pass required

Note that the top level “user-facing” features like interfaces, ready-to-use layers and optimisers are reduced because we spent a large chunk of our time implementing the fundamental framework from the ground up. Now that that is finished, adding new layer and optimiser types is easy thanks to functional automatic differentiation (you only need to define the forward pass) and convenient abstraction (you don’t need to care which backend(s) are used).

#### The “Cost” of Creating a Machine Learning Framework

Excluding time, it’s quite cheap in the literal sense (electricity is effectively free when you live at home)*. Honestly, with some solid prior programming experience (so that the low-level programming part doesn’t become an issue), the whole ordeal isn’t difficult and is something most people could do, given enough time. A lot of time. Overall, it took us:

1. ~700 hours of introductory and ongoing research

2. ~2200 hours of development (planning, implementation, testing)

3. 2 souls, sold to the devil for less bugs (exchange rate may vary)

We have long since stopped properly counting, so take these numbers with a grain of salt, but they should be in the right ballpark, give or take a few souls.

*=dragon bug catchers not accounted for. Counselling rate may vary.

#### Conclusion & Final remarks

All in all, an undertaking like this is a considerable time investment of about a year without much else. It was overkill for a high school thesis from the first moment and we knew that. Through scope creep and pride it kept getting more elaborate, essentially taking up all of our available time and then some. In our opinion, in the end it was all worth it though, for now we have

* a solid understanding of how artificial intelligence works

* recognition and awards from national and international competitions

* and, most importantly, the bragging rights for actually having written a machine learning framework (where’s your framework?).

Having said that, we cannot recommend writing a machine learning framework from scratch — unless, for some reason, you wish to spend a few thousand hours re-inventing processes and systems that have already been perfected by many others with much more time and skill. There’s plenty of great tutorials, guides, introductions and course material for every level on Youtube, Reddit and various university websites. Any one of those are a more efficient and pleasant, albeit slightly less cool, way to learn machine learning. If it’s still your deepest desire to participate in the development of such a framework, you should consider joining the development efforts of well established frameworks for a similar effect with more benefit to the public.

