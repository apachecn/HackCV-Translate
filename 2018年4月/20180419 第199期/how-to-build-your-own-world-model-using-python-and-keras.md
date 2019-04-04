![](https://cdn-images-1.medium.com/max/1600/1*ynKeQFaQJn4NxmViJW4KAg.gif)

# Hallucinogenic Deep Reinforcement Learning Using Python and Keras



If Artificial Intelligence is your thing, you need to check this out:

[**World Models**
Interactive demo: Tap screen to override the agent's decisions. We explore building generative neural network models of…worldmodels.github.io](https://worldmodels.github.io/)[](https://worldmodels.github.io/)

[https://arxiv.org/abs/1803.10122](https://arxiv.org/abs/1803.10122)

In short, it’s a masterpiece, for three reasons:

1. It combines several deep/reinforcement learning techniques to produce an amazing result — the first known agent to solve the popular 'Car Racing' reinforcement learning environment.

2. It’s written in a very accessible style, so a great learning resource for anyone interested in cutting-edge AI

3. You can code the solution yourself

**This post is a step by step guide through the paper**.

We’ll cover the technical details and also walk through how you can get a version running on your own machine.

Similarly to my post on [AlphaZero](https://medium.com/applied-data-science/how-to-build-your-own-alphazero-ai-using-python-and-keras-7f664945c188), I’m not associated with the authors of the paper but just wanted to share my interpretation of their terrific work.

![](https://cdn-images-1.medium.com/max/1200/1*8eLAymZlVovZrB0Vel-nQg.png)

> First, a quick note about a new platform, The Network — a place where data scientists can find paid contract projects with businesses!

> Click ‘Register’ to start building your profile.

### Step 1: The Problem

We’re going to build a reinforcement learning algorithm (an ‘agent’) that gets good at driving a car around a 2D racetrack. This environment (Car Racing) is available through the [OpenAI Gym](https://gym.openai.com/)

At each time-step, the algorithm is fed an observation (a 64 x 64 pixel colour image of the car and immediate surroundings) and needs to return the next set of actions to take — specifically, the steering direction (-1 to 1), acceleration (0 to 1) and brake (0 to 1).

This action is then passed to the environment, which returns the next observation and the cycle starts again.

An agent scores 1000/N for each of the N track tiles visited and -0.1 for each time-step taken. For example, if the agent completes the track in 732 frames, the reward is 1000–0.1*732 = 926.8 points.

![](https://cdn-images-1.medium.com/max/1200/1*Yil5VbPHSoHcEOrnGOobUQ.gif)

Here’s an example of an agent that chooses the action [0,1 0] for the first 200 time-steps then something random…not a great driving strategy.

The aim is to train the agent to understand that it can use information from its surroundings to inform the next best action.

![](https://cdn-images-1.medium.com/max/1600/1*U8VHQ030UBt33BXOGjmtbQ.png)

### Step 2: The Solution

There is an excellent online [interactive explanation of the methodology](https://worldmodels.github.io/), written by the authors, so I won’t go into the same level of detail here, but instead will focus on a high-level summary of how the pieces fit together, with an analogy to real driving to explain why the solution intuitively makes sense.

The solution consists of three distinct parts, which are trained separately:

#### A Variational Autoencoder (VAE)

When you make decisions whilst driving, you don’t actively analyse every single ‘pixel’ in your view — instead your brain condenses the visual information into a smaller number of ‘latent’ entities, such as the straightness of the road, upcoming bends and your position relative to the road, to inform your next action.

**This is exactly what the VAE is trained to do** — condense the 64x64x3 (RGB) input image into a 32-dimensional latent vector (z) that follows a Gaussian distribution.

This is useful because the agent can now work with a much smaller representation of its surroundings and therefore can be more efficient in its learning.

#### A Recurrent Neural Network with Mixture Density Network output layer (MDN-RNN)

![](https://cdn-images-1.medium.com/max/1200/1*s-yfIRkx0T_gbS1SYaldkw.gif)

If you didn’t have an MDN-RNN component to your decision making, your driving might look something like this.

As you drive, each subsequent observation isn’t a complete surprise to you. You know that if the current observation suggests a left turn in the road and you turn the wheel left, you expect the next observation to show that you are still in line with the road.

**This forward thinking is the job of the RNN** — specifically this a Long Short-Term Memory Network (LSTM) with 256 hidden units. The vector of hidden states is represented by h.

Similarly to the VAE, the RNN tries to capture a latent understanding of the current state of the car in its environment, but this time with the aim of predicting what the next ‘z’ might look like, based on the previous ‘z’ and the previous action.

The MDN output layer simply allows for the fact that the next ‘z’ could actually be drawn from any one of several Gaussian distributions.

![](https://cdn-images-1.medium.com/max/1200/1*pJIaYNT1RWCRUq8IDKz-2Q.png)

The same technique was applied in [this](http://blog.otoro.net/2015/12/28/recurrent-net-dreams-up-fake-chinese-characters-in-vector-format-with-tensorflow/) article, by the same author, for handwriting generation, to describe the fact that the next pen point could land in any one of the red distinct areas.

Similarly, in the World Models paper, the next observed latent state could be drawn from any one of five Gaussian distributions.

#### The Controller

Up until this point, we haven’t mentioned anything about choosing an action. That responsibility lies with the Controller.

The Controller is simply a **densely connected neural network**, where the input is a concatenation of z (the current latent state from the VAE — length 32) and h (the hidden state of the RNN — length 256). The 3 output neurons correspond to the three actions and are scaled to fall in the appropriate ranges.

#### A dialogue

To understand the different roles of the three components and how they work together, we can imagine a dialogue between them:

![](https://cdn-images-1.medium.com/max/1600/1*_iOdKbAgXsHZ6vDIx6sOnw.png)

> VAE: (looks at latest 64*64*3 observation) This looks like a straight road, with a slight left bend approaching, with the car facing in the direction of the road (z).

> RNN: Based on that description (z) and the fact that the Controller chose to accelerate hard at the last time-step (action), I will update my hidden state (h) so that the next observation is predicted to still be a straight road, but with slightly more left turn in view.

> Controller: Based on the description from the VAE (z) and the current hidden state from the RNN (h) my neural network outputs next action to be [0.34, 0.8, 0].

This action is then passed to the environment, which returns an updated observation and the cycle begins again.

We’ll now look at how to set up an environment that allows you to train your own version of the agent for car racing.

Time for some code!

### Step 3: Set up your environment

If you’ve got a high-spec laptop, you can run the solution locally, but I’d recommend using [Google Cloud Compute](https://cloud.google.com/compute/) for access to powerful machines that you can use in short bursts.

The following has been tested on Linux (Ubuntu 16.04) — just change the relevant commands for package installation if you’re on Mac or Windows.

1. **Clone the**[repository](https://github.com/AppliedDataSciencePartners/WorldModels)

In the command line, navigate to the place you want to store the repository and enter the following:



The repository is adapted from the highly useful [estool](https://github.com/hardmaru/estool) library developed by David Ha, the first author of the World Models paper.

For the neural network training, this implementation uses [Keras](https://keras.io/) with a [Tensorflow](https://www.tensorflow.org/) backend, though in the original paper the authors used raw Tensorflow.

**2. Set up a virtual environment**

Create yourself a Python 3 virtual environment (I use virutalenv and virtualenvwrapper)



**2. Install packages**



**3. Install requirements.txt**



There are more here than required by the Car Racing example, but you’ll have everything installed in case you want to test out some of the other environments in Open AI gym, that require the additional packages.

![](https://cdn-images-1.medium.com/max/1600/1*yRdffHYsR1-cPqqxoKghKg.png)

### Step 4: Generate random rollouts

For the Car Racing environment, both the VAE and RNN can be on **random** rollout data — that is, observation data generated by randomly taking actions at each time-step. Actually, we use pseudo-random actions, which forces the car to accelerate initially, in order to get it off the start line.

Since the VAE and RNN are independent of the decision-making Controller, all we need to ensure is that we encounter a diverse range of observations and choose a diverse range of actions to save as training data.

To generate the random rollouts, run the following from the command line



or if you’re on a server without a display,



This will produce 2000 rollouts (saved in ten batches of 200), starting with batch number 0. Each rollout will be a maximum of 300 time-steps long

Two sets of files are saved in `./data`, (* is the batch number)

`obs_data_*.npy` (stores the 64*64*3 images as numpy arrays)

`action_data_*.npy `(stores the 3 dimensional actions)

![](https://cdn-images-1.medium.com/max/1600/1*cbMhKPxI2aLmQ1semQoRKw.png)

### Step 5: Train the VAE

Training the VAE only requires the `obs_data_*.npy` files. Make sure you’ve completed Step 4, so that these files exist in the `./data` folder.

From the command line, run:



This will train a new VAE on each batch of data from 0 to 9.

The model weights will be saved to `./vae/weights.h5`. The `--new_model` flag tells the script to train the model from scratch.

If there is an existing `weights.h5` in this folder and the `--new_model` flag is not specified, the script will load the weights from this file and continue training the existing model. This way, you can iteratively train your VAE in batches, rather than all in one go.

The VAE architecture specification in the `./vae/arch.py` file.

![](https://cdn-images-1.medium.com/max/1600/1*JKOI6lJJqSgJeN1W_25-gw.png)

### Step 6: Generate RNN data

Now that we have a trained VAE, we can use it to generate the training set for the RNN.

The RNN requires encoded image data (z) from the VAE and actions (a) as input and one time-step ahead encoded image data from the VAE as output.

You can generate this data by running:



This will take the `obs_data_*.npy` and `action_data_*.npy` files from batches 0 to 9 and convert them to the correct format required by the RNN for training.

Two sets of files will be saved in `./data`, (* is the batch number)

`rnn_input_*.npy` (stores the [z a] concatenated vectors)

`rnn_output_*.npy `(stores the z vector one time-step ahead)

![](https://cdn-images-1.medium.com/max/1600/1*MZVrVvObKi4ZxAtiOP0QFw.png)

### Step 7: Train the RNN

Training the RNN only requires the `rnn_input_*.npy` and `rnn_output_*.npy `files. Make sure you’ve completed Step 6, so that these files exist in the `./data` folder.

From the command line, run:



This will train a new RNN on each batch of data from 0 to 9.

The model weights will be saved to `./rnn/weights.h5`. The `--new_model` flag tells the script to train the model from scratch.

Similarly to the VAE, if there is an existing `weights.h5` in this folder and the `--new_model` flag is not specified, the script will load the weights from this file and continue training the existing model. This way, you can iteratively train your RNN in batches, rather than all in one go.

The RNN architecture specification is in the `./rnn/arch.py` file.

![](https://cdn-images-1.medium.com/max/1600/1*OlZ4gIht9DTeBTQr1gfZEg.png)

### Step 8: Train the Controller

Now for the fun part!

So far, we’ve just used deep learning to build a VAE that can condense high dimension images down to a low dimensional latent space and an RNN that can predict how the latent space will evolve over time. This was possible because we were able to create a training set for each, using random rollout data.

To train the controller, we’ll use a form of reinforcement learning, that utilises an evolutionary algorithm known called **CMA-ES (Covariance Matrix Adaptation — Evolution Strategy)**.

Since the input is a vector of dimension 288 (= 32 + 256) and the output a vector of dimension 3, we have 288 * 3 + 1 (bias) = 867 parameters to train.

![](https://cdn-images-1.medium.com/max/1200/1*405ZycvL6eDyHc33OeYqmw.gif)

CMA-ES works by first creating multiple randomly initialised copies of the 867 parameters (the ‘population’). It then tests each member of the population inside the environment and records its average score. In exactly the same principle as natural selection, the weights that generate the highest scores are allowed to ‘reproduce’ and spawn the next generation.

To start this process on your machine, run the following command, with the appropriate values for the arguments



or on a server without display:



`--num_worker 16` : set this to no more than number of cores available

`--num_work_trial 2` : the number of members of the population that each worker will test (`num_worker * num_work_trial` gives the total population size for each generation)

`--num_episode 4` : the number of episodes each member of the population will be scored against (i.e. the score will be the average reward across this number of episodes)

`--max_length 1000` : the maximum number of time-steps in an episode

`--eval_steps 25`: the number of generations between the evaluation of the best set of weights, across 100 episodes

`--init_opt ./controller/car_racing.cma.4.32.es.pk` By default, the controller will start from scratch each time it is run and save the current state of the process to a pickle file in the `controller` directory. This argument allows you to continue training from the last save point, by pointing it at the relevant file.

After each generation, the current state of the algorithm and the best set of weights will be output to the `./controller` folder.

![](https://cdn-images-1.medium.com/max/1600/1*T3ze0CbMI9EgQSYi1xExhw.png)

### Step 9: Visualise agent

At the point of writing, I’ve managed to train an agent to achieve an average score of **~833.13** after 200 generations of training. This was trained on Google Cloud using an Ubuntu 16.04, 18 vCPU, 67.5GB RAM machine with the steps and parameters given in this tutorial.

The authors of the paper managed to achieve an average score of **~906**, after 2000 generations of training, which is believed to be the highest score in this environment to date. This utilised a slightly higher spec set-up (e.g. 10,000 episodes of training data, 64 population size, 64 core machine, 16 episodes per trial etc.)

To visualise the current state of your Controller, simply run:



`--filename` : the path to the json of weights that you want to attach to the controller

`--render_mode` : render the environment on your screen

`--record_video` : outputs mp4 files into the `./video` folder, showing each episode

`--final_mode` : run a 100 episode test of your controller and output the average score.

Here’s a demo!

![](https://cdn-images-1.medium.com/max/1600/1*-N0F3x2Z3RDvdjEx2rldVg.gif)

![](https://cdn-images-1.medium.com/max/1600/1*8FybAfsBx6GLBLiY2RDY6g.png)

### Step 10: Hallucinogenic Learning

That’s already pretty cool — but the next part of the paper is mind-blowingly impressive and I think has major implications for AI.

The paper goes on to show an amazing result, through another environment, [DoomTakeCover](https://github.com/ppaquette/gym-doom). The object here is to move an agent to avoid fireballs and stay alive as long as possible.

The authors show how it is possible for the agent to actually **learn how to play the game within its own VAE / RNN inspired hallucinogenic dreams**, rather than inside the environment itself.

The only required addition is that the RNN is trained to also predict the probability of being killed in the next time-step. This way, the VAE / RNN combination can be wrapped up as an environment in its own right and used to train the Controller. This is the very concept of a ‘**World Model**’.

We could summarise the hallucinogenic learning as follows:

> The agent’s initial training data is nothing more than random interactions with the real environment. Through this, it builds up a latent understanding of how the world ‘works’ — its natural groupings, physics and how its own actions affect the state of the world.

> It can then use this understanding to establish an optimal strategy for a given task, without ever having to actually test it in the real world, because it can use its own mental model of the environment as the ‘playground’ for trying things out.

This could easily be a description of a baby learning to walk. There are striking similarities that perhaps run deeper than mere analogy, making this a truly fascinating area of research.

### Summary

Hopefully you find this article useful — let me know in the comments below if you find any typos or have questions about anything in the codebase or article and I’ll get back to you as soon as possible.

![](https://cdn-images-1.medium.com/max/1600/1*eDhCRrmRljKt6DegLuwgUg.png)

If you would like to learn more about how our company, [Applied Data Science](https://applied-data.science) develops innovative data science solutions for businesses, feel free to get in touch through our [website](https://applied-data.science) or directly through [LinkedIn](https://www.linkedin.com/in/davidtfoster/).

… and if you like this, feel free to leave a few hearty claps :)

Applied Data Science is a London based consultancy that implements end-to-end data science solutions for businesses, delivering measurable value. If you’re looking to do more with your data, let’s talk.

![](https://cdn-images-1.medium.com/max/1600/1*39gZCiEY2D2vJRzmrYQVNA.png)

