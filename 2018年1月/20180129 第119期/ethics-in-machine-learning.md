# Ethics in Machine Learning



On a not-very-sunny day in our Golden State of California, I sat down (virtually) with Dr. Hanie Sedghi to discuss the topic of ethics in Machine Learning. Born and raised in Iran, Hanie is a research scientist at [Google Brain](https://research.google.com/teams/brain/), based in Mountain View. Prior to joining Google Brain, Hanie worked at [Allen Institute for Artificial Intelligence](http://allenai.org/) in Seattle as a research scientist for two years. She earned her PhD from University of Southern California in Spring 2015. Her dissertation topic is [Stochastic Optimization in High Dimensions](http://digitallibrary.usc.edu/cdm/ref/collection/p15799coll3/id/610840).

![](https://cdn-images-1.medium.com/max/1200/1*gDE3N_jwVcUzrXtUsZQF7Q.jpeg)

Hanie moved to the US in 2010 after earning B.Sc. and M.Sc. degrees in Electrical Engineering from Iran’s most prestigious engineering school, Sharif University of Technology.

After introducing me to her cute Persian cat, Misha— she actually brought him from Iran — we jumped right in. As a side note, Hanie and I talked in our native language, Farsi. Below I have transcribed and translated our conversation, which Hanie reviewed for accuracy.

**Roya Pakzad: Hanie, tell me in your own words, what is fairness in machine learning?**

**Hanie Sedghi:** There are many examples and definitions. But let me start with one very well-known example: algorithmic bias in criminal justice systems. In this situation, if we disproportionately feed data about crimes committed by African Americans into a crime prediction model, then of course our model’s prediction will be biased against black communities. This is a skewed sample. On the other hand, if minority groups are underrepresented in our data samples, for example women or transgender individuals, then again, we can expect that our model will have a bias.

**Roya: Are there any mathematical solutions for addressing that?**

**Hanie:** To some extent yes. There are three different approaches:

1) Post processing in terms of calibration of our model. What this means is that, we calibrate classifiers parameters such that it has the same acceptance ratio for all subgroups of sensitive features, e.g. race, sex, etc.

2) Data resampling to remove skewed sample. But, for many reasons, collecting more data is not very easy and sometimes causes problems for individuals.

3) Causal reasoning: We capture different paths in a causal graph that can lead to the same observational data. This basically means to model possible factors such that sex, race and other sensitive features to make sure their impact is captured and does not directly affect the result variable.

**Roya: What are some of the challenges in modeling fairness in decision-making algorithms?**

**Hanie:** One very important issue is the lack of a concrete definition of “fairness.” I can tell you that there are a number of definitions and sometimes research groups are not on the same page when it comes to the definition of fairness. When you don’t have a clear definition, then how can you model it correctly?

The other issue is the need for collaboration between social scientists and AI researchers. You know, you can’t expect AI researchers themselves to come up with a clear understanding of fairness. Not only we need people in social sciences to collaborate with us in defining these words, but also we need to keep this collaboration all along to the end of the product research and development.

> “One very important issue is the lack of a concrete definition of fairness.”

But it’s important to note that some collaborations between AI researchers and social scientists are already underway. For example, Solon Barocas (Cornell University) and Moritz Hardt at UC Berkeley have been working on the issue of defining and modeling fairness in active collaboration with social scientists.

**Roya: But they are academics. They are not expected to meet corporate production deadlines or offer super-practical and financially-feasible solutions. They might not be under pressure from the AI competition that we are witnessing in the big tech companies. How do you think we can encourage data scientists and engineers to be aware of these issues who might be under tough deadline pressures?**

**Hanie:** Well, there should definitely be some sort of training on ethics and fairness for them.

**Roya: How to make sure those trainings are effective? I, myself went through some types of company’s mandatory online 45 minute quizzes on ethics in the workplace. I confess I clicked all answers until I get the right one, without being actually trained!**

**Hanie:** I understand. This is a question that I can’t answer but I also would really like to know how to achieve this. But I do know that, with respect to bias and fairness, as long as our training is in the form of someone lecturing about the basics of gender or racial bias in society, that training is not likely to be effective. When you warn us about the issues without giving any solutions or approaches to resolve it, then how can that training be fully effective?

I also think first we should have a practical guideline. Currently, there is not any comprehensive guideline (that is actually implementable) in the field of ethics in machine learning. Let me give you an example: differential privacy. It means making sure that the data you have do not reveal the identity of the persons to whom the data belongs. It has a clear mathematical definition. So, for example, once for the Netflix Prize competition, to build a recommender system, we, as researchers, were able to identify that their dataset violates differential privacy. But for fairness, we currently do not have this.Consequently, it cannot be easily taught and monitored, and audited.

**Roya: You are an Iranian and you finished your B.Sc. and M.Sc. in Electrical Engineering. You also worked in the industry. How do you see the state of AI research in Iran?**

**Hanie:** Well, I was in Iran recently and had a chance to talk to some AI researchers there. There are great start-ups working mostly on developing apps similar to everything we use here. But regarding AI research, the progress is behind. Machine Learning itself is still relatively a new field. One reason for the gap could be the US sanctions against access to certain research papers, and, more importantly, to cloud computing platforms.

**Roya: I’m sure you know better than me that there is a lack of representation of women in tech. There is also a high rate of women leaving the tech industry. How can we address both issues?**

**Hanie:** Regarding lack of representation, I think we should motivate girls from earlier ages in the STEM fields. There are great initiatives by Google and Twitter currently underway. For example, they invite young girls to their campuses and give them small projects that motivate them to think about machine learning as a field that is capable of solving many real world problems.

Having strong role models is important too. Sometimes this lack of representation leads to lack of self confidence in women. They might feel frustrated about not being heard and intellectually respected. Eventually, they might leave the field. So, it is very important to have strong female role models to look up too. One of my own childhood role models was Marie Curie. In ML also, I had great role models such as Anima Anandkumar and Jennifer Chayes. I learned a lot from their strong personalities and self-confidence.

**Roya: A final question: are you concerned about the future of AI?**

**Hanie:** I don’t believe in picturing AI as a devil. We are the teacher and we have control over what we teach.****I believe that Machine Learning has great potential to make our life easier, from simple tasks such as weather forecasting and AI assistants to self driving cars. But I do think that we still have a lot of work to make AI intelligence reach anything approaching the level of humans. For example, you can have your AI answer your questions but you cannot yet expect it to fully reason. There are still many many difficult unanswered challenges and that’s what making this field the most interesting to me!

We wrapped up here. This is the first conversation of my interview series for my newsletter [Humane AI](https://www.royapakzad.co/newsletter/). I will continue talking with both policy and technical experts in the field of ethics of AI in future installments. Tune in to know their opinions about many issues including cybersecurity and AI, private sector approaches in addressing such ethical challenges, Human rights and AI for social good, and much more. To subscribe to the newsletter, click [here](https://royapakzad.us17.list-manage.com/subscribe?u=9138308bb26620c53a0881c20&id=8152ed9f0c).

You can find Dr. Hanie Sedghi on [Twitter](https://twitter.com/HanieSedghi?lang=en) and [LinkedIn](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&cad=rja&uact=8&ved=0ahUKEwjylvP-sePYAhUV6WMKHQCEC08QFggzMAE&url=https://www.linkedin.com/in/hanie-sedghi-71bb2582&usg=AOvVaw0YEx3cTOS3g83S7Bs38Baa).

