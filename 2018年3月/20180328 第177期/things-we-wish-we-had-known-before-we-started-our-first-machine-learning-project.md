# Things I wish we had known before we started our first Machine Learning project

![](https://cdn-images-1.medium.com/max/1600/1*KFN1nsAu-Hj0GH4RMCS_cA.jpeg)

Anything new brings along with it many unknowns that we discover with time. After spending some time with any new technology we have a list of things we wish we could send back to our old selves. This is one of those lists that I wish I could send back to our team before we set out to build our first Machine Learning pipeline. Having this could have saved a lot of our time and hope that by sharing this it will save yours.

As I have already shared in one of my [earlier posts](http://www.tothenew.com/blog/when-you-take-your-machine-learning-models-to-production-for-real-time-predictions/) we were doing things with Apache Spark. So this will contain some general advice for building a machine learning pipeline and other things specific to Apache Spark.

### Hard to estimate

Like anything big and unknown it was hard to do a time estimate. We had an idea about what we would need to do to achieve our goal. But those ideas turned out to be wrong one by one as time went by. We had to accept that and make sure that we can iterate quickly.

> There are going to be big unknowns. Ensure that your team is able to iterate.

### Validate your data that it is tidy before you start

We have been collecting raw data for around 3 years when we started with the ML pipeline. We were pushing this raw data to our analytics store to turn it into aggregated data. We had not used the raw data for anything but were keeping it in case everything went down and we needed to rebuild our analytic storage. The raw data was in the form of CSV files. We had not noticed that the data had some problems. The problems happened because the code which was writing the files was changed over time so some bugs crept in which went unnoticed. While building the machine learning pipeline we did fix the bugs which were creating the problems in data. We also ended up writing code in Apache spark to do a cleanup of our historical data. The problem was that we found it in the middle of all the things rather than the beginning which increased the difficulty for our team.

> Make sure your data is ok before you start.

### Pre-process data once, train models on it a million times

To train our machine learning models we initially tried loading all of our data. The size of our data was in TBs. We found that trying to load all of it every time made it very slow to train. It also made iterations on improving models slow. We realized that we did not need to load all of the data every time. We were not using all the columns of our data. We did some pre-processing and created a new smaller data set which had the columns we needed for training our models. We also made sure that we didn’t delete the original data source which would serve as a back up in case we messed something up over time.

> Don’t mix ETL and model training. If you are training 1000 models you don’t want to do pre-processing 1000 times. Do pre-processing once, save somewhere and then use it for all your model training needs.

### Providing easy to explore access to various team members

As already mentioned we were storing our raw data for backup purposes in AWS S3. We hadn’t really kept it easy to explore from a data science perspective. That was just not the goal when we started dumping data in S3. But when starting work on ML we found that providing easy to explore access to everyone was critical.

Just giving read access wasn’t enough. People can’t just download TBs of data on their laptops, can they? Say someone did download TBs of data then what are they going to do on their laptops? People don’t carry around laptops with 32-cores usually to do processing of TBs of data in a reasonable amount of time. It is just wasting everyone’s time.

We found that using a notebook like environment backed by Apache Spark worked for that purpose. Example of notebooks are jupyter, zeppelin. Found jupyter to work nicely when we had persistent cluster. Zeppelin won out with AWS EMR (AWS’s managed spark cluster) due to the in-built integration.

> Giving read access to people to TBs and hoping that they can make sense out of out is plain ridiculous. You have to give them the right tools to be able to make sense out of it. Notebook like jupyter, zeppelin backed by cloud based spark cluster worked for us.

### Monitoring is must for Big data

When you are working with Big Data laws of Physics change. Just kidding but traditional ways of software engineering just don’t work. Normal programs takes minutes, big data can take hours to days depending on what you are doing and how you are doing it. But we don’t live in days when we have to wait days for our batch jobs to complete. Maybe a decade ago but not anymore.

Decreasing the completion time of batch jobs in case of Big Data is more complicated compared to traditional programming. Using cloud we can horizontally scale the machines that we are using and decrease the time. But should we increase the number of machines or change the machine type altogether? Are we CPU bound, RAM bound, network bound or disk bound? Where is our bottleneck in this distributed environment? That is the question that we need to answer to reduce the execution time.

For Apache Spark it was hard to figure out the machine types that were needed. Amazon EMR comes with ganglia that lets us monitor our cluster memory/CPU at a glance. But sometimes we had to go and check the underlying EC2 instances monitoring as ganglia was not perfect. Using both together helped. We found our ETL jobs had a different execution profile compared to the job profile for the jobs that were training the machine learning models. ETL took a lot of network and memory, ML training took more computation. We chose different instance types for the two types of jobs.

> CPU/memory/networking/IO monitoring is needed to optimize costs. We found that different jobs (ETL, ML) had different machine requirements.

### Need to benchmark machine learning predictions at the very beginning

Do you have latency requirements for predictions from your machine learning models? If yes, remember that finding whether it the framework’s trained models can satisfy your latency requirements. It is easy to get a basic theoretical grasp of the Maths involved underlying the model and think that it would be fast. Turns out that there are other things that could cause the predictions to not be as fast as you would expect theoretically. As smart people says — If it is not tested it won’t work.

Build a simple model and benchmark it. It could waste a lot of your time if you find that out after you built your pipeline. We found it the hard way with spark when we found that spark did not satisfy our latency requirements. We used a library called mleap to improve the prediction latency as shared in [earlier blog post](http://www.tothenew.com/blog/when-you-take-your-machine-learning-models-to-production-for-real-time-predictions/).

> If you have latency requirements make a simple model from the framework that you want to use. Accuracy, precision or any other metric does’t matter. Just benchmark it for prediction latency.

### S3 is not a Filesystem no matter how AWS makes it look

Using the GUI or CLI of AWS it becomes easy to forget that S3 is not a filesystem. It is an Object store. An object store, if you didn’t know what that is, is a key value store where the value is an object . The object could be json, image etc.

This distinction matters because renaming things within S3 is not as fast as they would be within a real Filesystem. If you move a object within a Filesystem it might be blazing fast depending on what you are using but do not expect the same in S3.

Why is this important? Because when write data to S3 through Apache spark then Apache spark writes temporary files and then moves them to the new key. It might be fine if you were using a Filesystem but not with S3 due to the reasons explained above. There is a setting in Apache spark which tells it to not write temporary files instead writing to the final output. We used that and it saved us a lot of time which Apache spark was spending in writing to AWS S3. We haven’t faced issues so far.

### Apache spark is mainly Scala based

If you are using Apache spark you should be aware that it is primarily Scala based. Java and Python API do work but the examples out there on the internet are mostly Scala based. If you ask for help on their mailing list people would be glad to help but mostly in Scala because that is the API which they are using.

We used Java as our technology stack is on Java. When we were starting out we neither had any expertise in machine learning nor Scala. We thought machine learning is essential for our business Scala is not. We could not have our team be dealing with the learning curve of Scala as well as Machine learning. These were just practical considerations to make sure the whole project did not become a disaster.

That worked well to make sure that we were able to get a final product. But when facing issues with Apache Spark it became an annoying problem. We had problems and we found solutions. Except the solutions were in Scala. Translating Scala to Java is not difficult. Translating Spark Scala to Spark Java is difficult. Because the APIs are more difficult to use in Java.

> If you don’t know Scala and want to use Spark Mllib then you may want to consider a compromise in terms of language of choice. It is not ideal engineering solution but a practical one. Remember software engineering happens in iterations. Make it work then make it better. Getting things to work in production is far more satisfying compared to trying to make a perfect solution that never goes to production.

### Knowledge sharing becomes important if you are working with a team

If you are integrating machine learning with already existing systems you would have to deal with other developers. Also you would need to talk to business people, operations people, marketing people etc. Unless you are working on a AI product most of the these people won’t have good understanding of machine learning. Because Machine learning is part of the bigger business solution not the whole solution they have to spend time with other stuff too. They cannot sit down and do courses to learn Machine learning.

Do some knowledge sharing on Machine learning. You don’t have to teach them the algorithms and stuff but you do need to de-mystify machine learning. Explain some of the common jargons involved in layman terms — train/test/validation set, model, algorithms etc. Just at a high level.

> It is easy to forget that Machine learning is full of jargons. You may be completely familiar with them but others are not. It may sound alien language to everyone else in the team and they may be getting confused. Not everyone has taken a course in ML.

### Versioning your data may be a good idea

You may want to built a versioning scheme for your data and make it possible to switch different model training code to use different data sets without a redeployment of your whole software. We created some models, tried them with some data and found that the data was not enough. The models were working but were not good enough.

So we built a versioning scheme for our data location so that we could train models on v1 and keep on generating next version and after we enough data in new version we could switch models training code to use the new data. We also made a UI which allowed us to control parameters of the machine learning algorithms, allowed us basic filtering for some specific parameters and be able to specify the amount of data that we wanted to use for the training. Basically make some of the things easily configurable via UI to ensure that making changes in the data used for training did not require re-deployments.

If you enjoyed this you might also like this where I share how [we choose a data warehouse](https://medium.com/infinity-aka-aseem/why-we-chose-snowflake-as-our-data-warehouse-c5964a00802a) for easy to explore access.



### Thanks for reading :) If you enjoyed it, hit that clap button below as many times as possible!

