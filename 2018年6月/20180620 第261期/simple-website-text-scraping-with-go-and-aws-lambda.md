# Simple Website Text Scraping with Go and AWS Lambda

![](https://cdn-images-1.medium.com/max/1600/1*bVjCHDdxSvAJrit52W_SoQ.jpeg)

Recently I needed to know when certain websites were updated with specific text. I decided to utilize AWS Lambda to save on cost of hosting a server, and use Go because it’s fast, and also because it’s one of the supported languages on AWS Lambda. I am also using AWS SES to send me e-mail notifications when results are found.

Bellow I’ll be showing you how to compile the Go script, setup the AWS Lambda function, and configure a cron type job to run the script every hour.

First, clone the repo contains the script.

[https://github.com/aaronvb/aws_lambda_go_scraper](https://github.com/aaronvb/aws_lambda_go_scraper)



Then we’ll build the Go script and zip it up for AWS Lambda.



Create an AWS Lambda function with the Go runtime, and select or create a role that has access to AWS SES. We’ll be using AWS SES to send out the e-mail notification.

Once the AWS Lambda function is created, upload the zip file and make sure the handler is set to `main`.

![](https://cdn-images-1.medium.com/max/1600/1*9ZmhVFoa-6aExvJssU6MaA.png)

Next, create 3 environment variables: `RECIPIENT` will be the email which receives the notification, `SENDER` which will be the email address that sends the notification, and last `SES_LOCATION` which is the location of your SES(ie: us-west-2).

![](https://cdn-images-1.medium.com/max/1600/1*8RWqMXKVrXHhVZ5PUjYTgQ.png)

Don’t forget to add the email address to SES and verify it so it can receive emails.

![](https://cdn-images-1.medium.com/max/1600/1*E304gm1-PjY512hkYSO33w.png)

Now we can create a test event. In the event data pass a JSON hash which has a key `urls` and a string value with the urls you want to scrape, separated by commas, and a key `words`, with a string value of comma separated words you wish to scrape.

Example:



![](https://cdn-images-1.medium.com/max/1600/1*a9KhHh7UMEsx2reSc2wDIA.png)

![](https://cdn-images-1.medium.com/max/1600/1*Z27m6X5eDJx_ynwh2g5gbg.png)

Click the test button and you should receive a successful function execution with logs and an email. The logs will contain the results, message ID from SES, and any errors while parsing or sending the email.

![](https://cdn-images-1.medium.com/max/1600/1*PJUQ1PgTMMjn12GCCqQiIQ.png)

### Let’s Automate This

Now that the AWS Lambda function is working, it’s time to automate this and have it run every hour. We’ll pick different words because we know those exist. Let’s pretend we want to know when my personal website will be updated with the words “swift, java, and angular.”

For this we’ll be using AWS CloudWatch events. So let’s head over there and create a new events rule.

![](https://cdn-images-1.medium.com/max/1600/1*7D-BzX42lIgLyQLwpywXRw.png)

First we set the schedule to a fixed rate of 1 hour. Next, choose the Lambda function we created earlier. And finally, the most important part, select Configure input > Constant (JSON text), and paste in the JSON with the data to send to our function (see code below).



Once you fill that in, click Configure details to name the rule and then create it. We now have the script running every hour, scraping our website, searching for the text we provided, and alerting us when it finds it.

