# Collecting Weather Data to Boost Data Science Models with Selenium

![](https://cdn-images-1.medium.com/max/1600/1*uPrE5g20T8rrnjNxe5RYnA.png)

Have you ever wonder how weather data would affect your data set or wanted to understand the historical weather patterns in your local city? Do you find it difficult to find weather data for your specific needs? I did.

I was working on a Kaggle competition involving geospatial data and I wanted more relevant data that was compatible with my data set. Unfortunately, I couldn’t find anything. I spent days looking and all I found was a website that would only provide weather condition such as whether or not the skies were cloudy or clear on a given day. That wasn’t enough for me.

TLDR: So instead, I built my own web scrapper, collected my own data, and applied it to my data set. The results led to a model that was more generalizable to new data sets. And now you can improve your results too with this tutorial.

I used Selenium to build a web scraper with Python to collect weather information such as Temperature, Humidity, Wind Speed, and Weather Conditions from [https://www.wunderground.com/](https://www.wunderground.com/).

![](https://cdn-images-1.medium.com/max/1600/1*DhYVh-isXMN2Sy_wQh55NQ.png)

What makes this website different from your typical websites is that the information is based on the weather conditions of nearby airports. While this sounds counter-intuitive, it is in fact generalizable to cities close to those airports and more informative than typical sites. This makes it easy to search for the closest airports using zip codes.

If you scroll all the way down to the bottom, you’ll see a gold mine of weather data for you to mine. Beautiful isn’t it? You can mine this for specific days or a sequence of days.

![](https://cdn-images-1.medium.com/max/1600/1*5i5oard-5P94UvSrSIrKLQ.png)

Here are my steps on how I collect the data from this table for a range of dates. You can skip to the bottom of the page if you want to see the full code snippet.

**Step 1**: Load the packages need



Selenium is a tool used to automate tasks and actions with a browser through programming languages such as Python or Java. You can think about it as controlling your web browser with commands that you provide it. We are going to use it to collect data from HTML webpages.

To initialize Selenium, you’ll need to load the webdriver at given path. (Here is the installation guide for Selenium using Python: [http://selenium-python.readthedocs.io/installation.html](http://selenium-python.readthedocs.io/installation.html))



webdriver.Chrome will load the required software to run Selenium on Chrome. driver.get() will open a new web browser that it can control.

**Step 2**: Get a sequence of dates to search for









I wrote the list_dates() function to produce a sequence of dates from a start date to an end date. This is useful because you won’t need to need to manually input dates.







The date_part() function is used to break down the dates produced by list_date() into Month, Day, and Year so that it can easily be sent to wunderground to search for a date to collect data from. For example, both functions will produce the following pandas dataframe:



![](https://cdn-images-1.medium.com/max/1600/1*0qP4jY8Rc7UAqtrW_TGCAQ.png)

**Step 3**: Search by location

Now that we have the date dictionary, you will need to provide a zip code so that the wunderground can pull up the weather information. You can automate this task with Selenium. If you go to the search bar, right-click it, and click inspect, you will get HTML tag information about that search bar. This is generalizable to any HTML-JavaScript-CSS feature on the web page. For example, here you can see that the unique ‘id’ for the search bar:

![](https://cdn-images-1.medium.com/max/1600/1*GHq0x3CYz0gIW3D24gLtPQ.png)

![](https://cdn-images-1.medium.com/max/1600/1*T2xIs1B4BwfHm1FVvVE_gg.png)

To send a zip code to the search bar, the following code will look for the search bar by a specific ‘id’ with ‘find_element_by_id’, clear the search bar, send the zip code, and finally submit it to wunderground to search for the city.



**Step 4**: Search for date

![](https://cdn-images-1.medium.com/max/1600/1*EgRqNDGota-V8sdeozPbLQ.png)

![](https://cdn-images-1.medium.com/max/1600/1*ed7z67kWGwjUZTorUd20mg.png)

You can similarly search for dates like Step 3. By looking for HTML tags associated to the ‘Weather History Date’ pull down menus, you’ll see that it contains class attributes such as month, day and year. We can use this to send information to search for those pull down menus. You can iterate through the pandas data frame produced from Step 2 to search the weather of a given date.





**Step 5**: Collect weather data

![](https://cdn-images-1.medium.com/max/1600/1*eonknv98XyGT7_-t3lTidg.png)

We’re now ready to collect the weather data. We can see that the weather table is contained in this div tag with unique ‘id’ of ‘observations_details’. We can use this information to scrape the whole weather table with a simple line of code:



**Step 6**: Bring it all together

Finally, we can combine all the previous steps into a compact script. I have also including some preprocessing steps and a function in runs everything with one function:



























Congratulations! Now you can collect your weather data with Selenium and enhance all your Kaggle / work / school / side projects without compromising validation scores! You can also use what you’ve from this and collect data from other sources too!

References:

1. [Weather Underground](https://www.wunderground.com/)

2. [Selenium Download Page](https://www.seleniumhq.org/download/)

3. [Personal GitHub Repository](https://github.com/davidkes/Weather-Data-Scaper)

