# Open Machine Learning Course. Topic 9. Part 1. Time series analysis in Python

![](https://cdn-images-1.medium.com/max/1600/1*hJ_P0bVRLNNjcc5st5lu-A.jpeg)

Hi there!

We continue our open machine learning course with a new article on time series.

Let’s take a look at how to work with time series in Python, what methods and models we can use for prediction; what’s double and triple exponential smoothing; what to do if stationarity is not you favorite game; how to build SARIMA and stay alive; how to make predictions using xgboost. All of this will be applied to (harsh) real world example.

### Article outline

1. Introduction
 — Basic definitions
 — Quality metrics

2. Move, smoothe, evaluate
 — Rolling window estimations
 — Exponential smoothing, Holt-Winters model
 — Time-series cross validation, parameters selection

3. Econometric approach
 — Stationarity, unit root
 — Getting rid of non-stationarity
 — SARIMA intuition and model building

4. Linear (and not quite) models on time series
 — Feature extraction
 — Linear models, feature importance
 — Regularization, feature selection
 — XGBoost

5. Assignment #9

The following content is better viewed and reproduced as a [Jupyter-notebook](http://nbviewer.jupyter.org/github/Yorko/mlcourse_open/blob/master/jupyter_english/topic09_time_series/topic9_part1_time_series_python.ipynb)

In my day to day job I encounter time series-connected tasks almost every day. The most frequent question is — what will happen with our metrics in the next day/week/month/etc. — how many players will install the app, how much time will they spend online, how many actions users will do, and so forth. We can approach prediction task using different methods, depending on the required quality of the prediction, length of the forecasted period, and, of course, time we have to choose features and tune parameters to achieve desired results.

### Introduction

> Small definition of time series:Time series — is a series of data points indexed (or listed or graphed) in time order.

Therefore data is organized around relatively deterministic timestamps, and therefore, compared to random samples, may contain additional information that we will try to extract.

Let’s import some libraries. First and foremost we will need [statsmodels](http://statsmodels.sourceforge.net/stable/) library that has tons of statistical modeling functions, including time series. For R afficionados (that had to move to python) statsmodels will definitely look familiar as it supports model definitions like ‘Wage ~ Age + Education’.



As an example let’s use some real mobile game data on hourly ads watched by players and daily in-game currency spent:



![](https://cdn-images-1.medium.com/max/1600/1*Xglh0CGdddyPT2PdXbfSUA.png)

![](https://cdn-images-1.medium.com/max/1600/1*l43t8fEGWuxx8KmDC0eHNw.png)

### Forecast quality metrics

Before actually forecasting, let’s understand how to measure the quality of predictions and have a look at the most common and widely used metrics

* [R squared](http://scikit-learn.org/stable/modules/model_evaluation.html#r2-score-the-coefficient-of-determination), coefficient of determination (in econometrics it can be interpreted as a percentage of variance explained by the model), (-inf, 1] `sklearn.metrics.r2_score`

* [Mean Absolute Error](http://scikit-learn.org/stable/modules/model_evaluation.html#mean-absolute-error), it is an interpretable metric because it has the same unit of measurement as the initial series, [0, +inf)
`sklearn.metrics.mean_absolute_error`

* [Median Absolute Error](http://scikit-learn.org/stable/modules/model_evaluation.html#median-absolute-error), again an interpretable metric, particularly interesting because it is robust to outliers, [0, +inf)
`sklearn.metrics.median_absolute_error`

* [Mean Squared Error](http://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-error), most commonly used, gives higher penalty to big mistakes and vise versa, [0, +inf)`sklearn.metrics.mean_squared_error`

* [Mean Squared Logarithmic Error](http://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-logarithmic-error), practically the same as MSE but we initially take logarithm of the series, as a result we give attention to small mistakes as well, usually is used when data has exponential trends, [0, +inf)
`sklearn.metrics.mean_squared_log_error`

* **Mean Absolute Percentage Error**, same as MAE but percentage, — very convenient when you want to explain the quality of the model to your management, [0, +inf), not implemented in sklearn



Excellent, now we know how to measure the quality of the forecasts, what metrics can we use and how to translate the results to the boss. Little thing is left — building the model.

### Move, smoothe, evaluate

Let’s start with a naive hypothesis — “tomorrow will be the same as today”, but instead of a model like ŷ(t)=y(t−1) (which is actually a great baseline for any time series prediction problems and sometimes it’s impossible to beat it with any model) we’ll assume that the future value of the variable depends on the average **n** of its previous values and therefore we’ll use **moving average**.

![](https://cdn-images-1.medium.com/max/1600/1*xKNEAvd2OU0T71Y7rZu0ZA.png)





Unfortunately we can’t make this prediction long-term — to get one for the next step we need the previous value to be actually observed. But moving average has another use case — smoothing of the original time series to indicate trends. Pandas has an implementation available `DataFrame.rolling(window).mean()`. The wider the window - the smoother will be the trend. In the case of the very noisy data, which can be very often encountered in finance, this procedure can help to detect common patterns.



Smoothing by last 4 hours
`plotMovingAverage(ads, 4)`

![](https://cdn-images-1.medium.com/max/1600/1*g-79VWtvLlJA5rcISd5pPw.png)

Smoothing by last 12 hours
`plotMovingAverage(ads, 12)`

![](https://cdn-images-1.medium.com/max/1600/1*T71bcR_9UBe8MbevTgYTEQ.png)

Smoothing by 24 hours — we get daily trend
`plotMovingAverage(ads, 24)`

![](https://cdn-images-1.medium.com/max/1600/1*Xfxf0YM_VDgUhpw_USfJGw.png)

As you can see, applying daily smoothing on hour data allowed us to clearly see the dynamics of ads watched. During the weekends the values are higher (weekends — time to play) and weekdays are generally lower.

We can also plot confidence intervals for our smoothed values
`plotMovingAverage(ads, 4, plot_intervals=True)`

![](https://cdn-images-1.medium.com/max/1600/1*FzVb4xiHV6J1aTTAM7OdQw.png)

And now let’s create a simple anomaly detection system with the help of the moving average. Unfortunately, in this particular series everything is more or less normal, so we’ll intentionally make one of the values abnormal in the dataframe `ads_anomaly`



Let’s see, if this simple method can catch the anomaly
`plotMovingAverage(ads_anomaly, 4, plot_intervals=True, plot_anomalies=True)`

![](https://cdn-images-1.medium.com/max/1600/1*RXW9AGTcitJgMS7L-YJ3ZQ.png)

Neat! What about the second series (with weekly smoothing)?
`plotMovingAverage(currency, 7, plot_intervals=True, plot_anomalies=True)`

![](https://cdn-images-1.medium.com/max/1600/1*ml1RV_36gA8hx_lFhRIqHA.png)

Oh no! Here is the downside of our simple approach — it did not catch monthly seasonality in our data and marked almost all 30-day peaks as an anomaly. If you don’t want to have that many false alarms — it’s best to consider more complex models.

**Weighted average** is a simple modification of the moving average, inside of which observations have different weights summing up to one, usually more recent observations have greater weight.

![](https://cdn-images-1.medium.com/max/1600/1*8IBEmsFCywApUE_joR0aqA.png)





### Exponential smoothing

And now let’s take a look at what happens if instead of weighting the last nn values of the time series we start weighting all available observations while exponentially decreasing weights as we move further back in historical data. There’s a formula of the simple [exponential smoothing](https://en.wikipedia.org/wiki/Exponential_smoothing) that will help us in that:

![](https://cdn-images-1.medium.com/max/1600/1*JJZyOq_GySQnx8N1ZZ-UfQ.png)

Here the model value is a weighted average between the current true value and the previous model values. The α weight is called a smoothing factor. It defines how quickly we will “forget” the last available true observation. The less α is the more influence previous model values have, and the smoother the series is.

Exponentiality is hiding in the recursivity of the function — we multiply each time (1−α) by the previous model value which, in its turn, also containes (1−α) and so forth until the very beginning.



![](https://cdn-images-1.medium.com/max/1600/1*iP_blo7a9pCJ9EB4ZcOYcw.png)

![](https://cdn-images-1.medium.com/max/1600/1*L_w3qDoexk4ba3LKMXcqwQ.png)

### Double exponential smoothing

Until now all we could get from our methods in the best case was just a single future point prediction (and also some nice smoothing), that’s cool but not enough, so let’s extend exponential smoothing so that we can predict two future points (of course, we also get some smoothing).

Series decomposition should help us — we obtain two components: intercept (also, level) ℓ and trend (also, slope) b. We’ve learnt to predict intercept (or expected series value) using previous methods, and now we will apply the same exponential smoothing to the trend, believing naively or perhaps not that the future direction of the time series changes depends on the previous weighted changes.

![](https://cdn-images-1.medium.com/max/1600/1*ws6cwxbpczBgTGYEZep79Q.png)

As a result we get a set of functions. The first one describes intercept, as before it depends on the current value of the series, and the second term is now split into previous values of the level and of the trend. The second function describes trend — it depends on the level changes at the current step and on the previous value of the trend. In this case β coefficient is a weight in the exponential smoothing. The final prediction is the sum of the model values of the intercept and trend.



![](https://cdn-images-1.medium.com/max/1600/1*c9XHSuc9d83bmopHJVnPLA.png)

![](https://cdn-images-1.medium.com/max/1600/1*LrBONoISc3sb1HvwJNoN9A.png)

Now we have to tune two parameters — α and β. The former is responsible for the series smoothing around trend, and the latter for the smoothing of the trend itself. The bigger the values, the more weight the latest observations will have and the less smoothed the model series will be. Combinations of the parameters may produce really weird results, especially if set manually. We’ll look into choosing parameters automatically in a bit, immediately after triple exponential smoothing.

### Triple exponential smoothing a.k.a. Holt-Winters

Hooray! We’ve successfully reached our next variant of exponential smoothing, this time triple.

The idea of this method is that we add another, third component — seasonality. This means we should’t use the method if our time series do not have seasonality, which is not the case in our example. Seasonal component in the model will explain repeated variations around intercept and trend, and it will be described by the length of the season, in other words by the period after which variations repeat. For each observation in the season there’s a separate component, for example, if the length of the season is 7 (weekly seasonality), we will have 7 seasonal components, one for each day of the week.

Now we get a new system:

![](https://cdn-images-1.medium.com/max/1600/1*tQUjJKDKmqjGPeQ9YqV_dQ.png)

Intercept now depends on the current value of the series minus corresponding seasonal component, trend stays unchanged, and the seasonal component depends on the current value of the series minus intercept and on the previous value of the component. Please take into account that the component is smoothed through all the available seasons, for example, if we have a Monday component then it will only be averaged with other Mondays. You can read more on how averaging works and how initial approximation of the trend and seasonal components is done [here](http://www.itl.nist.gov/div898/handbook/pmc/section4/pmc435.htm). Now that we have seasonal component we can predict not one and not even two but arbitrary mm future steps which is very encouraging.

Below is the code for a triple exponential smoothing model, also known by the last names of its creators — Charles Holt and his student Peter Winters. Additionally Brutlag method was included into the model to build confidence intervals:

![](https://cdn-images-1.medium.com/max/1600/1*6Cu5nv-COdclmdfXnAU02Q.png)

where T is the length of the season, d is the predicted deviation, and the other parameters were taken from the triple exponential smoothing. You can read more about the method and its applicability to anomalies detection in time series [here](http://fedcsis.org/proceedings/2012/pliks/118.pdf).



### Time series cross validation

Before we start building model let’s talk first about how to estimate model parameters automatically.

There’s nothing unusual here, as always we have to choose a loss function suitable for the task, that will tell us how close the model approximates data. Then using cross-validation we will evaluate our chosen loss function for given model parameters, calculate gradient, adjust model parameters and so forth, bravely descending to the global minimum of error.

The question is how to do cross-validation on time series, because, you know, time series do have time structure and one just can’t randomly mix values in a fold without preserving this structure, otherwise all time dependencies between observations will be lost. That’s why we will have to use a bit more tricky approach to optimization of the model parameters, I don’t know if there’s an official name to it but on [CrossValidated](https://stats.stackexchange.com/questions/14099/using-k-fold-cross-validation-for-time-series-model-selection), where one can find all the answers but the Answer to the Ultimate Question of Life, the Universe, and Everything, “cross-validation on a rolling basis” was proposed as a name.

The idea is rather simple — we train our model on a small segment of the time series, from the beginning until some **t**, make predictions for the next **t+n**steps and calculate an error. Then we expand our training sample until **t+n** value and make predictions from **t+n** until **t+2∗n**, and we continue moving our test segment of the time series until we hit the last available observation. As a result we have as many folds as many **n** will fit between the initial training sample and the last observation.

![](https://cdn-images-1.medium.com/max/1600/1*6ujHlGolRTGvspeUDRe1EA.png)

Now, knowing how to set cross-validation, we will find optimal parameters for the Holt-Winters model, recall that we have daily seasonality in ads, hence the `slen=24` parameter



In the Holt-Winters model, as well as in the other models of exponential smoothing, there’s a constraint on how big smoothing parameters could be, each of them is in the range from 0 to 1, therefore to minimize loss function we have to choose an algorithm that supports constraints on model parameters, in our case — Truncated Newton conjugate gradient.





Chart rendering code



![](https://cdn-images-1.medium.com/max/1600/1*5GDW3uGcxkgoUfAxWMN7eQ.png)

![](https://cdn-images-1.medium.com/max/1600/1*DSXPcnHnvnZRH8YijMOfMg.png)

Judging by the chart, our model was able to successfully approximate the initial time series, catching daily seasonality, overall downwards trend and even some anomalies. If you take a look at the modeled deviation, you can clearly see that the model reacts quite sharply to the changes in the structure of the series but then quickly returns deviation to the normal values, “forgetting” the past. This feature of the model allows us to quickly build anomaly detection systems even for quite noisy series without spending too much time and money on preparing data and training the model.



![](https://cdn-images-1.medium.com/max/1600/1*i8KR9tLBxRQ5nUFtHjUvgA.png)

We’ll apply the same algorithm for the second series which, as we know, has trend and 30-day seasonality





![](https://cdn-images-1.medium.com/max/1600/1*Km4pL2wIDQGRE3-12rLFCg.png)

Looks quite adequate, model has caught both upwards trend and seasonal spikes and overall fits our values nicely

![](https://cdn-images-1.medium.com/max/1600/1*b1D6G3IR4XUcP8oUvu6B0A.png)

![](https://cdn-images-1.medium.com/max/1600/1*G_Do4GDD2nAHjiXFyW-qaA.png)

### Econometric approach

### Stationarity

Before we start modeling we should mention such an important property of time series as [stationarity](https://en.wikipedia.org/wiki/Stationary_process).

If the process is stationary that means it doesn’t change its statistical properties over time, namely mean and variance do not change over time (constancy of variance is also called [homoscedasticity](https://en.wikipedia.org/wiki/Homoscedasticity)), also covariance function does not depend on the time (should only depend on the distance between observations). You can see this visually on the pictures from the post of [Sean Abu](http://www.seanabu.com/2016/03/22/time-series-seasonal-ARIMA-model-in-python/):

* The red graph below is not stationary because the mean increases over time.

![](https://cdn-images-1.medium.com/max/1600/0*qrYiVksz8g3drl5Z.png)

* We were unlucky with the variance, see the varying spread of values over time

![](https://cdn-images-1.medium.com/max/1600/0*fEqQDq_TaEqa511n.png)

* Finally, the covariance of the i th term and the (i + m) th term should not be a function of time. In the following graph, you will notice the spread becomes closer as the time increases. Hence, the covariance is not constant with time for the right chart.

![](https://cdn-images-1.medium.com/max/1600/1*qJs3g2f77flIXr6mFsbPmw.png)

So why stationarity is so important? Because it’s easy to make predictions on the stationary series as we assume that the future statistical properties will not be different from the currently observed. Most of the time series models in one way or the other model and predict those properties (mean or variance, for example), that’s why predictions would be wrong if the original series were not stationary. Unfortunately most of the time series we see outside of textbooks are non-stationary but we can (and should) change this.

So, to fight non-stationarity we have to know our enemy so to say. Let’s see how to detect it. To do that we will now take a look at the white noise and random walks and we will learn how to get from one to another for free, without registration and SMS.

White noise chart:



![](https://cdn-images-1.medium.com/max/1600/1*C95wWTnA2hE2H9sBmFzQDw.png)

So the process generated by standard normal distribution is stationary and oscillates around 0 with with deviation of 1. Now based on this process we will generate a new one where each next value will depend on the previous one: x(t)=ρ*x(t−1)+e(t)

Chart rendering code



![](https://cdn-images-1.medium.com/max/1600/1*7jeb47aq-0WmNndzO_jxHQ.png)

![](https://cdn-images-1.medium.com/max/1600/1*XLwz-8vtx51T9RvfhZ2e_A.png)

![](https://cdn-images-1.medium.com/max/1600/1*gMyyzlnnJHN1Nd1dZ8SL5g.png)

![](https://cdn-images-1.medium.com/max/1600/1*ZhIS_d1-htkJtZM3QrYAGw.png)

On the first chart you can see the same stationary white noise you’ve seen before. On the second one the value of ρρ increased to 0.6, as a result wider cycles appeared on the chart but overall it is still stationary. The third chart deviates even more from the 0 mean but still oscillates around it. Finally, the value of ρ equal to 1 gives us a random walk process — non-stationary time series.

This happens because after reaching the critical value the series x(t)=ρ*x(t−1)+e(t) does not return to its mean value. If we subtract x(t−1) from the left and the right side we will get x(t)−x(t−1)=(ρ−1)*x(t−1)+e(t), where the expression on the left is called the first difference. If ρ=1 then the first difference gives us stationary white noise e(t). This fact is the main idea of the [Dickey-Fuller test](https://en.wikipedia.org/wiki/Dickey–Fuller_test) for the stationarity of time series (presence of a unit root). If we can get stationary series from non-stationary using the first difference we call those series integrated of order 1. Null hypothesis of the test — time series is non-stationary, was rejected on the first three charts and was accepted on the last one. We’ve got to say that the first difference is not always enough to get stationary series as the process might be integrated of order d, d > 1 (and have multiple unit roots), in such cases the augmented Dickey-Fuller test is used that checks multiple lags at once.

We can fight non-stationarity using different approaches — various order differences, trend and seasonality removal, smoothing, also using transformations like Box-Cox or logarithmic.

### Getting rid of non-stationarity and building SARIMA

Now let’s build an ARIMA model by walking through all the circles of hell stages of making series stationary.

Chart rendering code



![](https://cdn-images-1.medium.com/max/1600/1*GSf3AiF0mNc9AScQsI1UTQ.png)

Surprisingly, initial series are stationary, Dickey-Fuller test rejected null hypothesis that a unit root is present. Actually, it can be seen on the plot itself — we don’t have a visible trend, so mean is constant, variance is pretty much stable throughout the series. The only thing left is seasonality which we have to deal with before modelling. To do so let’s take “seasonal difference” which means a simple subtraction of series from itself with a lag that equals the seasonal period.



![](https://cdn-images-1.medium.com/max/1600/1*qD4gwPbKj3hFAah3F84r2g.png)

That’s better, visible seasonality is gone, however autocorrelation function still has too many significant lags. To remove them we’ll take first differences — subtraction of series from itself with lag 1



![](https://cdn-images-1.medium.com/max/1600/1*t_5QyZgc8nuuVdRBKbV4iQ.png)

Perfect! Our series now look like something undescribable, oscillating around zero, Dickey-Fuller indicates that it’s stationary and the number of significant peaks in ACF has dropped. We can finally start modelling!

### ARIMA-family Crash-Course

A few words about the model. Letter by letter we’ll build the full name — **SARIMA(p,d,q)(P,D,Q,s)**, Seasonal Autoregression Moving Average model:

* **AR(p)**— autoregression model, i.e., regression of the time series onto itself. Basic assumption — current series values depend on its previous values with some lag (or several lags). The maximum lag in the model is referred to as **p**. To determine the initial **p** you need to have a look at PACF plot — find the biggest significant lag, after which **most** other lags are becoming not significant.

* **MA(q)** — moving average model. Without going into detail it models the error of the time series, again the assumption is — current error depends on the previous with some lag, which is referred to as **q**. Initial value can be found on ACF plot with the same logic.

Let’s have a small break and combine the first 4 letters:

**AR(p) + MA(q) = ARMA(p,q)**

What we have here is the Autoregressive–moving-average model! If the series is stationary, it can be approximated with those 4 letters. Shall we continue?

* **I(d)**— order of integration. It is simply the number of nonseasonal differences needed for making the series stationary. In our case it’s just 1, because we used first differences.

Adding this letter to four previous gives us **ARIMA** model which knows how to handle non-stationary data with the help of nonseasonal differences. Awesome, last letter left!

* **S(s)** — this letter is responsible for seasonality and equals the season period length of the series

After attaching the last letter we find out that instead of one additional parameter we get three in a row — **(P,D,Q)**

* **P** — order of autoregression for seasonal component of the model, again can be derived from PACF, but this time you need to look at the number of significant lags, which are the multiples of the season period length, for example, if the period equals 24 and looking at PACF we see 24-th and 48-th lags are significant, that means initial **P** should be 2.

* **Q**— same logic, but for the moving average model of the seasonal component, use ACF plot

* **D** — order of seasonal integration. Can be equal to 1 or 0, depending on whether seasonal differences were applied or not

Now, knowing how to set initial parameters, let’s have a look at the final plot once again and set the parameters:
`tsplot(ads_diff[24+1:], lags=60)`

![](https://cdn-images-1.medium.com/max/1600/1*t_5QyZgc8nuuVdRBKbV4iQ.png)

* **p** is most probably 4, since it’s the last significant lag on PACF after which most others are becoming not significant.

* **d** just equals 1, because we had first differences

* **q** should be somewhere around 4 as well as seen on ACF

* **P** might be 2, since 24-th and 48-th lags are somewhat significant on PACF

* **D** again equals 1 — we performed seasonal differentiation

* **Q** is probably 1, 24-th lag on ACF is significant, while 48-th is not

Now we want to test various models and see which one is better











![](https://cdn-images-1.medium.com/max/1600/1*ETjGK2zslGmrH3HHvvDUhQ.png)

Let’s inspect the residuals of the model
`tsplot(best_model.resid[24+1:], lags=60)`

![](https://cdn-images-1.medium.com/max/1600/1*q_bSXlMdmiwMA73fXOOK5A.png)

Well, it’s clear that the residuals are stationary, there are no apparent autocorrelations, let’s make predictions using our model



![](https://cdn-images-1.medium.com/max/1600/1*ccFZwDXwb2ZW1c8FuxRHLw.png)

In the end we got quite adequate predictions, our model on average was wrong by 4.01%, which is very very good, but overall costs of preparing data, making series stationary and brute-force parameters selecting might not be worth this accuracy.

### Linear (and not quite) models on time series

Small lyrical digression again. Often in my job I have to build models with the only principle guiding me known as [fast, good, cheap](http://fastgood.cheap/). That means some of the models will never be “production ready” as they demand too much time for the data preparation (for example, SARIMA), or require frequent re-training on new data (again, SARIMA), or are difficult to tune (good example — SARIMA), so it’s very often much easier to select a couple of features from the existing time series and build a simple linear regression or, say, a random forest. Good and cheap.

Maybe this approach is not backed up by theory, breaks different assumptions (like, Gauss-Markov theorem, especially about the errors being uncorrelated), but it’s very useful in practice and quite frequently used in machine learning competitions.

### Feature exctraction

Alright, model needs features and all we have is a 1-dimentional time series to work with. What features can we exctract?

**Lags of time series, of course**

**Window statistics:**

* Max/min value of series in a window

* Average/median value in a window

* Window variance

* etc.

**Date and time features:**

* Minute of an hour, hour of a day, day of the week, you get it

* Is this day a holiday? Maybe something special happened? Make it a boolean feature

**Target encoding**

**Forecasts from other models** (though we can lose the speed of prediction this way)

Let’s run through some of the methods and see what we can extract from our ads series

### Lags of time series

Shifting the series **n** steps back we get a feature column where the current value of time series is aligned with its value at the time **t−n**. If we make a 1 lag shift and train a model on that feature, the model will be able to forecast 1 step ahead having observed current state of the series. Increasing the lag, say, up to 6 will allow the model to make predictions 6 steps ahead, however it will use data, observed 6 steps back. If something fundamentally changes the series during that unobserved period, the model will not catch the changes and will return forecasts with big error. So, during the initial lag selection one has to find a balance between the optimal prediction quality and the length of forecasting horizon.



Wonderful, we got ourselves a dataset here, why don’t we train a model?



![](https://cdn-images-1.medium.com/max/1600/1*0hLmlcFVnKljGCe-jmZDPg.png)

![](https://cdn-images-1.medium.com/max/1600/1*9K0FULC9peB6_zRgRnQUyA.png)

Well, simple lags and linear regression gave us predictions that are not that far from SARIMA in quality. There are lot’s of unnecessary features, but we’ll do feature selection a bit later. Now let’s continue engineering!

We’ll add into our dataset hour, day of the week and boolean for the weekend. To do so we need to transform current dataframe index into `datetime` format and exctract `hour` and `weekday` out of it.



We can visualize the resulting features



![](https://cdn-images-1.medium.com/max/1600/1*-ZG-BbBqy_Xl1AjxW8jyYg.png)

Since now we have different scales of variables — thousands for lag features and tens for categorical, it’s reasonable to transform them into same scale to continue exploring feature importances and later — regularization.



![](https://cdn-images-1.medium.com/max/1600/1*0vAkMWquV4dTgewDg94JYQ.png)

![](https://cdn-images-1.medium.com/max/1600/1*NENact_8y6O1352EYzoG0A.png)

Test error goes down a little bit and judging by the coefficients plot we can say that `weekday` and `is_weekend` are rather useful features

### Target encoding

I’d like to add another variant of encoding categorical variables — by mean value. If it’s undesirable to explode dataset by using tons of dummy variables that can lead to the loss of information about the distance, and if they can’t be used as real values because of the conflicts like “0 hours < 23 hours”, then it’s possible to encode a variable with a little bit more interpretable values. Natural idea is to encode with the mean value of the target variable. In our example every day of the week and every hour of the day can be encoded by the corresponding average number of ads watched during that day or hour. It’s very important to make sure that the mean value is calculated over train set only (or over current cross-validation fold only), so that the model is not aware of the future.



Let’s have a look at hour averages



![](https://cdn-images-1.medium.com/max/1600/1*vnc2RhvCPIuPOVqTKKIjog.png)

Finally, put all the transformations together in a single function



![](https://cdn-images-1.medium.com/max/1600/1*ujaq7SFvHBxPDeF8XFQOmA.png)

![](https://cdn-images-1.medium.com/max/1600/1*S0Pr4flzWrwhveJE9AWoeA.png)

Here comes **overfitting**! `Hour_average` variable was so great on train dataset that the model decided to concentrate all its forces on it - as a result the quality of prediction dropped. This problem can be approached in a variety of ways, for example, we can calculate target encoding not for the whole train set, but for some window instead, that way encodings from the last observed window will probably describe current series state better. Or we can just drop it manually, since we're sure here it makes things only worse.



### Regularization and feature selection

As we already know, not all features are equally healthy, some may lead to overfitting and should be removed. Besides manual inspecting we can apply regularization. Two most popular regression models with regularization are Ridge and Lasso regressions. They both add some more constrains to our loss function.

In case of **Ridge regression** — those constrains are the sum of squares of coefficients, multiplied by the regularization coefficient. I.e. the bigger coefficient feature has — the bigger our loss will be, hence we will try to optimize the model while keeping coefficients fairly low.

As a result of such regularization which has a proud name **L2** we’ll have higher bias and lower variance, so the model will generalize better (at least that’s what we hope will happen).

Second model —**Lasso regression**, here we add to the loss function not squares but absolute values of the coefficients, as a result during the optimization process coefficients of unimportant features may become zeroes, so Lasso regression allows for automated feature selection. This regularization type is called **L1**.

First, make sure we have things to drop and data truly has highly correlated features



![](https://cdn-images-1.medium.com/max/1600/1*KTxkDrHFdl7E6xs3Hq5_YA.png)



![](https://cdn-images-1.medium.com/max/1600/1*mh_iu9oxsBSDwHwsCjZdRw.png)

![](https://cdn-images-1.medium.com/max/1600/1*F4boA6Z10qzeJq4cpmp4jw.png)

We can clearly see how coefficients are getting closer and closer to zero (thought never actually reach it) as their importance in the model drops



![](https://cdn-images-1.medium.com/max/1600/1*u8lR_FDvPV6rH0JqJR87sw.png)

![](https://cdn-images-1.medium.com/max/1600/1*Q9hvolN8hDV0xTZwlIBR2g.png)

Lasso regression turned out to be more conservative and removed 23-rd lag from most important features (and also dropped 5 features completely) which only made the quality of prediction better.

### Boosting

Why not try XGBoost now?

![](https://cdn-images-1.medium.com/max/1600/0*4XXSSYy4nYDDgNex.jpg)



![](https://cdn-images-1.medium.com/max/1600/1*dpguJye_sOtOFK746JVovg.png)

Here is the winner! The smallest error on the test set among all the models we’ve tried so far.

Yet this victory is decieving and it might not be the brightest idea to fit xgboost as soon as you get your hands over time series data. Generally tree-based models poorly handle trends in data, compared to linear models, so you have to detrend your series first or use some tricks to make the magic happen. Ideally — make the series stationary and then use XGBoost, for example, you can forecast trend separately with a linear model and then add predictions from xgboost to get final forecast.

### Conclusion

We got acquainted with different time series analysis and prediction methods and approaches. Unfortunately, or maybe luckily, there’s no silver bullet to solve this kind of problems. Methods developed in the 60s of the last century (and some even in the beginning of the XIX century) are still popular along with the LSTM and RNN (not covered in this article). Partially this is related to the fact that the prediction task as any other data related task is creative in so many aspects and definitely requires research. In spite of the large number of formal quality metrics and approaches to parameters estimation, it’s often required to seek and try something different for each time series. Last but not least the balance between quality and cost is important. As a good example SARIMA model mentioned here not once or twice can produce spectacular results after due tuning but might require many hours of tambourine dancing time series manipulation, as in the same time simple linear regression model can be build in 10 minutes giving more or less comparable results.

### Assignment #9

Full versions of assignments are announced each week in a new run of the course (October 1, 2018). Meanwhile, you can practice with a demo version: [Kaggle Kernel](https://www.kaggle.com/kashnitsky/assignment-9-time-series-analysis), [nbviewer](https://mlcourse.ai/notebooks/blob/master/jupyter_english/assignments_demo/assignment09_time_series.ipynb?flush_cache=true).

### Useful resources

* [Online textbook](https://people.duke.edu/~rnau/411home.htm) of the advanced statistical forecasting course of the Duke University — covers in details various smoothing techniques, linear and ARIMA models

* [Comparison of ARIMA and Random Forest time series models for prediction of avian influenza H5N1 outbreaks](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-15-276) — one of a few where random forest applicability to the tasks of time series forecasting is actively defended

* [Time Series Analysis (TSA) in Python — Linear Models to GARCH](http://www.blackarbs.com/blog/time-series-analysis-in-python-linear-models-to-garch/11/1/2016) ARIMA models family and their applicability to the task of modeling financial indicators (Brian Christopher)

Author: [Dmitry Sergeyev](https://github.com/DmitrySerg). Translated and edited by [Borys Zibrov](https://www.linkedin.com/in/borowis/), and [Yuanyuan Pao](https://www.linkedin.com/in/yuanyuanpao/).

