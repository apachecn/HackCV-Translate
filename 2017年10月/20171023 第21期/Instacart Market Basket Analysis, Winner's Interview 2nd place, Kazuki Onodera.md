# Instacart Market Basket Analysis, Winner's Interview: 2nd place, Kazuki Onodera

原文链接：[Instacart Market Basket Analysis, Winner's Interview: 2nd place, Kazuki Onodera](http://blog.kaggle.com/2017/09/21/instacart-market-basket-analysis-winners-interview-2nd-place-kazuki-onodera/?from=hackcv&hmsr=hackcv.com&utm_medium=hackcv.com&utm_source=hackcv.com)

Our recent [Instacart Market Basket Analysis competition](https://www.kaggle.com/c/instacart-market-basket-analysis) challenged Kagglers to predict which grocery products an Instacart consumer will purchase again and when. Imagine, for example, having milk ready to be added to your cart right when you run out, or knowing that it's time to stock up again on your favorite ice cream.

This focus on understanding temporal behavior patterns makes the problem fairly different from standard item recommendation, where user needs and preferences are often assumed to be relatively constant across short windows of time. Whereas Netflix might be fine assuming you want to watch another movie similar to the one you just watched, it's less clear that you'll want to reorder a fresh batch of almond butter or toilet paper if you bought them yesterday.

We interviewed [Kazuki Onodera](https://www.linkedin.com/in/kazuki-onodera-a55b8a66) (aka [ONODERA](https://www.kaggle.com/onodera) on Kaggle), a data scientist at Yahoo! JAPAN, to understand how he used complex feature engineering, gradient boosted tree models, and special modeling of the competition's F1 evaluation metric to win 2nd place.

![img](http://s5047.pcdn.co/wp-content/uploads/2017/09/onodera-1024x262.png)

# Basics

**What was your background prior to entering this challenge?**

I studied Economics in university, and I worked as a consultant in the financial industry for several years. In 2015, I won 2nd place in the KDD Cup 2015 challenge, where the goal of the challenge was to predict the probability that a student would drop out of a course in 10 days. Now I work as a data scientist for Yahoo! JAPAN.

**How did you get started competing on Kaggle?**

I joined Kaggle about 2 years ago after one of my colleagues mentioned it to me. My first competition was the Otto Product Classification Challenge. Since the features in that challenge were obfuscated, I couldn't perform any exploratory data analysis or feature engineering, unlike what I did here.

**What made you decide to enter this competition?**

First, I like e-commerce. I’m currently in charge of auction services at Yahoo! JAPAN.

Second, this competition seemed to have clean data, and I thought that there might be a lot of room for feature engineering. I believe my strength is feature engineering, so I thought I'd be able to achieve good results with this kind of data.

# Diving Into The Solution

## Problem Overview

The goal of this competition was to predict grocery *reorders*: given a user’s purchase history (a set of orders, and the products purchased within each order), which of their previously purchased products will they repurchase in their next order?

The problem is a little different from the general recommendation problem, where we often face a cold start issue of making predictions for new users and new items that we’ve never seen before. For example, a movie site may need to recommend new movies and make recommendations for new users.

![img](http://s5047.pcdn.co/wp-content/uploads/2017/09/Screen-Shot-2017-09-20-at-11.17.27-AM-300x194.png)

The sequential and time-based nature of the problem also makes it interesting: how do we take the time since a user last purchased an item into account? Do users have specific purchase patterns, and do they buy different kinds of items at different times of the day? And the competition’s F1 evaluation metric makes sure our models have both high precision and high recall.

## Main Approach

I used XGBoost to create two gradient boosted tree models:

1. **Predicting reorders** - which previously purchased products will be in the next order? This model depends on both the user and product.
2. **Predicting None** - will the user’s next order contain any previously purchased products? This model only depends on the user.

![img](http://s5047.pcdn.co/wp-content/uploads/2017/09/Screen-Shot-2017-09-20-at-11.19.45-AM-221x300.png)

![img](http://s5047.pcdn.co/wp-content/uploads/2017/09/Screen-Shot-2017-09-20-at-11.19.57-AM-191x300.png)

Here is a diagram of the model flow.

![img](http://s5047.pcdn.co/wp-content/uploads/2017/09/Screen-Shot-2017-09-20-at-11.22.15-AM-1024x484.png)

In words:

- The reorder prediction model uses XGBoost to create six different gradient boosted tree models (each GBDT uses a different random seed). I average their predictions together to get the probability that User A will repurchase Item B in their next order.
- The None prediction model uses XGBoost to create seventeen different models. 11 of these use an eta parameter (a step size shrinkage) set to 0.01, and the others use an eta parameter set to 0.002. I take a weighted average of these predictions to get the probability that User A won’t repurchase any items in their next order.
- To convert these probabilities into binary Yes/No scores of which items User A will repurchase in their next order, I feed them into a special F1 Score Maximization algorithm that I created, detailed below.

## Exploratory Data Analysis

Let's explore the data a little.

How hot are users? How many orders do they make?
![img](http://s5047.pcdn.co/wp-content/uploads/2017/09/p6-1.png)

![img](http://s5047.pcdn.co/wp-content/uploads/2017/09/p6-2-1.png)

How hot are items? How often are they ordered?

![img](http://s5047.pcdn.co/wp-content/uploads/2017/09/p7-1.png)

![img](http://s5047.pcdn.co/wp-content/uploads/2017/09/p7-2.png)

## Data Augmentation

One of my thoughts was that more data would help me make better predictions. Thus, I decided to augment the amount of data I could train on.

We were given three datasets:

- A "prior" dataset containing user purchase histories.
- Training and test datasets consisting of future orders that we could train and test our models on.

Rather than training my model only on the provided training set, I increased the amount of training data available to me by adding in each user's 3 most recent orders as well.

This is best illustrated by the figure below.

![img](http://s5047.pcdn.co/wp-content/uploads/2017/09/Screen-Shot-2017-09-20-at-11.33.51-AM.png)

Instead of only using the provided training set (“tr”), I also looked a short window back in time (the cells shaded in yellow) to gather more data.

## Feature Engineering

I created four types of features:

1. **User features** - what is this user like?
2. **Item features** - what is this item like?
3. **User x item features** - how does this user feel about this item?
4. **Datetime features** - what is this day and hour like?

Here are some of the ideas behind the features I created.

**User features**

- How often the user reordered items
- Time between orders
- Time of day the user visits
- Whether the user ordered organic, gluten-free, or Asian items in the past
- Features based on order sizes
- How many of the user’s orders contained no previously purchased items

**Item features**

- How often the item is purchased
- Position in the cart
- How many users buy it as "one shot" item
- Stats on the number of items that co-occur with this item
- Stats on the order streak
- Probability of being reordered within N orders
- Distribution of the day of week it is ordered
- Probability it is reordered after the first order
- Statistics around the time between orders

**User x Item features**

- Number of orders in which the user purchases the item
- Days since the user last purchased the item
- Streak (number of orders in a row the user has purchased the item)
- Position in the cart
- Whether the user already ordered the item today
- Co-occurrence statistics
- Replacement items

**Datetime features**

- Counts by day of week
- Counts by hour

For a full list of all the features I used and how they were generated, see my [Github repository](https://github.com/KazukiOnodera/Instacart).

### **Which features were the most useful?**

For the reorder prediction model, we can see that the most important features were...

![img](http://s5047.pcdn.co/wp-content/uploads/2017/09/p12.png)

To explain the top features:

- **total_buy_n5(User A, Item B)** is the total number of times User A bought Item B out of the 5 most recent orders.
- **total_buy_ratio_n5** is the proportion of A's 5 most recent orders in which A bought B.
- **useritem_order_days_max_n5**, described in more detail below, captures the longest that A has recently gone without buying B.
- **order_ratio_by_chance_n5** captures the proportion of recent orders in which A had the chance to buy B, and did indeed do so. (A "chance" refers to the number of opportunities the user had for buying the item after first encountering it. For example, if a user A had order numbers 1-5, and bought item B at order number 2, then the user had 4 chances to buy the item, at order numbers 2, 3, 4, and 5.)
- **useritem_order_days_median_n5** is the median number of days that A has recently gone without buying B.

(Note: the suffix "_n5" means "near5", i.e., features extracted from the 5 most recent orders.)

For the None prediction model, the most important features were…

![img](http://s5047.pcdn.co/wp-content/uploads/2017/09/p13.png)

- **useritem_sum_pos_cart-mean(User A)** is described in more detail below, and is a kind of measure of whether the user tends to buy a lot of items at once.
- **total_buy-max** is the maximum number of times the user has bought any item.
- **total_buy_ratio_n5-max** is the maximum proportion of the 5 most recent orders in which the user bought a certain item. For example, if there was an item the user bought in 4 out of their 5 most recent orders, but no other item more often than that, this feature would be 0.8.
- **total_buy-mean** is the mean number of times the user has bought any item.
- **t-1_reordered_ratio** is the proportion of items in the previous order that were repurchases.

## **Insights**

Here were some of my most important insights into the problem.

### **Important Finding for Reorders - #1**

Let’s think about the reordering problem. Common sense tells us that an item purchased many times in the past has a high probability of being reordered. However, there may be a pattern for when the item is *not* reordered. We can try to figure out this pattern and understand when a user doesn’t repurchase an item.

For example, consider the following user.

![img](http://s5047.pcdn.co/wp-content/uploads/2017/09/Screen-Shot-2017-09-20-at-11.37.37-AM.png)

This user pretty much always orders Cola. But at order number 8, the user didn’t. Why not? Probably because the user bought Fridge Pack Cola instead.

So I created features to capture this kind of behavior.

### **Important Finding for Reorders - #2**

**Days_since_last_order_this_item(User A, Item B)** is a feature I created that measures the number of days that have passed since User A last ordered Item B.

**Useritem_orders_days_max(User A, Item B)** is the maximum of the above feature across time, i.e., the longest that User A has ever gone without ordering B.

**Days_last_order-max(User A, Item B)** is the difference between these two features. So this feature tells us how ready the user is to repurchase the item.

Indeed, if we plot the distribution of the feature, we can see that it’s highly predictive of our target value.

![img](http://s5047.pcdn.co/wp-content/uploads/2017/09/p14-1.png)

### **Important Finding for Reorders - #3**

We already know that fruits are reordered more frequently than vegetables (see [3 Million Instacart Orders, Open Sourced](https://tech.instacart.com/3-million-instacart-orders-open-sourced-d40d29ead6f2?gi=6dbf60de80d1)). I wanted to know how often, so I made a item_10to1_ratio feature that’s defined as the reorder ratio after an item is ordered vs. not ordered.

![img](http://s5047.pcdn.co/wp-content/uploads/2017/09/p17-1.png)

### **Important Finding for None - #1**

**Useritem_sum_pos_cart(User A, Item B)** is the sum across orders of the position in User A’s cart that Item B falls into.

**Useritem_sum_pos_cart-mean(User A)** is the mean of the above feature across all items.

This feature says that users who don't buy many items all at once are more likely to be None.

![img](http://s5047.pcdn.co/wp-content/uploads/2017/09/p18.png)

### **Important Finding for None - #2**

**Total_buy-max(User A)** is the total number of times User A has purchased any item. We can see that it predicts whether or not a user will make a reorder.

![img](http://s5047.pcdn.co/wp-content/uploads/2017/09/p19.png)

### **Important Finding for None - #3**

**t-1_is_None(User A)** is a binary feature that says whether or not the user’s previous order was None (i.e., contained no reordered products).

If the previous order is None, then the next order will also be None with 30% probability.

![img](http://s5047.pcdn.co/wp-content/uploads/2017/09/p20.png)

## **F1 Maximization**

In this competition, the evaluation metric was an [F1 score](https://en.wikipedia.org/wiki/F1_score), which is a way of capturing both precision and recall in a single metric.

![img](http://s5047.pcdn.co/wp-content/uploads/2017/09/Screen-Shot-2017-09-20-at-1.21.43-PM.png)

Thus, instead of returning reorder probabilities, we need to convert them into binary 1/0 (Yes/No) numbers.

In order to perform this conversion, we need to know a threshold. At first, I used grid search to find a universal threshold of 0.2. However, then I saw comments on the Kaggle discussion boards suggesting that different orders should have different thresholds.

To understand why, let’s look at an example.

![img](http://s5047.pcdn.co/wp-content/uploads/2017/09/unnamed.png)

Take the order in the first row. Let’s say our model predicts that Item A will be reordered with 0.9 probability, and Item B will be reordered with 0.3 probability. If we predict that only A will be reordered, then our expected F1 score is 0.81; if we predict that only B will be reordered, then our expected F1 score is 0.21; and if we predict that both A and B will be reordered, then our expected F1 score is 0.71.

Thus, we should predict that Item A and only Item A will be reordered. This will happen if we use a threshold between 0.3 and 0.9.

Similarly, for the order in the second row, our optimal choice is to predict that Items A and B will both be reordered. This will happen is long as the threshold is less than 0.2 (the probability that Item B will be reordered).

What this illustrates is that each order should have its own threshold.

### **Finding Thresholds**

How do we determine this threshold? I wrote a simulation algorithm as follows.

Let’s say our model predicts that Item A will be reordered with probability 0.9, and Item B with probability 0.3. I then simulate 9,999 target labels (whether A and B will be ordered or not) using these probabilities. For example, the simulated labels might look like this.

![img](http://s5047.pcdn.co/wp-content/uploads/2017/09/Screen-Shot-2017-09-20-at-1.24.05-PM.png)

I then calculate the expected F1 score for each set of labels, starting from the highest probability items, and then adding items (e.g., [A], then [A, B], then [A, B, C], etc) until the F1 score peaks and then decreases.

![img](http://s5047.pcdn.co/wp-content/uploads/2017/09/Screen-Shot-2017-09-20-at-1.24.56-PM.png)

## **Predicting None**

One way to think about None is as the probability (1 - Item A) * (1 - Item B) * …

But another method is to try to predict None as a special case. By creating a None model and treating None as just another item, I was able to boost my F1 score from 0.400 to 0.407.

# **Words of wisdom**

**What have you taken away from this competition?**

All metrics can be hacked, I think. Especially metrics where we have to convert probabilities to binary scores. (Although metrics like AUC are rarely hacked.)

**Do you have any advice for those just getting started in data science?**

Join the competitions you like. But never give up before the end, and try every approach you come up with. I know it’s a tradeoff between sleep and your leaderboard ranking. It’s common for features that take a lot of time to construct to wind up doing nothing. But we can’t know the result if we don't do anything. So the most important thing is to participate in the delusion that you’ll get a better result if you try!