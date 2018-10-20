# 10 tips on using Jupyter Notebook

原文链接：[10 tips on using Jupyter Notebook](https://medium.com/@r_kierzkowski/10-tips-on-using-jupyter-notebook-abc0ba7028a4?from=hackcv&hmsr=hackcv.com&utm_medium=hackcv.com&utm_source=hackcv.com)

![img](https://cdn-images-1.medium.com/max/1600/1*LPnY8nOLg4S6_TG0DEXwsg.png)

[Jupyter Notebook](http://jupyter.org/) (a.k.a iPython Notebook) is brilliant coding tool. It is ideal for doing reproducible research. Here is my list of 10 tips on structuring Jupyter notebooks, I worked out over the time.

#### 1. Use virtualenv to create self-contained environment

You might be tempted to install all research libraries within your operating system and share them among all your projects. Soon you will discover that when you add some additional library it may update ones installed previously. Some of the other libraries will no longer work with newer versions. So when you go back to a previous project you will waste a lot of time trying to figure out what changed and how to fix it.

The solution is to use separate virtual environment for each of your projects. I recommend using [virtualenv](https://virtualenv.pypa.io/) via [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/). To avoid problems with resolving paths to the virtual environment you should install Jupyter in each environment separately.

#### 2. Go with Python 3

It’s better. Really.

#### 3. Include requirements.txt

When you have a separate environment for your project, it is a good idea to save the list of dependencies. It will save you a lot of time in the future. For example when you will want to recreate the environment.

```
$ pip freeze > requirements.txt
```

#### 4. Do all imports in the first cell

Do all your imports in the first cell of your notebook. It has two benefits. The dependencies and tools used are obvious at the first glance. When you restart the notebook server, you can have all your imports restored with a single re-run. It is especially useful when you don’t want to re-execute the entire notebook.

I also use this cell to define any filesystem paths used in the notebook.

#### 5. Start dirty and keep your draft

Start quick and dirty. The fastest you get to what you want to do, the better. *The inspiration is perishable* [[*Rework*](http://www.amazon.com/gp/product/B002MUAJ2A/ref=as_li_ss_il?ie=UTF8&tag=sleepcodin-20&linkCode=as2&camp=1789&creative=390957&creativeASIN=B002MUAJ2A), by Jason Fried]*.* But when you notice that you start stepping on your own toes, that you are no longer effective and the development become clumsy, it is time to organize the notebook. Start over, copy the good code, rewrite and generalize bad one, but whatever you do: KEEP THE DRAFT NOTEBOOK!

#### 6. Wrap cell content in a function

Many of the notebook cells will look like this:

```
parameter1 = 1.0
parameter2 = 100
step1 = X * parameter1
step1 * parameter2
```

There are parameters at the beginning of the cell. You change them and re-execute the cell or you even copy the entire cell and modify parameters. There are some intermediate computations and at the end, there is a line to display the results.

It’s ok in the draft. But after a while it becomes unmanageable. You got plenty of intermediary variables trashing a global namespace. You lose the steps that led you to the current parameter choices.

Instead, you can wrap it all in one a function:

```
def computation(parameter1=1.0, parameter2=100):
    step1 = X * parameter1
    return step1 * parameter2
computation()
...
computation(parameter1=10.0)
```

You can modify the parameters and re-execute in a separate cell, keeping the history of changes. The intermediary steps will no longer trash the global namespace and consume memory.

#### 7. Use joblib for caching output

You thought your neural network for three days and now you are ready to build on top of it. But you forgot to plug your laptop to a power source and it runs out of batteries. So you scream: Why didn’t I pickle!? The answer is: because it is pain in the back. Managing file names, checking if the file exists, saving, loading… What to do instead? Use [joblib](https://pythonhosted.org/joblib/).

```
from sklearn.externals.joblib import Memory
memory = Memory(cachedir='/tmp', verbose=0)
@memory.cache
def computation(p1, p2):
    ...
```

With three lines of code, you get caching of the output of any function. Joblib traces parameters passed to a function, and if the function has been called with the same parameters it returns the return value cached on a disk.

#### 8. Make sections of the notebook self-contained

Make sections of your notebook loosely bound. Use as little global variables as possible. If you wrap your cells in functions and you use joblib for caching, it is really inexpensive to call same code within each section. It’s better than making code reliable on the variables created several cells above.

In general, try to limit the number of cells you have to re-run after the restart to continue on your work.

#### 9. Reuse variable names.

Don’t use long variable names. When you get a chance re-use existing ones. It is contrary to the advice I would give when developing other kinds of software, but in case of a notebook this approach works better.

Let me illustrate it with an example. Let’s assume that your algorithms need a list of clusters. You try various versions of clustering and algorithms. Your code can look like this:

```
clusters_kmeans_k10 = KMeans(k=10).fit_predict(X)
clusters_kmean_k5 = KMeans(k=5).fit_predict(X)
# many cells further
algorithm1(clusters_kmeans_k10)
algorithm2(clusters_kmeans_k10)
algorithm1(clusters_kmeans_k5)
algorithm2(clusters_kmeans_k5)
```

But instead you can use joblib cached function and re-use variables:

```
@memory.cache
def kmeans(X, k):
    return KMeans(k=k).fit_predict(X)
# many cells further
clusters = kmeans(X, k=10)
algorithm1(clusters)
algorithm2(clusters)
clusters = kmeans(X, k=5)
algorithm1(clusters)
algorithm2(clusters)
```

#### 10. Use assertions to test utility functions

When you create some utility function, create short tests using *assert* keyword. For example:

```
def norm_scale(X, axis=0):
    mx = np.max(X, axis=axis)
    mi = np.min(X, axis=axis)
    epsilon = 10**-32
    return (X — mi) / (np.abs(mi) + mx + epsilon)
norm = norm_scale(X)
assert np.min(norm) >= 0
assert np.max(norm) <= 1
```

Here are my tips? What are yours? How do you organize your notebooks?



![img](https://cdn-images-1.medium.com/max/1600/1*QCV7h713dLgy5COZTyBLdQ@2x.png)