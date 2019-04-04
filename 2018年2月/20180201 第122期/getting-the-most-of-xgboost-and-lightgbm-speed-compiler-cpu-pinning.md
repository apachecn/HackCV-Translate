# Getting the most of xgboost and LightGBM speed: Compiler, CPU pinning

**Currently, xgboost and LightGBM are the two best performing machine learning algorithms for large datasets** (both in speed and metric performance). They **scale very well up to billion of observations and/or elements** (ex: Reputation dataset, [53,181,000,000 elements](https://sites.google.com/view/lauraepp/new-benchmarks)).

**xgboost and LightGBM were made primarily for speed**: it is better to **iterate quickly at high accuracy to try more different things**, than waiting your neural network to finish after hours.

However, although they can be used on large datasets, **the question of scalability was**[partially answered](https://medium.com/data-design/benchmarking-xgboost-5ghz-i7-7700k-vs-20-core-xeon-ivy-bridge-and-kvm-vmware-virtualization-293807a13f1c): how well xgboost and LightGBM are scaling? **Do they prefer high frequency cores or more cores?**

* **xgboost exact** likes both many cores and high frequency, with a preference on both

* **xgboost fast histogram** needs high frequency

* **LightGBM** likes both many cores and high frequency, with a preference on high frequency

As we already know the answer to this question, we are going to look up for a more exotic situation: **changing the compiler, and pinning CPU**.

**Are xgboost and LightGBM faster by swapping the compiler from MinGW to Visual Studio? Is CPU pinning a good thing to do?**

This was also partially answered in [this GitHub issue](https://github.com/Microsoft/LightGBM/issues/749). Therefore, we are back with our Windows machine to do some benchmarks.

**Interactive documents**:

* [xgboost and LightGBM raw data](https://benchmark.laurae.design/speed_r_perf_analysis.html)

* [Visual Studio 2017 vs MinGW 4.9](https://benchmark.laurae.design/speed_r_vs_mingw.html)

* [CPU Roaming vs CPU Pinning](https://benchmark.laurae.design/speed_r_roaming_pinning_cpu.html)

* [GPU xgboost raw data](https://benchmark.laurae.design/speed_r_perf_gpu_analysis.html)

In the conclusion, an opening to **GPU xgboost** was included.

### A quick review on the definition of a compiler and CPU pinning

#### Defining a compiler and CPU pinning

![](https://cdn-images-1.medium.com/max/1200/0*X41dC85-9aXFv3jK.png)

![](https://cdn-images-1.medium.com/max/1200/1*GadFVd3t3tIzCsZvtmQ3lA.png)

* **Compiler**: the compiler **transforms the code of a source language into a code of a target language (usually to generate an executable)**. They are similar to a **translator**, and we all know translators do **not have the same level of performance**: some are providing gibberish words, some are providing excellent translations, which in turns make your **interpretation of words slower or quicker**.

* **CPU pinning**: CPU pinning is the **binding of a process (or thread) to a specific range of CPU cores**. This way, **the process will not roam anywhere as easily as it could without CPU pinning**. When the **process roams across CPUs, it incurs significantly higher RAM and cache latency**: this is even more severe with **multi-socket CPUs**.

**CPU pinning is also named CPU affinity**, although the **wording is inexact**(“affinity” could mean “preference”, although it is not in this case: it is **“this process uses this range and only this range of CPU cores”**).

#### Benchmarking the differences

We are going to benchmark the difference between compilers and CPU pinning, for each number of threads available (1 to 56) on our server:

* **Two compilers** to test: **Visual Studio** (Windows’ native) and **MinGW** (gcc)

* **Two CPU behaviors**: **CPU roaming** (no pinning) and **CPU pinning** (by socket, then by physical core, then by hyperthreaded core).

The latter means the following: if we have 2 sockets, 4 physical cores on each socket, and hyperthreaded activated, we will try to **contain all CPUs in one socket**, first adding physical (yellow) cores, then adding logical (orange) cores:

![](https://cdn-images-1.medium.com/max/1600/1*_Z2mPQvpuna7yMa5tURfbg.png)

We are benchmarking xgboost and LightGBM under the following **environment**:

* CPU: Dual Intel Xeon E5–2697v3 (14 cores, 28 threads, 3.6 GHz singlethread, 3.1 GHz multithread)

* RAM: 128GB RAM DDR4 2133 MHz

* GPU: none

* OS: Windows Server 2012 R2 Datacenter, without Meltdown/Spectre patch

* R version: default 3.4.3

* Compiler: Visual Studio 2017, MinGW 4.9 (R)

* xgboost: commit 3f3f54b (Jan 16, 2018, 5:16 PM GMT+1)

* LightGBM: commit 3dc5716 (Jan 18, 2018, 2:16 AM GMT+1)

The **dataset**:

* [Kaggle Bosch training dataset](https://www.kaggle.com/c/bosch-production-line-performance)

* Number of observations: 1,183,747

* Number of features: 969

* Sparsity: approx 81%

The **algorithm parameters**:

* Number of boosting iterations: 200

* Learning rate: 0.05

* Maximum depth: 8

* Maximum leaves: 255

* Max bins: 255

* Minimum hessian: 1

* xgboost only: fast histogram, depth-wise

* LightGBM only: minimum split loss of 1 (due to loss-guided optimization)

Each run were repeated at least twice, up to 10 times. It took approximately 1 week to run the benchmark, thanks to having so many threads!!!

### Benchmark Results

#### Reminder: xgboost and LightGBM does not scale linearly at all.

xgboost is up to 154% faster than a single thread, while LightGBM is up to 1,116% faster than a single thread.

If you have a workstation…:

* If you have 56 threads, do not expect that 56 threads to be 5,500% more efficient than 1 thread (it will not train 55x times faster).

* If you have 28 cores, do not expect that 28 threads to be 2,700% more efficient than 1 thread (it will not train 27x times faster).

* **If you have a small dataset, do not expect lot of threads to scale well** (it will negatively scale).

**Showing the results taking the best case scenario**(Visual Studio, Roaming CPUs) below:

#### Compiler Performance

**By far, Visual Studio is the compiler to go on Windows.** It is worth installing [Visual C++ Build Tools](http://landinghub.visualstudio.com/visual-cpp-build-tools) to get the fastest training speed possible.

**With roaming CPUs**:

**With CPU pinning**:

#### CPU pinning Performance

**CPU pinning increases the performance of xgboost with MinGW significantly. Otherwise, we are seeing performance degradation.**

Story morale:

* **Use CPU pinning if you are using xgboost with MinGW.**

* Another case: if you are training **parallel xgboost and LightGBM on the same machine, pin the CPUs in order to make sure CPU cache effects can trigger properly**(ex: if you are training 4 xgboost models at the same time on a 4 core machine, pin each model process to a separate core).

With **Visual Studio**:

With **MinGW**:

### Conclusion

**Using Visual Studio without CPU pinning seems the best choice by far.**

The recommendations for the power users wanting the most of their xgboost/LightGBM:

* **Use Visual Studio whenever possible**

* **Train models without CPU pinning**

* And attempt to get higher CPU frequencies…

If you were **forced to use xgboost in Windows, then force CPU pinning to increase the performance**.

If you have **single models to train, GPU xgboost** seems the way to go due to how stable it became today. **You do not even need a powerful server, even a laptop’s NVIDIA 1050 Ti outperforms our monster server.**

> For curious, using a NVIDIA 1050 Ti (1.75 GHz) on a laptop with GPU xgboost, it takes 92 seconds to train a model. That’s 28 seconds faster than the fastest xgboost (Visual Studio + CPU pinning + 9 physical cores). An overclocked workstation would slash that time to about 60 seconds.

Find below the **most brutal comparison in efficiency**, when using xgboost and CPU pinning:

Next part: [Investigating xgboost Exact scalability](https://medium.com/@Laurae2/investigating-xgboost-exact-scalability-d562b2b501c0)

