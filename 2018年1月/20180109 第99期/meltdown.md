# The Meltdown bug and the KPTI patch: How does it impact ML performance?

![](https://cdn-images-1.medium.com/max/1600/1*ra9lenkHMnnO2un2hUjPdQ.png)

At the start of 2018, the internet learned of two new serious exploits affecting major processor vendors, dubbed [Meltdown](https://meltdownattack.com) and [Spectre](https://spectreattack.com/). These exploits are bugs in processors’ speculative execution that allows an attacker to read (and potentially exectute) memory locations outside of their respective process, meaning that programs could read sensitive data in other software’s memory.

In order to fix the bug, the Linux kernel merged a patch known as KAISER or PTI (page table isolation), which effectively fixes the Meltdown attack. However, this patch induces a hit to performance, with people reporting **anywhere from a 5% to**[35%](https://siliconangle.com/blog/2018/01/02/intel-patches-critical-processor-security-bug-fix-imposes-35-performance-hit/)**reduction in CPU performance** across the board (and [some synthetic benchmarks even showing >50% performance drops](https://www.phoronix.com/scan.php?page=article&item=linux-415-x86pti&num=2)).

However, the PTI performance issues are very largely dependent on the task at hand — it’s possible such large drops are only present in synthetic benchmarks such as FSMark. So the question is: what performance hit will we see in machine learning applications?

#### The Setup

In order to compare performance with and without PTI, I setup a fresh Ubuntu 16.04 machine with intel microcode installed, and compared the latest kernel installed automatically on Ubuntu 16.04 (4.10.0–42-generic) with the latest mainline kernel release (4.15.0–041500rc6-generic) which has the PTI patch. I used Anaconda with Python 3.6 (and extra packages from pip) to perform the testing.

The rig I used for testing included a Intel Core i7–5820K (Haswell-E, stock clocks) and 64GB of DDR4 @ 2400MHz. If people are interested, I can get my hands on more Intel CPUs to test across a wider range of generations. It’s worth noting that AMD processors do not have the PTI patch enabled, as they are immune to the Meltdown attack — so **no performance hit should be expected if you’re on AMD** 🙂.

### The Results

First of all, we see a very slight decrease in performance across the board, but a large decrease in inference on models with convolutional layers. Specifically with AlexNet, the forward pass is about 5% slower, but backpropagation speed is almost the same — which is why the performance hit to training is about half of that of inference.

In terms of raw operations with Keras, fully connected and LSTM layers take almost no performance hit but convolutions have a large 10% decrease.

For Alexnet and MNIST benchmarks, I used the [TensorFlow tutorial models](https://github.com/tensorflow/models/tree/master/tutorials/image) while for Keras I used a randomly initialised model with several of the layer in question and measured inference speed on random data. It’s worth noting these benchmarks were run entirely on CPU.

I used Scikit-learn here to measure performance across ‘classical’ ML and data science algorithms. Here, we see a much bigger performance decrease compared to NNs, with PCA and Linear/LogisticRegression being most heavily affected. The reason for this decrease is probably due to some math being very heavily impacted — as discussed in the NumPy benchmarks below.

Interestingly, kNearestNeighbour is completely unaffected by PTI and actually appears to perform slightly better on the newer kernel. This is probably just within margin of error, but it’s possible some other kernel improvements helped speed it up slightly.

I also threw in a benchmark of pandas.read_csv() from a file cached in memory to see how much CSV parsing speed is decreased with PTI — about 6% reading the [Bosch Kaggle competition dataset](https://www.kaggle.com/c/bosch-production-line-performance) (2GB, 1M rows, 1K columns, float, 80% missing).

All the scikit-learn benchmarks were also computed on the Bosch dataset — I find it is generally good for ML benchmarks because it has large, normalised and well-formed data (although kNN and Kmeans were computed on a subset since the full data would have taken too long).

These benchmarks are probably the most synthetic here, testing the speed of only a single scipy operation. However, these results show us that the performance hit of PTI is extremely task-dependent. Here we can see that most operations are only minorly affected, with dot product and FFT taking a small hit to performance.

SVD, LU decomposition and QR decomposition all take massive hits to performance when PTI is enabled, with **QR decomposition decreasing by 37% from 190GFLOPS to 110GFLOPS**. This probably helps explain the performance decrease in PCA (which relies heavily on SVD) and linear regression (which relies heavily on QR decomposition).

These benchmarks were done using Intel’s own [ibench](https://github.com/IntelPython/ibench) package — just using Anaconda instead of Intel’s python distribution.

XGBoost gives us some interesting results here. For the most part, when working with a low number of threads, XGBoost has a neglible performance decrease from PTI, regardless of whether the slow exact method or fast histogram method is used.

However, when using very many threads, the CPU is working on many more columns at the same time and the speed using PTI falls off a cliff.

This isn’t a perfect representation of how XGBoost will perform on a high number of cores (as this is running 40 threads on 12 logical cores), but it gives an indication that PTI has a bigger impact when the CPU is working on many things at once. Unfortunately I don’t have access to any high-core-count servers where I can modify the kernel, so I can’t get more in-depth results.

Like with scikit-learn, these benchmarks were conducted on the Bosch dataset.

### Conclusion

The main thing to take away from this is that PTI’s performance impact is very much task-dependent — some tasks are unaffected and some have up to a whopping 40% reduction in performance. Overall, I think the impact is smaller than I expected, as it’s only a few applications that are heavily affected.

Thanks for reading! I hope this helps shed some light on what to expect when the kernel updates reach you. 😊 If you have any questions, requests for more benchmarks to run or want to see the tests repeated on different CPUs, let me know!

