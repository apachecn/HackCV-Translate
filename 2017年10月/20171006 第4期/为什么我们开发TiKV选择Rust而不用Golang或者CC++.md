# 为什么我们开发TiKV选择Rust而不用Golang或者C/C++?

原文链接：[Why did we choose Rust over Golang or C/C++ to develop TiKV?](https://pingcap.github.io/blog/2017/09/26/whyrust/?from=hackcv&hmsr=hackcv.com&utm_medium=hackcv.com&utm_source=hackcv.com)

## 什么是Rust

[Rust](https://en.wikipedia.org/wiki/Rust_(programming_language)) 是由Mozilla Research赞助的系统编程语言，自2015年5月的1.0版本以来，它以6周的发布周期快速稳定地运行。

请参阅下面的列表来了解最吸引我们的一些特性:

- Rust的设计原则类似C++的 [没有消耗的抽象](https://blog.rust-lang.org/2015/05/11/traits.html) and [RAII ](https://rustbyexample.com/scope/raii.html)。
- 最少的运行时间和绑定高效的C使得Rust能和C和C++一样高效，因此因此非常适合高性能最重要的系统编程领域。
- 编译期间，强大的类型系统和独特的生命周期管理有助于内存管理，确保内存和线程的安全并且在编译后会让程序运行的非常迅速。
- Rust提供字符串的匹配和像函数式编程语言那样的类型推断，使得代码简单、优雅。
- 宏和特性允许Rust高度抽象，在工程期间，特别是在设计到库的时候节省了相当多的样板。

## Rust系统

由于Cargo这个优秀个包管理工具，Rust有很多类型的库，比如，用于HTTP的Hyper，异步I/O的Tokio和mio，基本上涵盖了构建后端应用程序的所需要的所有库。

一般来说，Rust主要用于开发具有高性能服务器的应用程序。

## 使用Rust

作为一个新的编程语言，Rust是独特的，仅举几个使用Rust的项目：

- Dropbox的后端分布式存储系统
- [Servo](https://github.com/servo/servo), Firefox的新内核
- [Redox](https://github.com/redox-os/redox), 新的操作系统
- [TiKV](https://github.com/pingcap/tikv),  [TiDB](https://github.com/pingcap/tidb)的存储层, 由 [PingCAP](https://pingcap.com/index)开发的分布式数据库



TiKV 是一个键值对分布式数据库，它是TiDB项目的核心组件，是Google Spanner的开源实现。在这个博客，我将会说出为什么我们选择Rust从头开始构建这样一个大型分布式存储项目。

在过去的很长一段时间，C或者C++主要开发类似数据库这样的基础设施软件，Java和Golang存在GC抖动等问题，尤其是在读/写压力较高的情况下。一方面，Goroutine的轻量级线程和Golang迷人的功能以其运行时切换上下文的额外开销为代价，明显降低了开发并发程序的复杂性。

TiKV起源于2015年底。我们的团队在不同语言选择中苦苦挣扎，如Pure Go，Go + Cgo，C ++ 11或Rust。

- **Pure Go:** 我们核新团队对于Go语言有着丰富的经验，TiDB的SQL层是用Go开发的，Go的性能很高。然而，当设计到存储层的开发的时候，Pure Go是第一个排除的语言，原因是：我们决定使用RocksDB作为底层，用C ++编写，Go中现有的LSM-Tree实现（如goleveldb）没有RocksDB那样成熟。

- **Cgo:**如果我们使用Go，我们就要用Cgo来搭桥，但是Cgo有自己的问题。2015年年底，如果在Go代码中调用Cgo而不是在与Goroutine相同的线程中调用Cgo，性能可能会受到很大影响。此外，数据库需要频繁调用底层存储库，即RocksDB，如果每次调用RocksDB函数时都需要额外的开销，那么效率非常低。当然，可以引入一些变通方法来扩大调用Cgo的吞吐量，比如，在一定时间内为Cgo进行批量打包，这将减少单个请求的消耗而且抵消Cgo的开销。但是，GC问题还没有完全解决，这样实施可能会非常困难。在存储层，我们希望内存的速度尽可能的高效。Hacky的解决方法，例如广泛使用syscall.Mmap或对象重用可能会损害代码的可读性。

- **C++11:** C ++ 11应该完全没有问题。 RocksDB是使用C ++ 11开发的。但鉴于团队背景和我们想要做的事情，我们没有选择C ++ 11。以下为理由：

  1. 核心团队的成员是有着丰富的C++项目经验的资深C++开发者，但是像悬挂指针，内存泄漏或数据竞争这样的大型项目中看似不可避免的问题让他们对这个想法感到不寒而栗。当然，如果有良好的指导，或者严格的代码审查和编码规则，这些问题的可能性可能会降低很多。
  2. C++有大量且复杂的编程范式以及太多的技巧，它需要额外的成本来统一编码风格，特别是当有越来越多的新成员可能不熟悉C ++。经过多年使用GC语言后，很难花时间手动管理内存。
  3. 缺乏包管理和CI工具。自动化工具对于大的项目来说非常重要，因为关系到开发效率和迭代的速度，更重要的是，C ++库远远不够，其中一些需要由我们自己创建。

- **Rust:** Rust的1.0版本于2015年5月发布，具有一些迷人的功能：

  1. 内存安全
  2. LLVM赋予的高性能。运行时几乎与C ++没有什么不同。它还与C / C ++包有密切关系。
  3. Cargo, 强大的包管理工具
  4. 现代语法
  5. 几乎一致的故障排除和性能调整体验。我们可以直接重用一些我们已经非常熟悉的工具，比如perf。
  6. FFI（外部函数接口），直接调用RocksDB中的C API而不会丢失。

  内存安全是第一个而且也是最重要的原因。如前文所提，对于C++的老手而言，内存还礼和数据竞争的问题似乎变得很容易。但我相信Rust正在做的最大的解决方案是在编译器中加入约束并从一开始就解决它。对于大型的项目，永远不要用人来保证质量，人非圣贤，孰能无过。尽管对于Rust来说可能有点困难，但是我认为i这是完全值得的。此外，Rust是一种非常现代的编程语言，具有非凡的类型系统，模式建模，强大的宏，特征等。一旦熟悉它，就可以大大提高效率，这可能与我们选择C ++计算调试时间的效率相同。根据我们的经验，对于Rust零经验的软件工程师需要大概1个月来进行开发。经验丰富的Rust工程师和Golang工程师之间的效率几乎相同。

总而言之，作为一种新兴的编程语言，Rust似乎对中国的大多数开发人员来说都是新手，但它已成为C / C ++最有前途的挑战者。Rust也被称为“最受喜爱”的技术在 [StackOverflow’s 2016 developer survey](http://techbeacon.com/highlights-stack-overflow-2016-developer-survey).。因此从长远来看，Rust将在内存安全性和性能最重要的场景中大放异彩。
