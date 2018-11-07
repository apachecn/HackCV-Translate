# NOSQL数据库中的分布式算法

原文链接：[DISTRIBUTED ALGORITHMS IN NOSQL DATABASES](https://highlyscalable.wordpress.com/2012/09/18/distributed-algorithms-in-nosql-databases/?from=hackcv&hmsr=hackcv.com&utm_medium=hackcv.com&utm_source=hackcv.com)

可扩展性是NoSQL运动的主要驱动因素之一。因此，它包括分布式系统协调，故障转移，资源管理以及许多其他功能。这听起来像一把大伞，然而确实如此。 虽然很难说NoSQL运动是否将根本性的新技术引入分布式数据处理，但它引发了大量的实际研究，以及关于协议和算法不同组合的实际试验。已被证明具有实用效率的关系型数据库管理系统在这期间逐渐崭露头角。在本文中，我试图或多或少地系统性描述与NoSQL数据库中的分布式操作相关的技术。

接下来，我们将研究许多分布式策略，例如故障检测中的复制，这可能在数据库中发生。下面以粗体突出显示的这些活动分为三个主要部分

接下来，我们将研究许多分布式策略，例如故障检测中的复制，这可能在数据库中发生。下面以粗体突出显示的这些活动分为三个主要部分：

- 数据一致性。 从历史上看，NoSQL为了为地理位置分散的系统，低延迟或高可用性应用程序提供服务,其非常注重在一致性，容错性和性能之间进行权衡。 从根本上说，这些权衡都是围绕数据一致性进行，因此本节主要是关于 **数据复制** 和 **数据修复**。
- 数据分布。 数据库应适应不同的数据分布，集群拓扑和硬件配置。 在本节中，我们将讨论如何 **分发数据或再平衡数据** 才能快速处理故障，提供持久性保证，查询高效以及系统资源（如内存和硬盘空间）在整个群集中均匀使用。
- 系统协调。 协调技术就好像 **领导者选举** ，它在许多数据库中用于实现容错和强大的数据一致性。 然而，即使是分散的数据库通常也会跟踪它们的全局状态， **检测故障和拓扑变化**。 本节介绍几种用于保持系统连贯性状态的重要技术。

## 数据一致性
众所周知，在分布式系统(geographically distributed systems)或在可能是网络分区或延迟的其他环境中，数据库的分区部分必须独立操作，因此在不牺牲一致性的情况下通常不可能保持高可用性，这就是CAP定理。 然而，在分布式系统中一致性是十分昂贵，因为它可以交换到不仅仅是可用性， 一致性会经常涉及多个权衡。 为了研究这些权衡，我们首先注意到分布式系统中的一致性问题是由于耦合数据的复制和空间的分离引起的，因此我们必须从目标开始，探寻复制的必备属性：
  - 可用性。在网络分区的情况下使得数据库的独立部分可以提供读/写请求。
  - 读写延迟。读/写请求是具有最小延迟的进程。
  - 读/写扩展性。 可以跨多个节点均衡读/写负载。
  - 容错性。 提供读/写请求的处理不依赖于任何一个特定节点。  
  - 数据持久性。 特定条件下节点故障不会导致数据丢失。
  - 一致性。 一致性是一个比以前更复杂的属性，因此我们必须详细讨论不同的情况。 从理论上深入理解的一致性和并发模型会超出本文所讲内容，因此我们使用了非常精简的简单属性框架。
  - 读-写一致性。从读写的角度来看，数据库的基本目标是使得副本趋同的时间最小化（即将更新传递到所有副本时间）并保证最终的一致性。 除了这些弱保证之外，还有一些更强的一致性特点：
    - 写后读的一致性。 一次写操作对数据项X的影响将会被后继的读操作所看到。
    - 读后读一致性。 如果某个客户端读取数据项X的值，则关于X的后续读操作将始终返回（与第一次）相同或是更新后的值。
  - 写-写一致性。 在数据库分区的情况下会出现写冲突，因此数据库应该以某种方式处理这些冲突，或保证在不同分区不会处理并发写入。 从这个角度来看数据库可以提供了几种不同的一致性模型：
    - 原子级写入。 如果希望数据库提供的API中写入请求只能对值进行独立原子性赋值，可以选择每个数据的“最新”版本来避免写冲突。 这保证了所有节点最终都将使用相同版本的数据，而忽略可能受网络故障和延迟影响的更新顺序。 最后通过时间戳或特定于应用程序的度量来指定数据版本，例如Cassandra就使用的这种方法。
    - 原子级读-改-写。 应用程序可能需要执行读-改-写序列而不是单独的原子级写操作。 如果两个客户端读取相同版本的数据，修改它并同时回写，按照原子写模型，时间上比较靠后的那一次更新将会覆盖前一次。 此行为在可能不太合适（例如，如果两个客户端都向列表添加值）。 数据库可以提供至少两种解决方案：
      - 冲突预防。读-改-写可以被认为是事务的特定情况，因此像分布式锁或PAXOS [20,21]这样的一致性协议都是一种解决方案。这是一种通用技术，他支持原子读-改-写操作和任意分区级事务。另一种方法是完全阻止分布式并发写操作，并将特定数据项的所有写操作路由到单个节点（全局主节点或分片主节点）。为了防止冲突，数据库必须在网络分区的情况下牺牲可用性，停用其他所有分区而只留下一个用。这种方法用于许多具有强一致性保证的系统（例如大多数RDBMS，HBase，MongoDB）。
      - 冲突检测。数据库跟踪更新的并发冲突，然后要么回滚其中一个冲突更新，要么保留两个版本交给客户端解决。通常使用向量时钟[19]（是乐观锁的一种广义形式）或通过保留整个版本历史来跟踪并发更新。这种方法用于Riak，Voldemort，CouchDB等系统。

现在让我们仔细看看常用的复制技术，并根据所描述的特点对它们进行分类。 下面的第一个图描绘了不同技术之间的逻辑关系及其在一致性-扩展性-可用性-延迟性在权衡系统（consistency-scalability-availability-latency tradeoffs）中的坐标。 第二个图详细说明了每种技术。

[![consistency-plot-3](https://highlyscalable.files.wordpress.com/2012/09/consistency-plot-3.png?w=805)](https://highlyscalable.files.wordpress.com/2012/09/consistency-plot-3.png)

[![consistency-catalog](https://highlyscalable.files.wordpress.com/2012/09/consistency-catalog.png?w=805)](https://highlyscalable.files.wordpress.com/2012/09/consistency-catalog.png)复制因子4：这里假设读/写协调器（ read/write coordinator）可以是外部客户端或数据库中的代理节点。

让我们对一致性保证技术从弱到强一一介绍：

- （A，anti-entropy(反熵)）最弱的一致性保证，其策略是：Writer选择任意的节点副本更新，如果新数据还没有通过后台的反熵协议传递（更多关于下一节中的反熵协议）到读的那个节点，Reader查看到的依然是旧数据。这种方法的主要特性是：
  - 高传播延迟使其对数据同步非常不切实际，因此它通常仅用作辅助后台进程，用于检测和修复的计划外出现的不一致。但是，像Cassandra这样的数据库在传播有关数据库拓扑和其他元数据的信息时，使用反熵作为主要方式。
  - 一致性保证很差：即使没有故障，写冲突和读写差异也很可能出现。
  - 针对网络分区有出色的可用性和鲁棒性。此模式提供了良好的性能，因为单个更新由异步批处理替换。
  - 较弱的持久性保证。因为新数据最初存储在单个副本节点上。
- （B）对先前模式的有一个明显改进：一旦任一副本收到更新请求，就会异步地向所有（可用）副本发送更新，它可以被认为是一种有针对性的反熵。
  - 与纯反熵相比，这大大提高了一致性并且性能损失相对较小。 但是，相比强一致性和持久性保证保还有一定差距。
  - 如果由于网络故障或节点故障/替换而使得一些副本变得暂时不可用，则副本最终应通过反熵过程传递更新。
- （C）在前面的模式中，使用提示移交技术（hinted handoff technique）可以更好地处理故障[8]。 针对不可用节点的更新将记录在读/写协调器或任何其他节点上，并提示一旦原节点可用，就应将其传递回去。 这改善了持久性保证和副本收敛时间。
- （D，ReadOne-WriteOne）在延迟更新传播之前，提示移交（hinted handoff）的载体可能就会失效，因此必须通过所谓的读取修复来保障一致性。 每次读取（或随机选择的读取）都会触发异步进程，该异步进程向所有副本节点请求一份摘要（一种签名/哈希），并在检测到不一致性时，使之统一各个节点的数据摘要。 我们使用术语ReadOne-WriteOne来命名结合了A、B、C、D的技术，它们都不提供严格的一致性保证但足够有效，可以在实践中作为一种独立的方法使用。
- （E，Read Quorum Write Quorum）上面的策略是可以减少副本的收敛时间的启发式增强功能（译者注：heuristic  enhancements,基于直观或经验构造的增强方法，如神经网络就是一种启发式的算法）。为了保障更强的一致性（甚至超越最终一致性），必须牺牲可用性来保证读写集之间的重叠。常见做法是同步写入W副本而不是一个，然后在读取过程中同时读取R个副本而不是一个。
  - 首先，通过配置W并保障其大于1来确定持久性。
  - 其次，由于R + W> N，同步写入的集合将与读取期间接触的集合重叠（在上图中W = 2，R = 3，N = 4），因此读者将至少读到一个新的副本并选择它作为结果。如果依次执行读写请求（例如，对单个用户的写完再读），则可以保证一致性，但不保证全局读后读取一致性。根据下图中的示例来理解为什么读取可能不一致，图中虽然R = 2，W = 2，N = 3（R + W >N），但写操作对两个副本的更新是非事务性的，因此客户端可以在写操作未完成时读取，可能读到一新一旧或两旧值：
[![consistency-concurrent-quorum](https://highlyscalable.files.wordpress.com/2012/09/consistency-concurrent-quorum.png?w=805)](https://highlyscalable.files.wordpress.com/2012/09/consistency-concurrent-quorum.png)

-  - 对不同的读延迟和写持久性的需要，可以通过改变R和W的值调整。
  - 如果W <= N / 2，同时写入可以写到不相交的若干个节点（如，写操作A写前N/2个，B写后N/2个）。而设置W> N / 2则可确保使用回滚模型在原子级读改写时及时检测到冲突。
  - 严格来说，这种模式尽管它可以容忍单独节点的故障，但对网络分区容错性不好。 实际上，像sloppy quorum [8]这样的方法使用标准的Quorum机制通过牺牲一致性以提高在某些情况下的可用性。

- （F，Read All Write Quorum）对于读一致性问题可以通过在读取期间联系所有副本（readers可以获取数据或检查摘要）得以减轻。在出现了新的版本数据时，它确保了马上就可以在至少一个节点上看到最新数据。但是在网络分区这种情况下，这种保证可能就不起作用了。

- （G，Master-Slave）上述的技术通常用于提供原子写入或冲突检测一致性的读改写的这类级别。要实现冲突预防级别，必须使用一种集中管理或用锁。最简单的策略是使用主从异步复制。所有需要写入的特定数据项都被路由到中央节点，然后在中心节点按顺序执行，但这会使得主节点成为瓶颈，因此必须将数据划分为独立的分片，从而实现可扩展性。

- （H，Transactional Read Quorum Write Quorum and Read One Write All）Quorum机制也可以通过事务控制技术来避免写冲突。众所周知的方法是使用两阶段提交协议。但两阶段提交并不完全可靠，因为协调器（coordinator）发生故障会导致资源阻塞。 PAXOS提交协议[20,21]是一种更可靠的替代方案，但会损失一点性能。在此基础上，我们最终得到了Read One Write All的方法，即把所有副本的更新放在一个事务中，这种方法提供了强容错一致性但会损失掉一些性能和可用性。

上面分析中的一些权衡有必要再强调一下：

- **一致-可用性权衡**。这种严格的权衡取决于CAP定理。在网络分区的情况下，数据库要么只留下一个，要么接受数据冲突的可能性。
- **一致-可扩展性权衡**。可以看出，即使读写一致性的保证严重限制了副本集可扩展性，但在的原子级写入的模式中以相对可扩展的方式，写冲突依然是可以解决。原子读改写模型通过给数据加上临时性的全局锁来避免冲突。这表明， *数据或操作之间的依赖，即使是很小范围内或很短时间的，也会损害扩展性*。所以精心设计数据模型，[careful data modeling](https://highlyscalable.wordpress.com/2012/03/01/nosql-data-modeling-techniques/)将数据分片分开存放对于扩展性非常重要。
- **一致性延迟权衡**。如上所示，为保障数据库提供强一致性或持久性，如今存在使用Read-All和Write-All技术的趋势。但这些保证显然与请求延迟成反比。Quorum技术则是一个折中方法。
- **故障转移一致性/可扩展性/延迟权衡**。有趣的是故障转移与一致性/可伸缩性/延迟之间的取舍冲突并不十分严重。在合理的性能/一致性损失下，通常可以容忍高达N/2个节点的故障。但是，这种权衡是明显可见的，例如两阶段提交和PAXOS协议之间的差异就体现的很明显，而另一个例子是在提升某些一致性保证的能力，例如使用粘性会话进行读写操作，这会使故障转移变得复杂[22]

###反熵协议，Gossip算法

现在我们开始研究一下几个问题？：

*这里有一组节点，每个数据项都复制到一个节点子集中。每个节点也会提供更新请求 即使它没有与其他节点的网络连接。每个节点周期性地将其状态与其他节点同步，如果长时间不进行更新，则所有副本将逐渐变得一致。 这种同步过程是怎样的？当触发同步时，如何选择同步对象？数据交换协议是什么？让我们假设两个节点总是通过选择最新版本或保留两个版本以进一步应用程序端解析的方式来合并他们的数据版本。*

这个问题常见于数据一致性维护以及集群状态同步（如集群成员信息传播等）类似情况。尽管可以通过全局协调器（global coordinator）提供的数据库监控器和创建全局同步计划来解决这类问题，但分布式数据库可以提供更好的容错率。其主要思想是利用精心设计的传染协议（well-studied epidemic protocols [7]）。这种协议虽然相对简单，但具有很好的收敛时间，并几乎能够容忍任何节点的失效和网络分区。尽管有许多类型的传染算法，我们只关注反熵协议，因为NoSQL数据库都在使用它。

反熵协议假设同步是由固定的调度执行的 - 每个节点定期随机的选择其他节点以一些规则交换数据库内容，消除差异。这里有三种反熵协议：push,pull,和push-pull。其中push一些是随机选择一个节点然后把当前数据发送出去。实际上，把整个数据库都push输出是很笨的方法，因此节点通常按照下图所示协议工资：

[![gossips](https://highlyscalable.files.wordpress.com/2012/09/gossips.png?w=805)](https://highlyscalable.files.wordpress.com/2012/09/gossips.png)

节点A是同步发起点，它准备了一段类似于数据指纹的数据摘要（一组校验和）。节点B接受这段数据摘要，并与B的本地数据进行对比然后将不同的地方发回给A。最后A发送一个更新给B，B再更新自己的数据。Pull和Push-pull协议就与此类似，就如上图所示。

反熵协议提供了足够好的收敛时间和可扩展性。下图显示了在100个节点的集群中传播更新的模拟结果。 在每次迭代中，每个节点联系一个随机选择的节点。

[![epidemic-dynamics](https://highlyscalable.files.wordpress.com/2012/09/epidemic-dynamics.png?w=805)](https://highlyscalable.files.wordpress.com/2012/09/epidemic-dynamics.png)

我们发现 pull方式比push方式收敛效果更好，这点我们可以从理论上证明[7]。同时，push有一个“收敛尾巴”的问题，即经过多次迭代后大部分节点都涉及收敛依然有小部分节点不受影响。相比于原始的push方法或pull方法,Push-Pull方法可以极大的提高效率，因此它经常在实际中使用。因为反熵的平均时间是随着集群的规模以对数函数的形式增长的，因而他是可扩展的。

尽管这些技术看起来很简单，但是很多研究[5]是关于不同约束条件下反熵协议的性能。其中一个是利用网络拓扑的知识来通过更有效的模式替换随机节点的选择[10]；在网络带宽有限的条件下，调整传输速率或使用先进的规则选择要同步的数据[9]；摘要的计算也是一个难点，因此数据库可以只维护最近更新的日志以促进摘要计算。

### 最终一致性的数据类型

在上一节，我们假设 *两个节点总能合并他们的数据*。但要解决更新冲突并不是一个简单的任务，要想使得所有副本收敛到一个语义正确的值是难以想象的困难。已删除的项目可以在 Amazon Dynamo数据库中重现出现就是一个很著名的例子[8]。

让我们看一个简单的例子来说明这个问题：一个数据库维护一个逻辑上的全局计数器，每个数据库节点都可以进行增加/减少技术操作。尽管每个节点可以维护他们自己的本地计数器并使用单个标量值表示（其状态），但是这些本地计数器的标量值不能简单的进行加减合并。假设这样一个例子：这里有3个节点A,B,C每个节点执行一次加操作。如果A从B中pull得一个值，并把其添加到本地副本，C从B中pull得一个值，在从A中pull得一个值，然后C以标量值4结束，这是不正确的。解决这个问题有一种可能的方法，就是通过使用类似于向量时钟这样的数据结构[19]来维护每个节点的一对计数器[1]：????


```
`class` `Counter {``   ``int``[] plus``   ``int``[] minus``   ``int` `NODE_ID` `   ``increment() {``      ``plus[NODE_ID]++``   ``}` `   ``decrement() {``      ``minus[NODE_ID]++``   ``}` `   ``get() {``      ``return` `sum(plus) – sum(minus)``   ``}` `   ``merge(Counter other) {``      ``for` `i in ``1``..MAX_ID {``         ``plus[i] = max(plus[i], other.plus[i])``         ``minus[i] = max(minus[i], other.minus[i])``      ``}``   ``}``}`
```

Cassandra 使用了类似的方法[11]。他可能设计了一种更加复杂的最终一致性的数据结构，其利用基于状态或基于操作的副本原则。例如，[1]中就提及了一系列这样的数据结构，包含：

 - 计数器（加减操作）
 - 集合（添加移除操作）
 - 图（增加边/点，移除边/点的操作）
 - 列表（插入某位置，移除某位置的操作）

然而，最终一致性的数据类型的功能是有限的，并且还会带来额外开销

## 数据放置

这一节我们关注在分布式数据库下的数据放置的有关算法。这些算法负责把数据项映射到相应的物理节点，把数据从一个节点迁移到另一个节点以及整个数据库中RAM等资源的全局分配的问题。

### Rebalancing
### 均衡数据
让我们从一个简单的协议开始，这个协议是关于提供群集节点之间的无中断数据迁移的。这个任务会出现在如集群扩展（新节点的加入），故障转移（一些节点宕机），或负载均衡（数据在节点中的分布变得不均衡）等场景。正如下图A中的情形-有三个节点，数据随机分布在三个节点上（我们假设数据都仅仅是key-value模型，并不包含其他的很多信息）：

[![rebalancing](https://highlyscalable.files.wordpress.com/2012/09/rebalancing.png?w=805)](https://highlyscalable.files.wordpress.com/2012/09/rebalancing.png)

如果有一个没有内部支持数据均衡的数据库，他部署很多数据库实例在每个节点上，如B图所示。这需要手动进行集群扩展，停掉要迁移的数据库实例，把它转移到新节点上，再在新节点上启动，如图C所示。尽管一个自动化的数据库能够监控到每条记录，许多系统包括MongoDB,Oracle Coherence, and 即将到来的Redis Cluster仍然使用的是自动均衡技术，即，将数据分片作为最小迁移单元以提高效率。很明显，与提供均匀负载分布的节点数相比，许多分片数量应该非常大。根据在迁移分片期间将客户端从导出节点重定向到导入节点这个简单协议所说，他可以用来来完成无中断分片迁移。 下图描述了将在Redis Cluster中实现的get（key）逻辑的状态机：

[![redis-rebalancing-protocol](https://highlyscalable.files.wordpress.com/2012/09/redis-rebalancing-protocol.png?w=805)](https://highlyscalable.files.wordpress.com/2012/09/redis-rebalancing-protocol.png)

假设每个节点都知道集群的拓扑结构并能够讲任何key映射到相应的分片，也能把数据分片映射到节点。如果节点判断请求key属于一个本地分片，然后在本地查找（上图中的上方方框）。如果节点确定所请求的key属于另一个节点X，它向客户端发送永久重定向命令（上图中的下方方框）。 永久重定向表明客户端能够缓存分片和节点之间的映射。 如果正在进行分片迁移，则导出节点和导入节点会相应地标记此分片并开始移动其记录，分别锁定每个记录。 导出节点首先在本地查找key，如果未找到，则将客户端重定向到导入节点，假设key已经迁移。 此重定向是一次性的，不应缓存。 导入节点在本地处理重定向，但常规查询将永久重定向，直到迁移未完成。

### 动态环境中的分片和复制

我们需要解决的下一个问题就是如何将记录映射到物理节点。最直接的方法就是有一个key的范围表，在这个表中每个范围都被赋值给一个节点或使用类似于 *NodeID = hash(key) % TotalNodes* 的方法。但是，基于哈希取模的方法并不能明确的解决群集重新配置的问题，因为增加或移除的节点会使得整个集群中的全部数据彻底重排，继而导致复制和故障转移变得很难进行。

有很多种方法可以从复制和故障转移的角度进行增强。其中最著名的方法是consistent hashing算法。网上有很多关于consistent hashing算法的描述，因此我提供了一种最基础的描述。下图描述的就是 consistent hashing的基本思想：

[![consistent-hashing](https://highlyscalable.files.wordpress.com/2012/09/consistent-hashing.png?w=805)](https://highlyscalable.files.wordpress.com/2012/09/consistent-hashing.png)

consistent hashing是基于键值存储的映射模式 - 他将键值（通常使用哈希键值）映射到物理节点。哈希键值的空间是一片有序固定的二进制字符串空间，因此很明显每个键范围都分配给某个节点，如图(A)所示，3个节点A、B、C。为了副本复制，将键值空间闭合成一个环，然后沿环顺时针前进直到所有副本映射到空间中，如图(B)所示。换句话说，Y应该防止在节点B因为他的键值对应于B的范围，第一个副本应该放置在C，第二个副本应该放在A。

这种结构的好处在于添加和移除节点十分方便，因为他只会引起相邻区域的数据均衡。如图(C)所示，添加节点D只会影响X而对Y无影响。类似地，移除节点B（节点B发生故障）只会影响Y以及X中的副本。但是，正如[8]中指出的一样，这种做法在带来好处的同时也有一定的坏处，他可能容易过载 - 所有的负载都进行的均衡都由相邻区域处理，这使得他们复制大量数据。这个问题可以通过映射到一组范围而非一个范围来缓解，如图(D)所示。这种一种权衡 - 他做到了均衡负载，但相比于模块映射，这种映射使得总均衡数量适当降低。

对于相对小一点的数据库集群就不会有问题，研究如何在对等网络中将数据放置与网络路由结合起来很有意思。一个比较好的例子是Chord算法，它使环的完整性让步于单个节点的查找效率。Chord算法也使用了环映射键到节点的理念，在这方面和一致性hash很相似。不同的是，一个特定节点维护一个短列表，列表中的节点在环上的逻辑位置是指数增长的（如下图）。这使得可以使用二分搜索只需要几次网络跳跃就可以定位一个键。
在非常大的部署中，维持哈希环的完整和连贯是很难的。尽管这并不是数据库的典型问题，因为对于相对较小的集群，研究如何在数据放置与在对等网络中网络路由想结合是很有趣的。Chord算法就是一个很好的例子，它牺牲环的完整性而提高单个节点的查找效率。Chord算法也使用了环映射键值到节点的概念，在这方面和一致性哈希很相似。不同的是，一个特定节点维护一个含有其他节点的短列表，列表中的节点在环上的逻辑位置是指数增长的（如下图）。我们可以使用二分搜索只需要几次网络跳跃就可以定位一个键：

[![chord](https://highlyscalable.files.wordpress.com/2012/09/chord.png?w=416&h=300)](https://highlyscalable.files.wordpress.com/2012/09/chord.png)

这张图描述了由16个节点组成的集群，并阐明了节点A如何寻找到一个放在节点D上的键值。(A)描述了路由，(B)描述了关于节点A、B、C节点的局部图像。更多关于在分布式系统下数据复制的信息可以在[15]中找到。

### 多属性数据分片

当只需要通过主键访问数据项时，一致性哈希提供了一个高效的数据放置策略，但是当需要根据多个属性查询的时候，事情就变得复杂的多。最直接的方法（比如MongoDB所使用的方法）就是不考虑其他属性直接用主键来分布数据。但这样做的后果是依据主键的查询可以被路由到一些限制的节点上，而其他的查询就需要遍历集群的所有节点。这种在查询效率上的不平衡会造成一下几个问题：

*假设有一个数据集，其中的每条数据都有若干的属性和相应的值。那么有没有一种数据分布数据的策略，依靠有限的一些节点就可以查询前述属性的任意子集？*

HyperDex数据库提供了一种解决方案。其基本思想是把每一个属性都作为多维空间中的一个坐标轴，然后将空间中的区域映射到物理节点上。查询就是空间中的一个超平面，他与空间中的子集区域相交，因此这片相交的子集区域就应该与查询相关。来思考一下下面的例子[6]：

[![hyperspace-sharding](https://highlyscalable.files.wordpress.com/2012/09/hyperspace-sharding.png?w=366&h=296)](https://highlyscalable.files.wordpress.com/2012/09/hyperspace-sharding.png)

每个数据项都是一个用户信息，有三个属性First Name, Last Name, and Phone Number。这些属性建立了一个三维空间，一种可能的数据放置策略就是把每个象限映射到一个物理节点。像“First Name = John”这样的查询就对应于与四个象限相交的平面，因此只有四个节点会涉及这次查询。而有两个限制的查询相当于空间中的一条直线，这个查询会与两个象限相交（如上图所示），因此只有两个节点涉及这次查询。

这种方法的有一个缺陷：维度空间会随着属性数量指数增长。在一次查询的限制条件（属性）只有寥寥几个的时候，就会涉及到太多的空间区域，相应的会涉及到多台服务器。缓解这种缺陷的一个方法是将一个具有多个属性的数据项拆分为多个子项，并将这些子项映射到几个独立子空间而不是一个大的超空间：

[![hyperspace-sharding-2](https://highlyscalable.files.wordpress.com/2012/09/hyperspace-sharding-2.png?w=805)](https://highlyscalable.files.wordpress.com/2012/09/hyperspace-sharding-2.png)

这样提供了更“细”的查询到节点的映射，但同时也增加了集群协调的复杂性，因为一个数据项将会分布在若干个独立子空间中，而每个独立子空间又有自己的物理节点。数据更新时就必须考虑事物问题。更多关于这个技术以及实施细节请看[6]。

### 钝化副本

那些具有很强随机读取要求的APP需要将所有数据存入内存。对于这种情况，独立的主从副本（如MongoDB）进行数据分片，通常需要至少两倍的内存空间，因为每个数据都要在主节点和从节点上各有一份。为了在主节点失效时起到代替作用，从节点必须拥有和主节点一样的内存大小。然而假设系统能够容忍短时间的中断或性能下降，也可以不要分片，从而减少内存需求。

下图描述了4个节点上的16个分片，每个分片都有一份存在内存，副本存在硬盘上：

[![replica-passivation](https://highlyscalable.files.wordpress.com/2012/09/replica-passivation.png?w=805)](https://highlyscalable.files.wordpress.com/2012/09/replica-passivation.png)

灰色的箭头突出了节点2上的分片副本。其他节点上的分片也是这样复制的。红色箭头表示当节点二故障后钝化副本如何载入内存。集群内副本的均匀分布使得只留一小部分内存就可以存放在节点失效下激活的副本。如上图，集群只需要预留1/3的内存就可以承受单个节点的失效。值得注意的是，值得注意的是，激活副本（从磁盘加载到RAM）需要一些时间；另外，故障恢复期间激活副本会导致性能暂时性能下降或相应数据中断。

## 系统协调

这一小节我们讨论关于系统协调的一些技术。分布式协调是极大的一块领域，在过去的数十年里很多人对他进行深入研究。而在这篇文章中我们只考虑一些实用的技术。更多与分布锁，一致性协议以及其他基础技术相关的综述可以在相关网站和书上找到[17,18,21]。

### 故障检测

故障检测在任何有容错性的分布式系统中都是基本组成部分。实际上，所有的故障检测协议都是基于心跳通讯机制（a heartbeat mssages），这个机制很简单，被监控的组件定期发送心跳新给监控进程（或监控进程轮询被监控的组件），当有一段时间没有收到心跳信息就认为（被监控的组件）失效了。然而，真正的分布式系统还应该有一些其他的功能要求：

- 自适应性。对于在集群拓扑、负载或带宽中的动态变化，临时网络故障或延迟，故障检测应该有极强的鲁棒性。这是一个很大的难题，因为我们没有办法分别一个长时间没有响应的进程到底是不是真的失效了[13]。因此，故障检测总需要在故障检测时间（即多长时间被认为是真的故障）和虚假警报率之间做权衡。在这场权衡中，参数应当是自动地动态变化。
- 灵活性。乍一看，故障检测只需要输出一个表明被监控进程是否处于工作状态的布尔值即可，但在实际应用中这是不够的。我们来看看[12]中的一个类似于Hadoop MapReduce的例子。有一个分布式的应用程序，他由一个主节点和若干个工作节点组成。主节点有一系列的工作任务，他把这些工作分配给这些工作节点。主节点可以分辨不同“故障等级”。首先，如果一个主节点怀疑某个工作节点挂了，他将停止给这个工作节点提交新的任务。然后，随着时间的流逝依然没有心跳信息，主节点将在这个工作节点上处理的任务重新提交给其他的工作节点。最后，当主节点完全确信这个节点挂了后，他将释放相应的所有资源。
- 可扩展性和鲁棒性。作为系统进程，故障检测应随着系统的扩大而扩展。他也应该具有鲁棒性和一致性，比如：即使在发生通讯故障的情况下，系统中的所有节点也应该做出一致性的判断，（判断出哪些节点依然在运行，哪些节点是发生了故障）

所谓的累计失效检测器（Phi Accrual Failure Detector）[12]可以解决前两个问题，Cassandra对他进行了一些修改并应用在自己的产品中[16]。其基本的工作流程如下（见下图）：

- 对于每个被监控的资源，监视器记录心跳信息到达的时间Ti
- 在统计估计（Statistics Estimation）区间中，不断计算最近到达时间（即大小为W的滑动窗口）的均值和方差。
- 假设到达时间的分布已知（下图包含正态分布公式），我们可以计算当前心跳延迟的概率分布（心跳延迟：当前时间t_now和上次到达时间Tc之间的差值）。用这个概率来计算发生故障的置信度。正如[12]所建议的，为了使用方便，心跳延迟的概率可以通过对数方程重新调整。输出1意味着判断错误（认为节点失效）的概率是10%，2意味着1%，以此类推。

[![phi-accrual-failure-detector](https://highlyscalable.files.wordpress.com/2012/09/phi-accrual-failure-detector.png?w=805)](https://highlyscalable.files.wordpress.com/2012/09/phi-accrual-failure-detector.png)

可扩展性的需求可以通过对监视区域按照重要程度分层来解决，区域分层既可以有效阻止心跳信息在网络中泛滥[14]还可以通过谣言传播协议（gossip protocol）或者中央容错库（central fault-tolerant repository）同步。如下图所示（6个故障检测器组成了两个区域，他们通过谣言传播协议或者像ZooKeeper这样的鲁棒性库来联系）：

[![monitoring-zones](https://highlyscalable.files.wordpress.com/2012/09/monitoring-zones.png?w=336&h=293)](https://highlyscalable.files.wordpress.com/2012/09/monitoring-zones.png)

### 协调者竞选

协调者竞选是保证数据库强一致性的主要技术手段。。首先，它可以组织主从结构的系统中主节点的故障转移（failover）。其次，在网络分区的情况下，它可以断开处于少数的那部分节点，以避免写冲突。

Bully算法是一个相对简单的协调者竞争算法。 MongoDB使用了这个算法来选择副本集的主副本（leaders）。Bully算法主要思想是集群的每个成员都可以宣布自己是协调者并告诉其他节点。其他节点可以选择接受这个宣布或拒绝并加入竞争，最终没有竞争者的节点将称为协调者。这些节点使用一些属性来决定谁胜谁负。这些属性可以是静态ID也可以是某个度量如最后一次事物ID（最新的节点会胜出）。

下图就是一个bully算法的例子。其中静态ID作为相对度量，ID值大的会胜出：

1. 最初，集群中五个节点，其中节点5是全局接受的协调器。
2. 让我们假设节点5挂了，节点2和3同时监测到这一情况 。这两个节点开始竞争并发送竞争消息给ID更大的节点。
3. 节点4将节点2和3都淘汰了，而节点3将节点2淘汰了。
4. 假设这时节点1页监测到节点5故障，然后发送竞争小心给其他ID更大的节点。
5. 节点2，3和4都淘汰了节点1
6. 节点4也发送了竞争消息给节点5
7. 节点5没有响应，因此节点4宣布他自己成为了协调者并把这个消息告诉其他节点。

[![bully-algorithm](https://highlyscalable.files.wordpress.com/2012/09/bully-algorithm.png?w=805)](https://highlyscalable.files.wordpress.com/2012/09/bully-algorithm.png)

协调器选举过程可以计算参与其中的多个节点，并检查至少有一半的节点是否参加。这保证了在网络分区的情况下只有一个分区可以选择协调器。


## 参考文献

1. [M. Shapiro et al. A Comprehensive Study of Convergent and Commutative Replicated Data Types](http://hal.inria.fr/docs/00/55/55/88/PDF/techreport.pdf)
2. [I. Stoica et al. Chord: A Scalable Peer-to-peer  Lookup Service  for Internet Applications](http://pdos.csail.mit.edu/papers/chord:sigcomm01/chord_sigcomm.pdf)
3. [R. J. Honicky, E.L.Miller. Replication Under Scalable Hashing: A Family of Algorithms for Scalable Decentralized Data Distribution](http://www.ssrc.ucsc.edu/Papers/honicky-ipdps04.pdf)
4. [G. Shah. Distributed Data Structures for Peer-to-Peer Systems](http://cs-www.cs.yale.edu/homes/shah/pubs/thesis.pdf)
5. [A. Montresor, Gossip Protocols for Large-Scale Distributed Systems](http://sbrc2010.inf.ufrgs.br/resources/presentations/tutorial/tutorial-montresor.pdf)
6. [R. Escriva, B. Wong, E.G. Sirer. HyperDex: A Distributed, Searchable Key-Value Store](http://hyperdex.org/papers/hyperdex.pdf)
7. [A. Demers et al. Epidemic Algorithms for Replicated Database Maintenance](http://net.pku.edu.cn/~course/cs501/2009/reading/1987-SPDC-Epidemic%20algorithms%20for%20replicated%20database%20maintenance.pdf)
8. [G. DeCandia, et al. Dynamo: Amazon’s Highly Available Key-value Store](http://www.read.seas.harvard.edu/~kohler/class/cs239-w08/decandia07dynamo.pdf)
9. [R. van Resesse et al. Efficient Reconciliation and Flow Control for Anti-Entropy Protocols](http://www.cs.cornell.edu/home/rvr/papers/flowgossip.pdf)
10. [S. Ranganathan et al. Gossip-Style Failure Detection and Distributed Consensus for Scalable Heterogeneous Clusters](http://www.hcs.ufl.edu/pubs/CC2000.pdf)
11. <http://www.slideshare.net/kakugawa/distributed-counters-in-cassandra-cassandra-summit-2010>
12. [N. Hayashibara, X. Defago, R. Yared, T. Katayama.  The Phi Accrual Failure Detector](http://cassandra-shawn.googlecode.com/files/The%20Phi%20Accrual%20Failure%20Detector.pdf)
13. [M.J. Fischer, N.A. Lynch, and M.S. Paterson. Impossibility of Distributed Consensus with One Faulty Process](http://www.cs.mcgill.ca/~carl/impossible.pdf)
14. [N. Hayashibara, A. Cherif, T. Katayama. Failure Detectors for Large-Scale Distributed Systems](http://ddg.jaist.ac.jp/pub/HCK02.pdf)
15. M. Leslie, J. Davies, and T. Huffman. A Comparison Of Replication Strategies for Reliable Decentralised Storage
16. [A. Lakshman, P.Malik. Cassandra – A Decentralized Structured Storage System](http://www.cs.cornell.edu/projects/ladis2009/papers/lakshman-ladis2009.pdf)
17. N. A. Lynch.  Distributed Algorithms
18. G. Tel. Introduction to Distributed Algorithms
19. <http://basho.com/blog/technical/2010/04/05/why-vector-clocks-are-hard/>
20. [L. Lamport. Paxos Made Simple](http://research.microsoft.com/en-us/um/people/lamport/pubs/paxos-simple.pdf)
21. [J. Chase. Distributed Systems, Failures, and Consensus ](http://www.cs.duke.edu/courses/fall07/cps212/consensus.pdf)
22. [W. Vogels. Eventualy Consistent – Revisited](http://www.allthingsdistributed.com/2008/12/eventually_consistent.html)
23. [J. C. Corbett et al. Spanner: Google’s Globally-Distributed Database](http://static.googleusercontent.com/external_content/untrusted_dlcp/research.google.com/en//archive/spanner-osdi2012.pdf)
