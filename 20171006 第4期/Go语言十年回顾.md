### Go语言十年回顾

原文链接：[Go: Ten years and climbing](https://commandcenter.blogspot.jp/2017/09/go-ten-years-and-climbing.html?from=hackcv&hmsr=hackcv.com&utm_medium=hackcv.com&utm_source=hackcv.com)

[September 21, 2017](https://commandcenter.blogspot.com/2017/09/go-ten-years-and-climbing.html)





[![img](https://1.bp.blogspot.com/-aGrvoIrHLeE/WcQu3Y2fE_I/AAAAAAAApsw/0lYHJ9InDUAtARMUhf0kBUKxCrEUkmrVgCLcBGAs/s320/gophers10th.jpg)](https://1.bp.blogspot.com/-aGrvoIrHLeE/WcQu3Y2fE_I/AAAAAAAApsw/0lYHJ9InDUAtARMUhf0kBUKxCrEUkmrVgCLcBGAs/s1600/gophers10th.jpg)

Drawing Copyright ©2017 [Renee French](http://reneefrench.io/)



本周是Go创立十周年。

最初的讨论是在2007年9月20日星期四下午进行的。Robert Griesemer，Rob Pike和Ken Thompson在第二天下午2点在谷歌山区43号楼雅温得会议室举行了有组织的会议。第一封邮件有一些关于设计的信息，这个语言的名称出现在25号。

​	主题：回复：前卫讨论

​	发件人：Rob 'Commander' Pike 

​	日期：2007年9月25号 星期二 3:12PM

​	收件人：Robert Griesemer, Ken Thompson

​	

​	在开车回家的路上，我有一些想法。

 1. 名称

 2. 'go',你可以使用这个名字的原因是因为有一些很棒的属性，非常简短，容易拼写。工具：goc,gol,goa。如果有调试器/解释器，它可以被称为'go'。后缀是.go 

    ......   

​	

(值得一提的是，该语言称为Go，"golang"来自网站地址(go.com已经是Disney网站)但不是该语言的正确名称)



Go项目将其生日确定为2009年11月10日，那天是开源项目发布的时间，最初在code.google.com上，后来迁移到Github上。但就目前而言，两年前的概念和这个语言保持一致，这让我们不仅见证了历史上的一些早期时间，而且是我们能够有更进一步、更为长远的观点。



Go开发的第一个重大惊喜就是收到了这封邮件:

​	主题：Go的gcc前端

​	发件人： Ian Lance Taylor 

​	日期：2008年6月7号 星期六 7:06PM

​	收件人：Robert Griesemer, Rob Pike, Ken Thompson

​	我的一位同事给我展示了http://.../go_lang.html ，似乎是一个很有意思的语言，我为它写了一个gcc前端。当然，它缺少很多功能，但它确实编译了网页上的主要筛选代码。



一个盟友(Ian)和和第二个编译器(gccgo)的到来不仅令我们震惊，而且我们备受鼓舞。 对于锁定规范和库的过程来说，拥有该语言的第二种实现是至关重要的，这有助于保证Go[承诺](https://golang.org/doc/go1compat)的高可移植性。.



尽管Ian的办公室并不远，但是我们在这封信之前并没有见过他，自那之后，他一直是设计和实施该语言及其工具的核心参与者。



Russ Cox也在2008年加入了刚起步的Go团队，并且带来了一些自己的技巧。Russ发现让Go方法具有通用性是正确的，意味着一个函数可以有多个方法，于是就有了[http.HandlerFunc](https://golang.org/pkg/net/http/#HandlerFunc) 的想法，这是我们都意想不到的结果。Russ也推广了更多的想法，比如 [io.Reader](https://golang.org/pkg/io/#Reader) 和 [io.Writer](https://golang.org/pkg/io/#Writer) 接口，他们展示了所有的I/O的结构。



Jini Kim是我们的产品经理，他请来了安全专家Adam Langley帮助我们走向市场。Adam做了很多不广为人知的事情，包括创建原始的[golang.org](https://golang.org/)网页界面和 [构建仪表盘](https://build.golang.org/)，但他最大的贡献在于加密库。起初，对于我们中的一些人而已，它在规模和复杂性方面仿佛不成比例，但它最后应用于如此重要的网络和安全软件，这让它们成为了Go故事的重要组成部分。像 [Cloudflare](https://www.cloudflare.com/)这样的网络基础设施公司在很大程度上得益于Adam为Go做的贡献，而且互联网更适合它，Go也是，我们感谢他。

实际上许多公司很早就开始使用Go ，尤其是初创公司。他们中的一些成为云计算的强者，其中一个创业公司，现在叫作[Docker](https://www.docker.com/), 使用Go并且推动容器的计算, 然后导致了别的成果，比如 [Kubernetes](https://kubernetes.io/). 如今可以说Go是一个容器语言，这是另一个完全不可没有想到的结果。



Go在云计算的领域作用更大。 Donnie Berkholz在2014年三月在 [RedMonk](https://redmonk.com/)  [声称](http://redmonk.com/dberkholz/2014/03/18/go-the-emerging-language-of-cloud-infrastructure/) Go是"云计算基础设施的新兴语言"，差不多同一时间， [Apcera](https://www.apcera.com/) 的Derek Collison表明Go已经是云语言了。那可能并不是事实，但正如 伯克霍尔兹 所 使用的“新兴”一词 暗示的那样，它正在成为事实。



如今，Go以称为云语言，仅仅十年的时间就成为了如此庞大的并且不断发展的语言，甚至这个行业的主宰语言，这对于人们来说梦寐以求的成功。如果你觉得"主宰"这个词用的过了，那就看看中国的互联网吧。在一段时间内，通过 [Google趋势图](https://trends.google.com/trends/explore?q=golang) 向我们展示了Go在中国广泛的应用，这似乎是一个错误，但是每一个参加过中国的Go大会的人都可以证明，这些数据是真实的。Go在中国非常重要。



总的来说，Go十年来的经历有许多里程碑的事情。最令我们惊讶的是我们目前的位置： [保守估计](https://research.swtch.com/gophercount) 表明至少有50万Go程序员，当命名为Go的邮件发送时，Go有50万的gophers的想法听起来很荒谬。然而我们在这，而且在继续增长。



说到gophers，看看 [Renee French](http://reneefrench.io/)对吉祥物Go gopher想法，成为一个很有趣的创作，而且也是世界各地Go程序员的象征。很多最大的Go会议被称作GopherCons，因为他们聚集了来自世界各地的Gopher。



Gopher会议正在发展着。 [第一次会议](https://www.youtube.com/playlist?list=PLE7tQUdRKcyb-k4TMNm2K59-sVlUJumw7) 在三年前，如今在世界各地，也就有很多会议，加上无数小地方的[见面会](https://www.meetup.com/topics/golang/)"。在任何一天，在世界上某个地方某些人聚在一起分享一些想法。



回首Go过去十年的设计和发展，反观Go社区的发展，一切令人惊讶。大量的会议和聚会，Go项目和贡献者的名单不断增加，大量开源存储库托管Go的代码，大量的公司使用Go，这些都是令人惊讶的。



对于我们三个Robert、 Rob和 Ken来说我们想要使我们的编程更为轻松，见证我们的事业的开始对我们来说已经非常开心。

下一个十年会给我们带来些什么呢？



*- Rob Pike, with Robert Griesemer and Ken Thompson*
