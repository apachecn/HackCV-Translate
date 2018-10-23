# Go中的管道模式

原文链接：[Pipeline Patterns in Go](https://medium.com/statuscode/pipeline-patterns-in-go-a37bb3a7e61d?from=hackcv&hmsr=hackcv.com)

## 具有错误处理和取消的管道

### 介绍

在[围棋博客](https://blog.golang.org/pipelines)中由萨米尔Ajmani写的管道模式的优秀文章。在本文中，Golang中的管道模式得到了扩展，改进了错误处理和取消。如果你还没有阅读过该文章，那么你至少应该熟悉*什么是管道？*

### 管道错误处理和取消

Go Blog中引入的管道功能提供了并发性，组合和一些错误处理。但是，它遗漏了管道中经常需要的一些功能。

1. 如果我们的任何管道阶段遇到不可恢复的错误，则应将错误返回给管道的调用者，并且所有关联的goroutine应该快速退出以避免不必要的CPU和内存使用。
2. 管道可能已创建以响应RPC或Web请求。如果此类管道操作非常耗时，我们可能会发现原始请求被取消，因为调用者不想等待足够长的时间。这种取消通常使用上下文库传递给Go函数，我们希望能够以相同的方式取消整个管道。

使用本文中描述的管道模式可以优雅地实现这两个目标。

### 管道的示例

#### 源管道阶段

首先，这是一个示例管道源函数。它类似于`gen`Go Blog 的功能，但它包括错误处理和取消。此函数只是从片中读取字符串并将它们写入Go通道。为了进行错误处理，它认为某些情况是不可恢复的错误。

```go
func lineListSource(ctx context.Context, lines ...string) (
  <-chan string, <-chan error, error) {
  if len(lines) == 0 {
    // Handle an error that occurs before the goroutine begins.
    // 处理在goroutine开始之前发生的错误。
    return nil, nil, errors.Errorf("no lines provided")
  }
  out := make(chan string)
  errc := make(chan error, 1)
  go func() {
    defer close(out)
    defer close(errc)
    for lineIndex, line := range lines {
      if line == "" {
        // Handle an error that occurs during the goroutine.]
        // 处理goroutine期间发生的错误。
        errc <- errors.Errorf("line %v is empty", lineIndex+1)
        return
      }
      // Send the data to the output channel but return early
      // if the context has been cancelled.
      // 将数据发送到输出通道但提前返回
      // 如果上下文已被取消。
      select {
      case out <- line:
      case <-ctx.Done():
        return
      }
    }
  }()
  return out, errc, nil
}
```

这里有一些重要的事情需要注意。

1. 在创建goroutine之前可能会发生错误。在这种情况下，返回的错误是非零的，其他返回的值是nil。
2. 一旦goroutine启动，goroutine内可能会发生错误。在这种情况下，错误被发送到错误通道`errc`并且退出goroutine。由于错误通道的容量正好是1，因此发送单个错误永远不会阻塞。
3. `out`使用`select`上下文的`Done`函数将输出数据发送到通道，以便在取消上下文时，goroutine不会永久阻塞并且可以退出。
4. 创建输出通道`out`和错误通道`errc`，并保证由此功能关闭。这是确保goroutines不泄漏的重要模式。

#### 转变管道阶段

管道转变阶段从前一级获得输入，并将输出写入后续阶段。换句话说，任何管道阶段既不是源头也不是终点。下面的示例函数将每个输入字符串解析为具有特定基数的整数。

```go
func lineParser(ctx context.Context, base int, in <-chan string) (
  <-chan int64, <-chan error, error) {
  if base < 2 {
    // Handle an error that occurs before the goroutine begins.
    return nil, nil, errors.Errorf("invalid base %v", base)
  }
  out := make(chan int64)
  errc := make(chan error, 1)
  go func() {
    defer close(out)
    defer close(errc)
    for line := range in {
      n, err := strconv.ParseInt(line, base, 64)
      if err != nil {
        // Handle an error that occurs during the goroutine.
        errc <- err
        return
      }
      // Send the data to the output channel but return early 
      // if the context has been cancelled.
      select {
      case out <- n:
      case <-ctx.Done():
        return
      }
    }
  }()
  return out, errc, nil
}
```

这几乎与源函数`lineListSource`相同。主要区别在于输入数据来自`in`字符串通道，而不是来自一串字符串。

#### 水槽管道阶段

最后一种管道功能是接收阶段。它消耗先前阶段的输入，但不将输出发送到后续流水线阶段。下面是接收函数的示例，只是将接收的输入打印到stdout。

```go
func sink(ctx context.Context, in <-chan int64) (
  <-chan error, error) {
  errc := make(chan error, 1)
  go func() {
    defer close(errc)
    for n := range in {
      if n >= 100 {
        // Handle an error that occurs during the goroutine.
        errc <- errors.Errorf("number %v is too large", n)
        return
      }
      fmt.Printf("sink: %v\n", n)
    }
  }()
  return errc, nil
}
```

在这个例子中，不存在函数将在goroutine之外返回错误的情况，但函数仍然返回错误值（总是nil）以与其他管道函数保持一致，并允许之后用更复杂的初始化来更改方法的签名。

还应该注意的是，即使我们有这样一个简单的不能产生错误的goroutine，`errc`仍然需要错误通道，因为错误通道的关闭表示goroutine的完成。

#### 建立和执行管道

以下是建立和执行管道的方法。

```go
func runSimplePipeline(base int, lines []string) error {
  ctx, cancelFunc := context.WithCancel(context.Background())
  defer cancelFunc()
  var errcList []<-chan error
  // Source pipeline stage.
  linec, errc, err := lineListSource(ctx, lines...)
  if err != nil {
    return err
  }
  errcList = append(errcList, errc)
  // Transformer pipeline stage.
  numberc, errc, err := lineParser(ctx, base, linec)
  if err != nil {
    return err
  }
  errcList = append(errcList, errc)
  // Sink pipeline stage.
  errc, err = sink(ctx, numberc)
  if err != nil {
    return err
  }
  errcList = append(errcList, errc)
  fmt.Println("Pipeline started. Waiting for pipeline to complete.")
  return WaitForPipeline(errcList...)
}
```

本质上，调用每个管道函数来初始化该特定阶段。这种初始化包括通过Go通道连接组件并启动但不等待goroutine。

一旦初始化了所有管道功能，我们就等待整个管道完成。这是通过将从每个管道函数返回的一片错误通道，传递给我们的管道辅助函数来完成`WaitForPipeline`。

`WaitForPipeline`只是监视每个错误通道，如果在任何错误通道上收到非零错误，它将返回第一个遇到的错误。否则，当所有错误通道都已关闭时，它将返回nil错误。

```go
// WaitForPipeline waits for results from all error channels.
// It returns early on the first error.
func WaitForPipeline(errs ...<-chan error) error {
  errc := MergeErrors(errs...)
  for err := range errc {
    if err != nil {
     return err
    }
  }
  return nil
}
```

`WaitForPipeline`使用另一个管道辅助函数`MergeErrors`，它本身基于Go Blog 的`merge`函数。

```go
// MergeErrors merges multiple channels of errors.
// Based on https://blog.golang.org/pipelines.
func MergeErrors(cs ...<-chan error) <-chan error {  
  var wg sync.WaitGroup
  // We must ensure that the output channel has the capacity to
  // hold as many errors
  // as there are error channels. 
  // This will ensure that it never blocks, even
  // if WaitForPipeline returns early.
  out := make(chan error, len(cs))
  // Start an output goroutine for each input channel in cs.  output
  // copies values from c to out until c is closed, then calls
  // wg.Done.
  output := func(c <-chan error) {
    for n := range c {
      out <- n
    }
    wg.Done()
  }
  wg.Add(len(cs))
  for _, c := range cs {
    go output(c)
  }
  // Start a goroutine to close out once all the output goroutines
  // are done.  This must start after the wg.Add call.
  go func() {
    wg.Wait()
    close(out)
  }()
  return out
}
```

### 如何处理错误

当`runSimplePipeline`运行时，创建一个新的可删除上下文对象，然后调用`defer cancelFunc()`以保证上下文的函数返回前取消。请注意，如果这是一个RPC或Web请求，而不是这个演示函数，我们只使用提供的上下文而不是创建一个新的上下文。

接下来，初始化每个流水线阶段并启动相关的goroutine。如果在任何管道功能的初始化期间发生错误，`runSimplePipeline`将立即返回，导致延迟取消函数的执行，取消上下文。

如果任何goroutine已经启动（即至少一个管道功能已成功初始化），则可以在从输入通道接收或发送到输出通道时阻止它们。由于流水线阶段在退出时关闭其输出通道，因此在输入通道上阻塞的任何后续流水线阶段都将被解锁。如果在发送到输出通道时阻止任何goroutine，因为`select`用于发送到输出通道，它们将在取消上下文时被解除阻塞。

如果所有管道功能都已成功初始化，那么我们就进入了该`WaitForPipeline`功能。无论是否发生错误，当它返回时，再次执行延迟取消函数，取消上下文，并允许管道中的任何goroutine被解除阻塞并退出。

应该注意的是，在发生错误的情况下，一些管道goroutine可能在`runSimplePipeline`返回到调用地方之后继续执行一小段时间。这是因为`WaitForPipeline`如果遇到错误，将立即返回。这通常不是问题，但如果您的管道功能有副作用，您应该考虑此行为的影响。您可以使用一个 `sync.WaitGroup`来保证在`runSimplePipeline`返回之前终止管道goroutine 。

### 如何处理取消

如果在管道之外取消上下文，可能是因为取消了RPC或Web请求，则取消将传播到所有管道函数。源阶段应该快速识别取消的上下文，关闭其输出和错误通道，然后返回。如果输出通道在其输入通道上被阻塞，则输出通道的关闭将解锁后续阶段。这将持续到所有管道阶段退出为止。假设没有发生错误，`WaitForPipeline`将返回nil错误。最后，在`runSimplePipeline`返回之前，它将继续运行`cancelFunc`。与两次关闭通道（导致恐慌）不同，调用取消函数可以安全地执行多次，后续调用将是不做操作。

如你所见，无论是由于错误还是取消了上下文，终止过程几乎相同。事实上，如果在取消上下文*后* goroutine中发生错误，它将被正确报告`WaitForPipeline`。这允许管道阶段确定是否应将取消的上下文报告为错误。

### 复杂的管道

到目前为止，我们的示例管道非常简单，包括源，转换器和接收器。我们不仅可以在源和接收器之间添加任意数量的变换器，而且我们的管道可以像Go Blog中所描述的那样扇出（fan-out）和扇入（fan-in）。

这是一个扇出管道转换器，它将输入复制并发送到两个输出通道。

```go
func splitter(ctx context.Context, in <-chan int64) (
    <-chan int64, <-chan int64, <-chan error, error) {
  out1 := make(chan int64)
  out2 := make(chan int64)
  errc := make(chan error, 1)
  go func() {
    defer close(out1)
    defer close(out2)
    defer close(errc)
    for n := range in {
      // Send the data to the output channel 1 but return early 
      // if the context has been cancelled.
      select {
      case out1 <- n:
      case <-ctx.Done():
        return
      }
      // Send the data to the output channel 2 but return early 
      // if the context has been cancelled.
      select {
      case out2 <- n:
      case <-ctx.Done():
        return
      }
    }
  }()
  return out1, out2, errc, nil
}
```

使用它的更复杂的管道如下图所示。

```
                                  / squarer -> sink
lines -> lineParser -> splitter -|
                                  \ sink
```

下面的代码创建了这个管道。

```go
func runComplexPipeline(base int, lines []string) error {
  ctx, cancelFunc := context.WithCancel(context.Background())
  defer cancelFunc()
  var errcList []<-chan error
  // Source pipeline stage.
  linec, errc, err := lineListSource(ctx, lines...)
  if err != nil {
    return err
  }
  errcList = append(errcList, errc)
  // Transformer pipeline stage 1.
  numberc, errc, err := lineParser(ctx, base, linec)
  if err != nil {
    return err
  }
  errcList = append(errcList, errc)
  // Transformer pipeline stage 2.
  numberc1, numberc2, errc, err := splitter(ctx, numberc)
  if err != nil {
    return err
  }
  errcList = append(errcList, errc)
  // Transformer pipeline stage 3.
  numberc3, errc, err := squarer(ctx, numberc1)
  if err != nil {
    return err
  }
  errcList = append(errcList, errc)
  // Sink pipeline stage 1.
  errc, err = sink(ctx, numberc3)
  if err != nil {
    return err
  }
  errcList = append(errcList, errc)
  // Sink pipeline stage 2.
  errc, err = sink(ctx, numberc2)
  if err != nil {
   return err
  }
  errcList = append(errcList, errc)
  fmt.Println("Pipeline started. Waiting for pipeline to complete.")
  return WaitForPipeline(errcList...)
}
```

### 下载并运行代码

可以从[此处](https://gist.github.com/claudiofahey/3afcf4f4fb3d8d3b35cadb100d4fb9b7)下载本文中的所有代码。它包括`main`演示各种场景的功能。运行它的输出如下所示。

```bash
$ go run -race pipeline_demo.go 
runSimplePipeline: base=10, lines=[3 2 1]
Pipeline started. Waiting for pipeline to complete.
sink: 3
sink: 2
sink: 1
runSimplePipeline: base=1, lines=[3 2 1]
invalid base 1
runSimplePipeline: base=2, lines=[1010 1100 1000]
Pipeline started. Waiting for pipeline to complete.
sink: 10
sink: 12
sink: 8
runSimplePipeline: base=2, lines=[1010 1100 2000 1111]
Pipeline started. Waiting for pipeline to complete.
sink: 10
sink: 12
strconv.ParseInt: parsing "2000": invalid syntax
runSimplePipeline: base=10, lines=[1 10 100 1000]
Pipeline started. Waiting for pipeline to complete.
sink: 1
sink: 10
number 100 is too large
runComplexPipeline: base=10, lines=[5 4 3]
Pipeline started. Waiting for pipeline to complete.
sink: 25
sink: 5
sink: 4
sink: 3
sink: 16
sink: 9
runPipelineWithTimeout
Pipeline started. Waiting for pipeline to complete.
sink: 86
sink: 86
sink: 92
sink: 40
sink: 4
sink: 54
sink: 30
sink: 64
sink: 11
sink: 76
Cancelling context.
```





### Further Reading

- [Go Concurrency Patterns: Pipelines and cancellation](https://blog.golang.org/pipelines)
- [Go Concurrency Patterns: Context](https://blog.golang.org/context)
- [Errors Package](https://godoc.org/github.com/pkg/errors)