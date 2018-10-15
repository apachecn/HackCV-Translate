# Pipeline Patterns in Go

原文链接：[Pipeline Patterns in Go](https://medium.com/statuscode/pipeline-patterns-in-go-a37bb3a7e61d?from=hackcv&hmsr=hackcv.com)

### Introduction

The [Go Blog](https://blog.golang.org/pipelines) has an excellent article on the pipeline pattern written by Sameer Ajmani. In this article, that pipeline pattern in Golang is extended with improved error-handling and cancellation. If you haven’t already read that article, you should at least be familiar with the section *What is a pipeline?*

### Pipeline Error Handling and Cancellation

The pipeline functionality introduced in the Go Blog provides concurrency, composition, and some error-handling. However, it leaves out a few features often desired from a pipeline.

1. If any of our pipeline stages encounter a non-recoverable error, the error should be returned to the caller of the pipeline and all associated goroutines should quickly exit to avoid unnecessary CPU and memory usage.
2. The pipeline may have been created in response to an RPC or web request. If such pipeline operations are time-consuming, we may find that the original request was cancelled because the caller didn’t want to wait long enough. Such a cancellation is usually communicated to Go functions using the context library and we would like to be able to cancel the entire pipeline in the same way.

These two goals are achieved gracefully using the pipeline pattern described in this article.

### An Example Pipeline

#### Source Pipeline Stage

To start with, here is an example pipeline source function. It is similar to the `gen` function from the Go Blog but it includes error handling and cancellation. This function simply reads strings from a slice and writes them to a Go channel. In order to exercise error handling, it considers certain situations to be unrecoverable errors.

```
func lineListSource(ctx context.Context, lines ...string) (
  <-chan string, <-chan error, error) {
  if len(lines) == 0 {
    // Handle an error that occurs before the goroutine begins.
    return nil, nil, errors.Errorf("no lines provided")
  }
  out := make(chan string)
  errc := make(chan error, 1)
  go func() {
    defer close(out)
    defer close(errc)
    for lineIndex, line := range lines {
      if line == "" {
        // Handle an error that occurs during the goroutine.
        errc <- errors.Errorf("line %v is empty", lineIndex+1)
        return
      }
      // Send the data to the output channel but return early
      // if the context has been cancelled.
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

There are a few important things to notice here.

1. An error may occur before the goroutine is created. In this case, the returned error is non-nil and the other returned values are nil.
2. Once the goroutine starts, an error may occur within the goroutine. In this case, the error is sent to the error channel `errc` and the goroutine exits. Since the capacity of the error channel is exactly one, sending a single error will never block.
3. The output data is sent to the `out` channel using `select` along with the context’s `Done` function so that if the context is cancelled, the goroutine doesn’t block forever and it can exit.
4. The output channel `out` and the error channel `errc` are created and guaranteed to be closed by this function. This is an important pattern to ensure that goroutines do not leak.

#### Transformer Pipeline Stage

A pipeline transformer stage gets input from a previous stage and writes output to a subsequent stage. In other words, it is any pipeline stage that is neither a source nor a sink. The example function below parses each input string into an integer with a specific base.

```
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

This is nearly identical to the source function `lineListSource`. The main difference is that the input data comes from a channel `in` of strings, not from a slice of strings.

#### Sink Pipeline Stage

A final type of pipeline function is the sink stage. It consumes input from prior stages but does not send output to subsequent pipeline stages. The example sink function below simply prints the received input to stdout.

```
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

In this example, there is no case in which the function will return an error outside of the goroutine but the function still returns an error value (always nil) for consistency with other pipeline functions, and to allow for more complex initialization in the future without changing the method’s signature.

It also should be noted that even if we had such a simple goroutine that it could not produce an error, the error channel `errc` would still be required as the closing of the error channel signals the completion of the goroutine.

#### Building and Executing the Pipeline

Here’s how to build and execute the pipeline.

```
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

Essentially, each of the pipeline functions is called to initialize that particular stage. This initialization includes wiring up the component through the Go channels and starting, but not waiting for, the goroutine.

Once all pipeline functions have been initialized, we wait for the entire pipeline to complete. This is done by passing a slice of error channels returned from each pipeline function to our pipeline helper function `WaitForPipeline`.

`WaitForPipeline` simply monitors each of the error channels, and if a non-nil error is received on any of them, it returns the first encountered error. Otherwise, when all error channels have been closed, it returns a nil error.

```
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

`WaitForPipeline` uses another pipeline helper function `MergeErrors` which is itself based on the `merge` function from the Go Blog.

```
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

### How Errors are Handled

When `runSimplePipeline` runs, a new cancellable context object is created and then `defer cancelFunc()` is called to guarantee that the context is cancelled before the function returns. Note that if this were an RPC or web request, as opposed to this demonstration function, we would just use the provided context instead of creating a new one.

Next, each pipeline stage is initialized and the associated goroutine is started. If an error occurs during the initialization of any pipeline function, `runSimplePipeline` will immediately return, causing the deferred cancellation function to execute, cancelling the context.

If any goroutines have already started (i.e. at least one pipeline function successfully initialized), they may be blocked on either receiving from an input channel or sending to an output channel. Since pipeline stages close their output channels when they exit, any subsequent pipeline stages blocked on the input channel will become unblocked. If any goroutines were blocked on sending to an output channel, because `select` is used for sending to output channels, they will become unblocked when the context is cancelled.

If all pipeline functions have been initialized successfully, then we made it to the `WaitForPipeline` function. Regardless of whether an error occurs, when it returns, again the deferred cancellation function executes, cancelling the context, and allowing any goroutines in the pipeline to become unblocked and exit.

It should be noted that in the event of an error, it is possible for some pipeline goroutines to continue to execute for a short period after `runSimplePipeline`returns to the caller. This is because `WaitForPipeline` will return immediately if it encounters an error. This is generally not a problem but you should consider the effect of this behavior if your pipeline function has side effects. You may be able to use a `sync.WaitGroup` to guarantee the termination of pipeline goroutines before `runSimplePipeline` returns.

### How Cancellation is Handled

In the event that the context is cancelled outside of the pipeline, perhaps as a result of a cancelled RPC or web request, the cancellation will propagate to all pipeline functions. The source stage should quickly recognize the cancelled context, close its output and error channels, and return. The closure of the output channel will unblock the subsequent stage if it is blocked on its input channel. This will continue until all pipeline stages have exited. Assuming that no errors have occurred, `WaitForPipeline` will return a nil error. Finally, before `runSimplePipeline` returns, it will run `cancelFunc` which was deferred. Unlike closing a channel twice (resulting in a panic), calling the cancel function is safe to do more than once and subsequent calls will be no-ops.

As you can see, the termination process is nearly identical whether due to an error or a cancelled context. In fact, if an error occurs in a goroutine *after* the context is cancelled, it will be correctly reported by `WaitForPipeline`. This allows a pipeline stage to determine whether a cancelled context should be reported as an error or not.

### Complex Pipelines

So far our example pipeline has been quite simple, consisting of a source, a transformer, and a sink. Not only can we add any number of transformers between the source and sink, but our pipeline can fan-out and fan-in as described in the Go Blog.

Here is a fan-out pipeline transformer that duplicates and sends the inputs to two output channels.

```
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

A more complex pipeline that utilizes it is shown graphically below.

```
                                  / squarer -> sink
lines -> lineParser -> splitter -|
                                  \ sink
```

The code below creates this pipeline.

```
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

### Download and Run the Code

All of the code in this article can be downloaded from [here](https://gist.github.com/claudiofahey/3afcf4f4fb3d8d3b35cadb100d4fb9b7). It includes a `main`function that demonstrates various scenarios. The output of running it is shown below.

```
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