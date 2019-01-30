## 用Go建立神经网络

原文链接：[BUILDING A NEURAL NET FROM SCRATCH IN GO](https://www.datadan.io/building-a-neural-net-from-scratch-in-go/?from=hackcv&hmsr=hackcv.com&utm_medium=hackcv.com&utm_source=hackcv.com)

我很高兴我的新书[Machine Learning with Go](https://www.packtpub.com/big-data-and-business-intelligence/machine-learning-go)现已推出！写这本书让我可以全面了解Go中机器学习的现状，我很高兴看到社区如何成长！

[![img](https://www.datadan.io/content/images/2017/10/book_wide.png)](https://www.packtpub.com/big-data-and-business-intelligence/machine-learning-go)

在书中（包含我自己的启发），我决定用Go从零开始构建一个神经网络。事实证明，这很容易，我认为在这里分享我的小神经网络是一件很棒的事。

下面显示的所有代码和数据都可以在[GitHub](https://github.com/dwhitena/gophernet)上找到。

(如果您有兴趣利用已有的Go包装进行机器学习，请查看[所有优秀的现有软件包](https://github.com/gopherdata/resources/tree/master/tooling)，并观看Chris Benson最近在GolangUK关于Go中深度学习的[演讲](https://youtu.be/CHzMEamGZDA))

## 目标

在Go中完成构建神经网络的任务有多种方法，但我想遵循以下准则：

- **没有cgo** - 我希望我的小神经网络可以很好地编译为静态链接的二进制文件，我还想强调Go中本机可用的数字功能。
- **gonum矩阵输入** - 我希望为我的神经网络提供供应矩阵进行训练，类似于如何为大多数Python机器学习功能提供`numpy`数组。
- **可变数量的节点** - 虽然我只在这里说明一个架构，但我希望我的代码是灵活的，这样我就可以调整每层中的节点数量以用于其他场景。

## 网络架构

我们将在此示例中使用的基本网络体系结构包括输入层，单个隐藏层和输出层：

![img](https://www.datadan.io/content/images/2017/09/B05151_Chapter_08_05-1.png)

这种类型的单层神经网络可能不是很“深”，但它已被证明对绝大多数简单的分类任务非常有用。在我们的例子中，我们将根据着名的鸢尾花数据集训练我们的模型来对[著名的鸢尾花数据集合](https://en.wikipedia.org/wiki/Iris_flower_data_set)进行分类。这应该足以以高精度解决该问题。

网络中的每个**节点**将接收一个或多个输入，将它们线性地组合在一起（使用**权重**和**偏差**），然后应用非线性激活函数。通过优化权重和偏差，通过称为[**反向传播**](https://en.wikipedia.org/wiki/Backpropagation)的过程，我们将能够模仿我们的输入（花的测量）和我们想要预测的（花的种类）之间的关系。然后，我们将能够通过优化网络提供新输入（即，我们将**向前**它们**传播**）以预测相应的输出。

（如果你是神经网络的新手，你也可以查看[这个很棒的介绍](https://ujjwalkarn.me/2016/08/09/quick-intro-neural-networks/)，当然你还可以阅读[机器学习与Go]中的相关部分(https://www.packtpub.com/big-data-and-business-intelligence/machine-learning-go)。）

## 定义有用的函数和类型

在深入反向传播和前向传播之前，让我们定义一些类型，这将有助于我们使用我们的模型：

```go
// neuralNet contains all of the information
// that defines a trained neural network.
type neuralNet struct {
        config  neuralNetConfig
        wHidden *mat.Dense
        bHidden *mat.Dense
        wOut    *mat.Dense
        bOut    *mat.Dense
}

// neuralNetConfig defines our neural network
// architecture and learning parameters.
type neuralNetConfig struct {
        inputNeurons  int
        outputNeurons int
        hiddenNeurons int
        numEpochs     int
        learningRate  float64
}

// newNetwork initializes a new neural network.
func newNetwork(config neuralNetConfig) *neuralNet {
        return &neuralNet{config: config}
}
```

我们还需要定义我们的激活函数及其衍生函数，我们将在反向传播过程中使用它。激活函数有很多，但在这里我们将使用[sigmoid函数](http://mathworld.wolfram.com/SigmoidFunction.html)。该函数具有各种优点，包括概率解释和对其衍生的方便表达。

```go
// sigmoid implements the sigmoid function
// for use in activation functions.
func sigmoid(x float64) float64 {
        return 1.0 / (1.0 + math.Exp(-x))
}

// sigmoidPrime implements the derivative
// of the sigmoid function for backpropagation.
func sigmoidPrime(x float64) float64 {
	return sigmoid(x) * (1.0 - sigmoid(x))
}
```

## 为训练实施反向传播

通过上面的定义，我们可以编写[backpropagation方法](https://en.wikipedia.org/wiki/Backpropagation)的实现来训练或优化我们网络的权重和偏差。反向传播方法包括：

1. 初始化我们的权重和偏差（例如，随机）。
2. 通过神经网络向前馈送训练数据以产生输出。
3. 将输出与正确的输出进行比较以获得错误。
4. 根据错误计算我们的权重和偏差的变化。
5. 通过网络传播更改。
6. 对于给定数量的**批次**重复步骤2-5或直到满足停止标准。

在步骤3-5中，我们将利用[**随机梯度下降**](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)（SGD）来确定我们的权重和偏差的更新。

为了实现这种网络训练，我在`neuralNet`上创建了一个方法，它将用两个矩阵作为输入，`x`和`y`。`x`将是我们数据集的特征（即独立变量），而'y`将代表我们试图预测的内容（即因变量）。我将在本文后面展示一些这样的例子，但是现在，让我们假设它们采用这种形式。

在这个函数中，我们首先随机初始化我们的权重和偏差，然后使用反向传播来优化权重和偏差：

```go
// train trains a neural network using backpropagation.
func (nn *neuralNet) train(x, y *mat.Dense) error {

	// Initialize biases/weights.
	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)

	wHidden := mat.NewDense(nn.config.inputNeurons, nn.config.hiddenNeurons, nil)
	bHidden := mat.NewDense(1, nn.config.hiddenNeurons, nil)
	wOut := mat.NewDense(nn.config.hiddenNeurons, nn.config.outputNeurons, nil)
	bOut := mat.NewDense(1, nn.config.outputNeurons, nil)

	wHiddenRaw := wHidden.RawMatrix().Data
	bHiddenRaw := bHidden.RawMatrix().Data
	wOutRaw := wOut.RawMatrix().Data
	bOutRaw := bOut.RawMatrix().Data

	for _, param := range [][]float64{
		wHiddenRaw,
		bHiddenRaw,
		wOutRaw,
		bOutRaw,
	} {
		for i := range param {
			param[i] = randGen.Float64()
		}
	}

	// Define the output of the neural network.
	output := new(mat.Dense)

	// Use backpropagation to adjust the weights and biases.
	if err := nn.backpropagate(x, y, wHidden, bHidden, wOut, bOut, output); err != nil {
		return err
	}

	// Define our trained neural network.
	nn.wHidden = wHidden
	nn.bHidden = bHidden
	nn.wOut = wOut
	nn.bOut = bOut

	return nil
}
```

反向传播的实际实现如下所示。**注意/警告**，为了表现地清晰和简单，我将在执行反向传播时创建一些矩阵。对于大型数据集，您可能希望对其进行优化以减少内存中的矩阵数。

```go
// backpropagate completes the backpropagation method.
func (nn *neuralNet) backpropagate(x, y, wHidden, bHidden, wOut, bOut, output *mat.Dense) error {

	// Loop over the number of epochs utilizing
	// backpropagation to train our model.
	for i := 0; i < nn.config.numEpochs; i++ {

		// Complete the feed forward process.
		hiddenLayerInput := new(mat.Dense)
		hiddenLayerInput.Mul(x, wHidden)
		addBHidden := func(_, col int, v float64) float64 { return v + bHidden.At(0, col) }
		hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

		hiddenLayerActivations := new(mat.Dense)
		applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
		hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

		outputLayerInput := new(mat.Dense)
		outputLayerInput.Mul(hiddenLayerActivations, wOut)
		addBOut := func(_, col int, v float64) float64 { return v + bOut.At(0, col) }
		outputLayerInput.Apply(addBOut, outputLayerInput)
		output.Apply(applySigmoid, outputLayerInput)

		// Complete the backpropagation.
		networkError := new(mat.Dense)
		networkError.Sub(y, output)

		slopeOutputLayer := new(mat.Dense)
		applySigmoidPrime := func(_, _ int, v float64) float64 { return sigmoidPrime(v) }
		slopeOutputLayer.Apply(applySigmoidPrime, output)
		slopeHiddenLayer := new(mat.Dense)
		slopeHiddenLayer.Apply(applySigmoidPrime, hiddenLayerActivations)

		dOutput := new(mat.Dense)
		dOutput.MulElem(networkError, slopeOutputLayer)
		errorAtHiddenLayer := new(mat.Dense)
		errorAtHiddenLayer.Mul(dOutput, wOut.T())

		dHiddenLayer := new(mat.Dense)
		dHiddenLayer.MulElem(errorAtHiddenLayer, slopeHiddenLayer)

		// Adjust the parameters.
		wOutAdj := new(mat.Dense)
		wOutAdj.Mul(hiddenLayerActivations.T(), dOutput)
		wOutAdj.Scale(nn.config.learningRate, wOutAdj)
		wOut.Add(wOut, wOutAdj)

		bOutAdj, err := sumAlongAxis(0, dOutput)
		if err != nil {
			return err
		}
		bOutAdj.Scale(nn.config.learningRate, bOutAdj)
		bOut.Add(bOut, bOutAdj)

		wHiddenAdj := new(mat.Dense)
		wHiddenAdj.Mul(x.T(), dHiddenLayer)
		wHiddenAdj.Scale(nn.config.learningRate, wHiddenAdj)
		wHidden.Add(wHidden, wHiddenAdj)

		bHiddenAdj, err := sumAlongAxis(0, dHiddenLayer)
		if err != nil {
			return err
		}
		bHiddenAdj.Scale(nn.config.learningRate, bHiddenAdj)
		bHidden.Add(bHidden, bHiddenAdj)
	}

	return nil
}
```

在这里，我们使用了一个辅助函数，它允许我们沿着矩阵的一个维度对值求和，保持其他维度的完整性：

```go
// sumAlongAxis sums a matrix along a particular dimension, 
// preserving the other dimension.
func sumAlongAxis(axis int, m *mat.Dense) (*mat.Dense, error) {

        numRows, numCols := m.Dims()

        var output *mat.Dense

        switch axis {
        case 0:
                data := make([]float64, numCols)
                for i := 0; i < numCols; i++ {
                        col := mat.Col(nil, i, m)
                        data[i] = floats.Sum(col)
                }
                output = mat.NewDense(1, numCols, data)
        case 1:
                data := make([]float64, numRows)
                for i := 0; i < numRows; i++ {
                        row := mat.Row(nil, i, m)
                        data[i] = floats.Sum(row)
                }
                output = mat.NewDense(numRows, 1, data)
        default:
                return nil, errors.New("invalid axis, must be 0 or 1")
        }

        return output, nil
}
```

## 实现前向预测

在训练我们的神经网络之后，我们将要用它来进行预测。为此，我们只需要通过网络向前提供一些给定的“x”值以产生输出。这看起来类似于反向传播的第一部分。除此之外，我们将返回生成的输出。

```go
// predict makes a prediction based on a trained
// neural network.
func (nn *neuralNet) predict(x *mat.Dense) (*mat.Dense, error) {

	// Check to make sure that our neuralNet value
	// represents a trained model.
	if nn.wHidden == nil || nn.wOut == nil {
		return nil, errors.New("the supplied weights are empty")
	}
	if nn.bHidden == nil || nn.bOut == nil {
		return nil, errors.New("the supplied biases are empty")
	}

	// Define the output of the neural network.
	output := new(mat.Dense)

	// Complete the feed forward process.
	hiddenLayerInput := new(mat.Dense)
	hiddenLayerInput.Mul(x, nn.wHidden)
	addBHidden := func(_, col int, v float64) float64 { return v + nn.bHidden.At(0, col) }
	hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

	hiddenLayerActivations := new(mat.Dense)
	applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
	hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

	outputLayerInput := new(mat.Dense)
	outputLayerInput.Mul(hiddenLayerActivations, nn.wOut)
	addBOut := func(_, col int, v float64) float64 { return v + nn.bOut.At(0, col) }
	outputLayerInput.Apply(addBOut, outputLayerInput)
	output.Apply(applySigmoid, outputLayerInput)

	return output, nil
}
```

#＃ 数据

好的，我们现在拥有训练和测试神经网络所需的构建模块。然而，在我们开始试图运行之前，让我们简要地看一下我将用来试验这个神经网络的数据。

我将要使用的数据是流行的[鸢尾花数据集](https://archive.ics.uci.edu/ml/datasets/iris)的略微转换版本。该数据集包括四组虹膜花测量（将成为我们的`x`值）以及鸢尾花种类的相应指示（将成为我们的`y`值）。为了利用我们的神经网络利用这个数据集，我稍微改变了数据集，使得物种值由三个二进制列表示（如果行对应于该物种，则为1，否则为0）。我还在测量中添加了一点随机噪声，试图混淆神经网络（因为这个问题很容易解决）：

```
$ head train.csv
sepal_length,sepal_width,petal_length,petal_width,setosa,virginica,versicolor
0.0833333333333,0.666666666667,0.0,0.0416666666667,1.0,0.0,0.0
0.722222222222,0.458333333333,0.694915254237,0.916666666667,0.0,1.0,0.0
0.666666666667,0.416666666667,0.677966101695,0.666666666667,0.0,0.0,1.0
0.777777777778,0.416666666667,0.830508474576,0.833333333333,0.0,1.0,0.0
0.666666666667,0.458333333333,0.779661016949,0.958333333333,0.0,1.0,0.0
0.388888888889,0.416666666667,0.542372881356,0.458333333333,0.0,0.0,1.0
0.666666666667,0.541666666667,0.796610169492,0.833333333333,0.0,1.0,0.0
0.305555555556,0.583333333333,0.0847457627119,0.125,1.0,0.0,0.0
0.416666666667,0.291666666667,0.525423728814,0.375,0.0,0.0,1.0
```

我还将用于训练和测试的数据（通过80/20比例分割）分别分为`train.csv`和`test.csv`。

## 全部放在一起

让我们把这个神经网络运用起来。为此，我们首先需要读取我们的训练数据，初始化`neuralNet`值，并调用`train()`方法：

```go
package main

import (
        "encoding/csv"
        "errors"
        "fmt"
        "log"
        "math"
        "math/rand"
        "os"
        "strconv"
        "time"

        "gonum.org/v1/gonum/floats"
        "gonum.org/v1/gonum/mat"
)

func main() {

        // Open the training dataset file.
        f, err := os.Open("data/train.csv")
        if err != nil {
                log.Fatal(err)
        }
        defer f.Close()

        // Create a new CSV reader reading from the opened file.
        reader := csv.NewReader(f)
        reader.FieldsPerRecord = 7

        // Read in all of the CSV records
        rawCSVData, err := reader.ReadAll()
        if err != nil {
                log.Fatal(err)
        }

        // inputsData and labelsData will hold all the
        // float values that will eventually be
        // used to form our matrices.
        inputsData := make([]float64, 4*len(rawCSVData))
        labelsData := make([]float64, 3*len(rawCSVData))

        // inputsIndex will track the current index of
        // inputs matrix values.
        var inputsIndex int
        var labelsIndex int

        // Sequentially move the rows into a slice of floats.
        for idx, record := range rawCSVData {

                // Skip the header row.
                if idx == 0 {
                        continue
                }

                // Loop over the float columns.
                for i, val := range record {

                        // Convert the value to a float.
                        parsedVal, err := strconv.ParseFloat(val, 64)
                        if err != nil {
                                log.Fatal(err)
                        }

                        // Add to the labelsData if relevant.
                        if i == 4 || i == 5 || i == 6 {
                                labelsData[labelsIndex] = parsedVal
                                labelsIndex++
                                continue
                        }

                        // Add the float value to the slice of floats.
                        inputsData[inputsIndex] = parsedVal
                        inputsIndex++
                }
        }

        // Form the matrices.
        inputs := mat.NewDense(len(rawCSVData), 4, inputsData)
        labels := mat.NewDense(len(rawCSVData), 3, labelsData)

        // Define our network architecture and
        // learning parameters.
        config := neuralNetConfig{
                inputNeurons:  4,
                outputNeurons: 3,
                hiddenNeurons: 3,
                numEpochs:     5000,
                learningRate:  0.3,
        }

        // Train the neural network.
        network := newNetwork(config)
        if err := network.train(inputs, labels); err != nil {
                log.Fatal(err)
        }

        ...

        // Testing discussed below

        ...

}
```

这给了我们一个已经训练好的神经网络。然后我们可以将测试数据解析为矩阵`testInputs`和`testLabels`（我将略过这些细节，因为它们与上面相同），使用我们的`predict()`方法来预测花种，以及将预测情况与实际物种进行比较。计算预测和准确性如下所示：

```go
func main() {

        ...

        // Training as shown above.

        ...

        // Parsing the test data into testInputs and testLabels.

        ...

        // Make the predictions using the trained model.
        predictions, err := network.predict(testInputs)
        if err != nil {
                log.Fatal(err)
        }

        // Calculate the accuracy of our model.
        var truePosNeg int
        numPreds, _ := predictions.Dims()
        for i := 0; i < numPreds; i++ {

                // Get the label.
                labelRow := mat.Row(nil, i, testLabels)
                var species int
                for idx, label := range labelRow {
                        if label == 1.0 {
                                species = idx
                                break
                        }
                }

                // Accumulate the true positive/negative count.
                if predictions.At(i, species) == floats.Max(mat.Row(nil, i, predictions)) {
                        truePosNeg++
                }
        }

        // Calculate the accuracy (subset accuracy).
        accuracy := float64(truePosNeg) / float64(numPreds)

        // Output the Accuracy value to standard out.
        fmt.Printf("\nAccuracy = %0.2f\n\n", accuracy)
}
```

## 结果

编译并运行完整的程序会产生类似于：

```
$ go build
$ ./gophernet

Accuracy = 0.97
```
哇噢！对于我们从头开始的神经网络，97％的准确度并不算太糟糕！ 当然，这个数字会因模型中的随机性而有所不同，但它通常表现得非常好。

我希望这对你来说是有益的和有趣的。所有的代码和数据都是[可在这里](https://github.com/dwhitena/gophernet)，所以自己试试吧！ 此外，如果您对Go for ML / AI和Data Science感兴趣，我强烈建议：

- 加入[Gophers Slack](https://invite.slack.golangbridge.org/)，并参与#data-science频道（我在那里@dwhitena）
- 检查所有伟大的Go ML / AI /数据工具[这里](https://github.com/gopherdata/resources/tree/master/tooling)
- 关注[GopherData博客/网站](http://gopherdata.io/)获取更多有趣的文章和社区信息