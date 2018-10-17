## BUILDING A NEURAL NET FROM SCRATCH IN GO

原文链接：[BUILDING A NEURAL NET FROM SCRATCH IN GO](https://www.datadan.io/building-a-neural-net-from-scratch-in-go/?from=hackcv&hmsr=hackcv.com&utm_medium=hackcv.com&utm_source=hackcv.com)

I'm super pumped that my new book [Machine Learning with Go](https://www.packtpub.com/big-data-and-business-intelligence/machine-learning-go) is now available! Writing the book allowed me to get a complete view of the current state of machine learning in Go, and let's just say that I'm pretty excited to see how the community growing!

[![img](https://www.datadan.io/content/images/2017/10/book_wide.png)](https://www.packtpub.com/big-data-and-business-intelligence/machine-learning-go)

In the book (and for my own edification), I decided that I would build a neural network from scratch in Go. Turns out, this is fairly easy, and I thought it would be great to share my little neural net here.

All the code and data shown below is available [on GitHub](https://github.com/dwhitena/gophernet).

(If you are interested in leveraging pre-existing Go packaging for machine learning, check out [all the great existing packages](https://github.com/gopherdata/resources/tree/master/tooling), and be sure to watch [Chris Benson's recent talk](https://youtu.be/CHzMEamGZDA) at GolangUK about Deep Learning in Go)

## Goals

There are a whole variety of ways to accomplish this task of building a neural net in Go, but I wanted to adhere to the following guidelines:

- **No cgo** - I want my little neural net to compile nicely to a statically linked binary, and I also want to highlight the numerical functionality that is available natively in Go.
- **gonum matrix input** - I want supply matrices to my neural network for training, similar to how you would supply `numpy` arrays to most Python machine learning functions.
- **Variable numbers of nodes** - Although I will only illustrate one architecture here, I wanted my code to be flexible, such that I could tweak the numbers of nodes in each layer for other scenarios.

## Network Architecture

The basic network architecture that we will utilize in this example includes an input layer, a single hidden layer, and an output layer:

![img](https://www.datadan.io/content/images/2017/09/B05151_Chapter_08_05-1.png)

This type of single layer neural net might not be very "deep," but it has proven to be very useful for a huge majority of simple classification tasks. In our case, we will be training our model to classify iris flowers based on the [famous iris flower dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set). This should be more than enough to solve that problem with a high degree of accuracy.

Each of the **nodes** in the network will take in one or more inputs, combine those together linearly (using **weights** and a **bias**), and then apply a non-linear **activation function**. By optimizing the weights and the biases, with a process called [**backpropagation**](https://en.wikipedia.org/wiki/Backpropagation), we will be able to mimic the relationships between our inputs (measurements of flowers) and what we are trying to predict (species of flowers). We will then be able to feed new inputs through the optimized network (i.e., we will **feed** them **forward**) to predict the corresponding output.

(If you are new to neural nets, you might also check out [this great intro](https://ujjwalkarn.me/2016/08/09/quick-intro-neural-networks/), or, of course, you can read the relevant section in [Machine Learning with Go](https://www.packtpub.com/big-data-and-business-intelligence/machine-learning-go).)

## Defining Useful Functions and Types

Before diving into the backpropagation and feeding forward, let's define a couple types that will help us as we work with our model:

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

We also need to define our activation function and it's derivative, which we will utilize during backpropagation. There are many choices for activation functions, but here we are going to use the [sigmoid function](http://mathworld.wolfram.com/SigmoidFunction.html). This function has various advantages, including probabilistic interpretations and a convenient expression for it's derivative.

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

## Implementing Backpropagation for Training

With the definitions above taken care of, we can write an implementation of the [backpropagation method](https://en.wikipedia.org/wiki/Backpropagation) for training, or optimizing, the weights and biases of our network. The backpropagation method involves:

1. Initializing our weights and biases (e.g., randomly).
2. Feeding training data through the neural net forward to produce output.
3. Comparing the output to the correct output to get errors.
4. Calculating changes to our weights and biases based on the errors.
5. Propagating the changes back through the network.
6. Repeating steps 2-5 for a given number of **epochs** or until a stopping criteria is satisfied.

In steps 3-5, we will utilize [**stochastic gradient descent**](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) (SGD) to determine the updates for our weights and biases.

To implement this network training, I created a method on `neuralNet` that would take pointers to two matrices as input, `x` and `y`. `x` will be the features of our data set (i.e., the independent variables) and `y` will represent what we are trying to predict (i.e., the dependent variable). I will show an example of these later in the article, but for now, let's assume that they take this form.

In this function, we first initialize our weights and biases randomly and then use backpropagation to optimize the weights and the biases:

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

The actual implementation of backpropagation is shown below. **Note/Warning**, for clarity and simplicity, I'm going to create a handful of matrices as I carry out the backpropagation. For large data sets, you would likely want to optimize this to reduce the number of matrices in memory.

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

Here we have utilized a helper function that allows us to sum values along one dimension of a matrix, keeping the other dimension intact:

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

## Implementing Feed Forward for Prediction

After training our neural net, we are going to want to use it to make predictions. To do this, we just need to feed some given `x` values forward through the network to produce an output. This looks similar to the first part of backpropagation. Except, here we are going to return the generated output.

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

## The Data

Ok, we now have the building blocks that we will need to train and test our neural network. However, before we go off and try to run this, let's take a brief look at the data that I will be using to experiment with this neural net.

The data I'm going to use is a slightly transformed version of the popular [iris data set](https://archive.ics.uci.edu/ml/datasets/iris). This data set includes sets of four iris flower measurements (what will become our `x` values) along with a corresponding indication of iris species (what will become our `y` values). To utilize this data set with our neural net, I have slightly transformed the data set, such that the species values are represented by three binary columns (1 if the row corresponds to that species, 0 otherwise). I have also added a little bit of random noise to the measurement to try and confuse the neural net (because this problem is pretty easy to solve otherwise):

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

I also split the data set up for training and testing (via an 80/20 split) into `train.csv` and `test.csv` respectively.

## Putting It All Together

Let's put this neural net to work. To do this, we first need to read our training data, initialize a `neuralNet` value, and call the `train()` method:

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

That gives us a trained neural net. We can then parse the test data into matrices `testInputs` and `testLabels` (I'll spare you these details as they are the same as above), use our `predict()`method to make predictions for the flower species, and then compare the predictions to the actual species. Calculating the predictions and accuracy looks like the following:

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

## Results

Compiling and running the full program results in something similar to:

```
$ go build
$ ./gophernet

Accuracy = 0.97
```

Woohoo! 97% accuracy isn't too shabby for our little from scratch neural net! Of course this number will vary due to the randomness in the model, but it does generally perform very nicely.

I hope this was informative and interesting for you. All of the code and data is [available here](https://github.com/dwhitena/gophernet), so try it out yourself! Also, if you are interested in Go for ML/AI and Data Science in general, I highly recommend:

- Joining [Gophers Slack](https://invite.slack.golangbridge.org/), and participating in the #data-science channel (I'm @dwhitena there)
- Checking out all the great Go ML/AI/data tooling [here](https://github.com/gopherdata/resources/tree/master/tooling)
- Following the [GopherData blog/website](http://gopherdata.io/) for more interesting articles and community information