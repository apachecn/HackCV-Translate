# CoreML, Keras and TensorFlow — a super simple end to end test

![](https://cdn-images-1.medium.com/max/1600/1*RgllFqzSHoURTYQAMtExew.png)

This is a not a full Deep Learning tutorial but just a log for a super simple end to end test about how to use Keras, TensorFlow and CoreML all together.

We will build together an iOS App in Swift that will use CoreML to predict a Weight from a given Height creating and importing in CoreML a Keras/TensorFlow linear regression model.

The source of the App as well the Python script to create and export the Keras/Tensorflow model using CoreML Tools are available at:

[**JacopoMangiavacchi/HeightWeightCoreMLKerasTest**
HeightWeightCoreMLKerasTest - Super simple end to end test of Keras, TensorFlow and CoreMLgithub.com](https://github.com/JacopoMangiavacchi/HeightWeightCoreMLKerasTest)[](https://github.com/JacopoMangiavacchi/HeightWeightCoreMLKerasTest)

The incredibly simple linear regression model we will build and use in this test is based on some height/weight statistics available on a very usefull Keras/Tensorflow course available on the Udemy platform: [https://www.udemy.com/zero-to-deep-learning/](https://www.udemy.com/zero-to-deep-learning/)

Honestly there is no real reason to use ML in this very basic scenario but anyway I decided to document this very simple but complete end to end test, with documented step by step instructions, because it definetly helped me to connect all the dots while learning new things and evolving from my “mobile-first” background to the “ai-first” era.

First thing first: Setup the ML environment on your desktop (macOS or Windows) or your container.

First thing to do is to download and intall Anaconda from this link [https://www.continuum.io/downloads](https://www.continuum.io/downloads)

Anaconda is a package manager, an environment manager, a Python distribution, and a collection of over 720 open source packages free and easy to install that help a lot in general on dealing with Python but in particular here on loading dataset for example .CSV files, easily plot graph and easily import all the Python package needed such as Keras, CoreML Tools and other.

Of course Anaconda is not strictly needed but I really recommend this.

Once you have Anaconda installed you can easily setup the environment creating an environment.yml file like this:



If you want you can import this environment.yml cloning the project repository on your local computer with the following command: `git clone https://github.com/JacopoMangiavacchi/HeightWeightCoreMLKerasTest.git`

Once you have the environment.yml file you can let Anaconda install Python 2.7, TensorFlow 1.1, Keras 2.0.6, CoreMLTools 0.6.3, Pandas and other Python usefull packages with this simple command: `conda env create`

Once the environment is created you can simply activate it executing this command on macOS/linux `source activate KerasTensorFlowCoreML` or on Windows `activate KerasTensorFlowCoreML`

Now you can finally launch the super usefull Jupyter Notebook editor with the command `jupyter notebook `and open your browser to this local url [http://localhost:8888](http://localhost:8888)

If you have cloned the Git repo you will already have on your folder a complete Jupyter Notebook to open: createModel.ipynb

Other wise you need to copy and paste the following Python code in a new Notebook.

In any case once you have the completed Notebook edited please execute all cells (Python instructions) to create, test and export the model.

### Create the Keras / Tensorflow model

#### Description of the Python Notebook instructions

As said the project is all about to create a very simple linear regression model for predicting Weight from a given Height.

First things to do is to import some Python utilities like Pandas and import the HeightWeight statistics file





Than you can plot a weight / height graph and understand just with your eyes the regression ..



.. and obtain a graph like this:

![](https://cdn-images-1.medium.com/max/1600/1*COXK4EVKtcpbxavQrWUWKw.png)

Now you can create, compile and train (fit) the simple linear regression model from this Height and Weight data using the following instructions to create, compile and train a model :













Super easily the model train an input an array of Floats (Heights) as X and an array of Floats (Weights) as y.

NB Once again this is not a Keras/Tensorflow tutorial. I just created here a simple Sequential linear regression model and train the entire dataset of heights/weight statistics available. This is clearly not the best practice but it was for me just the best way to use the less number of Python lines of code for creating the model !

Now if you want you can test the model directly in Python using the Keras frontend API ..



.. and Plot again the Weight / Height data plust the linear regression with Pandas ..



.. and obtain a graph like this:

![](https://cdn-images-1.medium.com/max/1600/1*rUG18BnV0yA7w9hNRiL0WQ.png)

### Export the CoreML model

Still using Jupyter Notebook or your favorite Python environment you need to use Apple open source CoreML Tools to convert and export the model as a CoreML model.

Very easily to use CoreMLTools you need to import it in Python and basically call the Keras converter giving a name, if you want, to the input (Height) and output (Weight) array parameters and save the CoreML model like in the following code snippet:



The CoreML Model will be saved in the current folder as HeightWeight_model.mlmodel

If you want you can test the CoreML conversion directly calling from Python the CoreML predict API verifing if the returning value make sense for you using a command like:



NB Once again please consider that we just simply train the entire dataset and we do not provide any testing in this super simple scenario.

Once you have succesfully created and exported the CoreML model you can proceed to free your storage cleaning the Python, Keras, TensorFlow, CoreMLTools environment with the following command: `source deactivate conda remove -y -n KerasTensorFlowCoreML — all`
 
The CoreML model will not be deleted and it will remain in your folder

### Build the iOS sample project

If you have cloned the Git repo you can now just open the iOS sample project HeightWeightCoreMLKerasTest.xcodeproj in XCode 9 and Build and Test the App on your iPhone or Simulator.

The iOS sample project use a Swift wrapper class (HeightWeightModelWrapper.swift) to incupsulate all CoreML API and simplify the usage of CoreML Multi Array and implement some utility like convert from Centimeters to Inches and Pounds to Kilos.

Using this simple model from Swift to predict Weight from a given Height is as simple as executing this two line of code!



If you want to create your own iOS Swift sample app for this Heigth/Weight model you could to open XCode 9, create an empty Single View App project and start building your own UI.

Once you have your UI you can just drag and drop the CoreML model above (HeightWeight_model.mlmodel) in your XCode 9 project but please remember to check that the Target membership with your project is selected for the .mlmodel file in the project as in the picture below:

![](https://cdn-images-1.medium.com/max/1600/1*o1Nd8WTtiMSeXOei9iiFoA.png)

CoreML, like any Apple technology, is a kind of magic to use and just importing the model in XCode and building the solution it automatically generaty a Swift Class to directly use in your iOS project for using the model and do your predictions.

The only small issues that I found here is that sometime the CoreML MLMultiArray multi-dimension arrays are basically plumbing codes that you can easily avoid to spread all over in your UIViewControllers or in general in your project Swift Classes and Structs. More generally speaking I found that it could be easier to develop your own Wrapper class for the CoreML generated Swift code in order to strongly decouplyng the Model from your client application code, directly just deal with native data type and if you want implement some utility code in your wrapper to further simplify the usage of the CoreML model.

In my super simple iOS App I have created for example the following Wrapper for the Heigth/Weight CoreML model that let you directly call a predict method passing just a Float (Heigth) and eventually convert from Centimeters to Inches and Pounds to Kilos:









Of course once again this is just very basic code to just provide the idea with no serious error handling, throwable methods and forced optional unwrapping.

### Extend the Model

Now, once you have built and tested in CoreML this super simple Model, it’s time to do something a little bit more complex.

We are going to extend this Weight/Height model adding some Parameters to differentiate our prediction based on the Sex.

Again the goal here is not to write a tutorial on Deep Learning / AI but just document step by step how Keras (TensorFlow) and CoreML integrate in a generic mobile “AI First” App.

If you have cloned the Git repo above you can simply open in Jupyter the Notebook extendModel.ipynb and execute all cells to generate and export the new Extended model.

Otherwise as below create your Python notebook, import some Python utilities like Pandas and import again the HeightWeight statistics file





Now if took a better look at the csv we can see that we also have a column describing the sex for each single weight-height entry. If we execute the command `df.head()` we will see the head of the csv data as below:

![](https://cdn-images-1.medium.com/max/1600/1*P2eGQCZbwpzKQ4AavebRew.png)

Now in order to use this Gender data in Keras/TensorFlow we need to translate these sex strings in some numeric values. We are using Python Pandas package here in order to obtain a new all numeric dataset. The sex strings are transformed in this way in 0 or 1 numeric values in the corresponding Female and Male columns.





![](https://cdn-images-1.medium.com/max/1600/1*DE89e7n58iyx29IbZX42xA.png)

Now we are going to recreate our X and y_true where y_true will still be the array of Weights but X will now be a 3 dimension array of Heights, Female and Male values.



Quite in the same way as before we will use now the TensorFlow backend to create, compile and train the new model. The only difference here is that we will tell to Keras that our model will have now a input shape of 3: Height, Female, Male









NB Once again this is not a Keras/Tensorflow tutorial. We just trained here the entire dataset of heights/weight statistics available. This is clearly not the best practice but it was for me just the best way to use the less number of Python lines of code for creating the model !

Now as before we will use CoreML Tools to export the model







The CoreML Model will be saved in the current folder as HeightWeightExtended_model.mlmodel

Now that we have generate and exported this new CoreML model we will import it in our previous XCode 9 project replacing the old one and we will change the UI in order to ask to the user also the Sex.

The CoreML Swift Wrapper we created before will turn back very useful this time to simply adapt to the new 3 dimension MLMultiArray that is used as new input value for our new prediction function.

A simple Swift Enum as in the code below will help indeed a lot to simply adapt the ViewControllers of our App to the new changes.

























Have fun now with your Height prediction and please implement and share a real CoreML scenario ;-)

