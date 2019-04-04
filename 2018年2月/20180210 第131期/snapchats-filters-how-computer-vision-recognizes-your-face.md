# Snapchat’s Filters: How computer vision recognizes your face



In those moments of boredom when you’re playing with [Snapchat](https://www.snapchat.com/)’s filters — sticking your tongue out, ghoulifying your features, and working out how to get the flower crown to fit exactly on your head — surely you’ve had a moment where you’ve wondered what’s going on, on a technical level — how Snapchat manages to match your face to the animations?

After two weeks of researching online, I feel grateful to have finally gotten a glimpse behind the curtain. It turns out that the product is an instance of computer vision application, which is the main fuel behind all kinds of facial recognition software.

#### The Technology

The technology came from a Ukrainian startup **Looksery**, which is an application that allowed users to modify their facial features during video chats and for photos. Snapchat acquired this Odesa-based face changing startup in September 2015 for $150 million dollars. That’s reportedly the largest tech acquisition in Ukrainian history.

![](https://cdn-images-1.medium.com/max/1600/1*weCroGAyHfoZDK1h7ZOoDA.png)

Their [augmented reality filters](http://www.supreality.com/2016/04/13/snapchats-augmented-reality/#comment-1138) tap into the large and rapidly growing field of computer vision.**Computer vision** can be thought of as a direct opposite of computer graphics. While computer graphics try to produce image models from 3D models, computer vision tries to create a 3D space from image data. Computer Vision is starting to be utilized more and more in our society. It is how you scan your checks and the data is extracted from the lines. It is how you can deposit checks with your phone. It is how Facebook knows who’s in your photos, how self-driving cars can avoid running over people and how you can give yourself a dodgy nose.

#### How Snapchat Filters Work

Looksery maintains their engineering more confidential, but every one can access their patents online. The specific area of Computer Vision that Snapchat filters use is called **Image processing**. Image processing is the transformation of an image by performing mathematical operations on each individual pixel on the provided picture.

**1 — Face Detection:**

The first step works like this: Given an input image or video frame, find out all present human faces and output their bounding box (i.e. The rectangle coordinates in the form: **X**, **Y**, **Width** & **Height**).

Face detection has been a solved problem since the early 2000s but faces some challenges including detecting [tiny](https://arxiv.org/pdf/1612.04402.pdf), [partial](https://arxiv.org/pdf/1603.09364.pdf) & [non-frontal](http://ieeexplore.ieee.org/abstract/document/5459421/) faces. The most widely used technique is a combination of Histogram of Oriented Gradients ([HOG](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients) for short) and Support Vector Machine ([SVM](https://en.wikipedia.org/wiki/Support_vector_machine)) that achieve mediocre to relatively good detection ratios given a good quality image but this method is not capable of real-time detection at least on the **CPU**.

![](https://cdn-images-1.medium.com/max/1600/1*2-ezOTxM6wRtRG2FJ3eCbw.jpeg)

Here is how the HOG/SVM detector works:

Given an input image, compute the [pyramidal](https://en.wikipedia.org/wiki/Pyramid_(image_processing)) representation of that image which is a pyramid of multi scaled downed version of the original image. For each entry on the pyramid, a sliding window approach is used. The sliding window concept is quite simple. By looping over an image with a constant step size, small image patches typically of size 64 x 128 pixels are extracted at different scales. For each patch, the algorithm makes a decision if it contains a face or not. The HOG is computed for the current window and passed to the SVM classifier (Linear or not) for the decision to take place (i.e. Face or not). When done with the pyramid, a [non-maxima suppression](http://users.ecs.soton.ac.uk/msn/book/new_demo/nonmax/) (NMS for short) operation usually take place in order to discard stacked rectangles. You can read more about the HOG/SVM combination [here](https://dsp.stackexchange.com/questions/19187/head-detection-using-hog-and-svm).

**2 — Facial Landmarks:**

This is the next step in our analysis phase and works as follows: For each detected face, output the local region coordinates for each member or facial feature of that face. This includes the eyes, bone, lips, nose, mouth,… coordinates usually in the form of points (**X**,**Y**).

Extracting facial landmarks is a relatively cheap operation for the CPU given a bounding box (i.e. Cropped image with the target face), but quite difficult to implement for the programmer unless some not-so-fast machine learning techniques such as training & running a classifier is used.

![](https://cdn-images-1.medium.com/max/1600/1*vRXrawyNUUa_mLYR1Vf-dQ.jpeg)

You can find out more about extracting facial landmarks [here](http://www.learnopencv.com/facial-landmark-detection/) or this PDF: [One millisecond face alignment with an ensemble of regression trees](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Kazemi_One_Millisecond_Face_2014_CVPR_paper.pdf). In some and obviously useful cases, face detection and landmarks extraction are combined into a single operation.

**3 — Image Processing**

Now that the face has been detected, Snapchat can use Image Processing to apply features onto a full face. However, they chose to go one step further and they want to find your facial features. This is done with the aid of the **Active Shape Model**.

The Active Shape Model is a facial model that has been trained by the manual marking of the borders of facial features on hundreds to thousands of images. Through machine learning, an “average face” is created and aligns this with the image that is provided. This average face, of course, does not fit exactly with the user’s face (we all have diverse faces), so after fitting the face, pixels around the edge of the “average face” are examined to look for differences in shading. Because of the training that the algorithm went through, (the Machine Learning process), it has a basic skeleton of how certain facial features should look, so it looks for a similar pattern in the given image. Even if some of the initial changes are wrong, by taking into account the position of other points that it has fixed, the algorithm will correct errors it made regarding where it thought certain aspects of your face are. The model then adjusts and creates a mesh (a 3D model that can shift and scale with your face).

![](https://cdn-images-1.medium.com/max/1600/1*4xU_GUpNdFXLnrtMxCJOIg.jpeg)

This whole facial/feature recognition process is done when you see that white net right before you choose your filter. The filters then distort certain areas of the provided face by enhancing them or adding something on top of them.

#### From Filters to Face-Swap

The updated version of Snapchat a few months back had the feature for swapping faces with a friend, whether in real time or by accessing some faces from your gallery. Notice how the face shapes are visible, that’s the position where the statistical model lies. It helps Snapchat to quickly align you and your friends face and swap the features.

After locating all your features, the application creates a mesh along your face that sticks to each point frame by frame. This mesh can now be edited and modified as Snapchat feels.

![](https://cdn-images-1.medium.com/max/1600/1*YzfgVRCIUjfTUYE23VH9Ag.jpeg)

Some lenses do much more by either asking you to raise your eyebrows or by opening your mouth. This is also fairly simple to think about, but it requires a lot more algorithms to imply.

* The inside of the mouth is dark, relatively. So that gives away the opening of the mouth.

* The changes on the eyebrows relative to the other facial features are taken into account when it figures out the user has raised the eyebrows.

Now as mentioned before, this technology is not new. But to perform all those processes in real time and on a mobile platform takes a lot of processing power along with some complicated algorithms. That’s why Snapchat thought it’s better to pay 150 million dollars to acquire Looksery instead of just building its platform.

#### Conclusion

I hope this was informative and tickled your curiosity like it did mine. For now, I’ll be exploring Snapchat Filters more deeply, testing out my favorite facial lens, knowing and appreciating all the computer vision that’s going on behind the scenes.

— —

If you enjoyed this piece, I’d love it if you hit the clap button 👏 so others might stumble upon it. You can find my own code on [GitHub](https://github.com/khanhnamle1994), and more of my writing and projects at [https://jameskle.com/](https://jameskle.com/). You can also follow me on [Twitter](https://twitter.com/@james_aka_yale), [email me directly](mailto:khanhle.1013@gmail.com) or [find me on LinkedIn](http://www.linkedin.com/in/khanhnamle94).

**Additional Resources:**

* [How do Snapchat filters work](https://www.technobyte.org/2016/11/snapchat-filters-work/) (Technobyte)

* [How Snapchat Filter Works — Behind The Scenes](http://techundred.com/how-snapchat-filter-work/) (TechHundred)

* [How Snapchat’s filters work](https://www.youtube.com/watch?v=Pc2aJxnmzh0) (Vox)

#### This story is published in The Startup, Medium’s largest entrepreneurship publication followed by 294,522+ people.

#### Subscribe to receive our top stories here.

