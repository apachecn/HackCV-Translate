# AVA: The Art and Science of Image Discovery at Netflix

Authored by — [Madeline](https://www.linkedin.com/in/madelinejriley),[ Lauren](https://www.linkedin.com/in/laurenmachado/), [Boris](https://www.linkedin.com/in/boris-roussabrov-a043991/), [Tim](https://www.linkedin.com/in/tim-branyen-3b8200a/), [Parth](https://www.linkedin.com/in/bhawalkar/), [Eugene](https://www.linkedin.com/in/eugene-jin-485a128/) and [Apurva](https://www.linkedin.com/in/apurvakansara/)

### Introduction

![](https://cdn-images-1.medium.com/max/1600/1*R-4laTXe2wtJC_cBGUeBSw.jpeg)

At Netflix, the Content Platform Engineering and Global Product Creative teams know that imagery plays an incredibly important role in how viewers find new shows and movies to watch. We take pride in surfacing the unique elements of a story that connect our audiences to diverse characters and story lines. As our Original content slate continues to expand, our technical experts are tasked with finding new ways to scale our resources and alleviate our creatives from the tedious and ever-increasing demands of digital merchandising. One of the ways in which we do this is by harvesting static image frames directly from our source videos to provide a more flexible source of raw artwork.

### The Business Case

Merchandising stills are static video frames taken directly from the source video content used to broaden the reach of a title on the Netflix service. Within a single one-hour episode of Stranger Things, there are nearly 86,000 static video frames.

Traditionally, these merchandising stills are selected by human curators or editors, and require an in-depth expertise of the source content that they’re intended to represent. We know through A/B testing that we can effectively drive increased viewing from expected and unexpected audience groups by exploring as many [representations of a title as possible.](https://medium.com/netflix-techblog/selecting-the-best-artwork-for-videos-through-a-b-testing-f6155c4595f6) When it comes to title key art, we like to test many artistic representations of a title in order to find the “right” artwork for the right audience. While this presents an exciting opportunity for innovation and testing, it simultaneously presents a very challenging expectation to scale this experience across every title in our growing global catalog.

### AVA

AVA is a collection of tools and algorithms designed to surface high quality imagery from the videos on our service. A single season of average TV show (about 10 episodes) contains nearly 9 million total frames. Asking creative editors to efficiently sift through that many frames of video to identify one frame that will capture an audience’s attention is tedious and ineffective. We set out to build a tool that quickly and effectively identifies which frames are the best moments to represent a title on the Netflix service.

![](https://cdn-images-1.medium.com/max/1600/0*hdzpQ8SvoOTcoVOF.)

To achieve this goal, we first came up with objective signals that we can measure for each and every frame of the video using Frame Annotations. As result, we can collect an effective representation of each frame of the video. Subsequently, we created ranking algorithms that allows us to rank a subset of frames that meets aesthetic, creative and diversity objectives to represent content accurately for various canvases of our product.

![](https://cdn-images-1.medium.com/max/1600/0*JX7Jvs3Sw77rWDy2.)

![](https://cdn-images-1.medium.com/max/1600/0*2XsXoykvK7FG3ty9.)

### Frame Annotation

As part of our automation pipeline, we process and annotate many different variables on each individual frame of video to best derive what the frame contains, and to understand why it is or isn’t important to the story. In order to scale horizontally and have predictable SLA for a growing catalog of content, we utilized the [Archer ](https://atscaleconference.com/videos/archer-a-distributed-computing-platform-for-media-processing/)framework to process our videos more efficiently. Archer allowed us to split the videos into smaller sized chunks that could each be processed in parallel. This has enabled us to scale by lending efficiency to our video processing pipelines, and allowing us to integrate more and more content intelligence algorithms into our tool sets.

![](https://cdn-images-1.medium.com/max/1600/0*AuzFOe4Q4znOWVEC.)

Every frame of video in a piece of content is processed through a series of computer vision algorithms to gather objective frame metadata, latent representation of frame, as well as some of the contextual metadata that those frame(s) contain. The annotation properties that we process and apply to our video frames can be roughly grouped into 3 main categories:

#### Visual Metadata

Typically these properties are objective, measurable, and mostly contained at the pixel-level. Some examples of visual properties are brightness, color, contrast, and motion blur.

![](https://cdn-images-1.medium.com/max/1600/0*6JN93Hwllw5SbE7r.)

#### Contextual Metadata

Contextual metadata is comprised of a combination of elements that are aggregated to derive meaning from the actions or movement of the actors, objects and camera in the frame. Some examples include;

* **Face detection** with facial landmarks tracking, pose estimation, and sentiment analysis — This allows us to estimate posture and sentiment of the subjects in the frame.

* **Motion estimation** — This allows us to estimate the amount of motion (both camera movement and subject movement) contained within a particular shot. This allows us to control for elements such as motion blur, as well as to identify camera movement that makes for compelling still imagery.

* **Camera shot identification** — (e.g. close up shot vs. dolly shot) This provides insight into the intentions of the cinematographer, allowing us to quickly identify and surface stylistic camera choices that provide insight into the mood, tone and genre of the title.

* **Object detection** — The detection of props and animated object segmentation allow us to attribute importance to non-human subjects in the frame.

![](https://cdn-images-1.medium.com/max/1600/0*fRgpOHd60Zs-qE7-.)

![](https://cdn-images-1.medium.com/max/1600/1*GaNbrMmBBn_8U7ebYr_I1A.gif)

#### Composition Metadata

Composition metadata refers to a special set of heuristic characteristics that we’ve identified and defined based on some of the core principles in photography, cinematography and visual aesthetic design. Some examples of composition are rule-of-third, depth-of-field and symmetry.

![](https://cdn-images-1.medium.com/max/1600/1*30vxgwzO6NiLdECfWC6X9Q.jpeg)

### Image Ranking

After we’ve processed and annotated every frame in a given video, the next step is to surface “the best” image candidates from those frames through an automated artwork pipeline. That way, when our creative teams are ready to begin work for a piece of content, they are automatically provided with a high quality image set to choose from. Below, we outline some of the key elements we use to surface the best images for a given title.

**Actors**

Actors play a very important role in artwork. One way we identify the key character for a given episode is by utilizing a combination of face clustering and actor recognition to prioritize main characters and de-prioritize secondary characters or extras. To accomplish this, we trained a deep-learning model to trace facial similarities from all qualifying candidate frames tagged with frame annotation to surface and rank the main actors of a given title without knowing anything about the cast members.

Beyond cast, we also take into account pose, facial landmarks, and the overall position of characters for a given cast member.

![](https://cdn-images-1.medium.com/max/1600/0*2hfTvgGuXe15FUUc.)

![](https://cdn-images-1.medium.com/max/1600/0*7Xhx8y_5jfeJ3VjC.)

**Frame Diversity**

Creative and visual diversity is a highly subjective discipline, as there are many different ways to perceive and define diversity in imagery. In the context of this solution, image diversity more specifically refers to the algorithms ability to capture the heuristic variance that naturally occurs within a single movie or episode. In doing so, we hope to provide designers and creatives with a scalable mechanism to quickly understand which visual elements are most representative of the title, and which elements are misrepresentative of the title. Some of the visual heuristic variables that we’ve incorporated into AVA to surface a diverse image set for a title include elements such as **camera shot types** (long shot vs medium shot), **visual similarity** (rule of thirds, brightness, contrast), **color** (colors that are most prominent), and **saliency maps** (to identify negative space and complexity). By combining these heuristic variables, we can effectively cluster image frames based on a custom vector for diversity. Furthermore, by incorporating several vectors, we’re able to construct a diversity index against which all candidate imagery for a given episode or movie, can be scored.

![](https://cdn-images-1.medium.com/max/1600/0*s3H0tZLBWgjv2TUP.)

**Filters for Maturity**

For content sensitivity and audience maturity reasons, we also needed to make sure we excluded frames containing harmful or offensive elements. Examples of editorial exclusion criteria are things like; sex/nudity, text, logos/unauthorized branding, and violence/gore. In order to de-prioritize frames containing these elements, we incorporated the probability of each of these variables as vectors, allowing us to quantify and ultimately attribute a lower score for these frames.

We additionally included elements such as title genre, content format, maturity rating, etc. as secondary elements or minor features and as feedback to the model for ranking prediction.

### Conclusion

In this techblog, we’ve provided an overview of our unique approach to surfacing meaningful images from video and enabling our creative teams to design stunning artwork every single day. AVA is a collection of tools and algorithms encapsulating the key intersections of computer vision combined with the core principles of filmmaking and photo editing.

Stay tuned for a follow up blog in which we’ll dive into programmatic artwork composition, an exciting new solution that’s responsible for much of the artwork you see on the Netflix service today!

Thank you.

If you have great or innovative ideas come join us on the [Content Platform Engineering team!](https://jobs.netflix.com/jobs/866146)

