# Install Tensorflow 1.8.0 with GPU from source on Ubuntu 18.04 Bionic Beaver

![](https://cdn-images-1.medium.com/max/1600/1*8faEHRYdnEntxt6N5plKng.png)

When I first put Linux, the first thing I wanted to do was install the Tensorflow GPU. I found an excellent guide. But even with this in mind, I spent more than 40 hours on the installation. This article I want to save you time, sharing your own experience. Below you will find updated guide from python36.com

**Note**: Have updated guide [here](https://medium.com/@Oysiyl/install-tensorflow-1-9-0-with-gpu-from-source-on-ubuntu-18-04-bionic-beaver-9822add0d454) (all important changes are made at step 12). For me, previous version tensorflow and bazel build took 2h 3min. Updated version build 1hrs 32min. It worth it to update as you think?

### Step 1: Update and Upgrade your system:



### Step 2: Verify You Have a CUDA-Capable GPU:



**Note GPU model. eg. GeForce 840M**

If you do not see any settings, update the PCI hardware database that Linux maintains by entering update-pciids (generally found in /sbin) at the command line and rerun the previous lspci command.

### Step 3: Verify You Have a Supported Version of Linux:

To determine which distribution and release number you’re running, type the following at the command line:



The x86_64 line indicates you are running on a 64-bit system which is supported by cuda 9.1.

### Step 4: Install Dependencies:

Required to compile from source:



### Step 5: Install linux kernel header:

Goto terminal and type:



You can get like “4.15.0–23-generic”. Note down linux kernel version.

In Bionic Beaver (18.04) you have 4.15 kernel. With this version you have chance to stuck with cuda installation. I searched on Google for many hours and was not solved this problem.

![](https://cdn-images-1.medium.com/max/1600/1*Me3HBvPbgu_bIhZ1gncDlw.jpeg)

To avoid this problem install 4.16 kernel:



Once you’ve downloaded all the above kernel files, now install them as follows. Linux-headers will also be installed with this command:



Once the installation is complete, reboot your machine and verify that the new kernel version is being used:



You must get something like this:

![](https://cdn-images-1.medium.com/max/1600/1*NfIagIz5qWZEdrjMg0CppQ.png)

And that’s it. You are now using a much more recent kernel version than the one installed by default with **Ubuntu**.

### Step 6: Install NVIDIA CUDA 9.2:

**Remove previous cuda installation(if you installed cuda before):**



**Install cuda:**



On step 6 when execute last command be careful!

At the first try usually script stuck at unitramfs (every time when I launch it on 4.15 kernel I saw what this unitramfs files not found and cuda installed wrong and not correctly). On kernel 4.16 you don`t stuck with this problem.







If this line not update for few minutes, open System Monitor and wait, when the load of the CPU cores will decrease. Don`t type immediately!

Then try this in terminal:

use ESC few times and then type: password + Enter + password + Enter..

If not helped:

use ESC few times and then type: password + Enter + password + Enter..

Be patient with yourself, typed password slowly. After 10 try use ESC and type again.

If you install cuda on a fresh system, you need to type password «blindly » just once. Else be prepared to do this twice: when build kernel and when see this message:







And you will succeed!

### Step 7: Reboot the system to load the NVIDIA drivers.

### Step 8: Go to terminal and type:





Check driver version probably Driver Version: 396.26

For now if you use nvidia-smi command you get temp for GPU and nothing more (no process found below). And you have low screen resolution because your nvidia-drivers not detect ligament (GPU-monitor).

![](https://cdn-images-1.medium.com/max/1600/1*kUEziG_VgjZCoFeu3Gw2Wg.png)

But you can fix this with Xorg!

Use this command to create Xorg:



That`s create a file a file xorg.conf in path: (etc/X11/xorg.conf). To change resolution you need just reboot your system. And you can go to step 9.

If this not helped change this file (xorg.conf). To do this use this command with parameters of your monitor. My command look like this:



Press Enter and you got



Just copy this and open xorg.conf:



Paste here (instead of modeline). Change HorizSync and VertRefresh too:

![](https://cdn-images-1.medium.com/max/1600/1*xb0C3P2iM6FKiHLXqMCVAQ.png)

And reboot after it. For now your screen resolution should be the same as before. For now you can type



And see this:

![](https://cdn-images-1.medium.com/max/1600/1*VwPYwhd80QoYbjSIOWXNuQ.png)

Now you can see temperature and other useful information about your GPU.

### Step 9: Install cuDNN 7.1.4:

NVIDIA cuDNN is a GPU-accelerated library of primitives for deep neural networks.

Goto [https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn) and download Login and agreement required

After login and accepting agreement.

Download the following:

cuDNN v7.1.4 Library for Linux

Goto downloaded folder and in terminal perform following:



### Step 10: Install NCCL 2.2.13:

NVIDIA Collective Communications Library (NCCL) implements multi-GPU and multi-node collective communication primitives that are performance optimized for NVIDIA GPUs

Go to [https://developer.nvidia.com/nccl](https://developer.nvidia.com/nccl) and attend survey to download Nvidia NCCL.

Download following after completing survey.



Go to downloaded folder and in terminal perform following:



### Step 11: Install Dependencies

**Isntall libcupti:**



**Python related:**

To install these packages for Python 2.7, issue the following command:



To install these packages for Python 3.n, issue the following command:



### Step 12: Configure Tensorflow from source:

**Download bazel(version 0.14):**



**Reload environment variables**



**Start the process of building TensorFlow by downloading latest tensorflow 1.8.0.**



Give python path in



Press enter two times



Now you need compute capability which we have noted at step 1 eg. 5.0. Go to that [link](https://developer.nvidia.com/cuda-gpus) and click on “CUDA-Enabled GeForce Products”. For example: if you have GPU on Pascal architecture your`s compute capability should be 6.1, if Maxwell — 5.2 and so on.





Configuration finished!

### Step 13: Build Tensorflow using bazel

The next step in the process to install tensorflow GPU version will be to build tensorflow using bazel. This process takes a fairly long time.

**To build a pip package for TensorFlow you would typically invoke the following command:**



**Note:**if you got error like unsupported platform then make sure you are running correct pip command associated with the python you used while configuring tensorflow build.



This process will take a lot of time. It may take 1–2 hours or maybe even more.For example, on my i5–4590 it take 2 hrs 3 min. Be ready to wait!

Also if you got error like Segmentation Fault then try again it usually worked.

The bazel build command builds a script named build_pip_package. Running this script as follows will build a .whl file within the tensorflow_pkg directory:

**To build whl file issue following command:**



**To install tensorflow with pip:**



for existing virtual environment:



With a new virtual environment using virtualenv:



for python 2: (use sudo if required)



for python 3: (use sudo if required)



Note : if you got error like unsupported platform then make sure you are running correct pip command associated with the python you used while configuring tensorflow build.

You can check pip version and associated python by following command



### Step 14: Verify Tensorflow installation

Run in terminal



If the system outputs the following, then you are ready to begin writing tensorflow programs:



Success! You have now successfully installed tensorflow 1.8.0 on your machine.

![](https://cdn-images-1.medium.com/max/1600/1*B9fjMDFs9WEsxZbJebUClQ.png)

This article based on awesome guide, which available [here](http://www.python36.com/how-to-install-tensorflow-gpu-with-cuda-9-2-for-python-on-ubuntu/). I am only supplemented it with the errors I encountered and how to avoid them.

Those answers maybe can help you during the process:

[https://www.tecmint.com/upgrade-kernel-in-ubuntu/](https://www.tecmint.com/upgrade-kernel-in-ubuntu/)

[https://devtalk.nvidia.com/default/topic/1036167/stuck-trying-to-intall-nvidia-390-ubuntu-18-04-lts-/?offset=10](https://devtalk.nvidia.com/default/topic/1036167/stuck-trying-to-intall-nvidia-390-ubuntu-18-04-lts-/?offset=10)

[https://askubuntu.com/questions/647708/cannot-edit-xorg-conf-permissions](https://askubuntu.com/questions/647708/cannot-edit-xorg-conf-permissions)

[https://askubuntu.com/questions/4253/getting-screen-resolution-correct-with-nvidia-drivers](https://askubuntu.com/questions/4253/getting-screen-resolution-correct-with-nvidia-drivers)

[https://unix.stackexchange.com/questions/387735/how-to-set-a-custom-resolution-with-nvidia-drivers-installed](https://unix.stackexchange.com/questions/387735/how-to-set-a-custom-resolution-with-nvidia-drivers-installed)

Thanks for your attention!

