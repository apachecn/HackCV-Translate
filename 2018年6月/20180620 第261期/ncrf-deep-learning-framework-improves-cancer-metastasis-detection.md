# NCRF Deep Learning Framework Improves Cancer Metastasis Detection

![](https://cdn-images-1.medium.com/max/1600/1*Plc8FcIEcyKKkSK88HF9LA.png)

The **Baidu Silicon Valley Artificial Intelligence Lab** has released a paper which proposes a neural conditional random field (NCRF) process for cancer metastasis detection on Whole Slide Images (WSIs). [Cancer Metastasis Detection With Neural Conditional Random Field. Medical Imaging with Deep Learning (MIDL), 2018, Yi Li and Wei Ping](https://openreview.net/forum?id=S1aY66iiM) has been accepted by the International Conference on Medical Imaging with Deep Learning (MIDL), which runs July 4 ‑ 6 in Amsterdam.
 
WSIs are outrageously large files, which usually contain billions of pixels and take up gigabytes of disk space. Pathologists must search these images for groups of tumor cells which can be smaller than 1000 pixels in diameter. Accordingly, to effectively detect metastasis in these massive files is complicated and time-consuming for human experts.

![](https://cdn-images-1.medium.com/max/1600/1*k_uQsjz3tVw89FyhdN5olQ.png)

Various deep learning based algorithms have been proposed to aid pathologists in effectively reviewing these slides. Most of the algorithms split the slide into smaller individual image patches, e.g. at 256x256 pixels. A deep convolutional neural network (CNN) is then trained to determine whether each small patch contains tumor cells or healthy cells. It can however be difficult to predict whether such a patch contains tumor cells without viewing its surroundings, especially when dealing with tumor/healthy boundary regions, and false positive predictions are often returned.

![](https://cdn-images-1.medium.com/max/1600/1*erYvHdYS6MCxDlPiefqE7Q.png)

The NCRF proposed in the paper addresses this issue by**including a grid of neighboring patches as input to provide context that can improve prediction of tumor cells or healthy cells**.

![](https://cdn-images-1.medium.com/max/1600/1*Plc8FcIEcyKKkSK88HF9LA.png)

Conditional random fields (CRF) are used to model the spatial correlations between neighboring patches. The entire NCRF can be trained end-to-end without any pre- or post-processing.
 
The major improvement this algorithm provides is that it returns far fewer false positives. The model achieved an average FROC score (a tumor localization rating) of 0.8096 on the Camelyon16 dataset, outperforming not only a professional pathologist (0.7240), but also the previous AI winner of the Camelyon16 challenge (0.8074). 
 
Further clinical study on larger datasets will be necessary to assess the proposed algorithm comprehensively.
 
With the help of better tumor detection algorithms, pathologists can be freed from searching through an entire slide and can focus more on tumor regions highlighted by the algorithm.
 
 NCRF is open sourced at GitHub ([https://github.com/baidu-research/NCRF](https://github.com/baidu-research/NCRF)).

**Author:** Mos Zhang | **Editor:** Michael Sarazen

**Follow us on Twitter**[@Synced_Global](https://twitter.com/Synced_Global)**for more AI updates!**

**Subscribe**[here ](https://t.co/d2OrjqTGDq)**to get insightful tech news, reviews and analysis!**

Let’s talk about AI and FinTech! Synced invites you to join our**DTalk Episode One: Deploying AI in Mobile-First Customer-facing Financial Products: A Tale of Two Cycles**. Jike Chong will share his ideas on employing AI techniques in FinTech business model. Register at [https://goo.gl/KKhHgv](https://t.co/mllmSDfdpU)! We hope to see you on June 21st in Silicon Valley.

