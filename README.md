# Multiple Object Tracking as ID Prediction

This is the official PyTorch implementation of our paper:

> Multiple Object Tracking as ID Prediction <br>
> [Ruopeng Gao](https://ruopenggao.com/), Yijun Zhang, [Limin Wang](https://wanglimin.github.io/)

## Overview

**TL; DR.** MOTIP proposes a new perspective to regard the multi-object tracking task as an ID prediction problem. 
It directly predicts the ID labels for each object in the tracking process, which is more straightforward and effective.
![Overview](./assets/overview.png)

**Abstract.** In Multiple Object Tracking (MOT), tracking-by-detection methods have stood the test for a long time, which split the process into two parts according to the definition: object detection and association. They leverage robust single-frame detectors and treat object association as a post-processing step through hand-crafted heuristic algorithms and surrogate tasks. However, the nature of heuristic techniques prevents end-to-end exploitation of training data, leading to increasingly cumbersome and challenging manual modification while facing complicated or novel scenarios. In this paper, we regard this object association task as an End-to-End in-context ID prediction problem and propose a streamlined baseline called MOTIP. Specifically, we form the target embeddings into historical trajectory information while considering the corresponding IDs as in-context prompts, then directly predict the ID labels for the objects in the current frame. Thanks to this end-to-end process, MOTIP can learn tracking capabilities straight from training data, freeing itself from burdensome hand-crafted algorithms. Without bells and whistles, our method achieves impressive state-of-the-art performance in complex scenarios like DanceTrack and SportsMOT, and it performs competitively with other transformer-based methods on MOT17. We believe that MOTIP demonstrates remarkable potential and can serve as a starting point for future research.


