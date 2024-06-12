# Multiple Object Tracking as ID Prediction

This is the official PyTorch implementation of our paper:

> ***[Multiple Object Tracking as ID Prediction](https://arxiv.org/abs/2403.16848)*** <br>
> :mortar_board: [Ruopeng Gao](https://ruopenggao.com/), Yijun Zhang, [Limin Wang](https://wanglimin.github.io/) <br>
> :e-mail: Primary contact: ruopenggao@gmail.com

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multiple-object-tracking-as-id-prediction/multi-object-tracking-on-dancetrack)](https://paperswithcode.com/sota/multi-object-tracking-on-dancetrack?p=multiple-object-tracking-as-id-prediction)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multiple-object-tracking-as-id-prediction/multiple-object-tracking-on-sportsmot)](https://paperswithcode.com/sota/multiple-object-tracking-on-sportsmot?p=multiple-object-tracking-as-id-prediction)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multiple-object-tracking-as-id-prediction/multi-object-tracking-on-mot17)](https://paperswithcode.com/sota/multi-object-tracking-on-mot17?p=multiple-object-tracking-as-id-prediction)<br>

## :mag: Overview

**TL; DR.** MOTIP proposes a new perspective to ***regard the multi-object tracking task as an ID prediction problem***. 
It directly predicts the ID labels for each object in the tracking process, which is more straightforward and effective.

![Overview](./assets/overview.png)

**Abstract.** In Multiple Object Tracking (MOT), tracking-by-detection methods have stood the test for a long time, which split the process into two parts according to the definition: object detection and association. They leverage robust single-frame detectors and treat object association as a post-processing step through hand-crafted heuristic algorithms and surrogate tasks. However, the nature of heuristic techniques prevents end-to-end exploitation of training data, leading to increasingly cumbersome and challenging manual modification while facing complicated or novel scenarios. In this paper, we regard this object association task as an End-to-End in-context ID prediction problem and propose a streamlined baseline called MOTIP. Specifically, we form the target embeddings into historical trajectory information while considering the corresponding IDs as in-context prompts, then directly predict the ID labels for the objects in the current frame. Thanks to this end-to-end process, MOTIP can learn tracking capabilities straight from training data, freeing itself from burdensome hand-crafted algorithms. Without bells and whistles, our method achieves impressive state-of-the-art performance in complex scenarios like DanceTrack and SportsMOT, and it performs competitively with other transformer-based methods on MOT17. We believe that MOTIP demonstrates remarkable potential and can serve as a starting point for future research.


## :fire: News

- <span style="font-variant-numeric: tabular-nums;">***2024.05.06***</span>: We release the training code and scripts :hugs:. The pre-training scripts will be released later :soon:. Now you can directly download pre-trained weights from the [Cloud :cloud:](https://drive.google.com/drive/folders/1O1HUxJJaDBORG6XEBk2QcWeXKqAblbxa?usp=drive_link).

- <span style="font-variant-numeric: tabular-nums;">***2024.03.28***</span>: We release the inference code, you can evaluate the model following the [instructions](#evaluation) :tada:. Our model weights and logs are available in the [Google Drive](https://drive.google.com/drive/folders/1LTBWHLHhrF0Ro7fgCdAkgu9sJUV_y-vw?usp=drive_link) :cloud:.

- <span style="font-variant-numeric: tabular-nums;">***2024.03.26***</span>: The paper is released on [arXiv](https://arxiv.org/abs/2403.16848), ~~the code will be available in several days.~~ Welcome to watch our repository for the latest updates :pushpin:.


## :chart_with_upwards_trend: ​Main Results

### :dancer: ​DanceTrack

| Method              | Training Data       | HOTA | DetA | AssA | MOTA | IDF1 | URLs                                                         |
| ------------------- | ------------------- | ---- | ---- | ---- | ---- | ---- | ------------------------------------------------------------ |
| MOTIP               | DT                  | 67.5 | 79.4 | 57.6 | 90.3 | 72.2 | [model](https://drive.google.com/file/d/1qNGN7RsDf6a3i5lwjb0V8v6mKzxaMh0G/view?usp=drive_link), [config](./configs/r50_deformable_detr_motip_dancetrack.yaml), [log](https://drive.google.com/file/d/1XRRBjw92bQk7FUGxmZSsrTf5BjXbL2pp/view?usp=drive_link) |
| MOTIP<sub>DAB</sub> | DT                  | 70.0 | 80.8 | 60.8 | 91.0 | 75.1 | [model](https://drive.google.com/file/d/1mVj_FgE4fGUaALZB3JEmiAlqFSnHHNLN/view?usp=drive_link), [config](./configs/r50_dab_deformable_detr_motip_dancetrack.yaml), [log](https://drive.google.com/file/d/1tACnXMvwNx1jq7EOTngcsb9KulU4125f/view?usp=drive_link) |
| MOTIP               | DT + CH             | 71.4 | 81.3 | 62.8 | 91.6 | 76.3 | [model](https://drive.google.com/file/d/1BDvk6dxJh7LPCvkVC4ycGWNm4-DyVkbf/view?usp=drive_link), [config](./configs/r50_deformable_detr_motip_dancetrack_joint_ch.yaml), [log](https://drive.google.com/file/d/1JBrj5Jq5PXf7_ZO7xyrTEBDMm8HNxpX4/view?usp=drive_link) |
| MOTIP               | DT<sup>*</sup> + CH | 73.7 | 82.6 | 65.9 | 92.7 | 78.4 | [model](https://drive.google.com/file/d/1cdfGY3iwGcQKqpgxePPrMg10Taro3n-G/view?usp=drive_link), [config](./configs/r50_deformable_detr_motip_dancetrack_trainval_joint_ch.yaml), [log](https://drive.google.com/file/d/112n-ziOG8qfvyH8WK8x2Sqa7BPfIDTr7/view?usp=drive_link) |

<details>
  <summary><i>NOTE</i></summary>
  <ol>
    <li>MOTIP is built upon original Deformable DETR, while MOTIP<sub>DAB</sub> is based on DAB-Deformable DETR.</li>
    <li>DT and CH are the abbreviations of DanceTrack and CrowdHuman respectively.</li>
    <li>DT<sup>*</sup> denotes we utilize both the training and validation set of DanceTrack for training.</li>
  </ol>
</details>


### :basketball: ​SportsMOT

| Method | Training Data      | HOTA | DetA | AssA | MOTA | IDF1 | URLs                                                         |
| ------ | ------------------ | ---- | ---- | ---- | ---- | ---- | ------------------------------------------------------------ |
| MOTIP  | Sports             | 71.9 | 83.4 | 62.0 | 92.9 | 75.0 | [model](https://drive.google.com/file/d/1NIw77CBt8xEoZxHrUg14vrPYBCXUUgq-/view?usp=drive_link), [config](./configs/r50_deformable_detr_motip_sportsmot.yaml), [log](https://drive.google.com/file/d/1SNZ60uxVCdU5Poza0fXztWSGaZifVdaD/view?usp=drive_link) |
| MOTIP  | Sports<sup>*</sup> | 75.2 | 86.5 | 65.4 | 96.1 | 78.2 | [model](https://drive.google.com/file/d/1DTQenGa5WuFLVi_z7-07jsHBjTpiYGv_/view?usp=drive_link), [config<sup>*</sup>](./configs/r50_deformable_detr_motip_sportsmot.yaml), [log](https://drive.google.com/file/d/14eqHQh8pFc8vxpGRF9CMNp5yeMIA-tXQ/view?usp=drive_link) |

<details>
  <summary><i>NOTE</i></summary>
  <ol>
    <li>Sports is the abbreviation of SportsMOT.</li>
    <li>Sports<sup>*</sup> denotes we utilize both the training and validation set of SportsMOT for training.</li>
    <li>config<sup>*</sup> represents the configuration that can be used for inference. The corresponding training config file has not been uploaded yet.</li>
  </ol>
</details>



### :walking: ​MOT17

| Method | Training Data | HOTA | DetA | AssA | MOTA | IDF1 | URLs                                                         |
| ------ | ------------- | ---- | ---- | ---- | ---- | ---- | ------------------------------------------------------------ |
| MOTIP  | MOT17 + CH    | 59.2 | 62.0 | 56.9 | 75.5 | 71.2 | [model](https://drive.google.com/file/d/1ZsojRYBCbH9u9m1C5leb1MwmBB42sox8/view?usp=drive_link), [config](./configs/r50_deformable_detr_motip_mot17.yaml), [log](https://drive.google.com/file/d/1RB0XasyMMJFziB5wuyT208jMBLW37CPM/view?usp=drive_link) |

<details>
  <summary><i>NOTE</i></summary>
  <ol>
    <li>CH is the abbreviation of CrowdHuman.</li>
  </ol>
</details>


## :dash: Quick Start

- See [INSTALL.md](./docs/INSTALL.md) for instructions of installing required components.
- See [DATASET.md](./docs/DATASET.md) for datasets download and preparation.
- See [PRETRAIN.md](./docs/PRETRAIN.md) for how to get pretrained DETR weights.
- See [GET_STARTED.md](./docs/GET_STARTED.md) for how to get started with our MOTIP.

## :pencil2: Citation

If you think this project is helpful, please feel free to leave a :star: and cite our paper:

```tex
@article{MOTIP,
  title={Multiple Object Tracking as ID Prediction},
  author={Gao, Ruopeng and Zhang, Yijun and Wang, Limin},
  journal={arXiv preprint arXiv:2403.16848},
  year={2024}
}
```

