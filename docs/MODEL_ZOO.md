# Model Zoo

## MOTIP

### DanceTrack

| Method | Extra Data | Traj Aug |                          Resources                           | HOTA | DetA | AssA |
| :----: | :--------: | :------: | :----------------------------------------------------------: | :--: | :--: | :--: |
| MOTIP  |  ***no***  |  *yes*   | [pre-train](https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_coco_dancetrack.pth) \| [config](../configs/r50_deformable_detr_motip_dancetrack.yaml) \| [checkpoint](https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_motip_dancetrack.pth) | 69.6 | 80.4 | 60.4 |
| MOTIP  |  ***no***  |   *no*   | [pre-train](https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_coco_dancetrack.pth) \| [config](../configs/r50_deformable_detr_motip_dancetrack_without_trajectory_augmentation.yaml) \| [checkpoint](https://github.com/MCG-NJU/MOTIP/releases/download/v0.2/r50_deformable_detr_motip_dancetrack_without_trajectory_augmentation.pth) | 65.2 | 80.4 | 53.1 |

### SportsMOT

| Method | Extra Data | Traj Aug |                          Resources                           | HOTA | DetA | AssA |
| :----: | :--------: | :------: | :----------------------------------------------------------: | :--: | :--: | :--: |
| MOTIP  |  ***no***  |  *yes*   | [pre-train](https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_coco_sportsmot.pth) \| [config](../configs/r50_deformable_detr_motip_sportsmot.yaml) \| [checkpoint](https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_motip_sportsmot.pth) | 72.6 | 83.5 | 63.2 |
| MOTIP  |  ***no***  |   *no*   | [pre-train](https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_coco_sportsmot.pth) \| [config](../configs/r50_deformable_detr_motip_sportsmot_without_trajectory_augmentation.yaml) \| [checkpoint](https://github.com/MCG-NJU/MOTIP/releases/download/v0.2/r50_deformable_detr_motip_sportsmot_without_trajectory_augmentation.pth) | 70.9 | 83.7 | 60.1 |

### BFT

| Method | Extra Data | Traj Aug |                          Resources                           | HOTA | DetA | AssA |
| :----: | :--------: | :------: | :----------------------------------------------------------: | :--: | :--: | :--: |
| MOTIP  |  ***no***  |  *yes*   | [pre-train](https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_coco_bft.pth) \| [config](../configs/r50_deformable_detr_motip_bft.yaml) \| [checkpoint](https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_motip_bft.pth) | 70.5 | 69.6 | 71.8 |
| MOTIP  |  ***no***  |   *no*   | [pre-train](https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_coco_bft.pth) \| [config](../configs/r50_deformable_detr_motip_bft_without_trajectory_augmentation.yaml) \| [checkpoint](https://github.com/MCG-NJU/MOTIP/releases/download/v0.2/r50_deformable_detr_motip_bft_without_trajectory_augmentation.pth) | 71.3 | 69.2 | 73.7 |

***NOTE:***

1. *Traj Aug* is an abbreviation for *Trajectory Augmentation* in the paper.
1. You could also load previous checkpoints for inference from [prev-engine branch](https://github.com/MCG-NJU/MOTIP/tree/prev-engine), using runtime parameter `--use-previous-checkpoint True`. You may need to pass additional parameters to bridge the difference in the experimental setups. Typically, `--rel-pe-length` and `--miss-tolerance`.
1. We present some experimental results not included in the paper, which we plan to discuss in the extended version of the article :soon:.

## DETR

You can directly download the pre-trained DETR weights used in our experiment here **(recommended)**. Or you can choose to follow the [guidance](./GET_STARTED.md) to perform pre-training yourself.

|             Model Name              | Target Dataset | Extra Data |                          Resources                           |
| :---------------------------------: | :------------: | :--------: | :----------------------------------------------------------: |
|      r50_deformable_detr_coco       |      COCO      |  ***no***  | [checkpoint](https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_coco.pth) |
| r50_deformable_detr_coco_dancetrack |   DanceTrack   |  ***no***  | [config](../configs/pretrain_r50_deformable_detr_dancetrack.yaml) \| [checkpoint](https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_coco_dancetrack.pth) |
| r50_deformable_detr_coco_sportsmot  |   SportsMOT    |  ***no***  | [config](../configs/pretrain_r50_deformable_detr_sportsmot.yaml) \| [checkpoint](https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_coco_sportsmot.pth) |
|    r50_deformable_detr_coco_bft     |      BFT       |  ***no***  | [config](../configs/pretrain_r50_deformable_detr_bft.yaml) \| [checkpoint](https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_coco_bft.pth) |



