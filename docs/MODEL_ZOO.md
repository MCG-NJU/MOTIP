# Model Zoo

## MOTIP

### DanceTrack

| Method | Extra Data | Traj Aug |                          Resources                           | HOTA | DetA | AssA |
| :----: | :--------: | :------: | :----------------------------------------------------------: | :--: | :--: | :--: |
| MOTIP  |  ***no***  |  *yes*   | [pre-train](https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_coco_dancetrack.pth) \| [config](../configs/r50_deformable_detr_motip_dancetrack.yaml) \| [checkpoint](https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_motip_dancetrack.pth) | 69.6 | 80.4 | 60.4 |

### SportsMOT

| Method | Extra Data | Traj Aug |                          Resources                           | HOTA | DetA | AssA |
| :----: | :--------: | :------: | :----------------------------------------------------------: | :--: | :--: | :--: |
| MOTIP  |  ***no***  |  *yes*   | [pre-train](https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_coco_sportsmot.pth) \| [config](../configs/r50_deformable_detr_motip_sportsmot.yaml) \| [checkpoint](https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_motip_sportsmot.pth) | 72.6 | 83.5 | 63.2 |

### BFT

| Method | Extra Data | Traj Aug |                          Resources                           | HOTA | DetA | AssA |
| :----: | :--------: | :------: | :----------------------------------------------------------: | :--: | :--: | :--: |
| MOTIP  |  ***no***  |  *yes*   | [pre-train](https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_coco_bft.pth) \| [config](../configs/r50_deformable_detr_motip_bft.yaml) \| [checkpoint](https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_motip_bft.pth) | 70.5 | 69.6 | 71.8 |

***NOTE:***

1. *Traj Aug* is an abbreviation for *Trajectory Augmentation* in the paper.

## DETR

You can directly download the pre-trained DETR weights used in our experiment here **(recommended)**. Or you can choose to follow the [guidance](./GET_STARTED.md) to perform pre-training yourself.

|             Model Name              | Target Dataset | Extra Data |                          Resources                           |
| :---------------------------------: | :------------: | :--------: | :----------------------------------------------------------: |
|      r50_deformable_detr_coco       |      COCO      |  ***no***  | [checkpoint](https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_coco.pth) |
| r50_deformable_detr_coco_dancetrack |   DanceTrack   |  ***no***  | [config](../configs/pretrain_r50_deformable_detr_dancetrack.yaml) \| [checkpoint](https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_coco_dancetrack.pth) |
| r50_deformable_detr_coco_sportsmot  |   SportsMOT    |  ***no***  | [config](../configs/pretrain_r50_deformable_detr_sportsmot.yaml) \| [checkpoint](https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_coco_sportsmot.pth) |
|    r50_deformable_detr_coco_bft     |      BFT       |  ***no***  | [config](../configs/pretrain_r50_deformable_detr_bft.yaml) \| [checkpoint](https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_coco_bft.pth) |



