# Get Started

In this documentation, we will primarily focus on training and inference of our MOTIP model on the relevant MOT benchmarks. All the configurations corresponding to our experiments are stored in the [configs](../configs/) folder. You can also customize the configuration files according to your own requirements.

## Pre-training

To expedite the training process, we’ll begin by pre-training the DETR component of the model. Typically, training the DETR model on a specific dataset (like DanceTrack, SportsMOT, etc.) is quite efficient, taking only a few hours.

### COCO Pre-trained Weights

:floppy_disk: ​Similar to many other methods (e.g., MOTR and MeMOTR), we also use COCO pre-trained DETR weights for initialization. You can obtain them from the following links:

- Deformable DETR: [[official repo](https://github.com/fundamentalvision/Deformable-DETR)] [[our repo](https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_coco.pth)]

### Pre-train DETR on Specific Datasets

To accelerate the convergence, we will first pre-train DETR on the corresponding dataset (target dataset) to serve as the initialization for subsequent MOTIP training.

#### Our Pre-trained Weights

:floppy_disk: **We recommend directly using our pre-trained DETR weights, which are stored in the [model zoo](./MODEL_ZOO.md#DETR).** If needed, you can pre-train it yourself using the script provided below.

You should put necessary pre-trained weights into `./pretrains/` directory as default.

#### Pre-training Scripts

**All our pre-train scripts follows the template script below.** You'll need to fill the `<placeholders>` according to your requirements:

```bash
accelerate launch --num_processes=8 train.py --data-root <data dir> --exp-name <exp name> --config-path <.yaml config file path>
```

For example, you can pre-train a Deformable-DETR model on DanceTrack as follows:

```bash
accelerate launch --num_processes=8 train.py --data-root ./datasets/ --exp-name pretrain_r50_deformable_detr_dancetrack --config-path ./configs/pretrain_r50_deformable_detr_dancetrack.yaml
```

#### Gradient Checkpoint

Please referring to [here](./GET_STARTED.md#gradient-checkpoint) to get more information.

## Training

Once you have the DETR pre-trained weights on the corresponding dataset (target dataset), you can use the following script to train your own MOTIP model.

### Training Scripts

**All our training scripts follow the template script below.** You'll need to fill the `<placeholders>` according to your requirements:

```shell
accelerate launch --num_processes=8 train.py --data-root <DATADIR> --exp-name <exp name> --config-path <.yaml config file path>
```

For example, you can the default model on DanceTrack as follows:

```shell
accelerate launch --num_processes=8 train.py --data-root ./datasets/ --exp-name r50_deformable_detr_motip_dancetrack --config-path ./configs/r50_deformable_detr_motip_dancetrack.yaml
```

*Using this script, you can achieve 69.5 HOTA on DanceTrack test set. There is a relatively high instability (~ 1.5) which is also encountered in other work (e.g., [OC-SORT](https://github.com/noahcao/OC_SORT), [MOTRv2](https://github.com/megvii-research/MOTRv2/issues/2), [MeMOTR](https://github.com/MCG-NJU/MeMOTR/issues/17)).*

### Gradient Checkpoint

If your GPUs have less than 24GB CUDA memory, we offer the gradient checkpoint technology. You can set `--detr-num-checkpoint-frames` to `2` (< 16GB) or `1` (< 12GB) to reduce the CUDA memory requirements.

## Inference

We have two different inference modes:

1. Without ground truth annotations (e.g. DanceTrack test, SportsMOT test), [submission scripts](#Submission) can generate tracker files for submission.
2. With ground truth annotations, [evaluation scripts](#Evaluation) can produce tracking results and obtain evaluation results.

:pushpin: **Different inference behaviors are controlled by the runtime parameter `--inference-mode`.**

### Submission

You can obtain the tracking results (tracker files) using the following **template script**:

```shell
accelerate launch --num_processes=8 submit_and_evaluate.py --data-root <DATADIR> --inference-mode submit --config-path <.yaml config file path> --inference-model <checkpoint path> --outputs-dir <outputs dir> --inference-dataset <dataset name> --inference-split <split name>
```

For example, you can get our default results on the DanceTrack test set as follows:

```shell
accelerate launch --num_processes=8 submit_and_evaluate.py --data-root ./datasets/ --inference-mode submit --config-path ./configs/r50_deformable_detr_motip_dancetrack.yaml --inference-model ./outputs/r50_deformable_detr_motip_dancetrack/r50_deformable_detr_motip_dancetrack.pth --outputs-dir ./outputs/r50_deformable_detr_motip_dancetrack/ --inference-dataset DanceTrack --inference-split test
```

:racing_car: You can add `--inference-dtype FP16` to the script to use float16 for inference. This can improve inference speed by over 30% with only a slight impact on tracking performance (about 0.5 HOTA on DanceTrack test).

### Evaluation

You can obtain both the tracking results (tracker files) and evaluation results using the following **template script**:

```shell
accelerate launch --num_processes=8 submit_and_evaluate.py --data-root <DATADIR> --inference-mode evaluate --config-path <.yaml config file path> --inference-model <checkpoint path> --outputs-dir <outputs dir> --inference-dataset <dataset name> --inference-split <split name>
```

For example, you can get the evaluation results on the DanceTrack val set as follows:

```shell
accelerate launch --num_processes=8 submit_and_evaluate.py --data-root ./datasets/ --inference-mode evaluate --config-path ./configs/r50_deformable_detr_motip_dancetrack.yaml --inference-model ./outputs/r50_deformable_detr_motip_dancetrack/r50_deformable_detr_motip_dancetrack.pth --outputs-dir ./outputs/r50_deformable_detr_motip_dancetrack/ --inference-dataset DanceTrack --inference-split val
```
