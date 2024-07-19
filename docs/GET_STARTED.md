# Get Started

In this documentation, we will primarily focus on training and inference of our MOTIP model on the relevant MOT benchmarks.

## Training

In the `./configs/` directory, we provide various configuration files. You can use different configuration files to train the MOTIP model with different settings on various datasets.

### Default Scripts

**All our training scripts follow the template script below.** You'll need to fill the `<placeholders>` according to your requirements:

```shell
python -m torch.distributed.run --nproc_per_node=8 main.py --mode train --use-distributed True --use-wandb False --config-path <config file path> --data-root <DATADIR> --outputs-dir <outputs dir>
```

For example, you can the default model on DanceTrack as follows:

```shell
python -m torch.distributed.run --nproc_per_node=8 main.py --mode train --use-distributed True --use-wandb False --config-path ./configs/r50_deformable_detr_motip_dancetrack.yaml --data-root ./datasets/ --outputs-dir ./outputs/r50_deformable_detr_motip_dancetrack/
```

Using this script, you can achieve 66.2 ~ 67.6 HOTA on DanceTrack test set. This relatively high instability (~ 1.5) is also encountered in other work (e.g., [OC-SORT](https://github.com/noahcao/OC_SORT), [MOTRv2](https://github.com/megvii-research/MOTRv2/issues/2), [MeMOTR](https://github.com/MCG-NJU/MeMOTR/issues/17)). We suggest that part of the reason comes from the DanceTrack dataset itself, because the final performance on the MOT17 or SportsMOT test set will be more stable (~ 0.2 HOTA and ~ 0.5 HOTA).

### Gradient Checkpoint

If your GPUs have less than 24GB CUDA memory, we offer the gradient checkpoint technology. You can set `--detr-checkpoint-frames` to `2` (< 16GB) or `1` (< 12GB) to reduce the CUDA memory requirements.

## Inference

### Submitting

To evaluate on most MOT benchmarks, you’ll need to submit tracking result files to the official server. You can obtain the tracking output using the following script:

```shell
python -m torch.distributed.run --nproc_per_node=<gpu num> main.py --mode submit --use-distributed True --use-wandb False --config-path <config file path> --data-root <DATADIR> --inference-model <checkpoint path> --outputs-dir <outputs dir> --inference-dataset <dataset name> --inference-split <dataset split>
```

For example, you can submit the model on DanceTrack test set as follows:

```shell
python -m torch.distributed.run --nproc_per_node=8 main.py --mode submit --use-distributed True --use-wandb False --config-path ./configs/r50_deformable_detr_motip_dancetrack.yaml --data-root ./datasets/ --inference-model ./outputs/r50_deformable_detr_motip_dancetrack.pth --outputs-dir ./outputs/dancetrack_trackers/ --inference-dataset DanceTrack --inference-split test
```

### Evaluation

:construction: The evaluation scripts will be available in the future.