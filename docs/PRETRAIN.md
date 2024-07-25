# Pretrain DETR

To expedite the training process, we’ll begin by pretraining the DETR component of the model. Typically, training the DETR model on a specific dataset (like DanceTrack, SportsMOT, etc.) doesn’t take too long (approximately a few hours).

## DETR Weights on COCO

:floppy_disk: Similar to many other methods (e.g., MOTR and MeMOTR), we also use COCO pretrained DETR weights for initialization. You can obtain them from the following link:

- Deformable DETR: [[official repo](https://github.com/fundamentalvision/Deformable-DETR)] [[our drive](https://drive.google.com/file/d/1cgoBIn-jHrEifWOJ4UZyQr93jeTGI-mA/view?usp=drive_link)]
- DAB-Deformable DETR: [[official repo](https://github.com/IDEA-Research/DAB-DETR)]

## Pretrain DETR on Specific Datasets

### Scripts

**All our pre-train scripts follows the template script below.** You'll need to fill the `<placeholders>` according to your requirements:

```bash
python -m torch.distributed.run --nproc_per_node=8 main.py --mode train --use-distributed True --use-wandb False --config-path <config file path> --data-root <DATADIR> --outputs-dir <outputs dir>
```

For example, you can pre-train a Deformable-DETR model on DanceTrack as follows:

```bash
python -m torch.distributed.run --nproc_per_node=8 main.py --mode train --use-distributed True --use-wandb False --config-path ./configs/pretrain_r50_deformable_detr_dancetrack.yaml --data-root ./datasets/ --outputs-dir ./outputs/pretrain_r50_deformable_detr_dancetrack/
```

### Gradient Checkpoint

Please referring to [here](./GET_STARTED.md#gradient-checkpoint) to get more information.

### Pretrained Weights

:floppy_disk: You can directly download the pretrained DETR weights we used in our experiments from [Google Drive :cloud:](https://drive.google.com/drive/folders/1O1HUxJJaDBORG6XEBk2QcWeXKqAblbxa?usp=drive_link). Then put them into `./pretrains/` directory.