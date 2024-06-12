# Pretrain DETR

To expedite the training process, we’ll begin by pretraining the DETR component of the model. Typically, training the DETR model on a specific dataset (like DanceTrack, SportsMOT, etc.) doesn’t take too long (approximately a few hours).

## DETR Weights on COCO

:floppy_disk: Similar to many other methods (e.g., MOTR and MeMOTR), we also use COCO pretrained DETR weights for initialization. You can obtain them from the following link:

- Deformable DETR: [[official repo](https://github.com/fundamentalvision/Deformable-DETR)] [[our drive](https://drive.google.com/file/d/1cgoBIn-jHrEifWOJ4UZyQr93jeTGI-mA/view?usp=drive_link)]
- DAB-Deformable DETR: [[official repo](https://github.com/IDEA-Research/DAB-DETR)]

## Pretrain DETR on Specific Datasets

### Scripts

:construction: We’ll upload the scripts for pretraining DETR on specific datasets in the future. For now, you can directly [download](https://drive.google.com/drive/folders/1O1HUxJJaDBORG6XEBk2QcWeXKqAblbxa?usp=drive_link) the pretrained DETR weights that are already available.

### Pretrained Weights

:floppy_disk: You can directly download the pretrained DETR weights we used in our experiments from [Google Drive :cloud:](https://drive.google.com/drive/folders/1O1HUxJJaDBORG6XEBk2QcWeXKqAblbxa?usp=drive_link). Then put them into `./pretrains/` directory.