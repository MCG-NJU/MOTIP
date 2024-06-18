# Data Preparation

:link: For all the datasets we used in our experiments, you can access them from the following public link:

- [DanceTrack](https://github.com/DanceTrack/DanceTrack)
- [SportsMOT](https://github.com/MCG-NJU/SportsMOT)
- [MOT17](https://motchallenge.net/data/MOT17/)
- [CrowdHuman](https://www.crowdhuman.org/)

## Generate GT files

For the MOT17 and CrowdHuman datasets, you’ll need to use the provided script to convert their ground truth files to the format we require:

- For MOT17: [gen_mot17_gts.py](../data/gen_mot17_gts.py)
- For CrowdHuman: [gen_crowdhuman_gts.py](../data/gen_crowdhuman_gts.py)

:pushpin: You need to modify the paths in the script according to your requirements.

## File Tree

```text
<DATADIR>/
  ├── DanceTrack/
  │ ├── train/
  │ ├── val/
  │ ├── test/
  │ ├── train_seqmap.txt
  │ ├── val_seqmap.txt
  │ └── test_seqmap.txt
  ├── SportsMOT/
  │ ├── train/
  │ ├── val/
  │ ├── test/
  │ ├── train_seqmap.txt
  │ ├── val_seqmap.txt
  │ └── test_seqmap.txt
  ├── MOT17/
  │ ├── images/
  │ │ ├── train/     # unzip from MOT17, you need to delete some files, see the below the instructions
  │ │ └── test/      # unzip from MOT17
  │ └── gts/
  │   └── train/     # generate by ./data/gen_mot17_gts.py
  └── CrowdHuman/
    ├── images/
    │ ├── train/     # unzip from CrowdHuman
    │ └── val/       # unzip from CrowdHuman
    └── gts/
      ├── train/     # generate by ./data/gen_crowdhuman_gts.py
      └── val/       # generate by ./data/gen_crowdhuman_gts.py
```

:warning: Since each video sequence in MOT17 is stored three times, for each training video sequence, you should delete the other two sequences to achieve deduplication. For instance, in my experiment, I only retained the ‘xxx-DPM’ sequences.”

## Q & A

- Q: Lack the `val_seqmap.txt` file of SportsMOT? </br>
  A: Refer to [The 'val_seqmap.txt' file of SportsMOT dataset · Issue #13](https://github.com/MCG-NJU/MOTIP/issues/13)