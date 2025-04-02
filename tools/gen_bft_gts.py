# Copyright (c) Ruopeng Gao. All Rights Reserved.
# About: This script is used to generate the ground truth of the BFT dataset.

import os
from shutil import move, copy
from PIL import Image
from configparser import ConfigParser


def mv_files(root: str, split: str):
    seq_names = os.listdir(os.path.join(root, split))
    for seq_name in seq_names:
        # 1. organize the image files in a folder "img1":
        img_names = os.listdir(os.path.join(root, split, seq_name))
        os.makedirs(os.path.join(root, split, seq_name, "img1"), exist_ok=True)
        for img_name in img_names:
            move(os.path.join(root, split, seq_name, img_name), os.path.join(root, split, seq_name, "img1", img_name))
        # 2. move the gt files:
        gt_path = os.path.join(root, "annotations_mot", split, seq_name + ".txt")
        os.makedirs(os.path.join(root, split, seq_name, "gt"), exist_ok=True)
        copy(gt_path, os.path.join(root, split, seq_name, "gt", "gt.txt"))
    return


def reformat_gts(root: str, split: str):
    seq_names = os.listdir(os.path.join(root, split))
    for seq_name in seq_names:
        gt_path = os.path.join(root, split, seq_name, "gt", "gt.txt")
        with open(gt_path, "r") as f:
            lines = f.readlines()
        os.remove(gt_path)
        with open(gt_path, "w") as f:
            for line in lines:
                line = line.strip().split(",")
                f.write(f"{int(line[0])},{int(line[1])},{int(line[2])},{int(line[3])},{int(line[4])},{int(line[5])},1,1,1\n")
    pass
    return


def gen_seqmap(root: str, split: str):
    seq_names = os.listdir(os.path.join(root, split))
    with open(os.path.join(root, f"{split}_seqmap.txt"), "w") as f:
        f.write("name")
        for seq_name in seq_names:
            f.write(f"\n{seq_name}")
    return


def gen_ini_files(root: str, split: str):
    seq_names = os.listdir(os.path.join(root, split))
    for seq_name in seq_names:
        img_files = os.listdir(os.path.join(root, split, seq_name, "img1"))
        seq_length = len(img_files)
        an_image = Image.open(os.path.join(root, split, seq_name, "img1", img_files[0]))
        seq_width, seq_height = an_image.size
        ini_config = ConfigParser()
        ini_config.add_section("Sequence")
        ini_config.set("Sequence", "name", seq_name)
        ini_config.set("Sequence", "imDir", "img1")
        ini_config.set("Sequence", "frameRate", "25")
        ini_config.set("Sequence", "seqLength", str(seq_length))
        ini_config.set("Sequence", "imWidth", str(seq_width))
        ini_config.set("Sequence", "imHeight", str(seq_height))
        ini_config.set("Sequence", "imExt", ".jpg")
        with open(os.path.join(root, split, seq_name, "seqinfo.ini"), "w") as f:
            ini_config.write(f)
        pass



if __name__ == '__main__':
    bft_root = "../datasets/BFT/"
    bft_splits = ["train", "val", "test"]

    for bft_split in bft_splits:
        mv_files(bft_root, bft_split)
        print(f"Done with {bft_split} set.")
    for bft_split in bft_splits:
        reformat_gts(bft_root, bft_split)
        print(f"Done with {bft_split} set.")
    for bft_split in bft_splits:
        gen_seqmap(bft_root, bft_split)
        print(f"Done with {bft_split} set.")
    for bft_split in bft_splits:
        gen_ini_files(bft_root, bft_split)
        print(f"Done with {bft_split} set.")

