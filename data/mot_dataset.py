# Copyright (c) Ruopeng Gao. All Rights Reserved.
# ------------------------------------------------------------------------
import os
import torch
import random
import data.transforms as T

from collections import defaultdict
from torch.utils.data import Dataset
from math import floor
from random import randint
from PIL import Image

import pandas as pd
import pycocotools.mask as mask_util
import numpy as np


class MOTDataset(Dataset):
    """
    Unified API for all MOT Datasets.
    """
    def __init__(self, config: dict):
        assert len(config["DATASETS"]) == len(config["DATASET_SPLITS"]), f"Get {len(config['DATASETS'])} datasets" \
                                                                         f"but {len(config['DATASET_SPLITS'])} splits."
        # Unified random state:
        multi_random_state = random.getstate()
        random.seed(config["SEED"])
        self.unified_random_state = random.getstate()
        random.setstate(multi_random_state)
        # Data path configs:
        self.data_root = config["DATA_ROOT"]
        # Data sampling setting:
        # Overall setting:
        self.sample_steps = config["SAMPLE_STEPS"]
        self.sample_lengths = config["SAMPLE_LENGTHS"]
        self.sample_modes = config["SAMPLE_MODES"]
        self.sample_intervals = config["SAMPLE_INTERVALS"]
        steps_len = len(self.sample_steps)
        self.sample_lengths = self.sample_lengths + self.sample_lengths[-1:] * (steps_len - len(self.sample_lengths))
        self.sample_modes = self.sample_modes + self.sample_modes[-1:] * (steps_len - len(self.sample_modes))
        self.sample_intervals = self.sample_intervals \
            + self.sample_intervals[-1:] * (steps_len - len(self.sample_intervals))
        assert len(self.sample_steps) == len(self.sample_lengths) \
               == len(self.sample_modes) == len(self.sample_intervals), f"Sampling setting varies in length."
        # Current setting:
        self.sample_length = None
        self.sample_mode = None
        self.sample_interval = None
        # Dataset structures:
        self.datasets = [
            self.get_dataset_structure(dataset=config["DATASETS"][_], split=config["DATASET_SPLITS"][_])
            for _ in range(len(config["DATASETS"]))
        ]
        if "DATASET_WEIGHTS" in config:
            self.dataset_weights = defaultdict(lambda: defaultdict(float))
            assert len(config["DATASETS"]) == len(config["DATASET_SPLITS"]) == len(config["DATASET_WEIGHTS"])
            for _ in range(len(config["DATASETS"])):
                self.dataset_weights[config["DATASETS"][_]][config["DATASET_SPLITS"][_]] = config["DATASET_WEIGHTS"][_]
                pass
        else:
            self.dataset_weights = None
        # Dataset infos, key just like: [DanceTrack][train][seq][frame], value is {image_path: _, gt: [_, _, ...]}
        # moreover, gts format is [frame, id, label, visibility, x_left, y_top, w, h]
        self.infos = self.get_dataset_infos()
        # Begin frames details, in format: [dataset, split, seq, frame] in tuple, for subsequent sampling step:
        self.sample_frames_begin = []
        # Default initialize:
        self.set_epoch(epoch=0)
        # Get augmentation transforms
        self.transforms = self.get_transforms(config=config)
        pass

    def __len__(self):
        return len(self.sample_frames_begin)

    def __getitem__(self, item):
        dataset, split, seq, begin = self.sample_frames_begin[item]
        frames_idx = self.sample_frames_idx(dataset=dataset, split=split, seq=seq, begin=begin)
        images, infos = self.get_multi_frames(dataset=dataset, split=split, seq=seq, frames=frames_idx)
        if self.transforms is not None:
            if infos[0]["dataset"] in ["CrowdHuman"]:   # static images
                images, infos = self.transforms["static"](images, infos)
            else:
                images, infos = self.transforms["video"](images, infos)
        assert all([len(info["boxes"]) > 0 for info in infos])
        return {
            # "images": stacked_images,
            "images": images,
            "infos": infos
        }

    def get_dataset_structure(self, dataset: str, split: str):
        dataset_dir = os.path.join(self.data_root, dataset)
        structure = {"dataset": dataset, "split": split}
        if dataset == "DanceTrack" or dataset == "SportsMOT":
            split_dir = os.path.join(dataset_dir, split)
            seq_names = os.listdir(split_dir)
            structure["seqs"] = {
                seq: {
                    "images_dir": os.path.join(split_dir, seq, "img1"),
                    "gt_path": os.path.join(split_dir, seq, "gt", "gt.txt"),
                    "images_name": os.listdir(os.path.join(split_dir, seq, "img1")),
                    "max_frame": max([int(_[:-4]) for _ in os.listdir(os.path.join(split_dir, seq, "img1"))])
                }
                for seq in seq_names
            }
        elif dataset == "CrowdHuman":
            split_dir = os.path.join(dataset_dir, split)
            seq_names = os.listdir(os.path.join(split_dir, "images"))
            seq_names = [_[:-4] for _ in seq_names]
            structure["seqs"] = {
                seq: {
                    "image_path": os.path.join(split_dir, "images", f"{seq}.jpg"),
                    "gt_path": os.path.join(split_dir, "gts", f"{seq}.txt"),
                }
                for seq in seq_names
            }
        elif dataset == "MOT17" or dataset == "MOT17_SPLIT":
            split_dir = os.path.join(dataset_dir, split)
            seq_names = os.listdir(split_dir)
            structure["seqs"] = {
                seq: {
                    "images_dir": os.path.join(split_dir, seq, "img1"),
                    "gt_path": os.path.join(split_dir, seq, "gt", "format_gt.txt"),
                    "images_name": os.listdir(os.path.join(split_dir, seq, "img1")),
                    "max_frame": max([int(_[:-4]) for _ in os.listdir(os.path.join(split_dir, seq, "img1"))])
                }
                for seq in seq_names
            }
        elif dataset == "MOT15_V2":
            split_dir = os.path.join(dataset_dir, split)
            seq_names = os.listdir(split_dir)
            structure["seqs"] = {
                seq: {
                    "images_dir": os.path.join(split_dir, seq, "img1"),
                    "gt_path": os.path.join(split_dir, seq, "gt", "gt.txt"),
                    "images_name": os.listdir(os.path.join(split_dir, seq, "img1")),
                    "max_frame": max([int(_[:-4]) for _ in os.listdir(os.path.join(split_dir, seq, "img1"))])
                }
                for seq in seq_names
            }
        else:
            raise NotImplementedError(f"Do not support dataset '{dataset}'.")
        return structure

    def get_dataset_infos(self):
        infos = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
        for dataset in self.datasets:
            seqs = dataset["seqs"]
            dataset_name = dataset["dataset"]
            for seq_name, seq in seqs.items():
                if "images_name" in seq:    # for true sequence
                    for frame in seq["images_name"]:
                        infos[dataset_name][dataset["split"]][seq_name][int(frame[:-4])]["image_path"] \
                            = os.path.join(seq["images_dir"], frame)
                        infos[dataset_name][dataset["split"]][seq_name][int(frame[:-4])]["gts"] = []
                else:                       # for a static image
                    infos[dataset_name][dataset["split"]][seq_name][0]["image_path"] = seq["image_path"]
                    infos[dataset_name][dataset["split"]][seq_name][0]["gts"] = []
                # Prepare GTs for different frames:
                gt_path = seq["gt_path"] if "gt_path" in seq else None
                gts_dir = seq["gts_dir"] if "gts_dir" in seq else None
                if gt_path is not None:
                    with open(gt_path, "r") as gt_file:
                        for line in gt_file:
                            line = line[:-1]
                            if dataset_name == "DanceTrack" or dataset_name == "SportsMOT":
                                # [frame, id, x, y, w, h, 1, 1, 1]
                                f, i, x, y, w, h, _, _, _ = line.split(",")
                                label = 0
                                v = 1
                            elif dataset_name == "MOT17" or dataset_name == "MOT17_SPLIT":
                                f, i, x, y, w, h, v = line.split(" ")
                                label = 0
                            elif dataset_name == "CrowdHuman":
                                f, i, x, y, w, h = line.split(" ")
                                label = 0
                                v = 1
                            else:
                                raise NotImplementedError(f"Can't analysis the gts of dataset '{dataset_name}'.")
                            # format, and write into infos
                            f, i, label = map(int, (f, i, label))
                            x, y, w, h, v = map(float, (x, y, w, h, v))
                            # assert v != 0.0, f"Visibility of object '{i}' in frame '{f}' is 0.0."
                            infos[dataset_name][dataset["split"]][seq_name][f]["gts"].append([
                                f, i, label, v, x, y, w, h
                            ])
                            pass
                else:
                    assert 0
                pass
            pass
        return infos

    def set_epoch(self, epoch: int):
        self.sample_frames_begin = []   # empty it
        for _ in range(len(self.sample_steps)):
            if epoch >= self.sample_steps[_]:
                self.sample_mode = self.sample_modes[_]
                self.sample_length = self.sample_lengths[_]
                self.sample_interval = self.sample_intervals[_]
                break

        for dataset in self.datasets:
            for seq in dataset["seqs"]:
                if dataset["dataset"] in ["CrowdHuman"]:    # keep all frames, since they are static images:
                    if self.dataset_weights is None:
                        self.sample_frames_begin.append(
                            (dataset["dataset"], dataset["split"], seq, 0)
                        )
                    else:
                        for _ in range(int(self.dataset_weights[dataset["dataset"]][dataset["split"]])):
                            self.sample_frames_begin.append(
                                (dataset["dataset"], dataset["split"], seq, 0)
                            )
                else:                                       # real video:
                    f_min = int(min(self.infos[dataset["dataset"]][dataset["split"]][seq].keys()))
                    f_max = int(max(self.infos[dataset["dataset"]][dataset["split"]][seq].keys()))
                    for f in range(f_min, f_max - (self.sample_length - 1) + 1):
                        if all([len(self.infos[dataset["dataset"]][dataset["split"]][seq][f + _]["gts"]) > 0
                                for _ in range(self.sample_length)]):   # make sure at least a legal seq with gts:
                            if self.dataset_weights is None:
                                self.sample_frames_begin.append(
                                    (dataset["dataset"], dataset["split"], seq, f)
                                )
                            else:
                                weight = self.dataset_weights[dataset["dataset"]][dataset["split"]]
                                # if isinstance(weight, int):
                                if weight >= 1.0:
                                    assert weight == int(weight), f"Weight '{weight}' is not an integer."
                                    weight = int(weight)
                                    for _ in range(weight):
                                        self.sample_frames_begin.append(
                                            (dataset["dataset"], dataset["split"], seq, f)
                                        )
                                elif isinstance(weight, float) and weight <= 1.0:
                                    multi_random_state = random.getstate()
                                    random.setstate(self.unified_random_state)
                                    if random.random() < weight:
                                        self.sample_frames_begin.append(
                                            (dataset["dataset"], dataset["split"], seq, f)
                                        )
                                    self.unified_random_state = random.getstate()
                                    random.setstate(multi_random_state)
                                else:
                                    raise NotImplementedError(f"Do not support dataset weight '{weight}'.")
        return

    def sample_frames_idx(self, dataset: str, split: str, seq: str, begin: int) -> list[int]:
        if self.sample_mode == "random_interval":
            if dataset in ["CrowdHuman"]:       # static images, repeat is all right:
                return [begin] * self.sample_length
            elif self.sample_length == 1:       # only train detection:
                return [begin]
            else:                               # real video, do something to sample:
                remain_frames = int(max(self.infos[dataset][split][seq].keys())) - begin
                max_interval = floor(remain_frames / (self.sample_length - 1))
                interval = min(randint(1, self.sample_interval), max_interval)      # legal interval
                frames_idx = [begin + interval * _ for _ in range(self.sample_length)]
                if not all([len(self.infos[dataset][split][seq][_]["gts"]) for _ in frames_idx]):
                    # In the sampling sequence, there is at least a frame's gt is empty, not friendly for training,
                    # make sure all frames have gt:
                    frames_idx = [begin + _ for _ in range(self.sample_length)]
        else:
            raise NotImplementedError(f"Do not support sample mode '{self.sample_mode}'.")
        return frames_idx

    def get_multi_frames(self, dataset: str, split: str, seq: str, frames: list[int]):
        return zip(*[self.get_single_frame(dataset=dataset, split=split, seq=seq, frame=frame) for frame in frames])

    def get_single_frame(self, dataset: str, split: str, seq: str, frame: int):
        image = Image.open(self.infos[dataset][split][seq][frame]["image_path"])
        info = dict()
        # Details about current image:
        info["image_path"] = self.infos[dataset][split][seq][frame]["image_path"]
        info["dataset"] = dataset
        info["split"] = split
        info["seq"] = seq
        info["frame"] = frame
        info["ori_width"], info["ori_height"] = image.size
        # GTs for current image:
        boxes, ids, labels, areas = list(), list(), list(), list()
        for _, i, label, _, x, y, w, h in self.infos[dataset][split][seq][frame]["gts"]:
            boxes.append([x, y, w, h])
            areas.append(w * h)
            ids.append(i)
            labels.append(label)
        assert len(boxes) == len(areas) == len(ids) == len(labels), f"GT for [{dataset}][{split}][{seq}][{frame}], " \
                                                                    f"different attributes have different length."
        assert len(boxes) > 0, f"GT for [{dataset}][{split}][{seq}][{frame}] is empty."
        info["boxes"] = torch.as_tensor(boxes, dtype=torch.float)   # in format [x, y, w, h]
        info["areas"] = torch.as_tensor(areas, dtype=torch.float)
        info["ids"] = torch.as_tensor(ids, dtype=torch.long)
        info["labels"] = torch.as_tensor(labels, dtype=torch.long)
        # Change boxes' format into [x1, y1, x2, y2]
        info["boxes"][:, 2:] += info["boxes"][:, :2]

        return image, info

    @staticmethod
    def get_transforms(config: dict):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        use_color_jitter_v2 = False if "AUG_COLOR_JITTER_V2" not in config else config["AUG_COLOR_JITTER_V2"]
        return {
            "video": T.MultiCompose([
                        T.MultiRandomHorizontalFlip(),
                        T.MultiRandomSelect(
                            T.MultiRandomResize(sizes=config["AUG_RESIZE_SCALES"], max_size=config["AUG_MAX_SIZE"]),
                            T.MultiCompose([
                                T.MultiRandomResize(sizes=config["AUG_RANDOM_RESIZE"]),
                                T.MultiRandomCrop(
                                    min_size=config["AUG_RANDOM_CROP_MIN"],
                                    max_size=config["AUG_RANDOM_CROP_MAX"],
                                    overflow_bbox=config["AUG_OVERFLOW_BBOX"]
                                ),
                                T.MultiRandomResize(sizes=config["AUG_RESIZE_SCALES"], max_size=config["AUG_MAX_SIZE"])
                            ])
                        ),
                        T.MultiColorJitter(
                            brightness=0.5,
                            contrast=0.5,
                            saturation=0.5,
                            hue=0.2,
                        ) if use_color_jitter_v2 else T.MultiHSV(),
                        T.MultiCompose([
                            T.MultiToTensor(),
                            T.MultiNormalize(mean=mean, std=std)
                        ]),
                        T.MultiReverseClip(reverse_clip=config["AUG_REVERSE_CLIP"])
                    ]),
            "static": T.MultiCompose([
                        T.MultiRandomHorizontalFlip(),
                        T.MultiRandomShift(config["AUG_RANDOM_SHIFT_MAX_RATIO"]),
                        T.MultiRandomSelect(
                            T.MultiRandomResize(sizes=config["AUG_RESIZE_SCALES"], max_size=config["AUG_MAX_SIZE"]),
                            T.MultiCompose([
                                T.MultiRandomResize(sizes=config["AUG_RANDOM_RESIZE"]),
                                T.MultiRandomCrop(
                                    min_size=config["AUG_RANDOM_CROP_MIN"],
                                    max_size=config["AUG_RANDOM_CROP_MAX"],
                                    overflow_bbox=config["AUG_OVERFLOW_BBOX"]
                                ),
                                T.MultiRandomResize(sizes=config["AUG_RESIZE_SCALES"], max_size=config["AUG_MAX_SIZE"])
                            ])
                        ),
                        T.MultiColorJitter(
                            brightness=0.5,
                            contrast=0.5,
                            saturation=0.5,
                            hue=0.2,
                        ) if use_color_jitter_v2 else T.MultiHSV(),
                        T.MultiCompose([
                            T.MultiToTensor(),
                            T.MultiNormalize(mean=mean, std=std)
                        ])
                    ])
        }


def build(config: dict) -> MOTDataset:
    return MOTDataset(
        config=config
    )
