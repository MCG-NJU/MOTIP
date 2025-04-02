# Copyright (c) Ruopeng Gao. All Rights Reserved.

import os
from collections import defaultdict

from .dancetrack import DanceTrack
from .util import append_annotation, is_legal
from PIL import Image


class CrowdHuman(DanceTrack):
    def __init__(
            self,
            data_root: str = "./datasets/",
            sub_dir: str = "CrowdHuman",
            split: str = "train",
            load_annotation: bool = True,
    ):
        super(CrowdHuman, self).__init__(
            data_root=data_root,
            sub_dir=sub_dir,
            split=split,
            load_annotation=load_annotation,
        )
        return

    def _get_sequence_infos(self):
        sequence_names = self._get_sequence_names()
        sequence_infos = dict()
        for sequence_name in sequence_names:
            sequence_image_dir = os.path.join(self.data_dir, self.split, "images")
            image_path = os.path.join(sequence_image_dir, f"{sequence_name}.jpg")
            image = Image.open(image_path)
            image_width, image_height = image.size
            sequence_infos[sequence_name] = {
                "width": image_width,
                "height": image_height,
                "length": 1,
                "is_static": True,
            }
        return sequence_infos

    def _get_image_paths(self):
        sequence_names = self._get_sequence_names()
        image_paths = defaultdict(list)
        for sequence_name in sequence_names:
            sequence_image_dir = os.path.join(self.data_dir, self.split, "images")
            for i in range(self.sequence_infos[sequence_name]["length"]):
                image_paths[sequence_name].append(os.path.join(sequence_image_dir, f"{sequence_name}.jpg"))
        return image_paths

    def _get_annotations(self):
        sequence_names = self._get_sequence_names()
        # Init the annotations:
        annotations = self._init_annotations(sequence_names)
        # Load the annotations:
        for sequence_name in sequence_names:
            sequence_gt_dir = os.path.join(self.data_dir, self.split, "gts")
            gt_file_path = os.path.join(sequence_gt_dir, f"{sequence_name}.txt")
            with open(gt_file_path, "r") as gt_file:
                for line in gt_file:
                    line = line.strip().split(" ")
                    frame_id, obj_id, x, y, w, h = line
                    frame_id, obj_id = map(int, [frame_id, obj_id])
                    x, y, w, h = map(float, [x, y, w, h])
                    bbox = [x, y, w, h]
                    category, visibility = 0, 1.0
                    ann_index = frame_id        # already 0-indexed for annotations
                    # Organized into the annotations:
                    annotations[sequence_name][ann_index] = append_annotation(
                        annotation=annotations[sequence_name][ann_index],
                        obj_id=obj_id,
                        category=category,
                        bbox=bbox,
                        visibility=visibility,
                    )
        # Determine whether each annotation is legal:
        for sequence_name in sequence_names:
            for i in range(self.sequence_infos[sequence_name]["length"]):
                annotations[sequence_name][i]["is_legal"] = is_legal(annotations[sequence_name][i])
        return annotations

    def _get_sequence_names(self):
        sequence_names = os.listdir(os.path.join(self.data_dir, self.split, "images"))
        for _ in range(len(sequence_names)):
            sequence_names[_] = sequence_names[_][:-4]
        return sequence_names
