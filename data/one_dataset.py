# Copyright (c) Ruopeng Gao. All Rights Reserved.

import os


class OneDataset:
    def __init__(
            self,
            data_root: str = "./datasets/",
            sub_dir: str = "OneDataset",
            split: str = "train",
            load_annotation: bool = True,
    ):
        self.data_dir = os.path.join(data_root, sub_dir)
        self.split = split
        self.load_annotation = load_annotation

        # Null data:
        self.sequence_infos, self.image_paths, self.annotations = None, None, None
        return

    def get_sequence_infos(self):
        return self.sequence_infos

    def get_image_paths(self):
        return self.image_paths

    def get_annotations(self):
        if self.load_annotation:
            return self.annotations
        else:
            raise ValueError("Annotations are not loaded.")
