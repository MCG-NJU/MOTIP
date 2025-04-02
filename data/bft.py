# Copyright (c) Ruopeng Gao. All Rights Reserved.

import os
from PIL import Image

from .dancetrack import DanceTrack


class BFT(DanceTrack):
    def __init__(
            self,
            data_root: str = "./datasets/",
            sub_dir: str = "BFT",
            split: str = "train",
            load_annotation: bool = True,
    ):
        super(BFT, self).__init__(
            data_root=data_root,
            sub_dir=sub_dir,
            split=split,
            load_annotation=load_annotation,
        )
        return

    @staticmethod
    def _get_image_path(sequence_dir, frame_idx):
        return os.path.join(sequence_dir, "img1", f"{frame_idx + 1:06d}.jpg")  # different from DanceTrack

    def _get_sequence_infos(self):
        sequence_names = self._get_sequence_names()
        sequence_infos = dict()
        for sequence_name in sequence_names:
            sequence_dir = self._get_sequence_dir(self.data_dir, self.split, sequence_name)
            img_names = os.listdir(os.path.join(sequence_dir, "img1"))
            # Get the length of the sequence:
            seq_len = len(img_names)
            # Get the width and height of the sequence:
            an_image = Image.open(os.path.join(sequence_dir, "img1", img_names[0]))
            width, height = an_image.size
            sequence_infos[sequence_name] = {
                "width": width,
                "height": height,
                "length": seq_len,
                "is_static": False,
            }
            pass
        return sequence_infos
