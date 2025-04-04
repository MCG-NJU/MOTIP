# Copyright (c) Ruopeng Gao. All Rights Reserved.

import torch
from PIL import Image
from torchvision.transforms import v2

from torch.utils.data import Dataset
from utils.nested_tensor import nested_tensor_from_tensor_list


class SeqDataset(Dataset):
    def __init__(
            self,
            seq_info,
            image_paths,
            max_shorter: int = 800,
            max_longer: int = 1536,
            size_divisibility: int = 0,
            dtype=torch.float32,
    ):
        self.seq_info = seq_info
        self.image_paths = image_paths
        self.max_shorter = max_shorter
        self.max_longer = max_longer
        self.size_divisibility = size_divisibility
        self.dtype = dtype

        self.transform = v2.Compose([
            v2.Resize(size=self.max_shorter, max_size=self.max_longer),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = self._load(self.image_paths[item])
        transformed_image = self.transform(image)
        if self.dtype != torch.float32:
            transformed_image = transformed_image.to(self.dtype)
        transformed_image = nested_tensor_from_tensor_list([transformed_image], self.size_divisibility)
        return transformed_image, self.image_paths[item]

    def seq_hw(self):
        return self.seq_info["height"], self.seq_info["width"]

    @staticmethod
    def _load(path):
        image = Image.open(path)
        return image
