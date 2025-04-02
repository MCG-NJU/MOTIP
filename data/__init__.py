# Copyright (c) Ruopeng Gao. All Rights Reserved.

from torch.utils.data import DataLoader

from .joint_dataset import JointDataset
from .transforms import build_transforms
from .util import collate_fn


def build_dataset(config: dict):
    return JointDataset(
        data_root=config["DATA_ROOT"],
        datasets=config["DATASETS"],
        splits=config["DATASET_SPLITS"],
        transforms=build_transforms(config),
        size_divisibility=config.get("SIZE_DIVISIBILITY", 0),
    )


def build_dataloader(dataset, batch_size: int, shuffle: bool, num_workers: int):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
