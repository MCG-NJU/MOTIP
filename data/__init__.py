# @Author       : Ruopeng Gao
# @Date         : 2022/7/6
# @Description  : Data operators, such as data read, dataset, dataloader.

"""
In this package, should include the main operators about dataset.
Mainly include below parts:
1. Use dataset API (For example, COCO API) or just simple code to read files.
2. Build a PyTorch Dataset.
3. Build a PyTorch DataLoader.
4. Maybe you should design a PyTorch Sampler class.
5. Probably you should design how to transform the data.
6. Sometimes, you should build a function 'collate_fn' to tell the dataloader how to aggregate a batch of data.

The above features can be achieved in a single .py file or multi of them.
"""
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader

from utils.utils import is_distributed
from .mot_dataset import build as build_mot_dataset
from .utils import collate_fn


def build_dataset(config: dict):
    return build_mot_dataset(config=config)


def build_sampler(dataset, shuffle: bool):
    if is_distributed():
        sampler = DistributedSampler(dataset=dataset, shuffle=shuffle)
    else:
        sampler = RandomSampler(dataset) if shuffle is True else SequentialSampler(dataset)
    return sampler


def build_dataloader(dataset, sampler, batch_size: int, num_workers: int):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
