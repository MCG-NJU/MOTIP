import os

import yaml
import torch
import torch.distributed
import random
import numpy as np
from accelerate import PartialState, DistributedType


accelerate_state = PartialState()


def set_seed(seed: int):
    seed = seed + distributed_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return


def is_distributed():
    return not (accelerate_state.distributed_type == DistributedType.NO)


def distributed_rank():
    return accelerate_state.process_index


def is_main_process():
    return accelerate_state.is_main_process


def distributed_world_size():
    return accelerate_state.num_processes

def distributed_device():
    return accelerate_state.device


def yaml_to_dict(path: str):
    """
    Read a yaml file into a dict.

    Args:
        path (str): The path of yaml file.

    Returns:
        A dict.
    """
    with open(path) as f:
        return yaml.load(f.read(), yaml.FullLoader)


def labels_to_one_hot(labels: np.ndarray, class_num: int):
    """
    Args:
        labels: Original labels.
        class_num:

    Returns:
        Labels in one-hot.
    """
    labels = labels.cpu()
    return np.eye(N=class_num)[labels].reshape((len(labels), -1))
    # return np.eye(N=class_num)[labels]


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


