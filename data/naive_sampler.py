# Copyright (c) Ruopeng Gao. All Rights Reserved.

import random
import torch
from math import floor
from torch.utils.data.sampler import Sampler

from .joint_dataset import JointDataset


class NaiveSampler(Sampler):
    def __init__(
            self,
            data_source: JointDataset,
            sample_steps: list,
            sample_lengths: list,
            sample_intervals: list,
            length_per_iteration: int,
            sample_mode: str = "random_interval",
            seed: int = 1025,
            data_weights: dict | None = None,
    ):
        super().__init__()
        self.data_source = data_source
        self.sample_steps = sample_steps
        self.sample_lengths = sample_lengths
        self.sample_intervals = sample_intervals
        self.length_per_iteration = length_per_iteration
        self.sample_mode = sample_mode
        self.seed = seed
        self.data_weights = data_weights
        # Check for these parameters:
        assert len(sample_steps) == len(sample_lengths) == len(sample_intervals), \
            "The lengths of sample_steps, sample_lengths, and sample_intervals should be the same."
        assert sample_mode in ["random_interval"], \
            f"The sample_mode {sample_mode} is not supported."
        for sample_length in sample_lengths:
            if self.length_per_iteration is not None:
                assert sample_length % length_per_iteration == 0, \
                    f"The sample_length {sample_length} should be divisible by {length_per_iteration}."

        # Init the sampling details:
        self.sample_infos = None    # a list of dict: (dataset, split, sequence_name, frame_idxs)
        return

    def prepare_for_epoch(self, epoch: int):
        """
        Decide the sampling details for the current "epoch".
        It should be called at the beginning of each epoch.
        Args:
            epoch: The current epoch.
        """
        # Make sure the random process is the same for each process:
        general_random_state = random.getstate()
        random.seed(self.seed + epoch)      # only for sampling process.
        ############################################################
        sample_infos = []
        sample_length: int | None = None
        sample_interval: int | None = None
        # First, calculate the sample length and interval:
        for _ in range(len(self.sample_steps)):
            if epoch >= self.sample_steps[_]:
                sample_length = self.sample_lengths[_]
                sample_interval = self.sample_intervals[_]
        assert sample_length is not None and sample_interval is not None, \
            f"The epoch {epoch} is not supported."
        # Second, get all legal samplings:
        for dataset in self.data_source.annotations:
            for split in self.data_source.annotations[dataset]:
                # Get the weights of the current dataset and split:
                if self.data_weights is not None:
                    if dataset in self.data_weights and split in self.data_weights[dataset]:
                        _weight = self.data_weights[dataset][split]
                    else:
                        _weight = 1
                else:
                    _weight = 1
                assert isinstance(_weight, int), f"The weight {_weight} should be an integer."
                # TODO: Add support for float weights.
                # Sampling:
                for sequence_name in self.data_source.annotations[dataset][split]:
                    for frame_id in range(self.data_source.sequence_infos[dataset][split][sequence_name]["length"]):
                        _sample_times = _weight
                        for _ in range(_sample_times):
                            if self.data_source.sequence_infos[dataset][split][sequence_name]["is_static"] is True:
                                # static image
                                if self.data_source.ann_is_legals[dataset][split][sequence_name][0]:
                                    sample_infos.append({
                                        "dataset": dataset,
                                        "split": split,
                                        "sequence": sequence_name,
                                        "frame_idxs": [0] * sample_length,
                                    })
                                else:
                                    pass
                            else:   # real-world video
                                if frame_id + sample_length <= self.data_source.sequence_infos[dataset][split][sequence_name]["length"]:
                                    begin_index = frame_id
                                    remain_frames = self.data_source.sequence_infos[dataset][split][sequence_name]["length"] - begin_index - 1
                                    max_interval = floor(remain_frames / (sample_length - 1)) if sample_length > 1 \
                                        else 1
                                    interval = min(random.randint(1, sample_interval), max_interval)
                                    frame_idxs = [begin_index + interval * _ for _ in range(sample_length)]
                                    _ = torch.tensor(frame_idxs, dtype=torch.int64)
                                    if self.data_source.ann_is_legals[dataset][split][sequence_name][_].all():
                                        # make sure in the sampling sequence,
                                        # all frame's annotation is legal,
                                        # which is friendly for training.
                                        sample_infos.append({
                                            "dataset": dataset,
                                            "split": split,
                                            "sequence": sequence_name,
                                            "frame_idxs": frame_idxs,
                                        })
        total_len = len(sample_infos)
        # Shuffle the samples
        # and only keep the first "total_len // (sample_length / self.length_per_iteration)" samples:
        if self.length_per_iteration is None:
            kept_len = total_len
        else:
            kept_len = int(total_len // (sample_length // self.length_per_iteration))
        # These three lines are copied from the official PyTorch code:
        g = torch.Generator()
        g.manual_seed(self.seed + epoch)
        indices = torch.randperm(kept_len, generator=g).tolist()
        self.sample_infos = [sample_infos[_] for _ in indices]

        # Recover the random state:
        random.setstate(general_random_state)
        ###########################
        return

    def __iter__(self):
        return iter(self.sample_infos)

    def __len__(self):
        return len(self.sample_infos)




