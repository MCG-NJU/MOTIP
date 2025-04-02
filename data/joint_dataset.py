# Copyright (c) Ruopeng Gao. All Rights Reserved.
# No matter how many datasets you used in your code,
# you should always use JointDataset to combine and organize them (even if you only used one dataset).

import torch
from collections import defaultdict
from torch.utils.data import Dataset
from PIL import Image

from .dancetrack import DanceTrack
from .sportsmot import SportsMOT
from .crowdhuman import CrowdHuman
from .bft import BFT


dataset_classes = {
    "DanceTrack": DanceTrack,
    "SportsMOT": SportsMOT,
    "CrowdHuman": CrowdHuman,
    "BFT": BFT,
}


class JointDataset(Dataset):
    def __init__(
            self,
            data_root: str,
            datasets: list,
            splits: list,
            transforms=None,
            **kwargs,
    ):
        """
        Args:
            data_root: The root directory of datasets.
            datasets: The list of dataset names, e.g., ["DanceTrack", "SportsMOT"].
            splits: The list of (dataset) split names, e.g., ["train", "train"].
        """
        super().__init__()
        assert len(datasets) == len(splits), "The number of datasets and splits should be the same."
        self.transforms = transforms

        # Handle the parameters **kwargs:
        self.size_divisibility = kwargs.get("size_divisibility", 0)

        # Load the datasets into "sequence_infos", "image_paths", and "annotations",
        # each of which is a dictionary with the dataset name and split as the key.
        # e.g., sequence_infos["DanceTrack"]["train"]["sequence_name"] = {}.
        self.sequence_infos = defaultdict(lambda: defaultdict(dict))
        self.image_paths = defaultdict(lambda: defaultdict(dict))
        self.annotations = defaultdict(lambda: defaultdict(dict))
        for dataset, split in zip(datasets, splits):
            try:
                dataset_class = dataset_classes[dataset](
                    data_root=data_root,
                    split=split,
                    load_annotation=True,
                )
                self.sequence_infos[dataset][split] = dataset_class.get_sequence_infos()
                self.image_paths[dataset][split] = dataset_class.get_image_paths()
                self.annotations[dataset][split] = dataset_class.get_annotations()
            except KeyError:
                raise AttributeError(f"Dataset {dataset} is not supported.")
        # Decouple the 'is_legal' attribute from the annotations,
        # I believe it is more flexible to check the legality of the annotations in the sampling process.
        self.ann_is_legals = self._decouple_is_legal()

        # Init the sampling details:
        # Here, they are not ready for sampling,
        # you should call "self.set_sample_details()" to prepare them.
        self.sample_begins: list | None = None      # a tuple: (dataset, split, sequence_name, begin_index)
        return

    def _decouple_is_legal(self):
        decoupled_is_legal = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for dataset in self.annotations:
            for split in self.annotations[dataset]:
                for sequence_name in self.annotations[dataset][split]:
                    for frame_id, annotation in enumerate(self.annotations[dataset][split][sequence_name]):
                        decoupled_is_legal[dataset][split][sequence_name].append(annotation["is_legal"])
        # Reformat the 'is_legal' attribute from a list to a tensor,
        # which is more convenient for the sampling process (calculation-friendly).
        decoupled_is_legal_in_tensor = defaultdict(lambda: defaultdict(lambda: defaultdict(torch.Tensor)))
        for dataset in decoupled_is_legal:
            for split in decoupled_is_legal[dataset]:
                for sequence_name in decoupled_is_legal[dataset][split]:
                    decoupled_is_legal_in_tensor[dataset][split][sequence_name] = torch.tensor(
                        decoupled_is_legal[dataset][split][sequence_name], dtype=torch.bool
                    )
        return decoupled_is_legal_in_tensor

    def set_sample_details(
            self,
            sample_length: int,
            sample_interval: int,
            sample_mode: str = "random_interval",
    ):
        """
        Set the details for sampling.
        Now we only have "self.sample_begins" to store the beginning of each legal sample.
        NOTE: You should call this function at the start of each epoch.
        Args:
            sample_length: The length of each sample.
            sample_interval: The interval between two adjacent samples, currently not used.
            sample_mode: The mode of sampling, e.g., "random_interval", "fixed_interval".
        """
        assert sample_mode in ["random_interval"], f"Sample mode '{sample_mode}' is not supported."
        self.sample_begins = list()
        for dataset in self.annotations:
            for split in self.annotations[dataset]:
                for sequence_name in self.annotations[dataset][split]:
                    for frame_id in range(self.sequence_infos[dataset][split][sequence_name]["length"]):
                        if self.sequence_infos[dataset][split][sequence_name]["is_static"] is True:     # static image
                            self.sample_begins.append((dataset, split, sequence_name, frame_id))
                        else:   # real-world video
                            if frame_id + sample_length <= self.sequence_infos[dataset][split][sequence_name]["length"]:
                                if self.ann_is_legals[dataset][split][sequence_name][frame_id: frame_id + sample_length].all():
                                    # TODO: We may support different sampling ratio for each dataset, need to add code.
                                    self.sample_begins.append((dataset, split, sequence_name, frame_id))
        return

    def __len__(self):
        assert self.sample_begins is not None, "Please use 'self.set_sample_details()' at the start of each epoch."
        return len(self.sample_begins)

    def __getitem__(self, info):
        dataset = info["dataset"]
        split = info["split"]
        sequence = info["sequence"]
        frame_idxs = info["frame_idxs"]
        # Get image paths:
        image_paths = [
            self.image_paths[dataset][split][sequence][frame_idx] for frame_idx in frame_idxs
        ]
        # Read images:
        # images = [
        #     read_image(image_path) for image_path in image_paths
        # ]   # a list of tensors, shape=(C, H, W), dtype=torch.uint8
        images = [
            Image.open(image_path) for image_path in image_paths
        ]  # a list of tensors, shape=(C, H, W), dtype=torch.uint8
        # images = torch.stack(images, dim=0)     # shape=(N, C, H, W), dtype=torch.uint8
        # Get annotations:
        annotations = [
            self.annotations[dataset][split][sequence][frame_idx] for frame_idx in frame_idxs
        ]   # "bbox", "category", "id", "visibility", "is_legal"
        # Get metas:
        metas = [
            {
                "dataset": dataset,
                "split": split,
                "sequence": sequence,
                "frame_idx": frame_idx,
                "is_static": self.sequence_infos[dataset][split][sequence]["is_static"],
                "is_begin": False,      # whether the frame is the beginning of a video clip
                "size_divisibility": self.size_divisibility,
            } for frame_idx in frame_idxs
        ]
        # Do some modifications:
        metas[0]["is_begin"] = True     # the first frame is the beginning of a video clip
        # Deep copy:
        annotations = [annotation.copy() for annotation in annotations]
        metas = [meta.copy() for meta in metas]

        # Apply transforms:
        if self.transforms is not None:
            images, annotations, metas = self.transforms(images, annotations, metas)
        # from .tools import visualize_a_batch
        # visualize_a_batch(images, annotations)
        return images, annotations, metas

    def statistics(self):
        """
        Return the statistics of the dataset, in a list.
        Each item is a string: "Dancetrack.train, 35 sequences, 40000 frames."
        """
        statistics = list()
        for dataset in self.sequence_infos:
            for split in self.sequence_infos[dataset]:
                num_sequences = len(self.sequence_infos[dataset][split])
                num_frames = sum([info["length"] for info in self.sequence_infos[dataset][split].values()])
                statistics.append(f"{dataset}.{split}, {num_sequences} sequences, {num_frames} frames.")
        return statistics
