# Copyright (c) Ruopeng Gao. All Rights Reserved.
# ------------------------------------------------------------------------
import torch
import einops
import torch.nn as nn

from utils.utils import is_distributed, distributed_world_size


class IDCriterion(nn.Module):
    def __init__(self, weight: float, gpu_average: bool):
        super().__init__()
        self.weight = weight
        self.gpu_average = gpu_average
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, outputs, targets):
        assert len(outputs) == 1, f"ID Criterion is only supported bs=1, but get bs={len(outputs)}"
        outputs = einops.rearrange(outputs, "b n c -> (b n) c")
        targets = einops.rearrange(targets, "b n c -> (b n) c")
        ce_loss = self.ce_loss(outputs, targets).sum()
        # Average:
        num_ids = len(outputs)
        num_ids = torch.as_tensor([num_ids], dtype=torch.float, device=outputs.device)
        if self.gpu_average:
            if is_distributed():
                torch.distributed.all_reduce(num_ids)
            num_ids = torch.clamp(num_ids / distributed_world_size(), min=1).item()
        return ce_loss / num_ids


def build(config: dict):
    return IDCriterion(
        weight=config["ID_LOSS_WEIGHT"],
        gpu_average=config["ID_LOSS_GPU_AVERAGE"]
    )
