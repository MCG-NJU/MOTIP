# Copyright (c) Ruopeng Gao. All Rights Reserved.

import torch

from typing import Optional, List


class NestedTensor(object):
    def __init__(self, tensors: torch.Tensor, mask: Optional[torch.Tensor]):
        """Args:
            tensors: Tensor, (B, C, H, W)
            mask: Tensor, (B, H, W)
        """
        assert tensors.shape[0] == mask.shape[0], \
            f"tensors have batch size {tensors.shape[0]} but get {mask.shape[0]} for mask."
        self.tensors = tensors
        self.mask = mask

    def to(self, device, non_blocking=False):
        """
        Args:
            device: Device
            non_blocking: like pin_memory=True, can speed up data access.
        """
        tensor = self.tensors.to(device, non_blocking=non_blocking)
        if self.mask is None:
            mask = None
        else:
            mask = self.mask.to(device, non_blocking=non_blocking)
        return NestedTensor(tensors=tensor, mask=mask)

    def decompose(self) -> [torch.Tensor, torch.Tensor]:
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, item):
        return NestedTensor(tensors=self.tensors[item], mask=self.mask[item])

    def index_select(self, dim: int, index: torch.Tensor):
        return NestedTensor(
            tensors=torch.index_select(input=self.tensors, dim=dim, index=index),
            mask=torch.index_select(input=self.mask, dim=dim, index=index),
        )

    def clone(self):
        return NestedTensor(tensors=self.tensors.clone(), mask=self.mask.clone())


def nested_tensor_from_tensor_list(tensor_list: List[torch.Tensor], size_divisibility: int = 0) -> NestedTensor:
    """
    Args:
        tensor_list: List of tensors, each tensor should have shape (C, H, W)
        size_divisibility:
    Returns:
    """
    assert tensor_list[0].dim() == 3, f"Tensor should have 3 dimensions, but get {tensor_list[0].dim()}"

    heights, widths = zip(*[t.shape[1:] for t in tensor_list])  # heights, widths = [H1, H2, ..., Hn], [W1, W2, ..., Wn]
    # Calculate the shape (max size) of the NestedTensor:
    # final              B                         C                                 H       W
    final_shape = [len(tensor_list)] + [tensor_list[0].shape[0]] + list(map(max, (heights, widths)))
    final_b, final_c, final_h, final_w = final_shape    # get the final shape of the NestedTensor
    # If size_divisibility > 0, we need to adjust the final_h and final_w to be divisible by size_divisibility
    if size_divisibility > 0:
        stride = size_divisibility
        final_h = (final_h + (stride - 1)) // stride * stride
        final_w = (final_w + (stride - 1)) // stride * stride
    final_shape = [final_b, final_c, final_h, final_w]
    # Get the dtype and device of the final NestedTensor:
    dtype = tensor_list[0].dtype
    device = tensor_list[0].device
    # Initialize the tensors and masks:
    tensor = torch.zeros(final_shape, dtype=dtype, device=device)
    mask = torch.ones((final_b, final_h, final_w), dtype=torch.bool, device=device)
    # Fill the tensor and mask one by one:
    for input_tensor, pad_tensor, pad_mask in zip(tensor_list, tensor, mask):
        assert input_tensor.shape[0] == final_shape[1], "Tensor channel size should be equal."
        pad_tensor[: input_tensor.shape[0], : input_tensor.shape[1], : input_tensor.shape[2]].copy_(input_tensor)
        pad_mask[: input_tensor.shape[1], : input_tensor.shape[2]] = False
    return NestedTensor(tensors=tensor, mask=mask)


def nested_tensor_index_select(nested_tensor: NestedTensor, dim: int, index: torch.Tensor):
    tensor, mask = nested_tensor.decompose()
    selected_tensor = torch.index_select(input=tensor, dim=dim, index=index).contiguous()
    selected_mask = torch.index_select(input=mask, dim=dim, index=index).contiguous()
    return NestedTensor(tensors=selected_tensor, mask=selected_mask)

