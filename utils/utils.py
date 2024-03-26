# @Author       : Ruopeng Gao
# @Date         : 2022/7/5
# @Description  : Some utils.
import os
import math
import random

import yaml
import torch
import torchvision
import torch.distributed
import random
import numpy as np
from typing import Any, Dict, Generator, ItemsView, List, Tuple


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
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return False
    return True


def distributed_rank():
    if not is_distributed():
        return 0
    else:
        return torch.distributed.get_rank()


def distributed_world_rank():
    if not is_distributed():
        return 0
    else:
        return torch.distributed.get_global_rank()


def is_main_process():
    return distributed_rank() == 0


def distributed_world_size():
    if is_distributed():
        return torch.distributed.get_world_size()
    else:
        return 1
        # raise RuntimeError("'world size' is not available when distributed mode is not started.")


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


def labels_to_one_hot(labels: torch.Tensor, class_num: int, device="cpu"):
    """
    Args:
        labels: Original labels.
        class_num:
        device:
    Returns:
        Labels in one-hot.
    """
    if len(labels) > 0:
        return torch.eye(n=class_num, device=device)[labels].reshape((len(labels), -1))
    else:
        # A hack for empty labels.
        return torch.empty((0, class_num), device=device)


def pos_to_pos_embed(pos, num_pos_feats: int = 64, temperature: int = 10000, scale: float = 2 * math.pi):
    """
    Args:
        pos: 0~1, position vector, (N, M) / (B, N, M)
        num_pos_feats:
        temperature:
        scale:

    Returns:
    """
    pos = pos * scale
    dim_i = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_i = temperature ** (2 * (torch.div(dim_i, 2, rounding_mode="trunc")) / num_pos_feats)
    # 10000^(2i/d_model)
    pos_embed = pos[..., None] / dim_i      # (N, M, n_feats) or (B, N, M, n_feats)
    pos_embed = torch.stack((pos_embed[..., 0::2].sin(), pos_embed[..., 1::2].cos()), dim=-1)
    pos_embed = torch.flatten(pos_embed, start_dim=-3)
    return pos_embed


def inverse_sigmoid(x, eps=1e-5):
    """
    if      x = 1/(1+exp(-y))
    then    y = ln(x/(1-x))
    Args:
        x:
        eps:

    Returns:

    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    # if float(torchvision.__version__[:3]) < 0.7:
    #     if input.numel() > 0:
    #         return torch.nn.functional.interpolate(
    #             input, size, scale_factor, mode, align_corners
    #         )
    #
    #     output_shape = _output_size(2, input, size, scale_factor)
    #     output_shape = list(input.shape[:-2]) + list(output_shape)
    #     if float(torchvision.__version__[:3]) < 0.5:
    #         return _NewEmptyTensorOp.apply(input, output_shape)
    #     return _new_empty_tensor(input, output_shape)
    # else:
    return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def combine_detr_outputs(detr_outputs1, detr_outputs2):
    if detr_outputs1 is None:
        return detr_outputs2
    if detr_outputs2 is None:
        return detr_outputs1
    combined_outputs = dict()
    combined_outputs["pred_logits"] = torch.cat([detr_outputs1["pred_logits"], detr_outputs2["pred_logits"]], dim=0)
    combined_outputs["pred_boxes"] = torch.cat([detr_outputs1["pred_boxes"], detr_outputs2["pred_boxes"]], dim=0)
    combined_outputs["outputs"] = torch.cat([detr_outputs1["outputs"], detr_outputs2["outputs"]], dim=0)
    combined_outputs["aux_outputs"] = [
        {
            "pred_logits": torch.cat([
                detr_outputs1["aux_outputs"][_]["pred_logits"],
                detr_outputs2["aux_outputs"][_]["pred_logits"]],
                dim=0
            ),
            "pred_boxes": torch.cat([
                detr_outputs1["aux_outputs"][_]["pred_boxes"],
                detr_outputs2["aux_outputs"][_]["pred_boxes"]],
                dim=0
            ),
        }
        for _ in range(len(detr_outputs1["aux_outputs"]))
    ]
    if "dn_meta" in detr_outputs1:  # for DINO?
        combined_outputs["dn_meta"] = {}
        combined_outputs["dn_meta"]["pad_size"] = detr_outputs1["dn_meta"]["pad_size"]
        combined_outputs["dn_meta"]["num_dn_group"] = detr_outputs1["dn_meta"]["num_dn_group"]
        combined_outputs["dn_meta"]["output_known_lbs_bboxes"] = {}
        combined_outputs["dn_meta"]["output_known_lbs_bboxes"]["pred_logits"] = torch.cat([
            detr_outputs1["dn_meta"]["output_known_lbs_bboxes"]["pred_logits"],
            detr_outputs2["dn_meta"]["output_known_lbs_bboxes"]["pred_logits"]],
            dim=0
        )
        combined_outputs["dn_meta"]["output_known_lbs_bboxes"]["pred_boxes"] = torch.cat([
            detr_outputs1["dn_meta"]["output_known_lbs_bboxes"]["pred_boxes"],
            detr_outputs2["dn_meta"]["output_known_lbs_bboxes"]["pred_boxes"]],
            dim=0
        )
        combined_outputs["dn_meta"]["output_known_lbs_bboxes"]["aux_outputs"] = [
            {
                "pred_logits": torch.cat([
                    detr_outputs1["dn_meta"]["output_known_lbs_bboxes"]["aux_outputs"][_]["pred_logits"],
                    detr_outputs2["dn_meta"]["output_known_lbs_bboxes"]["aux_outputs"][_]["pred_logits"]],
                    dim=0
                ),
                "pred_boxes": torch.cat([
                    detr_outputs1["dn_meta"]["output_known_lbs_bboxes"]["aux_outputs"][_]["pred_boxes"],
                    detr_outputs2["dn_meta"]["output_known_lbs_bboxes"]["aux_outputs"][_]["pred_boxes"]],
                    dim=0
                ),
            }
            for _ in range(len(detr_outputs1["dn_meta"]["output_known_lbs_bboxes"]["aux_outputs"]))
        ]
    if "interm_outputs" in detr_outputs1:
        combined_outputs["interm_outputs"] = {
            "pred_logits": torch.cat(
                [detr_outputs1["interm_outputs"]["pred_logits"], detr_outputs2["interm_outputs"]["pred_logits"]], dim=0
            ),
            "pred_boxes": torch.cat(
                [detr_outputs1["interm_outputs"]["pred_boxes"], detr_outputs2["interm_outputs"]["pred_boxes"]], dim=0
            )
        }
        combined_outputs["interm_outputs_for_matching_pre"] = {
            "pred_logits": torch.cat(
                [detr_outputs1["interm_outputs_for_matching_pre"]["pred_logits"], detr_outputs2["interm_outputs_for_matching_pre"]["pred_logits"]], dim=0
            ),
            "pred_boxes": torch.cat(
                [detr_outputs1["interm_outputs_for_matching_pre"]["pred_boxes"], detr_outputs2["interm_outputs_for_matching_pre"]["pred_boxes"]], dim=0
            )
        }
    return combined_outputs


def detr_outputs_index_select(detr_outputs, index, dim: int = 0):
    selected_detr_outputs = dict()
    selected_detr_outputs["pred_logits"] = torch.index_select(detr_outputs["pred_logits"], index=index, dim=dim).contiguous()
    selected_detr_outputs["pred_boxes"] = torch.index_select(detr_outputs["pred_boxes"], index=index, dim=dim).contiguous()
    selected_detr_outputs["outputs"] = torch.index_select(detr_outputs["outputs"], index=index, dim=dim).contiguous()
    selected_detr_outputs["aux_outputs"] = [
        {
            "pred_logits": torch.index_select(detr_outputs["aux_outputs"][_]["pred_logits"], index=index, dim=dim).contiguous(),
            "pred_boxes": torch.index_select(detr_outputs["aux_outputs"][_]["pred_boxes"], index=index, dim=dim).contiguous(),
        }
        for _ in range(len(detr_outputs["aux_outputs"]))
    ]
    if "dn_meta" in detr_outputs:
        selected_detr_outputs["dn_meta"] = {
            "pad_size": detr_outputs["dn_meta"]["pad_size"],
            "num_dn_group": detr_outputs["dn_meta"]["num_dn_group"],
            "output_known_lbs_bboxes": {
                "pred_logits": torch.index_select(detr_outputs["dn_meta"]["output_known_lbs_bboxes"]["pred_logits"], index=index, dim=dim).contiguous(),
                "pred_boxes": torch.index_select(detr_outputs["dn_meta"]["output_known_lbs_bboxes"]["pred_boxes"], index=index, dim=dim).contiguous(),
                "aux_outputs": [
                    {
                        "pred_logits": torch.index_select(detr_outputs["dn_meta"]["output_known_lbs_bboxes"]["aux_outputs"][_]["pred_logits"], index=index, dim=dim).contiguous(),
                        "pred_boxes": torch.index_select(detr_outputs["dn_meta"]["output_known_lbs_bboxes"]["aux_outputs"][_]["pred_boxes"], index=index, dim=dim).contiguous()
                    }
                    for _ in range(len(detr_outputs["dn_meta"]["output_known_lbs_bboxes"]["aux_outputs"]))
                ],
            }
        }
        pass
    if "interm_outputs" in detr_outputs:
        selected_detr_outputs["interm_outputs"] = {
            "pred_logits": torch.index_select(detr_outputs["interm_outputs"]["pred_logits"], index=index, dim=dim).contiguous(),
            "pred_boxes": torch.index_select(detr_outputs["interm_outputs"]["pred_boxes"], index=index, dim=dim).contiguous()
        }
        selected_detr_outputs["interm_outputs_for_matching_pre"] = {
            "pred_logits": torch.index_select(detr_outputs["interm_outputs_for_matching_pre"]["pred_logits"], index=index, dim=dim).contiguous(),
            "pred_boxes": torch.index_select(detr_outputs["interm_outputs_for_matching_pre"]["pred_boxes"], index=index, dim=dim).contiguous()
        }
    return selected_detr_outputs


def infos_to_detr_targets(infos: dict, device):
    targets = list()
    for info in infos:
        for _ in range(len(info)):
            targets.append({
                "boxes": info[_]["boxes"].to(device),
                "labels": info[_]["labels"].to(device)
            })
    return targets


def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    assert len(args) > 0 and all(
        len(a) == len(args[0]) for a in args
    ), "Batched iteration must have inputs of all the same size."
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size: (b + 1) * batch_size] for arg in args]


if __name__ == '__main__':
    config = yaml_to_dict("../configs/resnet18_mnist.yaml")


