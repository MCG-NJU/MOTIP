# Copyright (c) Ruopeng Gao. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from MOTR (https://github.com/megvii-research/MOTR)
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch
from torchvision.ops.boxes import box_area


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    boxes = [
        (x1 + x2) / 2,
        (y1 + y2) / 2,
        (x2 - x1),
        (y2 - y1)
    ]
    return torch.stack(boxes, dim=-1)


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    boxes = [
        (cx - 0.5 * w),
        (cy - 0.5 * h),
        (cx + 0.5 * w),
        (cy + 0.5 * h)
    ]
    return torch.stack(boxes, dim=-1)


def box_cxcywh_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    boxes = [
        (cx - 0.5 * w),
        (cy - 0.5 * h),
        w,
        h
    ]
    return torch.stack(boxes, dim=-1)


def box_iou_union(boxes1, boxes2):
    """
    Return the IoU and Union of two boxes, the input format should be x1y1x2y2 format.
    Args:
        boxes1:
        boxes2:

    Returns: iou, union
    """
    area1 = box_area(boxes1)    # [N, ]
    area2 = box_area(boxes2)    # [M, ]

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter  # [N, M]
    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou_union(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area
