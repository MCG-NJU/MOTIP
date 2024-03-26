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
import cv2
import copy
import torch
import random
import PIL.Image
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F

from math import floor, ceil
from utils.box_ops import box_xyxy_to_cxcywh


class MultiCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, infos):
        for tran in self.transforms:
            images, infos = tran(images, infos)
        return images, infos


class MultiRandomSelect:
    """
    P(transform1) = p, P(transform2) = (1-p).
    """
    def __init__(self, transform1, transform2, p: float = 0.5):
        self.transform1 = transform1
        self.transform2 = transform2
        self.p = p

    def __call__(self, images, infos):
        if random.random() < self.p:
            return self.transform1(images, infos)
        else:
            return self.transform2(images, infos)


class MultiRandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, images, infos):
        if random.random() < self.p:
            def hflip(image: PIL.Image, info: dict):
                image = F.hflip(image)
                w, h = image.size
                # x1y1            x1y1    x2y1
                #          =>          =>
                #     x2y2    x2y2            x1y2
                info["boxes"] = (info["boxes"][:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1])
                                 + torch.as_tensor([w, 0, w, 0]))
                return image, info
            images, infos = zip(*[hflip(img, info) for img, info in zip(images, infos)])
        return images, infos


class MultiRandomResize:
    def __init__(self, sizes: list | tuple, max_size: int | None = None):
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, images, infos):
        def resize(image, info, size, max_size):
            def get_new_hw(current_hw, new_short_size, new_long_max_size) -> (int, int):
                w, h = current_hw
                if new_long_max_size is not None:   # restrict the long side length:
                    min_wh, max_wh = float(min(w, h)), float(max(w, h))
                    if max_wh / min_wh * new_short_size > new_long_max_size:
                        new_short_size = int(floor(new_long_max_size * min_wh / max_wh))
                # Calculate the w and h:
                if w < h:
                    new_w = new_short_size
                    new_h = int(round(new_short_size * h / w))
                else:
                    new_h = new_short_size
                    new_w = int(round(new_short_size) * w / h)
                return new_h, new_w
            new_hw = get_new_hw(
                current_hw=image.size,
                new_short_size=size,
                new_long_max_size=max_size
            )
            resized_image = F.resize(image, new_hw)
            ratio_w, ratio_h = (float(x) / float(ori_x) for x, ori_x in zip(resized_image.size, image.size))
            # Resize the boxes and areas:
            info["boxes"] = info["boxes"] * torch.as_tensor([ratio_w, ratio_h, ratio_w, ratio_h])
            info["areas"] = info["areas"] * ratio_w * ratio_h
            return resized_image, info
        new_size = random.choice(self.sizes)    # the shorter side length.
        return zip(*[resize(image, info, size=new_size, max_size=self.max_size) for image, info in zip(images, infos)])


class MultiToTensor:
    def __call__(self, images, infos):
        tensor_images = list(map(F.to_tensor, images))
        return tensor_images, infos


class MultiNormalize:
    """
    images  -> use mean/std normalize
    boxes   -> x1y1x2y2 to cxcywh and in range [0, 1]
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images, infos):
        def normalize(image: torch.Tensor, info):
            assert isinstance(image, torch.Tensor), f"Image should be 'Tensor' type for normalization " \
                                                    f"but get '{type(image)}'."
            info["unnorm_image_tensor"] = image.clone()
            image = F.normalize(image, mean=self.mean, std=self.std)
            h, w = image.shape[-2:]     # last two dim is h and w
            info["boxes"] = box_xyxy_to_cxcywh(info["boxes"])
            info["boxes"] = info["boxes"] / torch.as_tensor([w, h, w, h])
            return image, info

        return zip(*[normalize(image, info) for image, info in zip(images, infos)])


class MultiRandomCrop:
    def __init__(self, min_size: int, max_size: int, overflow_bbox: bool = False):
        self.min_size = min_size
        self.max_size = max_size
        self.overflow_bbox = overflow_bbox

    def __call__(self, images, infos):
        uncropped_infos = copy.deepcopy(infos)
        def crop(image, info, ijhw):
            cropped_img = F.crop(image, *ijhw)
            i, j, h, w = ijhw
            boxes = info["boxes"].clone()           # copy, in format [x1, y1, x2, y2]
            boxes = boxes - torch.as_tensor([j, i, j, i])   # new bbox coordinates
            max_wh = torch.as_tensor([w, h])                # max legal coordinates for x/y
            boxes = torch.min(boxes.reshape(-1, 2, 2), max_wh)  # [[x1, y1], [x2, y2]] in a line, no over range max_wh.
            boxes = boxes.clamp(min=0)                          # do not less than 0, it's illegal.
            legal_idxs = torch.all(torch.as_tensor(boxes[:, 1, :] > boxes[:, 0, :]), dim=1)     # find legal boxes
            if torch.sum(legal_idxs) == 0:    # do not have any legal boxes, give up the crop process
                return image, info, False
            need_select_fields = ["labels", "ids", "boxes", "areas"]    # field that need to be select by legal idxs.
            if self.overflow_bbox:  # bbox overflow is legal:
                for field in need_select_fields:
                    info[field] = info[field][legal_idxs]
            else:                   # bbox should not overflow
                info["boxes"] = boxes.reshape(-1, 4)    # from [[x1, y1], [x2, y2]] to [x1, y1, x2, y2]
                boxes_wh = info["boxes"][:, [2, 3]] - info["boxes"][:, [0, 1]]
                boxes_areas = boxes_wh[:, 0] * boxes_wh[:, 1]
                info["areas"] = boxes_areas
                for field in need_select_fields:
                    info[field] = info[field][legal_idxs]
            return cropped_img, info, True
        crop_w = random.randint(self.min_size, min(images[0].width, self.max_size))
        crop_h = random.randint(self.min_size, min(images[0].height, self.max_size))
        crop_ijhw = T.RandomCrop.get_params(images[0], (crop_h, crop_w))    # in docs, it can use image in type PIL.
        cropped_images, infos, legals = zip(*[crop(image, info, ijhw=crop_ijhw) for image, info in zip(images, infos)])
        if all(legals):
            return cropped_images, infos
        else:
            # return images, infos
            return images, uncropped_infos
        # return zip(*[crop(image, info, ijhw=crop_ijhw) for image, info in zip(images, infos)])


class MultiRandomShift:
    def __init__(self, max_shift_ratio: float = 0.05, overflow_bbox: bool = False):
        self.max_shift_ratio = max_shift_ratio
        self.overflow_bbox = overflow_bbox

    def __call__(self, images, infos):
        def shift(image, info, region, target_h, target_w):
            # Crop the image and resize to target size (original size):
            cropped_image = F.crop(image, *region)
            cropped_image = F.resize(cropped_image, [target_h, target_w])
            top, left, height, width = region
            # Adjustment the bbox:
            boxes = info["boxes"].clone()
            boxes = boxes - torch.as_tensor([left, top, left, top])
            boxes *= torch.as_tensor([target_w / width, target_h / height, target_w / width, target_h / height])
            max_wh = torch.as_tensor([target_w, target_h])
            boxes = torch.min(boxes.reshape(-1, 2, 2), max_wh)
            boxes = boxes.clamp(min=0)
            legal_idxs = torch.all(torch.as_tensor(boxes[:, 1, :] > boxes[:, 0, :]), dim=1)
            if torch.sum(legal_idxs) == 0:    # do not have any legal boxes, give up the shift process
                return image, info
            if not self.overflow_bbox:
                info["boxes"] = boxes.reshape(-1, 4)
                boxes_wh = info["boxes"][:, [2, 3]] - info["boxes"][:, [0, 1]]
                boxes_areas = boxes_wh[:, 0] * boxes_wh[:, 1]
                info["areas"] = boxes_areas
            need_select_fields = ["labels", "ids", "boxes", "areas"]
            for field in need_select_fields:
                info[field] = info[field][legal_idxs]
            return cropped_image, info

        res_images, res_infos = [], []
        n_frames = len(images)
        w, h = images[0].size
        max_wh = max(w, h)
        max_shift = int(self.max_shift_ratio * max_wh)
        x_shift = ceil(max_shift * random.random())
        y_shift = ceil(max_shift * random.random())
        x_shift *= int(random.random() > 0.5)
        y_shift *= int(random.random() > 0.5)
        # First image do not need any shift:
        res_images.append(images[0])
        res_infos.append(infos[0])

        for i in range(1, n_frames):
            y_min = max(0, -y_shift)
            y_max = min(h, h - y_shift)
            x_min = max(0, -x_shift)
            x_max = max(w, w - x_shift)
            prev_image = res_images[i - 1].copy()
            prev_info = copy.deepcopy(res_infos[i - 1])
            shift_region = (int(y_min), int(x_min), int(y_max - y_min), int(x_max - x_min))
            # F.crop 使用的 region 要求是 top, left, height, width
            img_i, info_i = shift(image=prev_image, info=prev_info, region=shift_region, target_h=h, target_w=w)
            res_images.append(img_i)
            res_infos.append(info_i)

        if random.random() > 0.5:
            res_images.reverse()
            res_infos.reverse()
        return res_images, res_infos


class MultiHSV:
    """
    From YOLOX [https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/data/data_augment.py] and MOTRv2.
    """
    def __init__(self, hgain=5, sgain=30, vgain=30):
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, images, infos):
        hsv_augs = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain]
        hsv_augs *= np.random.randint(0, 2, 3)
        hsv_augs = hsv_augs.astype(np.int16)

        def hsv(image, info):
            image = np.array(image)
            img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.int16)

            img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
            img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
            img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)

            return cv2.cvtColor(img_hsv.astype(image.dtype), cv2.COLOR_HSV2RGB), info

        return zip(*[hsv(img, info) for img, info in zip(images, infos)])


class MultiReverseClip:
    def __init__(self, reverse_clip: bool = False):
        self.reverse_clip = reverse_clip

    def __call__(self, images, infos):
        if random.random() < self.reverse_clip:   # Reverse this clip.
            images = list(images)
            infos = list(infos)
            images.reverse()
            infos.reverse()
        return images, infos
