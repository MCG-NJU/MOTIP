# Copyright (c) Ruopeng Gao. All Rights Reserved.

import torch
import einops
from scipy.optimize import linear_sum_assignment

from structures.instances import Instances
from structures.ordered_set import OrderedSet
from utils.misc import distributed_device
from utils.box_ops import box_cxcywh_to_xywh
from models.misc import get_model


class RuntimeTracker:
    def __init__(
            self,
            model,
            # Sequence infos:
            sequence_hw: tuple,
            # Inference settings:
            use_sigmoid: bool = False,
            assignment_protocol: str = "hungarian",
            miss_tolerance: int = 30,
            det_thresh: float = 0.5,
            newborn_thresh: float = 0.5,
            id_thresh: float = 0.1,
            area_thresh: int = 0,
            only_detr: bool = False,
            dtype: torch.dtype = torch.float32,
    ):
        self.model = model
        self.model.eval()

        self.dtype = dtype

        self.use_sigmoid = use_sigmoid
        self.assignment_protocol = assignment_protocol.lower()
        self.miss_tolerance = miss_tolerance
        self.det_thresh = det_thresh
        self.newborn_thresh = newborn_thresh
        self.id_thresh = id_thresh
        self.area_thresh = area_thresh
        self.only_detr = only_detr
        self.num_id_vocabulary = get_model(model).num_id_vocabulary

        # Check for the legality of settings:
        assert self.assignment_protocol in ["hungarian", "id-max", "object-max", "object-priority", "id-priority"], \
            f"Assignment protocol {self.assignment_protocol} is not supported."

        self.bbox_unnorm = torch.tensor(
            [sequence_hw[1], sequence_hw[0], sequence_hw[1], sequence_hw[0]],
            dtype=torch.float32,
            device=distributed_device(),
        )

        # Trajectory fields:
        self.next_id = 0
        self.id_label_to_id = {}
        self.id_queue = OrderedSet()
        # Init id_queue:
        for i in range(self.num_id_vocabulary):
            self.id_queue.add(i)
        # All fields are in shape (T, N, ...)
        self.trajectory_features = torch.zeros(
            (0, 0, 256), dtype=dtype, device=distributed_device(),
        )
        self.trajectory_boxes = torch.zeros(
            (0, 0, 4), dtype=dtype, device=distributed_device(),
        )
        self.trajectory_id_labels = torch.zeros(
            (0, 0), dtype=torch.int64, device=distributed_device(),
        )
        self.trajectory_times = torch.zeros(
            (0, 0), dtype=dtype, device=distributed_device(),
        )
        self.trajectory_masks = torch.zeros(
            (0, 0), dtype=torch.bool, device=distributed_device(),
        )
        # self.trajectory_features = torch.zeros(())

        self.current_track_results = {}
        return

    @torch.no_grad()
    def update(self, image):
        detr_out = self.model(frames=image, part="detr")
        scores, categories, boxes, output_embeds = self._get_activate_detections(detr_out=detr_out)
        if self.only_detr:
            id_pred_labels = self.num_id_vocabulary * torch.ones(boxes.shape[0], dtype=torch.int64, device=boxes.device)
        else:
            id_pred_labels = self._get_id_pred_labels(boxes=boxes, output_embeds=output_embeds)
        # Filter out illegal newborn detections:
        keep_idxs = (id_pred_labels != self.num_id_vocabulary) | (scores > self.newborn_thresh)
        scores = scores[keep_idxs]
        categories = categories[keep_idxs]
        boxes = boxes[keep_idxs]
        output_embeds = output_embeds[keep_idxs]
        id_pred_labels = id_pred_labels[keep_idxs]

        # A hack implementation, before assign new id labels, update the id_queue to ensure the uniqueness of id labels:
        n_activate_id_labels = 0
        n_newborn_targets = 0
        for _ in range(len(id_pred_labels)):
            if id_pred_labels[_].item() != self.num_id_vocabulary:
                n_activate_id_labels += 1
                self.id_queue.add(id_pred_labels[_].item())
            else:
                n_newborn_targets += 1

        # Make sure the length of newborn instances is less than the length of remaining IDs:
        n_remaining_ids = len(self.id_queue) - n_activate_id_labels
        if n_newborn_targets > n_remaining_ids:
            keep_idxs = torch.ones(len(id_pred_labels), dtype=torch.bool, device=id_pred_labels.device)
            newborn_idxs = (id_pred_labels == self.num_id_vocabulary)
            newborn_keep_idxs = torch.ones(len(newborn_idxs), dtype=torch.bool, device=newborn_idxs.device)
            newborn_keep_idxs[n_remaining_ids:] = False
            keep_idxs[newborn_idxs] = newborn_keep_idxs
            scores = scores[keep_idxs]
            categories = categories[keep_idxs]
            boxes = boxes[keep_idxs]
            output_embeds = output_embeds[keep_idxs]
            id_pred_labels = id_pred_labels[keep_idxs]
        pass

        # Assign new id labels:
        id_labels = self._assign_newborn_id_labels(pred_id_labels=id_pred_labels)

        if len(torch.unique(id_labels)) != len(id_labels):
            print(id_labels, id_labels.shape)
            exit(-1)

        # Update the results:
        self.current_track_results = {
            "score": scores,
            "category": categories,
            # "bbox": boxes * self.bbox_unnorm,
            "bbox": box_cxcywh_to_xywh(boxes) * self.bbox_unnorm,
            "id": torch.tensor(
                [self.id_label_to_id[_] for _ in id_labels.tolist()], dtype=torch.int64,
            ),
        }

        # Update id_queue:
        for _ in range(len(id_labels)):
            self.id_queue.add(id_labels[_].item())

        # Update trajectory infos:
        self._update_trajectory_infos(boxes=boxes, output_embeds=output_embeds, id_labels=id_labels)

        # Filter out inactive tracks:
        self._filter_out_inactive_tracks()
        pass
        return

    def get_track_results(self):
        return self.current_track_results

    def _get_activate_detections(self, detr_out: dict):
        logits = detr_out["pred_logits"][0]
        boxes = detr_out["pred_boxes"][0]
        output_embeds = detr_out["outputs"][0]
        scores = logits.sigmoid()
        scores, categories = torch.max(scores, dim=-1)
        area = boxes[:, 2] * self.bbox_unnorm[2] * boxes[:, 3] * self.bbox_unnorm[3]
        activate_indices = (scores > self.det_thresh) & (area > self.area_thresh)
        # Selecting:
        # logits = logits[activate_indices]
        boxes = boxes[activate_indices]
        output_embeds = output_embeds[activate_indices]
        scores = scores[activate_indices]
        categories = categories[activate_indices]
        return scores, categories, boxes, output_embeds

    def _get_id_pred_labels(self, boxes: torch.Tensor, output_embeds: torch.Tensor):
        if self.trajectory_features.shape[0] == 0:
            return self.num_id_vocabulary * torch.ones(boxes.shape[0], dtype=torch.int64, device=boxes.device)
        else:
            # 1. prepare current infos:
            current_features = output_embeds[None, ...]     # (T, N, ...)
            current_boxes = boxes[None, ...]                # (T, N, 4)
            current_masks = torch.zeros((1, output_embeds.shape[0]), dtype=torch.bool, device=distributed_device())
            current_times = self.trajectory_times.shape[0] * torch.ones(
                (1, output_embeds.shape[0]), dtype=torch.int64, device=distributed_device(),
            )
            # 2. prepare seq_info:
            seq_info = {
                "trajectory_features": self.trajectory_features[None, None, ...],
                "trajectory_boxes": self.trajectory_boxes[None, None, ...],
                "trajectory_id_labels": self.trajectory_id_labels[None, None, ...],
                "trajectory_times": self.trajectory_times[None, None, ...],
                "trajectory_masks": self.trajectory_masks[None, None, ...],
                "unknown_features": current_features[None, None, ...],
                "unknown_boxes": current_boxes[None, None, ...],
                "unknown_masks": current_masks[None, None, ...],
                "unknown_times": current_times[None, None, ...],
            }
            # 3. forward:
            seq_info = self.model(seq_info=seq_info, part="trajectory_modeling")
            id_logits, _, _ = self.model(seq_info=seq_info, part="id_decoder")
            # 4. get scores:
            id_logits = id_logits[0, 0, 0]
            if not self.use_sigmoid:
                id_scores = id_logits.softmax(dim=-1)
            else:
                id_scores = id_logits.sigmoid()
            # 5. assign id labels:
            # Different assignment protocols:
            match self.assignment_protocol:
                case "hungarian": id_labels = self._hungarian_assignment(id_scores=id_scores)
                case "object-max": id_labels = self._object_max_assignment(id_scores=id_scores)
                case "id-max": id_labels = self._id_max_assignment(id_scores=id_scores)
                # case "object-priority": id_labels = self._object_priority_assignment(id_scores=id_scores)
                case _: raise NotImplementedError

            id_pred_labels = torch.tensor(id_labels, dtype=torch.int64, device=distributed_device())
            return id_pred_labels

    def _assign_newborn_id_labels(self, pred_id_labels: torch.Tensor):
        # 1. how many newborn instances?
        n_newborns = (pred_id_labels == self.num_id_vocabulary).sum().item()
        if n_newborns == 0:
            return pred_id_labels
        else:
            # 2. get available id labels from id_queue:
            newborn_id_labels = torch.tensor(
                list(self.id_queue)[:n_newborns], dtype=torch.int64, device=distributed_device(),
            )
            # 3. make sure these id labels are not in trajectory infos:
            trajectory_remove_idxs = torch.zeros(
                self.trajectory_id_labels.shape[1], dtype=torch.bool, device=distributed_device(),
            )
            for _ in range(len(newborn_id_labels)):
                if self.trajectory_id_labels.shape[0] > 0:
                    trajectory_remove_idxs |= (self.trajectory_id_labels[0] == newborn_id_labels[_])
                if newborn_id_labels[_].item() in self.id_label_to_id:
                    self.id_label_to_id.pop(newborn_id_labels[_].item())
            # remove from trajectory infos:
            self.trajectory_features = self.trajectory_features[:, ~trajectory_remove_idxs]
            self.trajectory_boxes = self.trajectory_boxes[:, ~trajectory_remove_idxs]
            self.trajectory_id_labels = self.trajectory_id_labels[:, ~trajectory_remove_idxs]
            self.trajectory_times = self.trajectory_times[:, ~trajectory_remove_idxs]
            self.trajectory_masks = self.trajectory_masks[:, ~trajectory_remove_idxs]
            # 4. assign id labels to newborn instances:
            pred_id_labels[pred_id_labels == self.num_id_vocabulary] = newborn_id_labels
            # 5. update id infos:
            for _ in range(len(newborn_id_labels)):
                self.id_label_to_id[newborn_id_labels[_].item()] = self.next_id
                self.next_id += 1

            return pred_id_labels

    def _update_trajectory_infos(self, boxes: torch.Tensor, output_embeds: torch.Tensor, id_labels: torch.Tensor):
        # 1. cut trajectory infos:
        self.trajectory_features = self.trajectory_features[-self.miss_tolerance + 2:, ...]
        self.trajectory_boxes = self.trajectory_boxes[-self.miss_tolerance + 2:, ...]
        self.trajectory_id_labels = self.trajectory_id_labels[-self.miss_tolerance + 2:, ...]
        self.trajectory_times = self.trajectory_times[-self.miss_tolerance + 2:, ...]
        self.trajectory_masks = self.trajectory_masks[-self.miss_tolerance + 2:, ...]
        # 2. find out all new instances:
        already_id_labels = set(self.trajectory_id_labels[0].tolist() if self.trajectory_id_labels.shape[0] > 0 else [])
        _id_labels = set(id_labels.tolist())
        newborn_id_labels = _id_labels - already_id_labels
        # 3. add newborn instances to trajectory infos:
        if len(newborn_id_labels) > 0:
            newborn_id_labels = torch.tensor(list(newborn_id_labels), dtype=torch.int64, device=distributed_device())
            _T = self.trajectory_id_labels.shape[0]
            _N = len(newborn_id_labels)
            _id_labels = einops.repeat(newborn_id_labels, 'n -> t n', t=_T)
            _boxes = torch.zeros((_T, _N, 4), dtype=self.dtype, device=distributed_device())
            _times = einops.repeat(
                torch.arange(_T, dtype=torch.int64, device=distributed_device()), 't -> t n', n=_N,
            )
            _features = torch.zeros(
                (_T, _N, 256), dtype=self.dtype, device=distributed_device(),
            )
            _masks = torch.ones((_T, _N), dtype=torch.bool, device=distributed_device())
            # 3.1. padding to trajectory infos:
            self.trajectory_id_labels = torch.cat([self.trajectory_id_labels, _id_labels], dim=1)
            self.trajectory_boxes = torch.cat([self.trajectory_boxes, _boxes], dim=1)
            self.trajectory_times = torch.cat([self.trajectory_times, _times], dim=1)
            self.trajectory_features = torch.cat([self.trajectory_features, _features], dim=1)
            self.trajectory_masks = torch.cat([self.trajectory_masks, _masks], dim=1)
        # 4. update trajectory infos:
        _N = self.trajectory_id_labels.shape[1]
        current_id_labels = self.trajectory_id_labels[0] if self.trajectory_id_labels.shape[0] > 0 else id_labels
        current_features = torch.zeros((_N, 256), dtype=self.dtype, device=distributed_device())
        current_boxes = torch.zeros((_N, 4), dtype=self.dtype, device=distributed_device())
        current_times = self.trajectory_id_labels.shape[0] * torch.ones((_N,), dtype=torch.int64, device=distributed_device())
        current_masks = torch.ones((_N,), dtype=torch.bool, device=distributed_device())
        # 4.1. find out the same id labels (matching):
        indices = torch.eq(current_id_labels[:, None], id_labels[None, :]).nonzero(as_tuple=False)
        current_idxs = indices[:, 0]
        idxs = indices[:, 1]
        # 4.2. fill in the infos:
        current_id_labels[current_idxs] = id_labels[idxs]
        current_features[current_idxs] = output_embeds[idxs]
        current_boxes[current_idxs] = boxes[idxs]
        current_masks[current_idxs] = False
        # 4.3. cat to trajectory infos:
        self.trajectory_features = torch.cat([self.trajectory_features, current_features[None, ...]], dim=0).contiguous()
        self.trajectory_boxes = torch.cat([self.trajectory_boxes, current_boxes[None, ...]], dim=0).contiguous()
        self.trajectory_id_labels = torch.cat([self.trajectory_id_labels, current_id_labels[None, ...]], dim=0).contiguous()
        self.trajectory_times = torch.cat([self.trajectory_times, current_times[None, ...]], dim=0).contiguous()
        self.trajectory_masks = torch.cat([self.trajectory_masks, current_masks[None, ...]], dim=0).contiguous()
        # 4.4. a hack implementation to fix "times":
        self.trajectory_times = einops.repeat(
            torch.arange(self.trajectory_times.shape[0], dtype=torch.int64, device=distributed_device()),
            't -> t n', n=self.trajectory_times.shape[1],
        ).contiguous().clone()
        return

    def _filter_out_inactive_tracks(self):
        is_active = torch.sum((~self.trajectory_masks).to(torch.int64), dim=0) > 0
        self.trajectory_features = self.trajectory_features[:, is_active]
        self.trajectory_boxes = self.trajectory_boxes[:, is_active]
        self.trajectory_id_labels = self.trajectory_id_labels[:, is_active]
        self.trajectory_times = self.trajectory_times[:, is_active]
        self.trajectory_masks = self.trajectory_masks[:, is_active]
        return

    def _hungarian_assignment(self, id_scores: torch.Tensor):
        id_labels = list()  # final ID labels
        if len(id_scores) > 1:
            id_scores_newborn_repeat = id_scores[:, -1:].repeat(1, len(id_scores) - 1)
            id_scores = torch.cat((id_scores, id_scores_newborn_repeat), dim=-1)
        trajectory_id_labels_set = set(self.trajectory_id_labels[0].tolist())
        match_rows, match_cols = linear_sum_assignment(1 - id_scores.cpu())
        for _ in range(len(match_rows)):
            _id = match_cols[_]
            if _id not in trajectory_id_labels_set:
                id_labels.append(self.num_id_vocabulary)
            elif _id >= self.num_id_vocabulary:
                id_labels.append(self.num_id_vocabulary)
            elif id_scores[match_rows[_], _id] < self.id_thresh:
                id_labels.append(self.num_id_vocabulary)
            else:
                id_labels.append(_id)
        return id_labels

    def _object_max_assignment(self, id_scores: torch.Tensor):
        id_labels = list()  # final ID labels
        trajectory_id_labels_set = set(self.trajectory_id_labels[0].tolist())   # all tracked ID labels

        object_max_confs, object_max_id_labels = torch.max(id_scores, dim=-1)   # get the target ID labels and confs
        # Get the max confs of each ID label:
        id_max_confs = dict()
        for conf, id_label in zip(object_max_confs.tolist(), object_max_id_labels.tolist()):
            if id_label not in id_max_confs:
                id_max_confs[id_label] = conf
            else:
                # if conf == id_max_confs[id_label]:  # a very rare case
                #     conf = conf - 0.0001
                id_max_confs[id_label] = max(id_max_confs[id_label], conf)
        if self.num_id_vocabulary in id_max_confs:
            id_max_confs[self.num_id_vocabulary] = 0.0  # special token

        # Assign ID labels:
        for _ in range(len(object_max_id_labels)):
            if object_max_id_labels[_].item() not in trajectory_id_labels_set:         # not in tracked IDs -> newborn
                id_labels.append(self.num_id_vocabulary)
            else:
                _id_label = object_max_id_labels[_].item()
                _conf = object_max_confs[_].item()
                if _conf < self.id_thresh or _conf < id_max_confs[_id_label]:  # low conf or not the max conf -> newborn
                    id_labels.append(self.num_id_vocabulary)
                elif _id_label in id_labels:
                    id_labels.append(self.num_id_vocabulary)
                else:                                                          # normal case
                    id_labels.append(_id_label)

        return id_labels

    def _id_max_assignment(self, id_scores: torch.Tensor):
        id_labels = [self.num_id_vocabulary] * len(id_scores)  # final ID labels
        trajectory_id_labels_set = set(self.trajectory_id_labels[0].tolist())   # all tracked ID labels

        id_max_confs, id_max_obj_idxs = torch.max(id_scores, dim=0)
        # Get the max confs of each object:
        object_max_confs = dict()
        for conf, object_idx in zip(id_max_confs.tolist(), id_max_obj_idxs.tolist()):
            if object_idx not in object_max_confs:
                object_max_confs[object_idx] = conf
            else:
                if conf == object_max_confs[object_idx]:    # a very rare case
                    conf = conf - 0.0001
                object_max_confs[object_idx] = max(object_max_confs[object_idx], conf)

        # Assign ID labels:
        for _ in range(len(id_max_obj_idxs)):
            _obj_idx, _id_label, _conf = id_max_obj_idxs[_].item(), _, id_max_confs[_].item()
            if _conf < self.id_thresh or _conf < object_max_confs[_obj_idx]:
                pass
            elif _id_label not in trajectory_id_labels_set:
                pass
            else:
                id_labels[_obj_idx] = _id_label

        return id_labels
