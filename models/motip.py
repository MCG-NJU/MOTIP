# Copyright (c) RuopengGao. All Rights Reserved.
# About:
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from torch.utils.checkpoint import checkpoint
from .seq_decoder import SeqDecoder
from .deformable_detr.deformable_detr import build as build_deformable_detr
from .dab_deformable_detr.dab_deformable_detr import build_dab_deformable_detr
from structures.instances import Instances
from structures.args import Args
from utils.utils import batch_iterator, combine_detr_outputs
from collections import deque
from structures.ordered_set import OrderedSet


class MOTIP(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        self.num_id_vocabulary = config["NUM_ID_VOCABULARY"]    # how many id words
        self.training_num_id = config["NUM_ID_VOCABULARY"] if "TRAINING_NUM_ID" not in config else config["TRAINING_NUM_ID"]
        self.num_classes = config["NUM_CLASSES"]
        self.max_temporal_length = config["MAX_TEMPORAL_LENGTH"] if "MAX_TEMPORAL_LENGTH" in config \
            else config["MAX_TEMPORAL_PE_LENGTH"]

        # DETR Framework:
        self.detr_framework = config["DETR_FRAMEWORK"]
        # Backbone:
        self.backbone_type = config["BACKBONE"]
        self.lr_backbone = config["LR"] * config["LR_BACKBONE_SCALE"]
        self.backbone_dilation = config["DILATION"]
        # DETR settings:
        self.detr_num_queries = config["DETR_NUM_QUERIES"]
        self.detr_num_feature_levels = config["DETR_NUM_FEATURE_LEVELS"]
        self.detr_aux_loss = config["DETR_AUX_LOSS"]
        self.detr_with_box_refine = config["DETR_WITH_BOX_REFINE"]
        self.detr_two_stage = config["DETR_TWO_STAGE"]
        self.detr_masks = config["DETR_MASKS"]
        self.detr_hidden_dim = config["DETR_HIDDEN_DIM"]
        self.detr_position_embedding = config["DETR_PE"]
        self.detr_nheads = config["DETR_NUM_HEADS"]
        self.detr_enc_layers = config["DETR_ENC_LAYERS"]
        self.detr_dec_layers = config["DETR_DEC_LAYERS"]
        self.detr_dropout = config["DETR_DROPOUT"]
        self.detr_dim_feedforward = config["DETR_DIM_FEEDFORWARD"]
        self.detr_dec_n_points = config["DETR_DEC_N_POINTS"]
        self.detr_enc_n_points = config["DETR_ENC_N_POINTS"]
        self.detr_cls_loss_coef = config["DETR_CLS_LOSS_COEF"]
        self.detr_bbox_loss_coef = config["DETR_BBOX_LOSS_COEF"]
        self.detr_giou_loss_coef = config["DETR_GIOU_LOSS_COEF"]
        self.detr_set_cost_class = config["DETR_CLS_LOSS_COEF"] if "DETR_SET_COST_CLASS" not in config else config["DETR_SET_COST_CLASS"]
        self.detr_set_cost_bbox = config["DETR_BBOX_LOSS_COEF"] if "DETR_SET_COST_BBOX" not in config else config["DETR_SET_COST_BBOX"]
        self.detr_set_cost_giou = config["DETR_GIOU_LOSS_COEF"] if "DETR_SET_COST_GIOU" not in config else config["DETR_SET_COST_GIOU"]
        self.detr_focal_alpha = config["DETR_FOCAL_ALPHA"]
        self.device = config["DEVICE"]

        self.only_detr = config["TRAIN_STAGE"] == "only_detr"

        # Prepare args for detr
        detr_args = Args()
        detr_args.num_classes = self.num_classes
        detr_args.device = self.device
        detr_args.num_queries = self.detr_num_queries
        detr_args.num_feature_levels = self.detr_num_feature_levels
        detr_args.aux_loss = self.detr_aux_loss
        detr_args.with_box_refine = self.detr_with_box_refine
        detr_args.two_stage = self.detr_two_stage
        detr_args.hidden_dim = self.detr_hidden_dim
        detr_args.backbone = self.backbone_type
        detr_args.lr_backbone = self.lr_backbone
        detr_args.dilation = self.backbone_dilation
        detr_args.masks = self.detr_masks
        detr_args.position_embedding = self.detr_position_embedding
        detr_args.nheads = self.detr_nheads
        detr_args.enc_layers = self.detr_enc_layers
        detr_args.dec_layers = self.detr_dec_layers
        detr_args.dim_feedforward = self.detr_dim_feedforward
        detr_args.dropout = self.detr_dropout
        detr_args.dec_n_points = self.detr_dec_n_points
        detr_args.enc_n_points = self.detr_enc_n_points
        detr_args.cls_loss_coef = self.detr_cls_loss_coef
        detr_args.bbox_loss_coef = self.detr_bbox_loss_coef
        detr_args.giou_loss_coef = self.detr_giou_loss_coef
        detr_args.focal_alpha = self.detr_focal_alpha
        # Three hack implementation:
        detr_args.set_cost_class = self.detr_set_cost_class
        detr_args.set_cost_bbox = self.detr_set_cost_bbox
        detr_args.set_cost_giou = self.detr_set_cost_giou
        if self.detr_framework == "Deformable-DETR":
            # DETR model and criterion:
            self.detr, self.detr_criterion, _ = build_deformable_detr(detr_args)
        elif self.detr_framework == "DAB-Deformable-DETR":
            detr_args.num_patterns = 0
            detr_args.random_refpoints_xy = False
            self.detr, self.detr_criterion, _ = build_dab_deformable_detr(detr_args)
            # TODO: We will upload the DAB-DETR code soon.
        else:
            raise RuntimeError(f"Unknown DETR framework: {self.detr_framework}.")
        # ID Label Criterion:
        self.id_criterion = nn.CrossEntropyLoss()
        self.id_loss_weight = config["ID_LOSS_WEIGHT"]

        # Seq Decoder:
        if self.only_detr is False:
            self.seq_decoder = SeqDecoder(
                detr_hidden_dim=config["DETR_HIDDEN_DIM"],
                hidden_dim=256 if "SEQ_HIDDEN_DIM" not in config else config["SEQ_HIDDEN_DIM"],
                dim_feedforward=512 if "SEQ_DIM_FEEDFORWARD" not in config else config["SEQ_DIM_FEEDFORWARD"],
                num_heads=8 if "SEQ_NUM_HEADS" not in config else config["SEQ_NUM_HEADS"],
                dropout=0.0,
                n_id_decoder_layers=config["ID_DECODER_LAYERS"],
                num_id_vocabulary=self.num_id_vocabulary,
                training_num_id=config["NUM_ID_VOCABULARY"] if "TRAINING_NUM_ID" not in config else config["TRAINING_NUM_ID"],
                device=self.device,
                max_temporal_length=config["MAX_TEMPORAL_LENGTH"],
            )

    def get_id_word_from_tgt(self, tgt):
        return self.embed_to_word(tgt[:, :, 256:])

    @torch.no_grad()
    def inference(
            self,
            trajectory_history: deque[Instances],
            num_id_vocabulary: int,
            ids_to_results: dict,
            current_id: int,
            id_deque: OrderedSet,
            id_thresh: float = 0.1,
            newborn_thresh: float = 0.5,
    ):
        """
        :param trajectory_history: Historical trajectories.
        :param num_id_vocabulary: Number of ID vocabulary, K in the paper.
        :param ids_to_results: Mapping from ID word index to ID label in tracker files.
        :param current_id: Current next ID label of tracker files.
        :param id_deque: OrderedSet of ID words, may be recycled.
        :param id_thresh: ID threshold.
        :param newborn_thresh: Newborn threshold,
                               only the conf higher than this threshold will be considered as a newborn target.
        :return:
        """
        deque_max_length = trajectory_history.maxlen
        trajectory_history_list = list(trajectory_history)
        trajectory = trajectory_history_list[:-1]
        current = trajectory_history_list[-1:]

        # NEED TO KNOW:
        # 1. "ids" is the final ID words for current frame, it is a list.
        #    If a target does not have a corresponding ID word, it will be assigned as -1 in "ids".
        # 2. "new_id" is the ID words that need to be assigned to the new targets, also a list.
        # 3. "current" is the objects in the current frame.

        n_targets_in_frames = [len(_) for _ in trajectory_history_list]
        num_history_tokens, num_current_tokens = sum(n_targets_in_frames[:-1]), sum(n_targets_in_frames[-1:])
        if num_history_tokens == 0:     # no history tokens
            ids = [-1] * num_current_tokens
        elif num_current_tokens == 0:   # no current tokens
            ids = []
            return ids, trajectory_history, ids_to_results, current_id, id_deque, None  # directly return
        else:                           # normal process:
            trajectory_id_set = set(torch.cat([_.ids for _ in trajectory_history_list[:-1]], dim=0).cpu().tolist())
            # Seq Decoding:
            pred_id_words, _ = self.seq_decoder(
                track_seqs=[trajectory_history_list]
            )
            id_confs = torch.softmax(pred_id_words, dim=-1)     # [1, N, K + 1]
            id_confs = id_confs[0]                              # [N, K + 1]

            ids = list()
            newborn_repeat = id_confs[:, -1:].repeat(1, len(id_confs) - 1)
            extended_id_confs = torch.cat((id_confs, newborn_repeat), dim=-1)
            match_rows, match_cols = linear_sum_assignment(1 - extended_id_confs.cpu())         # simple and efficient
            for _ in range(len(match_rows)):
                _id = match_cols[_]
                if _id not in trajectory_id_set:
                    ids.append(-1)
                elif _id >= num_id_vocabulary:
                    ids.append(-1)
                elif id_confs[match_rows[_], _id].item() < id_thresh:
                    ids.append(-1)
                else:
                    ids.append(_id)

            # Here is a customized implementation for ID assignment,
            # as an alternative to the Hungarian algorithm.
            # However, the Hungarian algorithm is more efficient and simpler (off-the-shelf package).
            # These two implementations only brings a slight difference in performance.
            # In our practice, < 0.3 HOTA on DanceTrack, < 0.1 HOTA on MOT17.
            # each_id_max_confs = torch.max(id_confs, dim=0).values
            # ids = list()
            # for i in range(len(id_confs)):
            #     target_id_confs, target_ids = torch.topk(id_confs[i], k=len(id_confs[0]))
            #     target_id = None    # final target ID word index
            #     for c in range(len(target_id_confs)):
            #         _id, _conf = target_ids[c].item(), target_id_confs[c].item()
            #         if _id == num_id_vocabulary:        # newborn
            #             target_id = -1
            #             break
            #         if _conf < id_thresh:
            #             break                           # early stop
            #         if _conf < each_id_max_confs[_id].item():
            #             continue                        # not the best choice
            #         else:
            #             if _id == num_id_vocabulary:
            #                 target_id = -1
            #             elif _id not in trajectory_id_set:
            #                 target_id = -1
            #             else:
            #                 target_id = _id
            #                 each_id_max_confs[_id] = 1.01   # hack implementation, avoid double assign
            #             break
            #     if target_id is None:
            #         target_id = -1
            #     ids.append(target_id)

        # Update the ID deque:
        for _id in ids:
            if _id != -1:
                id_deque.add(_id)

        # Filter the newborn targets, True means marked as newborn but not reach the newborn threshold:
        newborn_neg_filter = ((torch.tensor(ids).to(current[0].confs.device) == -1)
                              & (current[0].confs <= newborn_thresh).reshape(-1, ))

        if torch.sum(~newborn_neg_filter) > num_id_vocabulary:
            # The legal objects are too many, we need to filter out some of them.
            # Warning: This should not happen in normal cases.
            #          If it happens, you may increase the ID vocabulary size.
            print(f"[Warning!] There are too many objects, N={torch.sum(~newborn_neg_filter)}. ")
            already_ids_num = torch.sum(torch.tensor(ids) != -1)
            newborn_index = torch.tensor(ids).to(current[0].confs.device) == -1
            confs = current[0].confs.reshape(-1, ) * newborn_index.to(float)
            newborn_num_in_legal = num_id_vocabulary - already_ids_num
            index = torch.topk(confs, k=newborn_num_in_legal, dim=0).indices
            newborn_neg_filter_from_topk = torch.tensor(ids).to(current[0].confs.device) == -1
            newborn_neg_filter_from_topk[index] = False
            legal_newborn_neg_filter = newborn_neg_filter | newborn_neg_filter_from_topk
            newborn_neg_filter = legal_newborn_neg_filter
            print(f"[Warning!] Because the newborn objects are too many, "
                  f"we only keep {newborn_num_in_legal} newborn objects with highest confs. "
                  f"Already assigned {already_ids_num} IDs. "
                  f"Now we have {torch.sum(~newborn_neg_filter)} IDs.")

        # Just a check!
        assert torch.sum(~newborn_neg_filter) <= num_id_vocabulary, f"Too many IDs: {torch.sum(~newborn_neg_filter)}."

        # Remove the illegal newborn targets (conf < newborn_thresh):
        ids = torch.tensor(ids)[~newborn_neg_filter.cpu()].tolist()
        current[0] = current[0][~newborn_neg_filter]

        num_new_id = ids.count(-1)      # how many new ID words need to be assigned

        if num_new_id > 0:              # assign new ID words
            id_deque_list = list(id_deque)
            if len(id_deque_list) + num_new_id <= num_id_vocabulary:
                # The ID dictionary is not fully used, we can directly assign new ID words.
                new_ids = [len(id_deque_list) + _ for _ in range(num_new_id)]   # ID dictionary index (ID words)
            else:
                # The ID dictionary is fully used, we need to recycle some ID words.
                if len(id_deque_list) < num_id_vocabulary:
                    # There are still some empty slots in the ID dictionary,
                    # we can directly assign these clear_id_num_new_id new ID words.
                    clear_num_new_id = num_id_vocabulary - len(id_deque_list)
                    conflict_num_new_id = num_new_id - clear_num_new_id
                    new_ids = [len(id_deque_list) + _ for _ in range(clear_num_new_id)]
                else:
                    # There are no empty slots in the ID dictionary,
                    # we need to recycle conflict_num_new_id ID words.
                    conflict_num_new_id = num_new_id
                    new_ids = []
                # Recycled ID words:
                conflict_new_id = id_deque_list[:conflict_num_new_id]
                # As we need to recycle some ID words in conflict_new_id,
                # we need to remove the corresponding tracklets in the trajectory.
                for _ in range(len(trajectory)):
                    conflict_index = torch.zeros([len(trajectory[_]), ], dtype=torch.bool,
                                                 device=trajectory[_].ids.device)     # init
                    for _id in conflict_new_id:
                        conflict_index = conflict_index | (trajectory[_].ids == _id)
                    trajectory[_] = trajectory[_][~conflict_index]
                new_ids = new_ids + conflict_new_id     # assign the recycled ID words to "new_ids"

            # Update the corresponding mapping from ID words to ID labels (in tracker outputs):
            for _id in new_ids:
                ids_to_results[_id] = current_id
                current_id += 1
                id_deque.add(_id)

            # Insert the new_ids into the ids list:
            new_id_idx = 0
            ori_ids = ids
            ids = []
            for _ in ori_ids:
                if _ == -1:     # new id need to add:
                    ids.append(new_ids[new_id_idx])
                    new_id_idx += 1
                else:
                    ids.append(_)

        current[0].ids = torch.tensor(ids, dtype=torch.long, device=current[0].ids.device)
        trajectory_history_list = trajectory + current
        trajectory_history = deque(trajectory_history_list, maxlen=deque_max_length)
        assert len(ids) == len(set(ids)), f"ids is not unique: ids={ids}."
        return ids, trajectory_history, ids_to_results, current_id, id_deque, ~newborn_neg_filter
        # We will remove some illegal newborn targets in the outer function,
        # based on the "newborn_neg_filter" flags.

    def forward_train(
            self,
            track_history: list[list[Instances]],
            traj_drop_ratio: float,
            traj_switch_ratio: float,
            use_checkpoint: bool = False,
    ):
        assert len(track_history) == 1, f"Only BS=1 is supported."

        pred_id_words, gt_id_words = self.seq_decoder(
            track_seqs=track_history,
            traj_drop_ratio=traj_drop_ratio,
            traj_switch_ratio=traj_switch_ratio,
            use_checkpoint=use_checkpoint,
        )

        return pred_id_words, gt_id_words

    def add_random_id_words_to_instances(self, instances: list[Instances]):
        # assert len(instances) == 1  # only for bs=1
        ids = torch.cat([instance.ids for instance in instances], dim=0)
        ids_unique = torch.unique(ids)

        if len(ids_unique) > self.training_num_id:
            keep_index = torch.randperm(len(ids_unique))[:self.training_num_id]
            ids_unique = ids_unique[keep_index]
            pass
        id_words_unique = torch.randperm(n=self.num_id_vocabulary)[:len(ids_unique)]
        id_to_word = {
            i.item(): w.item() for i, w in zip(ids_unique, id_words_unique)
        }
        already_id_set = set()
        for t in range(len(instances)):
            id_words, id_labels = [], []
            for _ in range(len(instances[t])):
                i = instances[t].ids[_].item()
                if i in id_to_word:
                    id_words.append(id_to_word[i])
                else:   # handle the case that the number of objects exceeds the length of ID dictionary
                    id_words.append(-1)
                    id_labels.append(-1)
                    continue
                if i in already_id_set:
                    id_labels.append(id_to_word[i])
                else:
                    id_labels.append(self.num_id_vocabulary)
                    already_id_set.add(i)
            instances[t].id_words = torch.tensor(id_words, dtype=torch.long)
            instances[t].id_labels = torch.tensor(id_labels, dtype=torch.long)
            ins_keep_index = instances[t].id_words != -1
            instances[t] = instances[t][ins_keep_index]
        return

    def forward(self, frames, detr_checkpoint_frames: int | None = None, targets=None, max_pad=None):
        if detr_checkpoint_frames is not None:
            # Checkpoint will only be used in the training stage.
            detr_outputs = None
            for batch_frames in batch_iterator(detr_checkpoint_frames, frames):
                batch_frames = batch_frames[0]
                if targets is None:
                    _ = checkpoint(self.detr, batch_frames, use_reentrant=False)
                else:
                    if detr_outputs is None:
                        begin = 0
                    else:
                        begin = detr_outputs["outputs"].shape[0]
                    _ = checkpoint(self.detr, batch_frames, targets[begin:begin+batch_frames.tensors.shape[0]], max_pad, use_reentrant=False)
                if detr_outputs is None:
                    detr_outputs = _
                else:
                    detr_outputs = combine_detr_outputs(detr_outputs, _)
        else:
            if targets is None:
                detr_outputs = self.detr(frames)
            else:
                detr_outputs = self.detr(frames, targets, max_pad)

        return detr_outputs


def build(config: dict):
    return MOTIP(config=config)
