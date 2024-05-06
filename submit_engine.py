# Copyright (c) RuopengGao. All Rights Reserved.
# About:
import os
import json

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data import DataLoader
from data.seq_dataset import SeqDataset
from utils.nested_tensor import tensor_list_to_nested_tensor
from models.utils import get_model
from utils.box_ops import box_cxcywh_to_xyxy
from collections import deque
from structures.instances import Instances
from structures.ordered_set import OrderedSet
from log.logger import Logger
from utils.utils import yaml_to_dict, is_distributed, distributed_rank, distributed_world_size
from models import build_model
from models.utils import load_checkpoint


def submit(config: dict, logger: Logger):
    """
    Submit a model for a specific dataset.
    :param config:
    :param logger:
    :return:
    """
    if config["INFERENCE_CONFIG_PATH"] is None:
        model_config = config
    else:
        model_config = yaml_to_dict(path=config["INFERENCE_CONFIG_PATH"])
    model = build_model(config=model_config)
    load_checkpoint(model, path=config["INFERENCE_MODEL"])

    if is_distributed():
        model = DDP(model, device_ids=[distributed_rank()])

    if config["INFERENCE_GROUP"] is not None:
        submit_outputs_dir = os.path.join(config["OUTPUTS_DIR"], config["MODE"], config["INFERENCE_GROUP"],
                                          config["INFERENCE_SPLIT"],
                                          f'{config["INFERENCE_MODEL"].split("/")[-1][:-4]}')
    else:
        submit_outputs_dir = os.path.join(config["OUTPUTS_DIR"], config["MODE"], "default",
                                          config["INFERENCE_SPLIT"],
                                          f'{config["INFERENCE_MODEL"].split("/")[-1][:-4]}')

    # 需要调度整个 submit 流程
    submit_one_epoch(
        config=config,
        model=model,
        logger=logger,
        dataset=config["INFERENCE_DATASET"],
        data_split=config["INFERENCE_SPLIT"],
        outputs_dir=submit_outputs_dir,
        only_detr=config["INFERENCE_ONLY_DETR"]
    )

    logger.print(log=f"Finish submit process for model '{config['INFERENCE_MODEL']}' on the {config['INFERENCE_DATASET']} {config['INFERENCE_SPLIT']} set, outputs are write to '{os.path.join(submit_outputs_dir, 'tracker')}/.'")
    logger.save_log_to_file(
        log=f"Finish submit process for model '{config['INFERENCE_MODEL']}' on the {config['INFERENCE_DATASET']} {config['INFERENCE_SPLIT']} set, outputs are write to '{os.path.join(submit_outputs_dir, 'tracker')}/.'",
        filename="log.txt",
        mode="a"
    )

    return


@torch.no_grad()
def submit_one_epoch(config: dict, model: nn.Module,
                     logger: Logger, dataset: str, data_split: str,
                     outputs_dir: str, only_detr: bool = False):
    model.eval()

    all_seq_names = get_seq_names(data_root=config["DATA_ROOT"], dataset=dataset, data_split=data_split)
    seq_names = [all_seq_names[_] for _ in range(len(all_seq_names))
                 if _ % distributed_world_size() == distributed_rank()]

    if len(seq_names) > 0:
        for seq in seq_names:
            submit_one_seq(
                model=model, dataset=dataset,
                seq_dir=os.path.join(config["DATA_ROOT"], dataset, data_split, seq),
                only_detr=only_detr, max_temporal_length=config["MAX_TEMPORAL_LENGTH"],
                outputs_dir=outputs_dir,
                det_thresh=config["DET_THRESH"],
                newborn_thresh=config["DET_THRESH"] if "NEWBORN_THRESH" not in config else config["NEWBORN_THRESH"],
                area_thresh=config["AREA_THRESH"], id_thresh=config["ID_THRESH"],
                image_max_size=config["INFERENCE_MAX_SIZE"] if "INFERENCE_MAX_SIZE" in config else 1333,
                inference_ensemble=config["INFERENCE_ENSEMBLE"] if "INFERENCE_ENSEMBLE" in config else 0,
            )
    else:   # fake submit, will not write any outputs.
        submit_one_seq(
            model=model, dataset=dataset,
            seq_dir=os.path.join(config["DATA_ROOT"], dataset, data_split, all_seq_names[0]),
            only_detr=only_detr, max_temporal_length=config["MAX_TEMPORAL_LENGTH"],
            outputs_dir=outputs_dir,
            det_thresh=config["DET_THRESH"],
            newborn_thresh=config["DET_THRESH"] if "NEWBORN_THRESH" not in config else config["NEWBORN_THRESH"],
            area_thresh=config["AREA_THRESH"], id_thresh=config["ID_THRESH"],
            image_max_size=config["INFERENCE_MAX_SIZE"] if "INFERENCE_MAX_SIZE" in config else 1333,
            fake_submit=True,
            inference_ensemble=config["INFERENCE_ENSEMBLE"] if "INFERENCE_ENSEMBLE" in config else 0,
        )

    if is_distributed():
        torch.distributed.barrier()

    return


@torch.no_grad()
def submit_one_seq(
            model: nn.Module, dataset: str, seq_dir: str, outputs_dir: str,
            only_detr: bool, max_temporal_length: int = 0,
            det_thresh: float = 0.5, newborn_thresh: float = 0.5, area_thresh: float = 100, id_thresh: float = 0.1,
            image_max_size: int = 1333,
            fake_submit: bool = False,
            inference_ensemble: int = 0,
        ):
    os.makedirs(os.path.join(outputs_dir, "tracker"), exist_ok=True)
    seq_dataset = SeqDataset(seq_dir=seq_dir, dataset=dataset, width=image_max_size)
    seq_dataloader = DataLoader(seq_dataset, batch_size=1, num_workers=4, shuffle=False)
    # seq_name = seq_dir.split("/")[-1]
    seq_name = os.path.split(seq_dir)[-1]
    device = model.device
    current_id = 0
    ids_to_results = {}
    id_deque = OrderedSet()     # an ID deque for inference, the ID will be recycled if the dictionary is not enough.

    # Trajectory history:
    if only_detr:
        trajectory_history = None
    else:
        trajectory_history = deque(maxlen=max_temporal_length)

    if fake_submit:
        print(f"[Fake] Start >> Submit seq {seq_name.split('/')[-1]}, {len(seq_dataloader)} frames ......")
    else:
        print(f"Start >> Submit seq {seq_name.split('/')[-1]}, {len(seq_dataloader)} frames ......")

    for i, (image, ori_image) in enumerate(seq_dataloader):
        ori_h, ori_w = ori_image.shape[1], ori_image.shape[2]
        frame = tensor_list_to_nested_tensor([image[0]]).to(device)
        detr_outputs = model(frames=frame)
        detr_logits = detr_outputs["pred_logits"]
        detr_scores = torch.max(detr_logits, dim=-1).values.sigmoid()
        detr_det_idxs = detr_scores > det_thresh        # filter by the detection threshold
        detr_det_logits = detr_logits[detr_det_idxs]
        detr_det_labels = torch.max(detr_det_logits, dim=-1).indices
        detr_det_boxes = detr_outputs["pred_boxes"][detr_det_idxs]
        detr_det_outputs = detr_outputs["outputs"][detr_det_idxs]   # detr output embeddings
        area_legal_idxs = (detr_det_boxes[:, 2] * ori_w * detr_det_boxes[:, 3] * ori_h) > area_thresh   # filter by area
        detr_det_outputs = detr_det_outputs[area_legal_idxs]
        detr_det_boxes = detr_det_boxes[area_legal_idxs]
        detr_det_logits = detr_det_logits[area_legal_idxs]
        detr_det_labels = detr_det_labels[area_legal_idxs]

        # De-normalize to target image size:
        box_results = detr_det_boxes.cpu() * torch.tensor([ori_w, ori_h, ori_w, ori_h])
        box_results = box_cxcywh_to_xyxy(boxes=box_results)

        if only_detr is False:
            if len(box_results) > get_model(model).num_id_vocabulary:
                print(f"[Carefully!] we only support {get_model(model).num_id_vocabulary} ids, "
                      f"but get {len(box_results)} detections in seq {seq_name.split('/')[-1]} {i+1}th frame.")

        # Decoding the current objects' IDs
        if only_detr is False:
            assert max_temporal_length - 1 > 0, f"MOTIP need at least T=1 trajectory history, " \
                                                f"but get T={max_temporal_length - 1} history in Eval setting."
            current_tracks = Instances(image_size=(0, 0))
            current_tracks.boxes = detr_det_boxes
            current_tracks.outputs = detr_det_outputs
            current_tracks.ids = torch.tensor([get_model(model).num_id_vocabulary] * len(current_tracks),
                                              dtype=torch.long, device=current_tracks.outputs.device)
            current_tracks.confs = detr_det_logits.sigmoid()
            trajectory_history.append(current_tracks)
            if len(trajectory_history) == 1:    # first frame, do not need decoding:
                newborn_filter = (trajectory_history[0].confs > newborn_thresh).reshape(-1, )   # filter by newborn
                trajectory_history[0] = trajectory_history[0][newborn_filter]
                box_results = box_results[newborn_filter.cpu()]
                ids = torch.tensor([current_id + _ for _ in range(len(trajectory_history[-1]))],
                                   dtype=torch.long, device=current_tracks.outputs.device)
                trajectory_history[-1].ids = ids
                for _ in ids:
                    ids_to_results[_.item()] = current_id
                    current_id += 1
                id_results = []
                for _ in ids:
                    id_results.append(ids_to_results[_.item()])
                    id_deque.add(_.item())
                id_results = torch.tensor(id_results, dtype=torch.long)
            else:
                ids, trajectory_history, ids_to_results, current_id, id_deque, boxes_keep = get_model(model).inference(
                    trajectory_history=trajectory_history,
                    num_id_vocabulary=get_model(model).num_id_vocabulary,
                    ids_to_results=ids_to_results,
                    current_id=current_id,
                    id_deque=id_deque,
                    id_thresh=id_thresh,
                    newborn_thresh=newborn_thresh,
                    inference_ensemble=inference_ensemble,
                )   # already update the trajectory history/ids_to_results/current_id/id_deque in this function
                id_results = []
                for _ in ids:
                    id_results.append(ids_to_results[_])
                id_results = torch.tensor(id_results, dtype=torch.long)
                if boxes_keep is not None:
                    box_results = box_results[boxes_keep.cpu()]
        else:   # only detr, ID is just +1 for each detection.
            id_results = torch.tensor([current_id + _ for _ in range(len(box_results))], dtype=torch.long)
            current_id += len(id_results)

        # Output to tracker file:
        if fake_submit is False:
            # Write the outputs to the tracker file:
            result_file_path = os.path.join(outputs_dir, "tracker", f"{seq_name}.txt")
            with open(result_file_path, "a") as file:
                assert len(id_results) == len(box_results), f"Boxes and IDs should in the same length, " \
                                                            f"but get len(IDs)={len(id_results)} and " \
                                                            f"len(Boxes)={len(box_results)}"
                for obj_id, box in zip(id_results, box_results):
                    obj_id = int(obj_id.item())
                    x1, y1, x2, y2 = box.tolist()
                    if dataset in ["DanceTrack", "MOT17", "SportsMOT", "MOT17_SPLIT", "MOT15", "MOT15_V2"]:
                        result_line = f"{i + 1}," \
                                      f"{obj_id}," \
                                      f"{x1},{y1},{x2 - x1},{y2 - y1},1,-1,-1,-1\n"
                    else:
                        raise NotImplementedError(f"Do not know the outputs format of dataset '{dataset}'.")
                    file.write(result_line)
    if fake_submit:
        print(f"[Fake] Finish >> Submit seq {seq_name.split('/')[-1]}. ")
    else:
        print(f"Finish >> Submit seq {seq_name.split('/')[-1]}. ")
    return


def get_seq_names(data_root: str, dataset: str, data_split: str):
    if dataset in ["DanceTrack", "SportsMOT", "MOT17", "MOT17_SPLIT"]:
        dataset_dir = os.path.join(data_root, dataset, data_split)
        return sorted(os.listdir(dataset_dir))
    else:
        raise NotImplementedError(f"Do not support dataset '{dataset}' for eval dataset.")
