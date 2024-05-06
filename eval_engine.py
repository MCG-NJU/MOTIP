# Copyright (c) Ruopeng Gao. All Rights Reserved.
# ------------------------------------------------------------------------
import os

import torch
import torch.distributed
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel as DDP

from models import build_model
from models.utils import load_checkpoint
from log.logger import Logger, ProgressLogger
from log.log import Metrics
from utils.utils import is_distributed, distributed_rank, yaml_to_dict, \
    distributed_world_size, is_main_process, distributed_world_rank
from submit_engine import submit_one_seq, get_seq_names


def evaluate(config: dict, logger: Logger):
    """
    Evaluate a model.

    Args:
        config:
        logger:
    Returns:

    """
    model_config = yaml_to_dict(path=config["INFERENCE_CONFIG_PATH"])
    model = build_model(config=model_config)
    load_checkpoint(model, path=config["INFERENCE_MODEL"])

    # If DDP:
    if is_distributed():
        model = DDP(model, device_ids=[distributed_rank()])

    if config["INFERENCE_GROUP"] is not None:
        eval_outputs_dir = os.path.join(config["OUTPUTS_DIR"], config["MODE"], config["INFERENCE_GROUP"],
                                        config["INFERENCE_SPLIT"],
                                        f'{config["INFERENCE_MODEL"].split("/")[-1][:-4]}')
    else:
        eval_outputs_dir = os.path.join(config["OUTPUTS_DIR"], config["MODE"], "default", config["INFERENCE_SPLIT"],
                                        f'{config["INFERENCE_MODEL"].split("/")[-1][:-4]}')
    eval_metrics = evaluate_one_epoch(
        config=config,
        model=model,
        logger=logger,
        dataset=config["INFERENCE_DATASET"],
        data_split=config["INFERENCE_SPLIT"],
        outputs_dir=eval_outputs_dir,
        only_detr=config["INFERENCE_ONLY_DETR"]
    )
    eval_metrics.sync()
    logger.save_metrics(
        metrics=eval_metrics,
        prompt=f"[Eval Checkpoint '{config['INFERENCE_MODEL']}'] ",
        fmt="{global_average:.4f}",
        statistic=None
    )

    return


@torch.no_grad()
def evaluate_one_epoch(config: dict, model: nn.Module,
                       logger: Logger, dataset: str, data_split: str,
                       outputs_dir: str, only_detr: bool = False):
    model.eval()
    metrics = Metrics()
    device = config["DEVICE"]

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
    else:
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

    tracker_dir = os.path.join(outputs_dir, "tracker")
    dataset_dir = os.path.join(config["DATA_ROOT"], dataset)
    if dataset in ["DanceTrack", "SportsMOT"]:
        gt_dir = os.path.join(dataset_dir, data_split)
    elif dataset in ["MOT17_SPLIT", "MOT15", "MOT15_V2", "MOT17"]:
        gt_dir = os.path.join(dataset_dir, data_split)
    else:
        raise NotImplementedError(f"Do not support to find the gt_dir for dataset '{dataset}'.")

    if is_distributed():
        torch.distributed.barrier()

    if is_main_process():
        # Need to eval the submit tracker:
        if dataset == "DanceTrack" or dataset == "SportsMOT":
            os.system(f"python3 TrackEval/scripts/run_mot_challenge.py --SPLIT_TO_EVAL {data_split}  "
                      f"--METRICS HOTA CLEAR Identity  --GT_FOLDER {gt_dir} "
                      f"--SEQMAP_FILE {os.path.join(dataset_dir, f'{data_split}_seqmap.txt')} "
                      f"--SKIP_SPLIT_FOL True --TRACKERS_TO_EVAL '' --TRACKER_SUB_FOLDER ''  --USE_PARALLEL True "
                      f"--NUM_PARALLEL_CORES 8 --PLOT_CURVES False "
                      f"--TRACKERS_FOLDER {tracker_dir}")
        elif dataset == "MOT17" and data_split == "test":
            os.system(f"python3 TrackEval/scripts/run_mot_challenge.py --SPLIT_TO_EVAL {data_split}  "
                      f"--METRICS HOTA CLEAR Identity  --GT_FOLDER {gt_dir} "
                      f"--SEQMAP_FILE {os.path.join(dataset_dir, f'{data_split}_seqmap.txt')} "
                      f"--SKIP_SPLIT_FOL True --TRACKERS_TO_EVAL '' --TRACKER_SUB_FOLDER ''  --USE_PARALLEL True "
                      f"--NUM_PARALLEL_CORES 8 --PLOT_CURVES False "
                      f"--TRACKERS_FOLDER {tracker_dir}")
        elif dataset == "MOT17_SPLIT" or dataset == "MOT17":
            os.system(f"python3 TrackEval/scripts/run_mot_challenge.py --SPLIT_TO_EVAL {data_split}  "
                      f"--METRICS HOTA CLEAR Identity  --GT_FOLDER {gt_dir} "
                      f"--SEQMAP_FILE {os.path.join(dataset_dir, f'{data_split}_seqmap.txt')} "
                      f"--SKIP_SPLIT_FOL True --TRACKERS_TO_EVAL '' --TRACKER_SUB_FOLDER ''  --USE_PARALLEL True "
                      f"--NUM_PARALLEL_CORES 8 --PLOT_CURVES False "
                      f"--TRACKERS_FOLDER {tracker_dir} --BENCHMARK MOT17")
        elif dataset == "MOT15" or dataset == "MOT15_V2":
            os.system(f"python3 TrackEval/scripts/run_mot_challenge.py --SPLIT_TO_EVAL {data_split}  "
                      f"--METRICS HOTA CLEAR Identity  --GT_FOLDER {gt_dir} "
                      f"--SEQMAP_FILE {os.path.join(dataset_dir, f'{data_split}_seqmap.txt')} "
                      f"--SKIP_SPLIT_FOL True --TRACKERS_TO_EVAL '' --TRACKER_SUB_FOLDER ''  --USE_PARALLEL True "
                      f"--NUM_PARALLEL_CORES 8 --PLOT_CURVES False "
                      f"--TRACKERS_FOLDER {tracker_dir} --BENCHMARK MOT15")
        else:
            raise NotImplementedError(f"Do not support to eval the results for dataset '{dataset}'.")

    if is_distributed():
        torch.distributed.barrier()
    # Get eval Metrics:
    eval_metric_path = os.path.join(tracker_dir, "pedestrian_summary.txt")
    eval_metrics_dict = get_eval_metrics_dict(metric_path=eval_metric_path)
    metrics["HOTA"].update(eval_metrics_dict["HOTA"])
    metrics["DetA"].update(eval_metrics_dict["DetA"])
    metrics["AssA"].update(eval_metrics_dict["AssA"])
    metrics["DetPr"].update(eval_metrics_dict["DetPr"])
    metrics["DetRe"].update(eval_metrics_dict["DetRe"])
    metrics["AssPr"].update(eval_metrics_dict["AssPr"])
    metrics["AssRe"].update(eval_metrics_dict["AssRe"])
    metrics["MOTA"].update(eval_metrics_dict["MOTA"])
    metrics["IDF1"].update(eval_metrics_dict["IDF1"])

    return metrics


def get_eval_metrics_dict(metric_path: str):
    with open(metric_path) as f:
        metric_names = f.readline()[:-1].split(" ")
        metric_values = f.readline()[:-1].split(" ")
    metrics = {
        n: float(v) for n, v in zip(metric_names, metric_values)
    }
    return metrics
