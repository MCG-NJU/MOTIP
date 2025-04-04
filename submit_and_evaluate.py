# Copyright (c) Ruopeng Gao. All Rights Reserved.
# About: Submit or evaluate the model.

import os
import time
import torch
import subprocess
from accelerate import Accelerator
from accelerate.state import PartialState
from torch.utils.data import DataLoader

from runtime_option import runtime_option
from utils.misc import yaml_to_dict
from configs.util import load_super_config, update_config
from log.logger import Logger
from data.joint_dataset import dataset_classes
from data.seq_dataset import SeqDataset
from models.runtime_tracker import RuntimeTracker
from log.log import Metrics
from models.motip import build as build_motip
from models.misc import load_checkpoint


def submit_and_evaluate(config: dict):
    # Init Accelerator at beginning:
    accelerator = Accelerator()
    state = PartialState()

    mode = config["INFERENCE_MODE"]
    assert mode in ["submit", "evaluate"], f"Mode {mode} is not supported."
    # Generate the output dir:
    assert "OUTPUTS_DIR" in config and config["OUTPUTS_DIR"] is not None, "OUTPUTS_DIR is not set."
    outputs_dir = config["OUTPUTS_DIR"]
    inference_group = config["INFERENCE_GROUP"]
    inference_dataset = config["INFERENCE_DATASET"]
    inference_split = config["INFERENCE_SPLIT"]
    inference_model = config["INFERENCE_MODEL"]
    _inference_model_name = os.path.split(inference_model)[-1][:-4]
    outputs_dir = os.path.join(
        outputs_dir, mode, inference_group, inference_dataset, inference_split, _inference_model_name
    )
    _is_outputs_dir_exist = os.path.exists(outputs_dir)
    accelerator.wait_for_everyone()
    os.makedirs(outputs_dir, exist_ok=True)

    # Init Logger, do not use wandb:
    logger = Logger(
        logdir=str(outputs_dir),
        use_wandb=False,
        config=config,
        # exp_owner=config["EXP_OWNER"],
        # exp_project=config["EXP_PROJECT"],
        # exp_group=config["EXP_GROUP"],
        # exp_name=config["EXP_NAME"],
    )
    # Log runtime config:
    logger.config(config=config)
    # Log other infos:
    logger.info(
        f"{mode.capitalize()} model: {inference_model}, inference dataset: {inference_dataset}, "
        f"inference split: {inference_split}, inference group: {inference_group}."
    )
    if _is_outputs_dir_exist:
        logger.warning(f"Outputs dir '{outputs_dir}' already exists, may overwrite the existing files.")
        time.sleep(5)   # wait for 5 seconds, give the user a chance to cancel.
    else:
        logger.info(f"Outputs dir '{outputs_dir}' created.")

    model, _ = build_motip(config=config)

    load_checkpoint(model, path=config["INFERENCE_MODEL"])

    model = accelerator.prepare(model)

    metrics = submit_and_evaluate_one_model(
        is_evaluate=config["INFERENCE_MODE"] == "evaluate",
        accelerator=accelerator,
        state=state,
        logger=logger,
        model=model,
        data_root=config["DATA_ROOT"],
        dataset=config["INFERENCE_DATASET"],
        data_split=config["INFERENCE_SPLIT"],
        outputs_dir=outputs_dir,
        image_max_longer=config["INFERENCE_MAX_LONGER"],    # the max shorter side of the image is set to 800 by default
        size_divisibility=config.get("SIZE_DIVISIBILITY", 0),
        use_sigmoid=config.get("USE_FOCAL_LOSS", False),
        assignment_protocol=config.get("ASSIGNMENT_PROTOCOL", "hungarian"),
        miss_tolerance=config["MISS_TOLERANCE"],
        det_thresh=config["DET_THRESH"],
        newborn_thresh=config["NEWBORN_THRESH"],
        id_thresh=config["ID_THRESH"],
        area_thresh=config.get("AREA_THRESH", 0),
        inference_only_detr=config["INFERENCE_ONLY_DETR"] if config["INFERENCE_ONLY_DETR"] is not None
        else config["ONLY_DETR"],
        dtype=config.get("INFERENCE_DTYPE", "FP32"),
    )

    if metrics is not None:
        metrics.sync()
        logger.metrics(
            log=f"Finish evaluation for model '{inference_model}', dataset '{inference_dataset}', "
                f"split '{inference_split}', group '{inference_group}': ",
            metrics=metrics,
            fmt="{global_average:.4f}",
        )
    return


def submit_and_evaluate_one_model(
        is_evaluate: bool,
        accelerator: Accelerator,
        state: PartialState,
        logger: Logger,
        model,
        data_root: str,
        dataset: str,
        data_split: str,
        # Outputs:
        outputs_dir: str,
        # Parameters with defaults:
        image_max_shorter: int = 800,
        image_max_longer: int = 1536,
        size_divisibility: int = 0,
        use_sigmoid: bool = False,
        assignment_protocol: str = "hungarian",
        miss_tolerance: int = 30,
        det_thresh: float = 0.5,
        newborn_thresh: float = 0.5,
        id_thresh: float = 0.1,
        area_thresh: int = 0,
        inference_only_detr: bool = False,
        dtype: str = "FP32",
):
    # Build the datasets:
    inference_dataset = dataset_classes[dataset](
        data_root=data_root,
        split=data_split,
        load_annotation=False,
    )
    # Set the dtype during inference:
    match dtype:
        case "FP32": dtype=torch.float32
        case "FP16": dtype=torch.float16
        case _: raise ValueError(f"Unknown dtype '{dtype}'.")
    # Filter out the sequences that will not be processed in this GPU (if we have multiple GPUs):
    _inference_sequence_names = list(inference_dataset.sequence_infos.keys())
    _inference_sequence_names.sort()
    # If we have multiple GPUs, we need to filter out the sequences that will not be processed in this GPU:
    # However, there is a special case that the number of GPUs is larger than the number of sequences:
    if len(_inference_sequence_names) <= state.process_index:
        logger.info(
            log=f"Number of sequences is smaller than the number of processes, "
                f"a fake sequence will be processed on process {state.process_index}.",
            only_main=False,
        )
        inference_dataset.sequence_infos = {
            _inference_sequence_names[0]: inference_dataset.sequence_infos[_inference_sequence_names[0]]
        }
        inference_dataset.image_paths = {
            _inference_sequence_names[0]: inference_dataset.image_paths[_inference_sequence_names[0]]
        }
        is_fake = True
    else:
        for _ in range(len(_inference_sequence_names)):
            if _ % state.num_processes != state.process_index:
                inference_dataset.sequence_infos.pop(_inference_sequence_names[_])
                inference_dataset.image_paths.pop(_inference_sequence_names[_])
        is_fake = False

    # Process each sequence:
    for sequence_name in inference_dataset.sequence_infos.keys():
        # break
        sequence_dataset = SeqDataset(
            seq_info=inference_dataset.sequence_infos[sequence_name],
            image_paths=inference_dataset.image_paths[sequence_name],
            max_shorter=image_max_shorter,
            max_longer=image_max_longer,
            size_divisibility=size_divisibility,
            dtype=dtype,
        )
        sequence_loader = DataLoader(
            dataset=sequence_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=lambda x: x[0],
        )
        # sequence_loader = accelerator.prepare(sequence_loader)
        sequence_wh = sequence_dataset.seq_hw()
        runtime_tracker = RuntimeTracker(
            model=model,
            sequence_hw=sequence_wh,
            use_sigmoid=use_sigmoid,
            assignment_protocol=assignment_protocol,
            miss_tolerance=miss_tolerance,
            det_thresh=det_thresh,
            newborn_thresh=newborn_thresh,
            id_thresh=id_thresh,
            area_thresh=area_thresh,
            only_detr=inference_only_detr,
            dtype=dtype,
        )
        if is_fake:
            logger.info(
                f"Fake submitting sequence {sequence_name} with {len(sequence_loader)} frames.",
                only_main=False
            )
        else:
            logger.info(f"Submitting sequence {sequence_name} with {len(sequence_loader)} frames.", only_main=False)
        sequence_results, sequence_fps = get_results_of_one_sequence(
            runtime_tracker=runtime_tracker,
            sequence_loader=sequence_loader,
            logger=logger,
        )
        # Write the results to the submit file:
        if dataset in ["DanceTrack", "SportsMOT", "MOT17", "PersonPath22_Inference", "BFT"]:
            sequence_tracker_results = []
            for t in range(len(sequence_results)):
                for obj_id, score, category, bbox in zip(
                        sequence_results[t]["id"],
                        sequence_results[t]["score"],
                        sequence_results[t]["category"],
                        sequence_results[t]["bbox"],    # [x, y, w, h]
                ):
                    sequence_tracker_results.append(
                        f"{t + 1},{obj_id.item()},"
                        f"{bbox[0].item()},{bbox[1].item()},{bbox[2].item()},{bbox[3].item()},"
                        f"1,-1,-1,-1\n"
                    )
            if not is_fake:
                os.makedirs(os.path.join(outputs_dir, "tracker"), exist_ok=True)
                with open(os.path.join(outputs_dir, "tracker", f"{sequence_name}.txt"), "w") as submit_file:
                    submit_file.writelines(sequence_tracker_results)
                logger.success(f"Submit sequence {sequence_name} done, FPS: {sequence_fps:.2f}. "
                               f"Saved to {os.path.join(outputs_dir, 'tracker', f'{sequence_name}.txt')}.",
                               only_main=False)
            else:
                logger.success(f"Fake submit sequence {sequence_name} done, FPS: {sequence_fps:.2f}.", only_main=False)
            pass
        else:
            raise NotImplementedError(f"Do not support to submit the results for dataset '{dataset}'.")

    # Post-process for submitting and evaluation:
    accelerator.wait_for_everyone()
    if not is_evaluate:
        logger.success(
            log=f"Submit done. Saved to {os.path.join(outputs_dir, 'tracker')}",
            only_main=True,
        )
        return None
    else:
        if accelerator.is_main_process:
            logger.info(
                log=f"Start evaluation...",
                only_main=True,
            )
            # Prepare for evaluation:
            if dataset in ["DanceTrack", "SportsMOT", "MOT17", "BFT"]:
                gt_dir = os.path.join(data_root, dataset, data_split)
                tracker_dir = os.path.join(outputs_dir, "tracker")
            elif dataset in ["PersonPath22_Inference"]:
                gt_dir = os.path.join(data_root, dataset, "gts", "person_path_22-test")
                tracker_dir = os.path.join(outputs_dir, "tracker")
            else:
                raise NotImplementedError(f"Do not support to find the gt_dir for dataset '{dataset}'.")
            if dataset in ["DanceTrack", "SportsMOT", "BFT"] or (dataset in ["MOT17"] and data_split == "test"):
                args = {
                    "--SPLIT_TO_EVAL": data_split,
                    "--METRICS": ["HOTA", "CLEAR", "Identity"],
                    "--GT_FOLDER": gt_dir,
                    "--SEQMAP_FILE": os.path.join(data_root, dataset, f"{data_split}_seqmap.txt"),
                    "--SKIP_SPLIT_FOL": "True",
                    "--TRACKERS_TO_EVAL": "",
                    "--TRACKER_SUB_FOLDER": "",
                    "--USE_PARALLEL": "True",
                    "--NUM_PARALLEL_CORES": "8",
                    "--PLOT_CURVES": "False",
                    "--TRACKERS_FOLDER": tracker_dir,
                }
                cmd = ["python", "TrackEval/scripts/run_mot_challenge.py"]
            elif dataset in ["PersonPath22_Inference"]:
                args = {
                    "--SPLIT_TO_EVAL": data_split,
                    "--METRICS": ["HOTA", "CLEAR", "Identity"],
                    "--GT_FOLDER": gt_dir,
                    "--USE_PARALLEL": "True",
                    "--NUM_PARALLEL_CORES": "8",
                    "--TRACKERS_FOLDER": tracker_dir,
                    "--BENCHMARK": "person_path_22",
                    "--SEQMAP_FILE": os.path.join(data_root, dataset, "gts", "seqmaps", "person_path_22-test.txt"),
                    "--SKIP_SPLIT_FOL": "True",
                    "--TRACKER_SUB_FOLDER": "",
                    "--TRACKERS_TO_EVAL": "",
                }
                cmd = ["python", "TrackEval/scripts/run_person_path_22.py"]
            else:
                raise NotImplementedError(
                    f"Do not support to eval the results for dataset '{dataset}' split '{data_split}'."
                )
            for k, v in args.items():
                cmd.append(k)
                if isinstance(v, list):
                    cmd += v
                else:
                    cmd.append(v)
            # Run the eval script:
            _ = subprocess.run(
                cmd,
            )
            # Check if the eval script is done:
            if _.returncode == 0:
                logger.success("Evaluation script is done.", only_main=True)
            else:
                raise RuntimeError("Evaluation script failed.")
        # Wait for all processes:
        accelerator.wait_for_everyone()
        # Get the metrics:
        eval_metrics_path = os.path.join(outputs_dir, "tracker", "pedestrian_summary.txt")
        eval_metrics_dict = get_eval_metrics_dict(metric_path=eval_metrics_path)
        metrics = Metrics()
        metrics["HOTA"].update(eval_metrics_dict["HOTA"])
        metrics["DetA"].update(eval_metrics_dict["DetA"])
        metrics["AssA"].update(eval_metrics_dict["AssA"])
        metrics["DetPr"].update(eval_metrics_dict["DetPr"])
        metrics["DetRe"].update(eval_metrics_dict["DetRe"])
        metrics["AssPr"].update(eval_metrics_dict["AssPr"])
        metrics["AssRe"].update(eval_metrics_dict["AssRe"])
        metrics["MOTA"].update(eval_metrics_dict["MOTA"])
        metrics["IDF1"].update(eval_metrics_dict["IDF1"])
        logger.success(
            log=f"Get evaluation metrics from {eval_metrics_path}.",
            only_main=True,
        )

        return metrics


@torch.no_grad()
def get_results_of_one_sequence(
        logger: Logger,
        runtime_tracker: RuntimeTracker,
        sequence_loader: DataLoader,
):
    tracker_results = []
    assert len(sequence_loader) > 10, "The sequence loader is too short."
    for t, (image, image_path) in enumerate(sequence_loader):
        if t == 10:
            begin_time = time.time()
        image.tensors = image.tensors.cuda()
        image.mask = image.mask.cuda()
        # image = nested_tensor_from_tensor_list(tensor_list=[image[0]])
        runtime_tracker.update(image=image)
        _results = runtime_tracker.get_track_results()
        tracker_results.append(_results)
    fps = (len(sequence_loader) - 10) / (time.time() - begin_time)
    return tracker_results, fps


def get_eval_metrics_dict(metric_path: str):
    with open(metric_path) as f:
        metric_names = f.readline()[:-1].split(" ")
        metric_values = f.readline()[:-1].split(" ")
    metrics = {
        n: float(v) for n, v in zip(metric_names, metric_values)
    }
    return metrics


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Get runtime option:
    opt = runtime_option()
    cfg = yaml_to_dict(opt.config_path)

    # Loading super config:
    if opt.super_config_path is not None:  # the runtime option is priority
        cfg = load_super_config(cfg, opt.super_config_path)
    else:  # if not, use the default super config path in the config file
        cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"])

    # Combine the config and runtime into config dict:
    cfg = update_config(config=cfg, option=opt)

    # Call the "train_engine" function:
    submit_and_evaluate(config=cfg)
