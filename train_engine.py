# Copyright (c) Ruopeng Gao. All Rights Reserved.
# ------------------------------------------------------------------------
import os
import torch
import wandb
import torch.nn as nn
import torch.distributed

from einops import rearrange
from structures.instances import Instances
from torch.utils.data import DataLoader
from models import build_model
from models.motip import MOTIP
from models.utils import save_checkpoint, load_checkpoint, load_detr_pretrain, get_model
from models.criterion import build as build_id_criterion
from data import build_dataset, build_sampler, build_dataloader
from utils.utils import labels_to_one_hot, is_distributed, distributed_rank, \
    combine_detr_outputs, detr_outputs_index_select, infos_to_detr_targets, batch_iterator, is_main_process
from utils.nested_tensor import nested_tensor_index_select
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from log.logger import Logger, ProgressLogger
from log.log import Metrics, TPS
from eval_engine import evaluate_one_epoch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint


def train(config: dict, logger: Logger):
    # Dataset:
    dataset_train = build_dataset(config=config)

    # Model
    model = build_model(config=config)
    if config["DETR_PRETRAIN"] is not None:
        load_detr_pretrain(model=model, pretrain_path=config["DETR_PRETRAIN"], num_classes=config["NUM_CLASSES"])
        logger.print(f"Load DETR pretrain model from {config['DETR_PRETRAIN']}.")
    else:
        logger.print("No pre-trained detr used.")

    # For optimizer:
    param_groups = get_param_groups(model=model, config=config)
    optimizer = AdamW(params=param_groups, lr=config["LR"], weight_decay=config["WEIGHT_DECAY"])

    # Criterion (Loss Function):
    id_criterion = build_id_criterion(config=config)

    # Scheduler:
    if config["SCHEDULER_TYPE"] == "MultiStep":
        scheduler = MultiStepLR(optimizer, milestones=config["SCHEDULER_MILESTONES"],
                                gamma=config["SCHEDULER_GAMMA"])
    else:
        raise RuntimeError(f"Do not support scheduler type {config['SCHEDULER_TYPE']}.")

    # Train States:
    train_states = {
        "start_epoch": 0,
        "global_iter": 0
    }

    # For resume:
    if config["RESUME_MODEL"] is not None:  # need to resume from checkpoint
        load_checkpoint(
            model=model,
            path=config["RESUME_MODEL"],
            optimizer=optimizer if config["RESUME_OPTIMIZER"] else None,
            scheduler=scheduler if config["RESUME_SCHEDULER"] else None,
            states=train_states if config["RESUME_STATES"] else None
        )
        # Different processing on scheduler:
        if config["RESUME_SCHEDULER"]:
            scheduler.step()
        else:
            for i in range(0, train_states["start_epoch"]):
                scheduler.step()
        logger.print(f"Resume from model {config['RESUME_MODEL']}. "
                     f"Optimizer={config['RESUME_OPTIMIZER']}, Scheduler={config['RESUME_SCHEDULER']}, "
                     f"States={config['RESUME_STATES']}")
        logger.save_log_to_file(f"Resume from model {config['RESUME_MODEL']}. "
                                f"Optimizer={config['RESUME_OPTIMIZER']}, Scheduler={config['RESUME_SCHEDULER']}, "
                                f"States={config['RESUME_STATES']}", mode="a")

    # Distributed, every gpu will share the same parameters.
    if is_distributed():
        model = DDP(model, device_ids=[distributed_rank()])

    for epoch in range(train_states["start_epoch"], config["EPOCHS"]):
        epoch_start_timestamp = TPS.timestamp()
        dataset_train.set_epoch(epoch)
        sampler_train = build_sampler(dataset=dataset_train, shuffle=True)
        dataloader_train = build_dataloader(
            dataset=dataset_train,
            sampler=sampler_train,
            batch_size=config["BATCH_SIZE"],
            num_workers=config["NUM_WORKERS"]
        )
        if is_distributed():
            sampler_train.set_epoch(epoch)

        # Train one epoch:
        train_metrics = train_one_epoch(
            config=config, model=model, logger=logger,
            dataloader=dataloader_train, id_criterion=id_criterion,
            optimizer=optimizer, epoch=epoch, states=train_states,
            clip_max_norm=config["CLIP_MAX_NORM"], detr_num_train_frames=config["DETR_NUM_TRAIN_FRAMES"],
            detr_checkpoint_frames=config["DETR_CHECKPOINT_FRAMES"],
            lr_warmup_epochs=0 if "LR_WARMUP_EPOCHS" not in config else config["LR_WARMUP_EPOCHS"]
        )
        lr = optimizer.state_dict()["param_groups"][-1]["lr"]
        train_metrics["learning_rate"].update(lr)
        train_metrics["learning_rate"].sync()
        time_per_epoch = TPS.format(TPS.timestamp() - epoch_start_timestamp)
        logger.print_metrics(
            metrics=train_metrics,
            prompt=f"[Epoch {epoch} Finish] [Total Time: {time_per_epoch}] ",
            fmt="{global_average:.4f}"
        )
        logger.save_metrics(
            metrics=train_metrics,
            prompt=f"[Epoch {epoch} Finish] [Total Time: {time_per_epoch}] ",
            fmt="{global_average:.4f}",
            statistic="global_average",
            global_step=train_states["global_iter"],
            prefix="epoch",
            x_axis_step=epoch,
            x_axis_name="epoch"
        )

        # Save checkpoint.
        if (epoch + 1) % config["SAVE_CHECKPOINT_PER_EPOCH"] == 0:
            save_checkpoint(model=model,
                            path=os.path.join(config["OUTPUTS_DIR"], f"checkpoint_{epoch}.pth"),
                            states=train_states,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            only_detr=config["TRAIN_STAGE"] == "only_detr",
                            )
            if config["INFERENCE_DATASET"] is not None:
                if config["TRAIN_STAGE"] == "only_detr":
                    eval_metrics = evaluate_one_epoch(
                        config=config,
                        model=model,
                        logger=logger,
                        dataset=config["INFERENCE_DATASET"],
                        data_split=config["INFERENCE_SPLIT"],
                        outputs_dir=os.path.join(config["OUTPUTS_DIR"], config["MODE"],
                                                 "eval_during_train", config["INFERENCE_SPLIT"], f"epoch_{epoch}"),
                        only_detr=True
                    )
                else:
                    eval_metrics = evaluate_one_epoch(
                        config=config,
                        model=model,
                        logger=logger,
                        dataset=config["INFERENCE_DATASET"],
                        data_split=config["INFERENCE_SPLIT"],
                        outputs_dir=os.path.join(config["OUTPUTS_DIR"], config["MODE"],
                                                 "eval_during_train", config["INFERENCE_SPLIT"], f"epoch_{epoch}"),
                        only_detr=False
                    )
                eval_metrics.sync()
                logger.print_metrics(
                    metrics=eval_metrics,
                    prompt=f"[Epoch {epoch} Eval] ",
                    fmt="{global_average:.4f}"
                )
                logger.save_metrics(
                    metrics=eval_metrics,
                    prompt=f"[Epoch {epoch} Eval] ",
                    fmt="{global_average:.4f}",
                    statistic="global_average",
                    global_step=train_states["global_iter"],
                    prefix="epoch",
                    x_axis_step=epoch,
                    x_axis_name="epoch"
                )

        # Next step.
        scheduler.step()

    return


def train_one_epoch(config: dict, model: MOTIP, logger: Logger,
                    dataloader: DataLoader, id_criterion: nn.Module,
                    optimizer: torch.optim,
                    epoch: int, states: dict, clip_max_norm: float, detr_num_train_frames: int,
                    detr_checkpoint_frames: int = 0, lr_warmup_epochs: int = 0):
    model.train()
    metrics = Metrics()   # save metrics
    memory_optimized_detr_criterion = config["MEMORY_OPTIMIZED_DETR_CRITERION"]
    checkpoint_detr_criterion = config["CHECKPOINT_DETR_CRITERION"]
    auto_memory_optimized_detr_criterion = config["AUTO_MEMORY_OPTIMIZED_DETR_CRITERION"]

    tps = TPS()             # save time per step

    device = torch.device(config["DEVICE"])

    # Check train stage:
    assert config["TRAIN_STAGE"] in ["only_detr", "only_decoder", "joint"], \
        f"Illegal train stage '{config['TRAIN_STAGE']}'."

    model_without_ddp = get_model(model)
    detr_params = []
    other_params = []
    for name, param in model_without_ddp.named_parameters():
        if "detr" in name:
            detr_params.append(param)
        else:
            other_params.append(param)

    optimizer.zero_grad()   # init optim
    for i, batch in enumerate(dataloader):
        if epoch < lr_warmup_epochs:
            # Do lr warmup:
            lr_warmup(optimizer=optimizer, epoch=epoch, iteration=i,
                      orig_lr=config["LR"], warmup_epochs=lr_warmup_epochs, iter_per_epoch=len(dataloader))

        iter_start_timestamp = TPS.timestamp()

        # prepare some meta info
        num_gts = sum([len(info["boxes"]) for info in batch["infos"][0]])

        B, T = len(batch["images"]), len(batch["images"][0])
        detr_num_train_frames = min(detr_num_train_frames, T)
        frames = batch["nested_tensors"]    # (B, T, C, H, W) for tensors
        infos = batch["infos"]
        detr_targets = infos_to_detr_targets(infos=infos, device=device)
        random_frame_idxs = torch.randperm(T)
        argsort_random_frame_idx = torch.argsort(random_frame_idxs)
        argsort_random_frame_idx_repeat = torch.cat([argsort_random_frame_idx + b * T for b in range(B)])
        detr_train_frame_idxs = random_frame_idxs[:detr_num_train_frames]
        detr_no_grad_frame_idxs = random_frame_idxs[detr_num_train_frames:]
        # Prepare frames for training:
        detr_train_frames = nested_tensor_index_select(frames, dim=1, index=detr_train_frame_idxs)
        detr_no_grad_frames = nested_tensor_index_select(frames, dim=1, index=detr_no_grad_frame_idxs)
        # (B, T) to (B*T):
        detr_train_frames.tensors = rearrange(detr_train_frames.tensors, "b t c h w -> (b t) c h w")
        detr_train_frames.mask = rearrange(detr_train_frames.mask, "b t h w -> (b t) h w")
        detr_train_frames = detr_train_frames.to(device)
        detr_no_grad_frames.tensors = rearrange(detr_no_grad_frames.tensors, "b t c h w -> (b t) c h w")
        detr_no_grad_frames.mask = rearrange(detr_no_grad_frames.mask, "b t h w -> (b t) h w")
        detr_no_grad_frames = detr_no_grad_frames.to(device)
        detr_train_targets = detr_no_grad_targets = None
        # DETR forward:
        # Without Train:
        if T > detr_num_train_frames:
            with torch.no_grad():
                if detr_checkpoint_frames > 0 and len(detr_no_grad_frames) > detr_checkpoint_frames * 4:
                    # To reduce CUDA memory usage:
                    detr_no_grad_outputs = None
                    # detr_no_grad_adapter_outputs = None
                    for batch_frames in batch_iterator(detr_checkpoint_frames * 4, detr_no_grad_frames):
                        batch_frames = batch_frames[0]
                        _ = model(frames=batch_frames)
                        if detr_no_grad_outputs is None:
                            detr_no_grad_outputs = _
                        else:
                            detr_no_grad_outputs = combine_detr_outputs(detr_no_grad_outputs, _)
                else:
                    detr_no_grad_outputs = model(frames=detr_no_grad_frames)
        else:
            detr_no_grad_outputs = None
        # Train:
        if detr_num_train_frames > 0:
            if detr_checkpoint_frames == 0 or len(detr_train_frames) <= detr_checkpoint_frames:
                detr_train_outputs = model(frames=detr_train_frames)
            else:
                detr_train_outputs = model(frames=detr_train_frames, detr_checkpoint_frames=detr_checkpoint_frames)
        else:
            detr_train_outputs = None
        if T > detr_num_train_frames:
            detr_outputs = combine_detr_outputs(detr_train_outputs, detr_no_grad_outputs)
        else:
            detr_outputs = detr_train_outputs
        detr_outputs = detr_outputs_index_select(detr_outputs, index=argsort_random_frame_idx_repeat.to(device))

        if memory_optimized_detr_criterion or (auto_memory_optimized_detr_criterion and num_gts > 2400):
            train_detr_outputs = detr_outputs_index_select(detr_outputs, index=detr_train_frame_idxs.to(device))
            train_detr_targets = [detr_targets[_] for _ in detr_train_frame_idxs.tolist()]
            detr_loss_dict, _ = get_model(model).detr_criterion(outputs=train_detr_outputs, targets=train_detr_targets)
            match_idxs = []
            with torch.no_grad():
                idxs = torch.arange(0, len(detr_targets), device=device)
                for idx in batch_iterator(4, idxs):
                    idx = idx[0]
                    outputs_without_aux = {k: v for k, v in detr_outputs_index_select(detr_outputs, index=idx).items() if
                                           k != 'aux_outputs' and k != 'enc_outputs'}
                    m = get_model(model).detr_criterion.matcher(
                        outputs=outputs_without_aux,
                        targets=[detr_targets[_] for _ in idx.tolist()]
                    )
                    match_idxs += m
                    pass
            pass
        else:
            if checkpoint_detr_criterion:
                detr_loss_dict, match_idxs = checkpoint(
                    get_model(model).detr_criterion,
                    detr_outputs, detr_targets,
                    use_reentrant=False
                )
            else:
                detr_loss_dict, match_idxs = get_model(model).detr_criterion(outputs=detr_outputs, targets=detr_targets)

        if config["TRAIN_STAGE"] == "only_detr":    # only train detr part:
            id_loss = None
        else:
            # MOTIP processing:
            match_instances = generate_match_instances(
                match_idxs=match_idxs, infos=infos, detr_outputs=detr_outputs
            )
            assert len(match_instances) == 1, f"For simplicity, only the case of bs=1 is implemented."
            # Generate field 'id_words' for instances:
            get_model(model).add_random_id_words_to_instances(instances=match_instances[0])
            pred_id_words, gt_id_words = get_model(model).forward_train(
                track_history=match_instances,
                traj_drop_ratio=config["TRAJ_DROP_RATIO"],
                traj_switch_ratio=config["TRAJ_SWITCH_RATIO"] if "TRAJ_SWITCH_RATIO" in config else 0.0,
                use_checkpoint=config["SEQ_DECODER_CHECKPOINT"],
            )
            id_loss = id_criterion(pred_id_words, gt_id_words)

        # Calculate the overall loss for barkward processing:
        detr_weight_dict = get_model(model).detr_criterion.weight_dict
        detr_loss = sum(detr_loss_dict[k] * detr_weight_dict[k] for k in detr_loss_dict.keys() if k in detr_weight_dict)
        if config["TRAIN_STAGE"] == "only_detr":    # only need detr loss:
            loss = detr_loss.clone()
        else:
            loss = detr_loss + id_loss * id_criterion.weight

        # Backward the loss:
        loss /= config["ACCUMULATE_STEPS"]
        loss.backward()

        # Add metrics to Log:
        metrics["overall_loss"].update(loss.item() * config["ACCUMULATE_STEPS"])
        metrics["overall_detr_loss"].update(detr_loss.item())
        metrics["bbox_l1"].update(detr_loss_dict["loss_bbox"].item())
        metrics["bbox_giou"].update(detr_loss_dict["loss_giou"].item())
        metrics["cls_loss"].update(detr_loss_dict["loss_ce"].item())
        if config["TRAIN_STAGE"] != "only_detr":    # log about id branch is also need to be written:
            metrics["overall_id_loss"].update(id_loss.item() * id_criterion.weight)
            metrics["id_loss"].update(id_loss.item())

        # Parameters update:
        if (i + 1) % config["ACCUMULATE_STEPS"] == 0:
            if clip_max_norm > 0:
                detr_grad_norm = torch.nn.utils.clip_grad_norm_(detr_params, clip_max_norm)
                other_grad_norm = torch.nn.utils.clip_grad_norm_(other_params, clip_max_norm)
                metrics["detr_grad_norm"].update(detr_grad_norm.item())
                metrics["other_grad_norm"].update(other_grad_norm.item())
            else:
                pass
            optimizer.step()
            optimizer.zero_grad()

        iter_end_timestamp = TPS.timestamp()
        tps.update(iter_end_timestamp - iter_start_timestamp)
        eta = tps.eta(total_steps=len(dataloader), current_steps=i)

        if (i % config["OUTPUTS_PER_STEP"] == 0) or (i == len(dataloader) - 1):
            metrics["learning_rate"].clear()
            metrics["learning_rate"].update(optimizer.state_dict()["param_groups"][-1]["lr"])
            metrics.sync()
            logger.print_metrics(
                metrics=metrics,
                prompt=f"[Epoch: {epoch}] [{i}/{len(dataloader)}] [tps: {tps.average:.2f}s] [eta: {TPS.format(eta)}] "
            )
            logger.save_metrics(
                metrics=metrics,
                prompt=f"[Epoch: {epoch}] [{i}/{len(dataloader)}] [tps: {tps.average:.2f}s] ",
                global_step=states["global_iter"],
            )

        states["global_iter"] += 1

    states["start_epoch"] += 1

    return metrics


def generate_match_instances(match_idxs, infos, detr_outputs):
    match_instances = []
    B, T = len(infos), len(infos[0])
    for b in range(B):
        match_instances.append([])
        for t in range(T):
            flat_idx = b * T + t
            output_idxs, info_idxs = match_idxs[flat_idx]
            instances = Instances(image_size=(0, 0))
            instances.ids = infos[b][t]["ids"][info_idxs]
            instances.gt_boxes = infos[b][t]["boxes"][info_idxs]
            instances.pred_boxes = detr_outputs["pred_boxes"][flat_idx][output_idxs]
            instances.outputs = detr_outputs["outputs"][flat_idx][output_idxs]
            match_instances[b].append(instances)
    return match_instances


def get_param_groups(model: nn.Module, config) -> list[dict]:
    def match_names(name, key_names):
        for key in key_names:
            if key in name:
                return True
        return False
    # keywords
    backbone_names = config["LR_BACKBONE_NAMES"]
    linear_proj_names = config["LR_LINEAR_PROJ_NAMES"]
    dictionary_names = [] if "LR_DICTIONARY_NAMES" not in config else config["LR_DICTIONARY_NAMES"]
    _dictionary_scale = 1.0 if "LR_DICTIONARY_SCALE" not in config else config["LR_DICTIONARY_SCALE"]
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() if match_names(n, backbone_names) and p.requires_grad],
            "lr_scale": config["LR_BACKBONE_SCALE"],
            "lr": config["LR"] * config["LR_BACKBONE_SCALE"]
        },
        {
            "params": [p for n, p in model.named_parameters() if match_names(n, linear_proj_names) and p.requires_grad],
            "lr_scale": config["LR_LINEAR_PROJ_SCALE"],
            "lr": config["LR"] * config["LR_LINEAR_PROJ_SCALE"]
        },
        {
            "params": [p for n, p in model.named_parameters() if match_names(n, dictionary_names) and p.requires_grad],
            "lr_scale": _dictionary_scale,
            "lr": config["LR"] * _dictionary_scale
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if not match_names(n, backbone_names)
                       and not match_names(n, linear_proj_names)
                       and not match_names(n, dictionary_names)
                       and p.requires_grad],
        }
    ]
    return param_groups


def lr_warmup(optimizer, epoch: int, iteration: int, orig_lr: float, warmup_epochs: int, iter_per_epoch: int):
    # min_lr = 1e-8
    total_warmup_iters = warmup_epochs * iter_per_epoch
    current_lr_ratio = (epoch * iter_per_epoch + iteration + 1) / total_warmup_iters
    current_lr = orig_lr * current_lr_ratio
    for param_grop in optimizer.param_groups:
        if "lr_scale" in param_grop:
            param_grop["lr"] = current_lr * param_grop["lr_scale"]
        else:
            param_grop["lr"] = current_lr
        pass
    return
