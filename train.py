# Copyright (c) Ruopeng Gao. All Rights Reserved.

import os
import math
import torch
import einops
from accelerate import Accelerator
from accelerate.state import PartialState
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from collections import defaultdict
from torchvision.transforms import v2
from typing import Any, Generator, List

from models.motip import build as build_motip
from models.motip.id_criterion import build as build_id_criterion
from runtime_option import runtime_option
from utils.misc import yaml_to_dict, set_seed
from configs.util import load_super_config, update_config
from log.logger import Logger
from data import build_dataset
from data.naive_sampler import NaiveSampler
from data.util import collate_fn
from log.log import TPS, Metrics
from models.misc import load_detr_pretrain, save_checkpoint, load_checkpoint
from models.misc import get_model
from utils.nested_tensor import NestedTensor
from submit_and_evaluate import submit_and_evaluate_one_model


def train_engine(config: dict):
    # Init some settings:
    assert "EXP_NAME" in config and config["EXP_NAME"] is not None, "Please set the experiment name."
    outputs_dir = config["OUTPUTS_DIR"] if config["OUTPUTS_DIR"] is not None \
        else os.path.join("./outputs/", config["EXP_NAME"])

    # Init Accelerator at beginning:
    accelerator = Accelerator()
    state = PartialState()
    # Also, we set the seed:
    set_seed(config["SEED"])
    # Set the sharing strategy (to avoid error: too many open files):
    torch.multiprocessing.set_sharing_strategy('file_system')   # if not, raise error: too many open files.

    # Init Logger:
    logger = Logger(
        logdir=os.path.join(outputs_dir, "train"),
        use_wandb=config["USE_WANDB"],
        config=config,
        exp_owner=config["EXP_OWNER"],
        exp_project=config["EXP_PROJECT"],
        exp_group=config["EXP_GROUP"],
        exp_name=config["EXP_NAME"],
    )
    logger.info(f"We init the logger at {logger.logdir}.")
    if config["USE_WANDB"] is False:
        logger.warning("The wandb is not used in this experiment.")
    logger.info(f"The distributed type is {state.distributed_type}.")
    logger.config(config=config)

    # Build training dataset:
    train_dataset = build_dataset(config=config)
    logger.dataset(train_dataset)
    # Build training data sampler:
    if "DATASET_WEIGHTS" in config:
        data_weights = defaultdict(lambda: defaultdict())
        for _ in range(len(config["DATASET_WEIGHTS"])):
            data_weights[config["DATASETS"][_]][config["DATASET_SPLITS"][_]] = config["DATASET_WEIGHTS"][_]
        data_weights = dict(data_weights)
    else:
        data_weights = None
    train_sampler = NaiveSampler(
        data_source=train_dataset,
        sample_steps=config["SAMPLE_STEPS"],
        sample_lengths=config["SAMPLE_LENGTHS"],
        sample_intervals=config["SAMPLE_INTERVALS"],
        length_per_iteration=config["LENGTH_PER_ITERATION"],
        data_weights=data_weights,
    )
    # Build training data loader:
    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=config["BATCH_SIZE"],
        num_workers=config["NUM_WORKERS"],
        prefetch_factor=config["PREFETCH_FACTOR"] if config["NUM_WORKERS"] > 0 else None,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Init the training states:
    train_states = {
        "start_epoch": 0,
        "global_step": 0
    }

    # Build MOTIP model:
    model, detr_criterion = build_motip(config=config)
    # Load the pre-trained DETR:
    load_detr_pretrain(
        model=model, pretrain_path=config["DETR_PRETRAIN"], num_classes=config["NUM_CLASSES"],
        default_class_idx=config["DETR_DEFAULT_CLASS_IDX"] if "DETR_DEFAULT_CLASS_IDX" in config else None,
    )
    logger.success(
        log=f"Load the pre-trained DETR from '{config['DETR_PRETRAIN']}'. "
    )
    # Build Loss Function:
    id_criterion = build_id_criterion(config=config)

    # Build Optimizer:
    if config["DETR_NUM_TRAIN_FRAMES"] == 0:
        for n, p in model.named_parameters():
            if "detr" in n:
                p.requires_grad = False     # only train the MOTIP part.
    param_groups = get_param_groups(model, config)
    optimizer = AdamW(
        params=param_groups,
        lr=config["LR"],
        weight_decay=config["WEIGHT_DECAY"],
    )
    scheduler = MultiStepLR(
        optimizer=optimizer,
        milestones=config["SCHEDULER_MILESTONES"],
        gamma=config["SCHEDULER_GAMMA"],
    )

    # Other infos:
    only_detr = config["ONLY_DETR"]

    # Resuming:
    if config["RESUME_MODEL"] is not None:
        load_checkpoint(
            model=model,
            path=config["RESUME_MODEL"],
            optimizer=optimizer if config["RESUME_OPTIMIZER"] else None,
            scheduler=scheduler if config["RESUME_SCHEDULER"] else None,
            states=train_states,
        )
        # Different processing on scheduler:
        if config["RESUME_SCHEDULER"]:
            scheduler.step()
        else:
            for _ in range(0, train_states["start_epoch"]):
                scheduler.step()
        logger.success(
            log=f"Resume the model from '{config['RESUME_MODEL']}', "
                f"optimizer={config['RESUME_OPTIMIZER']}, "
                f"scheduler={config['RESUME_SCHEDULER']}, "
                f"states={train_states}. "
                f"Start from epoch {train_states['start_epoch']}, step {train_states['global_step']}."
        )

    train_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, model, optimizer,
        # device_placement=[False]        # whether to place the data on the device
    )

    for epoch in range(train_states["start_epoch"], config["EPOCHS"]):
        logger.info(log=f"Start training epoch {epoch}.")
        epoch_start_timestamp = TPS.timestamp()
        # Prepare the sampler for the current epoch:
        train_sampler.prepare_for_epoch(epoch=epoch)
        # Train one epoch:
        train_metrics = train_one_epoch(
            accelerator=accelerator,
            logger=logger,
            states=train_states,
            epoch=epoch,
            dataloader=train_dataloader,
            model=model,
            detr_criterion=detr_criterion,
            id_criterion=id_criterion,
            optimizer=optimizer,
            only_detr=only_detr,
            lr_warmup_epochs=config["LR_WARMUP_EPOCHS"],
            lr_warmup_tgt_lr=config["LR"],
            detr_num_train_frames=config["DETR_NUM_TRAIN_FRAMES"],
            detr_num_checkpoint_frames=config["DETR_NUM_CHECKPOINT_FRAMES"],
            detr_criterion_batch_len=config.get("DETR_CRITERION_BATCH_LEN", 10),
            use_decoder_checkpoint=config["USE_DECODER_CHECKPOINT"],
            accumulate_steps=config["ACCUMULATE_STEPS"],
            separate_clip_norm=config.get("SEPARATE_CLIP_NORM", True),
            max_clip_norm=config.get("MAX_CLIP_NORM", 0.1),
            use_accelerate_clip_norm=config.get("USE_ACCELERATE_CLIP_NORM", True),
            # For multi last checkpoints:
            outputs_dir=outputs_dir,
            is_last_epochs=(epoch == config["EPOCHS"] - 1),
            multi_last_checkpoints=config["MULTI_LAST_CHECKPOINTS"],
        )

        # Get learning rate:
        lr = optimizer.state_dict()["param_groups"][-1]["lr"]
        train_metrics["lr"].update(lr)
        train_metrics["lr"].sync()
        time_per_epoch = TPS.format(TPS.timestamp() - epoch_start_timestamp)
        logger.metrics(
            log=f"[Finish epoch: {epoch}] [Time: {time_per_epoch}] ",
            metrics=train_metrics,
            fmt="{global_average:.4f}",
            statistic="global_average",
            global_step=train_states["global_step"],
            prefix="epoch",
            x_axis_step=epoch,
            x_axis_name="epoch",
        )

        # Save checkpoint:
        if (epoch + 1) % config["SAVE_CHECKPOINT_PER_EPOCH"] == 0:
            save_checkpoint(
                model=model,
                path=os.path.join(outputs_dir, f"checkpoint_{epoch}.pth"),
                states=train_states,
                optimizer=optimizer,
                scheduler=scheduler,
                only_detr=only_detr,
            )
            if config["INFERENCE_DATASET"] is not None:
                assert config["INFERENCE_SPLIT"] is not None, f"Please set the INFERENCE_SPLIT for inference."
                eval_metrics = submit_and_evaluate_one_model(
                    is_evaluate=True,
                    accelerator=accelerator,
                    state=state,
                    logger=logger,
                    model=model,
                    data_root=config["DATA_ROOT"],
                    dataset=config["INFERENCE_DATASET"],
                    data_split=config["INFERENCE_SPLIT"],
                    outputs_dir=os.path.join(outputs_dir, "train", "eval_during_train", f"epoch_{epoch}"),
                    image_max_longer=config["INFERENCE_MAX_LONGER"],
                    size_divisibility=config.get("SIZE_DIVISIBILITY", 0),
                    miss_tolerance=config["MISS_TOLERANCE"],
                    use_sigmoid=config["USE_FOCAL_LOSS"] if "USE_FOCAL_LOSS" in config else False,
                    assignment_protocol=config["ASSIGNMENT_PROTOCOL"] if "ASSIGNMENT_PROTOCOL" in config else "hungarian",
                    det_thresh=config["DET_THRESH"],
                    newborn_thresh=config["NEWBORN_THRESH"],
                    id_thresh=config["ID_THRESH"],
                    area_thresh=config["AREA_THRESH"],
                    inference_only_detr=config["INFERENCE_ONLY_DETR"] if config["INFERENCE_ONLY_DETR"] is not None
                    else config["ONLY_DETR"],
                )
                eval_metrics.sync()
                logger.metrics(
                    log=f"[Eval epoch: {epoch}] ",
                    metrics=eval_metrics,
                    fmt="{global_average:.4f}",
                    statistic="global_average",
                    global_step=train_states["global_step"],
                    prefix="epoch",
                    x_axis_step=epoch,
                    x_axis_name="epoch",
                )

        logger.success(log=f"Finish training epoch {epoch}.")
        # Prepare for next step:
        scheduler.step()
    pass


def train_one_epoch(
        # Infos:
        accelerator: Accelerator,
        logger: Logger,
        states: dict,
        epoch: int,
        dataloader: DataLoader,
        model,
        detr_criterion,
        id_criterion,
        optimizer,
        only_detr,
        lr_warmup_epochs: int,
        lr_warmup_tgt_lr: float,
        detr_num_train_frames: int,
        detr_num_checkpoint_frames: int,
        detr_criterion_batch_len: int,
        use_decoder_checkpoint: bool,
        accumulate_steps: int = 1,
        separate_clip_norm: bool = True,
        max_clip_norm: float = 0.1,
        use_accelerate_clip_norm: bool = True,
        logging_interval: int = 20,
        # For multi last checkpoints:
        outputs_dir: str = None,
        is_last_epochs: bool = False,
        multi_last_checkpoints: int = 0,
):
    current_last_checkpoint_idx = 0

    model.train()
    tps = TPS()     # time per step
    metrics = Metrics()
    optimizer.zero_grad()
    step_timestamp = tps.timestamp()
    device = accelerator.device
    _B = dataloader.batch_sampler.batch_size
    _num_gts_per_frame = 0

    # Prepare for gradient clip norm:
    model_without_ddp = get_model(model)
    detr_params = []
    other_params = []
    for name, param in model_without_ddp.named_parameters():
        if "detr" in name:
            detr_params.append(param)
        else:
            other_params.append(param)

    for step, samples in enumerate(dataloader):
        images, annotations, metas = samples["images"], samples["annotations"], samples["metas"]
        # Normalize the images:
        # (Normally, it should be done in the dataloader, but here we do it in the training loop (on cuda).)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        images.tensors = v2.functional.to_dtype(images.tensors, dtype=torch.float32, scale=True)
        images.tensors = v2.functional.normalize(images.tensors, mean=mean, std=std)
        # A hack implementation to recover 0.0 in the masked regions:
        images.tensors = images.tensors * (~images.mask[:, :, None, ...]).to(torch.float32)
        images.tensors = images.tensors.contiguous()

        # Learning rate warmup:
        if epoch < lr_warmup_epochs:
            # Do warmup:
            lr_warmup(
                optimizer=optimizer,
                epoch=epoch, curr_iter=step, tgt_lr=lr_warmup_tgt_lr,
                warmup_epochs=lr_warmup_epochs, num_iter_per_epoch=len(dataloader),
            )

        _B, _T = len(annotations), len(annotations[0])
        detr_num_train_frames = min(detr_num_train_frames, _T)

        # Prepare the DETR targets from the annotations:
        detr_targets_flatten = annotations_to_flatten_detr_targets(annotations=annotations, device=device)

        # Select the training and no_grad frames:
        random_frame_idxs = torch.randperm(_T, device=device)   # use these random indices to select the frames.
        go_back_frame_idxs = torch.argsort(random_frame_idxs)   # use these indices to go back to the original order.
        go_back_frame_idxs_flatten = torch.cat([
            go_back_frame_idxs + _T * b for b in range(_B)
        ])      # only used for the DETR's criterion.
        # Split random_frame_idxs into training and no_grad frame indices:
        detr_train_frame_idxs = random_frame_idxs[:detr_num_train_frames]
        detr_no_grad_frame_idxs = random_frame_idxs[detr_num_train_frames:]

        detr_outputs_flatten_idxs = torch.arange(_B * _T, device=device)
        detr_outputs_flatten_idxs = einops.rearrange(detr_outputs_flatten_idxs, "(b t) -> b t", b=_B)
        detr_outputs_flatten_idxs = torch.cat([
            einops.rearrange(detr_outputs_flatten_idxs[:, :detr_num_train_frames], "b t -> (b t)"),
            einops.rearrange(detr_outputs_flatten_idxs[:, detr_num_train_frames:], "b t -> (b t)"),
        ], dim=0)
        detr_outputs_flatten_go_back_idxs = torch.argsort(detr_outputs_flatten_idxs)
        pass
        # Select the training and no_grad frames:
        detr_train_frames = nested_tensor_index_select(images, dim=1, index=detr_train_frame_idxs)
        detr_no_grad_frames = nested_tensor_index_select(images, dim=1, index=detr_no_grad_frame_idxs)

        # Prepare for the DETR forward function, turn the (B, T, ...) images to (B*T, ...) (or said flatten):
        detr_train_frames.tensors = einops.rearrange(detr_train_frames.tensors, "b t c h w -> (b t) c h w").contiguous()
        detr_train_frames.mask = einops.rearrange(detr_train_frames.mask, "b t h w -> (b t) h w").contiguous()
        detr_no_grad_frames.tensors = einops.rearrange(detr_no_grad_frames.tensors, "b t c h w -> (b t) c h w").contiguous()
        detr_no_grad_frames.mask = einops.rearrange(detr_no_grad_frames.mask, "b t h w -> (b t) h w").contiguous()

        # TODO: For DeNoise (e.g., in DINO-DETR),
        #       need to split the detr_targets_flatten into training and no_grad parts.

        # DETR forward:
        # 1. no_grad frames:
        if _T > detr_num_train_frames:      # do have no_grad frames (if not, skip this part)
            with torch.no_grad():
                if detr_num_checkpoint_frames == 0 or detr_num_checkpoint_frames * 4 >= len(detr_no_grad_frames):
                    # Directly forward the no_grad frames:
                    detr_no_grad_outputs = model(frames=detr_no_grad_frames, part="detr")
                else:
                    # Split the no_grad frames into batched iterations (reduce the memory usage):
                    detr_no_grad_outputs = None
                    for batch_samples in batch_iterator(
                        detr_num_checkpoint_frames * 4,
                        detr_no_grad_frames,
                    ):
                        batch_frames = batch_samples[0]
                        _ = model(frames=batch_frames, part="detr")
                        detr_no_grad_outputs = tensor_dict_cat(detr_no_grad_outputs, _, dim=0)
        else:                               # no no_grad frames
            detr_no_grad_outputs = None

        # 2. training frames:
        if detr_num_train_frames > 0:
            if detr_num_checkpoint_frames == 0 or detr_num_checkpoint_frames >= len(detr_train_frames):
                # Directly forward the training frames:
                detr_train_outputs = model(frames=detr_train_frames, part="detr")
            else:
                # Split the training frames into batched iterations (reduce the memory usage):
                detr_train_outputs = None
                for batch_samples in batch_iterator(
                    detr_num_checkpoint_frames,
                    detr_train_frames,
                ):
                    batch_frames = batch_samples[0]
                    _ = model(frames=batch_frames, part="detr", use_checkpoint=True)
                    detr_train_outputs = tensor_dict_cat(detr_train_outputs, _, dim=0)
        else:
            detr_train_outputs = None

        # Combine training and no_grad outputs:
        detr_outputs = tensor_dict_cat(detr_train_outputs, detr_no_grad_outputs, dim=0)
        # Recover the order of the outputs:
        detr_outputs = tensor_dict_index_select(detr_outputs, index=detr_outputs_flatten_go_back_idxs, dim=0)
        detr_outputs = tensor_dict_index_select(detr_outputs, index=go_back_frame_idxs_flatten, dim=0)

        # DETR criterion:
        detr_loss_dict, detr_indices = detr_criterion(outputs=detr_outputs, targets=detr_targets_flatten, batch_len=detr_criterion_batch_len)

        # Whether to only train the DETR, OR to train the MOTIP together:
        if not only_detr:
            _G, _, _N = annotations[0][0]["trajectory_id_labels"].shape
            # Need to prepare for MOTIP:
            seq_info = prepare_for_motip(
                detr_outputs=detr_outputs, annotations=annotations, detr_indices=detr_indices,
            )
            seq_info = model(seq_info=seq_info, part="trajectory_modeling")
            id_logits, id_gts, id_masks = model(
                seq_info=seq_info,
                part="id_decoder",
                use_decoder_checkpoint=use_decoder_checkpoint,
            )
            id_loss = id_criterion(id_logits=id_logits, id_labels=id_gts, id_masks=id_masks)
            _num_gts_per_frame = max(_num_gts_per_frame, id_gts.shape[-1])
            # print(f"Num of GTs per frame: {_num_gts_per_frame}")
            pass
        else:
            id_loss = None

        # Backward:
        with accelerator.autocast():
            detr_weight_dict = detr_criterion.weight_dict
            detr_loss = sum(
                detr_loss_dict[k] * detr_weight_dict[k] for k in detr_loss_dict.keys() if k in detr_weight_dict
            )
            loss = detr_loss + (id_loss if id_loss is not None else 0) * id_criterion.weight
            # Logging losses:
            metrics.update(name="loss", value=loss.item())
            metrics.update(name="detr_loss", value=detr_loss.item())
            if id_loss is not None:
                metrics.update(name="id_loss", value=id_loss.item())
            for k, v in detr_loss_dict.items():
                metrics.update(name=k, value=v.item())
            loss /= accumulate_steps
            accelerator.backward(loss)  # use this line to replace loss.backward()
            if (step + 1) % accumulate_steps == 0:
                if use_accelerate_clip_norm:
                    if separate_clip_norm:
                        detr_grad_norm = accelerator.clip_grad_norm_(detr_params, max_norm=max_clip_norm)
                        other_grad_norm = accelerator.clip_grad_norm_(other_params, max_norm=max_clip_norm)
                    else:
                        detr_grad_norm = other_grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=max_clip_norm)
                else:
                    if separate_clip_norm:
                        accelerator.unscale_gradients()
                        detr_grad_norm = torch.nn.utils.clip_grad_norm_(detr_params, max_clip_norm)
                        other_grad_norm = torch.nn.utils.clip_grad_norm_(other_params, max_clip_norm)
                    else:
                        accelerator.unscale_gradients()
                        detr_grad_norm = other_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_clip_norm)
                # Hack implementation to log grad_norm
                metrics.update(name="detr_grad_norm", value=detr_grad_norm.item())
                metrics.update(name="other_grad_norm", value=other_grad_norm.item())
                optimizer.step()
                optimizer.zero_grad()

        # Logging:
        tps.update(tps=tps.timestamp() - step_timestamp)
        step_timestamp = tps.timestamp()
        # Logging:
        if step % logging_interval == 0:
            # logger.info(f"[Epoch: {epoch}] [{step}/{total_steps}] [tps: {tps.average:.2f}s]")
            # Get learning rate for current step:
            _lr = optimizer.state_dict()["param_groups"][-1]["lr"]
            # Get the GPU memory usage:
            torch.cuda.synchronize()
            _cuda_memory = torch.cuda.max_memory_allocated(device) / 1024 / 1024
            _cuda_memory = torch.tensor([_cuda_memory], device=device)
            # _cuda_memory_reduce = accelerator.reduce(_cuda_memory, reduction="none")
            _gathered_cuda_memory = accelerator.gather(_cuda_memory)
            _max_cuda_memory = _gathered_cuda_memory.max().item()
            accelerator.wait_for_everyone()
            # Clear some values:
            metrics["lr"].clear()  # clear the learning rate value from last step
            metrics["max_cuda_mem(MB)"].clear()
            # Update them to the metrics:
            metrics.update(name="lr", value=_lr)
            metrics.update(name="max_cuda_mem(MB)", value=_max_cuda_memory)
            # Sync the metrics:
            metrics.sync()
            eta = tps.eta(total_steps=len(dataloader), current_steps=step)
            logger.metrics(
                log=f"[Epoch: {epoch}] [{step}/{len(dataloader)}] "
                    f"[tps: {tps.average:.2f}s] [eta: {TPS.format(eta)}] ",
                metrics=metrics,
                global_step=states["global_step"],
            )
        # For multi last checkpoints:
        if is_last_epochs and multi_last_checkpoints > 0:
            if (step + 1) == int(math.ceil((len(dataloader) / multi_last_checkpoints) * (current_last_checkpoint_idx + 1))):
                _dir = os.path.join(outputs_dir, "multi_last_checkpoints")
                os.makedirs(_dir, exist_ok=True)
                save_checkpoint(
                    model=model,
                    path=os.path.join(_dir, f"last_checkpoint_{current_last_checkpoint_idx}.pth"),
                    states=states,
                    optimizer=None,
                    scheduler=None,
                    only_detr=only_detr,
                )
                logger.info(
                    log=f"Save the last checkpoint {current_last_checkpoint_idx} at step {step}."
                )
                current_last_checkpoint_idx += 1
        # Update the counters:
        states["global_step"] += 1
    states["start_epoch"] += 1
    return metrics


def get_param_groups(model, config) -> list[dict]:
    def _match_names(_name, _key_names):
        for _k in _key_names:
            if _k in _name:
                return True
        return False

    # Keywords:
    backbone_names = config["LR_BACKBONE_NAMES"]
    linear_proj_names = config["LR_LINEAR_PROJ_NAMES"]
    dictionary_names = config["LR_DICTIONARY_NAMES"]
    pass
    # Param groups:
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() if _match_names(n, backbone_names) and p.requires_grad],
            "lr_scale": config["LR_BACKBONE_SCALE"],
            "lr": config["LR"] * config["LR_BACKBONE_SCALE"]
        },
        {
            "params": [p for n, p in model.named_parameters() if _match_names(n, linear_proj_names) and p.requires_grad],
            "lr_scale": config["LR_LINEAR_PROJ_SCALE"],
            "lr": config["LR"] * config["LR_LINEAR_PROJ_SCALE"]
        },
        {
            "params": [p for n, p in model.named_parameters() if _match_names(n, dictionary_names) and p.requires_grad],
            "lr_scale": config["LR_DICTIONARY_SCALE"],
            "lr": config["LR"] * config["LR_DICTIONARY_SCALE"]
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if not _match_names(n, backbone_names)
                       and not _match_names(n, linear_proj_names)
                       and not _match_names(n, dictionary_names)
                       and p.requires_grad],
        }
    ]
    return param_groups


def lr_warmup(optimizer, epoch: int, curr_iter: int, tgt_lr: float, warmup_epochs: int, num_iter_per_epoch: int):
    # min_lr = 1e-8
    total_warmup_iters = warmup_epochs * num_iter_per_epoch
    current_lr_ratio = (epoch * num_iter_per_epoch + curr_iter + 1) / total_warmup_iters
    current_lr = tgt_lr * current_lr_ratio
    for param_grop in optimizer.param_groups:
        if "lr_scale" in param_grop:
            param_grop["lr"] = current_lr * param_grop["lr_scale"]
        else:
            param_grop["lr"] = current_lr
        pass
    return


def annotations_to_flatten_detr_targets(annotations: list, device):
    """
    Args:
        annotations: annotations from the dataloader.
        device: move the targets to the device.

    Returns:
        A list of targets for the DETR model supervision, len=(B*T).
    """
    targets = []
    for annotation in annotations:      # scan by batch
        for ann in annotation:          # scan by frame
            targets.append(
                {
                    "boxes": ann["bbox"].to(device),
                    "labels": ann["category"].to(device),
                }
            )
    return targets


def nested_tensor_index_select(nested_tensor: NestedTensor, dim: int, index: torch.Tensor):
    tensors, mask = nested_tensor.decompose()
    _device = tensors.device
    index = index.to(_device)
    selected_tensors = torch.index_select(input=tensors, dim=dim, index=index).contiguous()
    selected_mask = torch.index_select(input=mask, dim=dim, index=index).contiguous()
    return NestedTensor(tensors=selected_tensors, mask=selected_mask)


def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    assert len(args) > 0 and all(
        len(a) == len(args[0]) for a in args
    ), "Batched iteration must have inputs of all the same size."
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size: (b + 1) * batch_size] for arg in args]


def tensor_dict_cat(tensor_dict1, tensor_dict2, dim=0):
    if tensor_dict1 is None or tensor_dict2 is None:
        assert tensor_dict1 is not None or tensor_dict2 is not None, "One of the tensor dict should be not None."
        return tensor_dict1 if tensor_dict2 is None else tensor_dict2
    else:
        res_tensor_dict = defaultdict()
        for k in tensor_dict1.keys():
            if isinstance(tensor_dict1[k], torch.Tensor):
                res_tensor_dict[k] = torch.cat([tensor_dict1[k], tensor_dict2[k]], dim=dim)
            elif isinstance(tensor_dict1[k], dict):
                res_tensor_dict[k] = tensor_dict_cat(tensor_dict1[k], tensor_dict2[k], dim=dim)
            elif isinstance(tensor_dict1[k], list):
                assert len(tensor_dict1[k]) == len(tensor_dict2[k]), "The list should have the same length."
                res_tensor_dict[k] = [
                    tensor_dict_cat(tensor_dict1[k][_], tensor_dict2[k][_], dim=dim)
                    for _ in range(len(tensor_dict1[k]))
                ]
            else:
                raise ValueError(f"Unsupported type {type(tensor_dict1[k])} in the tensor dict concat.")
        return dict(res_tensor_dict)


def tensor_dict_index_select(tensor_dict, index, dim=0):
    res_tensor_dict = defaultdict()
    for k in tensor_dict.keys():
        if isinstance(tensor_dict[k], torch.Tensor):
            res_tensor_dict[k] = torch.index_select(tensor_dict[k], index=index, dim=dim).contiguous()
        elif isinstance(tensor_dict[k], dict):
            res_tensor_dict[k] = tensor_dict_index_select(tensor_dict[k], index=index, dim=dim)
        elif isinstance(tensor_dict[k], list):
            res_tensor_dict[k] = [
                tensor_dict_index_select(tensor_dict[k][_], index=index, dim=dim)
                for _ in range(len(tensor_dict[k]))
            ]
        else:
            raise ValueError(f"Unsupported type {type(tensor_dict[k])} in the tensor dict index select.")
    return dict(res_tensor_dict)


def prepare_for_motip(detr_outputs, annotations, detr_indices):
    _B, _T = len(annotations), len(annotations[0])
    _G, _, _N = annotations[0][0]["trajectory_id_labels"].shape
    _device = detr_outputs["pred_logits"].device
    _feature_dim = detr_outputs["outputs"].shape[-1]
    # Init corresponding variables:
    trajectory_id_labels = - torch.ones((_B, _G, _T, _N), dtype=torch.int64, device=_device)
    trajectory_times = - torch.ones((_B, _G, _T, _N), dtype=torch.int64, device=_device)
    trajectory_masks = torch.ones((_B, _G, _T, _N), dtype=torch.bool, device=_device)
    trajectory_boxes = torch.zeros((_B, _G, _T, _N, 4), dtype=torch.float32, device=_device)
    trajectory_features = torch.zeros((_B, _G, _T, _N, _feature_dim), dtype=torch.float32, device=_device)
    unknown_id_labels = - torch.ones((_B, _G, _T, _N), dtype=torch.int64, device=_device)
    unknown_times = - torch.ones((_B, _G, _T, _N), dtype=torch.int64, device=_device)
    unknown_masks = torch.ones((_B, _G, _T, _N), dtype=torch.bool, device=_device)
    unknown_boxes = torch.zeros((_B, _G, _T, _N, 4), dtype=torch.float32, device=_device)
    unknown_features = torch.zeros((_B, _G, _T, _N, _feature_dim), dtype=torch.float32, device=_device)
    for b in range(_B):
        for t in range(_T):
            flatten_idx = b * _T + t
            go_back_detr_idxs = torch.argsort(detr_indices[flatten_idx][1])
            detr_output_embeds = detr_outputs["outputs"][flatten_idx][detr_indices[flatten_idx][0][go_back_detr_idxs]]
            detr_boxes = detr_outputs["pred_boxes"][flatten_idx][detr_indices[flatten_idx][0][go_back_detr_idxs]]
            # detr_output_embeds = einops.repeat(detr_output_embeds, "n d -> g n d", g=_G)
            # detr_boxes = einops.repeat(detr_boxes, "n d -> g n d", g=_G)
            for group in range(_G):
                _curr_traj_ann_idxs = annotations[b][t]["trajectory_ann_idxs"][group, 0, :]
                _curr_unk_ann_idxs = annotations[b][t]["unknown_ann_idxs"][group, 0, :]
                _curr_traj_masks = annotations[b][t]["trajectory_id_masks"][group, 0, :]
                _curr_unk_masks = annotations[b][t]["unknown_id_masks"][group, 0, :]
                # Fill the fields:
                trajectory_id_labels[b, group, t] = annotations[b][t]["trajectory_id_labels"][group, 0, :]
                unknown_id_labels[b, group, t] = annotations[b][t]["unknown_id_labels"][group, 0, :]
                trajectory_times[b, group, t] = annotations[b][t]["trajectory_times"][group, 0, :]
                unknown_times[b, group, t] = annotations[b][t]["unknown_times"][group, 0, :]
                trajectory_masks[b, group, t] = _curr_traj_masks
                unknown_masks[b, group, t] = _curr_unk_masks
                trajectory_features[b, group, t, ~_curr_traj_masks] = detr_output_embeds[_curr_traj_ann_idxs[~_curr_traj_masks]]
                unknown_features[b, group, t, ~_curr_unk_masks] = detr_output_embeds[_curr_unk_ann_idxs[~_curr_unk_masks]]
                trajectory_boxes[b, group, t, ~_curr_traj_masks] = detr_boxes[_curr_traj_ann_idxs[~_curr_traj_masks]]
                unknown_boxes[b, group, t, ~_curr_unk_masks] = detr_boxes[_curr_unk_ann_idxs[~_curr_unk_masks]]
                pass
            pass
    return {
        "trajectory_id_labels": trajectory_id_labels,
        "trajectory_times": trajectory_times,
        "trajectory_masks": trajectory_masks,
        "trajectory_boxes": trajectory_boxes,
        "trajectory_features": trajectory_features,
        "unknown_id_labels": unknown_id_labels,
        "unknown_times": unknown_times,
        "unknown_masks": unknown_masks,
        "unknown_boxes": unknown_boxes,
        "unknown_features": unknown_features,
    }


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # from issue: https://github.com/pytorch/pytorch/issues/11201
    # import torch.multiprocessing
    # torch.multiprocessing.set_sharing_strategy('file_system')

    # Get runtime option:
    opt = runtime_option()
    cfg = yaml_to_dict(opt.config_path)

    # Loading super config:
    if opt.super_config_path is not None:   # the runtime option is priority
        cfg = load_super_config(cfg, opt.super_config_path)
    else:                                   # if not, use the default super config path in the config file
        cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"])

    # Combine the config and runtime into config dict:
    cfg = update_config(config=cfg, option=opt)

    # Call the "train_engine" function:
    train_engine(config=cfg)
