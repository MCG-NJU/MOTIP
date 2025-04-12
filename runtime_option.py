# Copyright (c) Ruopeng Gao. All Rights Reserved.
import argparse


def runtime_option():
    """
    Build a parser that can set up runtime options, such as choose device, data path, and so on.
    Every option in this parser should appear in .yaml config file.

    Returns:
        A parser.
    """
    parser = argparse.ArgumentParser("Network training and evaluation script.", add_help=True)

    # Config file.
    parser.add_argument("--config-path", type=str, default="./configs/r50_deformable_detr_motip_dancetrack.yaml")
    parser.add_argument("--super-config-path", type=str)

    # About system.
    # parser.add_argument("--device", type=str, help="Device.")
    parser.add_argument("--num-workers", type=int, help="Number of workers.")
    parser.add_argument("--prefetch-factor", type=int)
    parser.add_argument("--seed", type=int)

    # About data.
    parser.add_argument("--data-root", type=str, help="Data root path.")
    parser.add_argument("--dataset-weights", nargs="*", type=int)

    # About evaluation.
    parser.add_argument("--eval-model", type=str, help="Eval model path.")

    # About outputs.
    parser.add_argument("--outputs-dir", type=str, help="Outputs dir.")
    parser.add_argument("--exp-name", type=str, help="Exp name.")
    parser.add_argument("--exp-group", type=str, help="Exp group, for wandb.")
    parser.add_argument("--use-wandb", type=str, help="Whether use wandb.")

    # About training settings:
    parser.add_argument("--resume-model", type=str, help="Resume training model path.")
    parser.add_argument("--resume-optimizer", type=str)
    parser.add_argument("--resume-scheduler", type=str)
    parser.add_argument("--detr-pretrain", type=str)
    parser.add_argument("--only-detr", type=str)

    # About sampling:
    parser.add_argument("--sample-steps", type=int, nargs="+")
    parser.add_argument("--sample-lengths", type=int, nargs="+")
    parser.add_argument("--sample-intervals", type=int, nargs="+")
    parser.add_argument("--length-per-iteration", type=int)

    # About augmentation:
    parser.add_argument("--aug-max-size", type=int)
    parser.add_argument("--aug-max-shift-ratio", type=float)
    parser.add_argument("--aug-resize-scales", nargs="+", type=int)
    parser.add_argument("--aug-brightness", type=float)
    parser.add_argument("--aug-contrast", type=float)
    parser.add_argument("--aug-saturation", type=float)
    parser.add_argument("--aug-hue", type=float)
    parser.add_argument("--aug-color-jitter-v2", type=str)
    parser.add_argument("--aug-num-groups", type=int)
    parser.add_argument("--aug-trajectory-occlusion-prob", type=float)
    parser.add_argument("--aug-trajectory-switch-prob", type=float)
    parser.add_argument("--detr-num-train-frames", type=int)
    parser.add_argument("--detr-num-checkpoint-frames", type=int)
    parser.add_argument("--detr-criterion-batch-len", type=int)
    parser.add_argument("--use-decoder-checkpoint", type=str)
    parser.add_argument("--use-aux-loss", type=str)
    parser.add_argument("--use-shared-aux-head", type=str)
    parser.add_argument("--use-focal-loss", type=str)

    # Model settings:
    parser.add_argument("--ffn-dim-ratio", type=int)
    parser.add_argument("--rel-pe-length", type=int)
    parser.add_argument("--id-dim", type=int)
    parser.add_argument("--num-id-decoder-layers", type=int)

    # About inference.
    parser.add_argument("--inference-model", type=str, help="Inference model path.")
    parser.add_argument("--inference-mode", type=str)
    parser.add_argument("--inference-dataset", type=str)
    parser.add_argument("--inference-split", type=str)
    parser.add_argument("--inference-group", type=str)
    parser.add_argument("--inference-max-longer", type=int)
    parser.add_argument("--inference-dtype", type=str)
    parser.add_argument("--assignment-protocol", type=str)
    parser.add_argument("--miss-tolerance", type=int)
    parser.add_argument("--det-thresh", type=float)
    parser.add_argument("--newborn-thresh", type=float)
    parser.add_argument("--id-thresh", type=float)
    parser.add_argument("--area-thresh", type=int)

    # Hyperparams.
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--accumulate-steps", type=int)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--lr-warmup-epochs", type=int)
    parser.add_argument("--lr-dictionary-scale", type=float)
    parser.add_argument("--scheduler-milestones", type=int, nargs="+")
    parser.add_argument("--id-loss-weight", type=float)
    parser.add_argument("--num-training-ids", type=int)
    parser.add_argument("--separate-clip-norm", type=str)
    parser.add_argument("--max-clip-norm", type=float)
    parser.add_argument("--use-accelerate-clip-norm", type=str)

    # Logging:
    parser.add_argument("--git-version", type=str)
    parser.add_argument("--save-checkpoint-per-epoch", type=int)

    parser.add_argument("--use-previous-checkpoint", type=str)

    return parser.parse_args()
