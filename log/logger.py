# logger.py

import argparse
import json
import os
import random
from typing import Any, Dict, Optional

import mlflow
import wandb
import yaml
from accelerate.state import PartialState
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from data.joint_dataset import JointDataset
from log.log import Metrics
from utils.misc import is_main_process

state = PartialState()


class ProgressLogger:
    def __init__(
        self, total_len: int, prompt: Optional[str] = None, only_main: bool = True
    ):
        """
        Init a progress logger.
        """
        self.only_main = only_main
        self.is_activate = (self.only_main and is_main_process()) or (
            not self.only_main
        )

        if self.is_activate:
            self.total_len = total_len
            self.tqdm = tqdm(total=total_len)
            self.prompt = prompt or ""
        else:
            self.total_len = 0
            self.tqdm = None
            self.prompt = ""

    def update(self, step_len: int = 1, **kwargs: Any):
        if self.is_activate:
            if self.prompt:
                self.tqdm.set_description(self.prompt)
            if kwargs:
                self.tqdm.set_postfix(**kwargs)
            self.tqdm.update(step_len)


class Logger:
    """
    Log information to disk, Weights & Biases, and MLflow.
    """

    def __init__(
        self,
        logdir: str,
        use_wandb: bool = False,
        use_mlflow: bool = False,
        config: Optional[Dict[str, Any]] = None,
        exp_owner: Optional[str] = None,
        exp_project: Optional[str] = None,
        exp_group: Optional[str] = None,
        exp_name: Optional[str] = None,
    ):
        self.logdir = logdir
        self.use_wandb = use_wandb
        self.use_mlflow = use_mlflow

        if is_main_process():
            os.makedirs(self.logdir, exist_ok=True)

            # -- W&B init --
            if self.use_wandb:
                assert (
                    config and exp_owner and exp_project and exp_name
                ), "Must set config, owner, project, and name for W&B."
                if exp_group is None:
                    self.warning("No W&B group specified; defaulting to 'default'.")
                    exp_group = "default"
                self.wandb = wandb.init(
                    dir=self.logdir,
                    project=exp_project,
                    group=exp_group,
                    entity=exp_owner,
                    name=exp_name,
                    config=config,
                )
            else:
                self.wandb = None

            if self.use_mlflow:
                mlflow.set_experiment(exp_project)
                self.mlflow_run = mlflow.start_run(
                    run_name=exp_name, tags={"group": exp_group, "owner": exp_owner}
                )

                # Print the experiment ID and run ID
                print(f"Experiment ID: {self.mlflow_run.info.experiment_id}")
                print(f"Run ID: {self.mlflow_run.info.run_id}")

                if config:
                    mlflow.log_params({k: str(v) for k, v in config.items()})
                    self._save_config(config, filename="config.yaml")
                    mlflow.log_artifact(os.path.join(self.logdir, "config.yaml"))

        else:
            self.wandb = None
            self.mlflow_run = None

    def config(self, config: Dict[str, Any]):
        """Print, save, and log config."""
        self._print_config(config=config)
        self._save_config(config=config, filename="config.yaml")
        if self.use_mlflow and self.mlflow_run:
            mlflow.log_params({k: str(v) for k, v in config.items()})
            mlflow.log_artifact(os.path.join(self.logdir, "config.yaml"))

    def dataset(self, dataset: JointDataset):
        """Log dataset statistics."""
        stats = dataset.statistics()
        for stat in stats:
            msg = f"[Loaded Data] {stat}"
            self._print(log=self._colorize("[Loaded Data]", "success") + " " + stat)
            self._save(log=msg)
            if self.use_mlflow and self.mlflow_run:
                # parse "name: value" if possible
                if ":" in stat:
                    name, val = stat.split(":", 1)
                    name = name.strip().replace(" ", "_")
                    val = val.strip()
                    try:
                        mlflow.log_metric(name, float(val))
                    except ValueError:
                        mlflow.log_param(name, val)

    @state.on_main_process
    def _print_config(self, config: Dict[str, Any]):
        print(self._colorize("[Runtime Config]", "success"), end=" ")
        for k, v in config.items():
            print(f"{k.lower()}: {v} | ", end="")
        print()

    @state.on_main_process
    def _save_config(self, config: Dict[str, Any], filename: str = "config.yaml"):
        self._write_dict_to_yaml(config, filename=filename, mode="w")

    def info(self, log: str, only_main: bool = True):
        self._print(
            log=f"{self._colorize('[INFO]', 'info')} {log}", only_main=only_main
        )
        self._save(log=f"[INFO] {log}", only_main=only_main)

    def warning(self, log: str, only_main: bool = True):
        self._print(
            log=f"{self._colorize('[WARNING]', 'warning')} {log}", only_main=only_main
        )
        self._save(log=f"[WARNING] {log}", only_main=only_main)

    def success(self, log: str, only_main: bool = True):
        self._print(
            log=f"{self._colorize('[SUCCESS]', 'success')} {log}", only_main=only_main
        )
        self._save(log=f"[SUCCESS] {log}", only_main=only_main)

    def metrics(
        self,
        log: str,
        metrics: Metrics,
        fmt: Optional[str] = "{average:.4f} ({global_average:.4f})",
        statistic: Optional[str] = "average",
        global_step: int = 0,
        prefix: Optional[str] = None,
        x_axis_step: Optional[int] = None,
        x_axis_name: Optional[str] = None,
        filename: str = "log.txt",
        file_mode: str = "a",
        only_main: bool = True,
    ):
        self.print_metrics(
            metrics,
            prompt=f"{self._colorize('[Metrics]', 'info')} {log}",
            fmt=fmt,
            only_main=only_main,
        )
        self.save_metrics(
            metrics=metrics,
            prompt=f"[Metrics] {log}",
            fmt=fmt,
            statistic=statistic,
            global_step=global_step,
            prefix=prefix,
            x_axis_step=x_axis_step,
            x_axis_name=x_axis_name,
            filename=filename,
            file_mode=file_mode,
            only_main=only_main,
        )

    def _print(self, log: str, only_main: bool = True):
        if self._is_to_do(only_main):
            print(log)

    def _save(
        self,
        log: str,
        filename: str = "log.txt",
        mode: str = "a",
        only_main: bool = True,
        end: str = "\n",
    ):
        if self._is_to_do(only_main):
            with open(os.path.join(self.logdir, filename), mode=mode) as f:
                f.write(log + end)

    def print_metrics(
        self,
        metrics: Metrics,
        prompt: str = "",
        fmt: str = "{average:.4f} ({global_average:.4f})",
        only_main: bool = True,
    ):
        if self._is_to_do(only_main):
            print(prompt, end="")
            print(metrics.fmt(fmt=fmt))

    def save_metrics_to_file(
        self,
        metrics: Metrics,
        prompt: str = "",
        fmt: str = "{average:.4f} ({global_average:.4f})",
        filename: str = "log.txt",
        mode: str = "a",
        only_main: bool = True,
    ):
        if self._is_to_do(only_main):
            log = f"{prompt}{metrics.fmt(fmt=fmt)}"
            self._save(log=log, filename=filename, mode=mode)

    def save_metrics(
        self,
        metrics: Metrics,
        prompt: str = "",
        fmt: Optional[str] = "{average:.4f} ({global_average:.4f})",
        statistic: Optional[str] = "average",
        global_step: int = 0,
        prefix: Optional[str] = None,
        x_axis_step: Optional[int] = None,
        x_axis_name: Optional[str] = None,
        filename: str = "log.txt",
        file_mode: str = "a",
        only_main: bool = True,
    ):
        # Save to text file
        if fmt is not None:
            self.save_metrics_to_file(
                metrics, prompt, fmt, filename, file_mode, only_main
            )

        # Log to W&B
        if statistic is not None and self.wandb:
            self.save_metrics_to_wandb(metrics, statistic, global_step, prefix)
            if x_axis_step is not None:
                if x_axis_name is None:
                    raise RuntimeError(
                        "If x_axis_step is set, x_axis_name must be provided."
                    )
                self.wandb_log({x_axis_name: x_axis_step}, step=global_step)

        # Log to MLflow
        if statistic is not None and self.use_mlflow and self.mlflow_run:
            for name, m in metrics.metrics.items():
                metric_name = f"{prefix + '_' if prefix else ''}{name}"
                value = getattr(m, statistic)
                mlflow.log_metric(metric_name, value, step=global_step)

    def _write_dict_to_yaml(self, x: Dict[str, Any], filename: str, mode: str = "w"):
        with open(os.path.join(self.logdir, filename), mode=mode) as f:
            yaml.dump(x, f, allow_unicode=True)

    def _write_dict_to_json(self, log: Dict[str, Any], filename: str, mode: str = "w"):
        with open(os.path.join(self.logdir, filename), mode=mode) as f:
            f.write(json.dumps(log, indent=4))
            f.write("\n")

    @staticmethod
    def _is_to_do(only_main: bool = True) -> bool:
        return is_main_process() or not only_main

    @staticmethod
    def _colorize(log: str, log_type: str) -> str:
        if log_type == "info":
            return f"\033[1;36m{log}\033[0m"
        elif log_type == "warning":
            return f"\033[1;33m{log}\033[0m"
        elif log_type == "error":
            return f"\033[1;31m{log}\033[0m"
        elif log_type == "success":
            return f"\033[1;32m{log}\033[0m"
        else:
            raise ValueError(f"Unknown log type: {log_type}.")

    def wandb_log(self, data: Dict[str, Any], step: int):
        if self.wandb:
            wandb.log(data=data, step=step)
        if self.use_mlflow and self.mlflow_run:
            for k, v in data.items():
                mlflow.log_metric(k, v, step=step)

    def save_metrics_to_wandb(
        self,
        metrics: Metrics,
        statistic: str = "average",
        global_step: int = 0,
        prefix: Optional[str] = None,
    ):
        for name, value in metrics.metrics.items():
            metric_name = f"{prefix + '_' if prefix else ''}{name}"
            metric_value = getattr(value, statistic)
            self.wandb_log({metric_name: metric_value}, step=global_step)

    def log_images(
        self,
        orig_path: str,
        gt_ann: Dict[str, Any],
        pred_ann: Dict[str, Any],
        sequence_name: str,
        frame_idx: int,
        max_sequences: int = 5,
    ):
        """
        Log three views for a single frame:
          1) GT boxes + instance masks
          2) Predicted boxes
          3) Predicted instance masks
        """
        if not self.use_mlflow and not self.wandb:
            return

        # load original
        img = Image.open(orig_path).convert("RGB")
        W, H = img.size

        # prepare output folder
        img_dir = os.path.join(self.logdir, "images", sequence_name)
        os.makedirs(img_dir, exist_ok=True)

        # utility: pick a random color per object
        def rand_color(i):
            random.seed(i)
            return tuple(random.randint(0, 255) for _ in range(3))

        # 1) GT overlay
        img_gt = img.copy()
        draw = ImageDraw.Draw(img_gt, "RGBA")
        font = ImageFont.load_default()
        for i, (bbox, mask_path, obj_id) in enumerate(
            zip(gt_ann["bbox"], gt_ann["mask_paths"], gt_ann["id"])
        ):
            color = rand_color(obj_id)
            # paste mask
            mask = Image.open(mask_path).convert("L").resize((W, H))
            overlay = Image.new("RGBA", (W, H), color + (128,))
            img_gt.paste(overlay, (0, 0), mask)
            # draw box + id
            x, y, w, h = bbox
            draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
            draw.text((x, y - 10), str(obj_id), fill=color, font=font)

        # 2) Predicted boxes
        img_pb = img.copy()
        draw2 = ImageDraw.Draw(img_pb)
        for i, (bbox, obj_id) in enumerate(zip(pred_ann["bbox"], pred_ann["id"])):
            color = rand_color(obj_id)
            x, y, w, h = bbox
            draw2.rectangle([x, y, x + w, y + h], outline=color, width=2)
            draw2.text((x, y - 10), str(obj_id), fill=color, font=font)

        # 3) Predicted masks
        img_pm = img.copy()
        draw3 = ImageDraw.Draw(img_pm, "RGBA")
        for i, (mask_arr, obj_id) in enumerate(zip(pred_ann["masks"], pred_ann["id"])):
            color = rand_color(obj_id)
            mask = Image.fromarray((mask_arr * 255).astype("uint8")).resize((W, H))
            overlay = Image.new("RGBA", (W, H), color + (128,))
            img_pm.paste(overlay, (0, 0), mask)
            # draw id
            # approximate box from mask
            xs, ys = mask_arr.nonzero()
            if xs.size and ys.size:
                x0, y0 = ys.min(), xs.min()
                draw3.text((x0, y0 - 10), str(obj_id), fill=color, font=font)

        # artifact paths
        p_gt = os.path.join(img_dir, f"frame{frame_idx:04d}_gt.png")
        p_pb = os.path.join(img_dir, f"frame{frame_idx:04d}_pred_boxes.png")
        p_pm = os.path.join(img_dir, f"frame{frame_idx:04d}_pred_masks.png")
        img_gt.save(p_gt)
        img_pb.save(p_pb)
        img_pm.save(p_pm)

        # log to MLflow
        if self.use_mlflow and self.mlflow_run:
            mlflow.log_image(
                img_gt,
                artifact_file=os.path.join(
                    "images", sequence_name, os.path.basename(p_gt)
                ),
            )
            mlflow.log_image(
                img_pb,
                artifact_file=os.path.join(
                    "images", sequence_name, os.path.basename(p_pb)
                ),
            )
            mlflow.log_image(
                img_pm,
                artifact_file=os.path.join(
                    "images", sequence_name, os.path.basename(p_pm)
                ),
            )

        # log to W&B
        if self.wandb:
            self.wandb.log({f"{sequence_name}/gt": wandb.Image(img_gt)}, commit=False)
            self.wandb.log(
                {f"{sequence_name}/pred_boxes": wandb.Image(img_pb)}, commit=False
            )
            self.wandb.log(
                {f"{sequence_name}/pred_masks": wandb.Image(img_pm)}, commit=False
            )


def parser_to_dict(parser: argparse.Namespace) -> Dict[str, Any]:
    """
    Transform argparse namespace to a dict of non-falsy values.
    """
    opts = {}
    for k, v in vars(parser).items():
        if v:
            opts[k] = v
    return opts
