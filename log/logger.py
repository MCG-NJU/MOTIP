# Copyright (c) Ruopeng Gao. All Rights Reserved.

import os
import json
import argparse
import yaml
import wandb

from tqdm import tqdm
from typing import Any

from log.log import Metrics
from utils.misc import is_main_process
from accelerate.state import PartialState
from data.joint_dataset import JointDataset


state = PartialState()


class ProgressLogger:
    def __init__(self, total_len: int, prompt: str = None, only_main: bool = True):
        """
        Init a progress logger.
        """
        self.only_main = only_main
        self.is_activate = (self.only_main and is_main_process()) or (self.only_main is False)

        if self.is_activate:
            self.total_len = total_len
            self.tqdm = tqdm(total=total_len)
            self.prompt = prompt
        else:
            self.total_len = None
            self.tqdm = None
            self.prompt = None

    def update(self, step_len: int, **kwargs: Any):
        if (self.only_main and is_main_process()) or (self.only_main is False):
            self.tqdm.set_description(self.prompt)
            self.tqdm.set_postfix(**kwargs)
            self.tqdm.update(step_len)
        else:
            return


class Logger:
    """
    Log information.
    """
    def __init__(
            self,
            logdir: str,
            use_wandb: bool,
            config: dict | None = None,       # log to wandb
            exp_owner: str | None = None,
            exp_project: str | None = None,
            exp_group: str | None = None,
            exp_name: str | None = None,
    ):
        self.logdir = logdir
        if is_main_process():
            os.makedirs(self.logdir, exist_ok=True)
            if use_wandb:       # init wandb
                assert config is not None, "Please set the config for the experiment."
                assert exp_owner is not None, "Please set the owner of the experiment."
                assert exp_project is not None, "Please set the project of the experiment."
                if exp_group is None:
                    self.warning("The group of the experiment is not set. Please make sure it is what you want.")
                    exp_group = "default"
                assert exp_name is not None, "Please set the name of the experiment."
                self.wandb = wandb.init(
                    dir=self.logdir,
                    project=exp_project,
                    group=exp_group,
                    entity=exp_owner,
                    name=exp_name,
                    config=config,
                )   # for more details, see https://docs.wandb.ai/ref/python/init
            else:
                self.wandb = None
        else:
            self.wandb = None
        return

    def config(self, config: dict):
        self._print_config(config=config)
        self._save_config(config=config, filename="config.yaml")
        return

    def dataset(self, dataset: JointDataset):
        dataset_statistics = dataset.statistics()
        for _statistic in dataset_statistics:
            self._print(log=f"{self._colorize(log='[Loaded Data]', log_type='success')} {_statistic}")
            self._save(log=f"[Loaded Data] {_statistic}")
        return

    @state.on_main_process
    def _print_config(self, config: dict):
        print(self._colorize(log="[Runtime Config]", log_type="success"), end=" ")
        for _ in config:
            print(f"{_.lower()}: {config[_]} | ", end="")
        print("", end="\n")

    @state.on_main_process
    def _save_config(self, config: dict, filename: str = "config.yaml"):
        self._write_dict_to_yaml(x=config, filename=filename, mode="w")
        return

    def info(self, log: str, only_main: bool = True):
        self._print(log=f"{self._colorize(log='[INFO]', log_type='info')} {log}", only_main=only_main)
        self._save(log=f"[INFO] {log}", only_main=only_main)
        return

    def warning(self, log: str, only_main: bool = True):
        self._print(log=f"{self._colorize(log='[WARNING]', log_type='warning')} {log}", only_main=only_main)
        self._save(log=f"[WARNING] {log}", only_main=only_main)
        return

    def success(self, log: str, only_main: bool = True):
        self._print(log=f"{self._colorize(log='[SUCCESS]', log_type='success')} {log}", only_main=only_main)
        self._save(log=f"[SUCCESS] {log}", only_main=only_main)
        return

    def metrics(
            self,
            log: str,
            metrics,
            fmt: None | str = "{average:.4f} ({global_average:.4f})",
            statistic: None | str = "average",
            global_step: int = 0,
            prefix: None | str = None,
            x_axis_step: None | int = None,
            x_axis_name: None | str = None,
            filename: str = "log.txt",
            file_mode: str = "a",
            only_main: bool = True
    ):
        self.print_metrics(
            metrics=metrics,
            prompt=f"{self._colorize(log='[Metrics]', log_type='info')} {log}",
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
        if self._is_to_do(only_main=only_main):
            print(log)

    def _save(self, log: str, filename: str = "log.txt", mode: str = "a", only_main: bool = True, end: str = "\n"):
        if self._is_to_do(only_main=only_main):
            with open(os.path.join(self.logdir, filename), mode=mode) as f:
                f.write(log + end)
        return

    def print_metrics(
            self, metrics: Metrics, prompt: str = "",
            fmt: str = "{average:.4f} ({global_average:.4f})",
            only_main: bool = True,
    ):
        if self._is_to_do(only_main):
            print(prompt, end="")
            print(metrics.fmt(fmt=fmt))
        return

    def save_metrics_to_file(self, metrics: Metrics, prompt: str = "",
                             fmt: str = "{average:.4f} ({global_average:.4f})",
                             filename: str = "log.txt", mode: str = "a", only_main: bool = True):
        if self._is_to_do(only_main):
            log = f"{prompt}{metrics.fmt(fmt=fmt)}"
            self._save(log=log, filename=filename, mode=mode)
        return

    def save_metrics(
            self,
            metrics: Metrics,
            prompt: str = "",
            fmt: None | str = "{average:.4f} ({global_average:.4f})",
            statistic: None | str = "average",
            global_step: int = 0,
            prefix: None | str = None,
            x_axis_step: None | int = None,
            x_axis_name: None | str = None,
            filename: str = "log.txt",
            file_mode: str = "a",
            only_main: bool = True,
    ):
        """
        Save the metrics into .txt/wandb.

        Args:
            metrics: The metrics to save.
            prompt: Prompt of logging metrics.
            fmt: Format for Metric Value. If fmt is None, we will not output these into the log.txt file.
            statistic: Which statistic is output to wandb.
                       If is None, we will not output these to wandb.
            global_step: Global step of metrics records, generally, we set it to total iter of model training.
            prefix: Prefix of all metrics.
                    If your metric name is "loss", prefix is "epoch", the final name of this metric is "epoch_loss".
            x_axis_step: A different X-axis value from global step.
            x_axis_name: Name of X-axis.
            filename: The filename for saving metrics log.
            file_mode: The file mode for saving, like "w" or "a".
            only_main: Only save the log in the main process.
        Returns:

        """
        if fmt is not None:
            self.save_metrics_to_file(
                metrics=metrics, prompt=prompt, fmt=fmt, filename=filename, mode=file_mode, only_main=only_main
            )
        if statistic is not None:
            if self.wandb:
                self.save_metrics_to_wandb(metrics=metrics, statistic=statistic,
                                           global_step=global_step, prefix=prefix)
                if x_axis_step is not None:
                    if x_axis_name is None:
                        raise RuntimeError(f"If you set x_axis_step, you should also set a valid x_axis_name.")
                    self.wandb_log(     # see https://github.com/wandb/wandb/issues/410 for more details.
                        data={x_axis_name: x_axis_step},
                        step=global_step
                    )
        return

    def _write_dict_to_yaml(self, x: dict, filename: str, mode: str = "w"):
        with open(os.path.join(self.logdir, filename), mode=mode) as f:
            yaml.dump(x, f, allow_unicode=True)
        return

    def _write_dict_to_json(self, log: dict, filename: str, mode: str = "w"):
        """
        Logger writes a dict log to a .json file.

        Args:
            log (dict): A dict log.
            filename (str): Log file's name.
            mode (str): File writing mode, "w" or "a".
        """
        with open(os.path.join(self.logdir, filename), mode=mode) as f:
            f.write(json.dumps(log, indent=4))
            f.write("\n")
        return

    @staticmethod
    def _is_to_do(only_main: bool = True):
        return is_main_process() or not only_main

    @staticmethod
    def _colorize(log: str, log_type: str):
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

    def wandb_log(self, data: dict, step: int):
        if self.wandb:
            wandb.log(data=data, step=step)
        return

    def save_metrics_to_wandb(self, metrics: Metrics, statistic: str = "average",
                              global_step: int = 0, prefix: None | str = None):
        for name, value in metrics.metrics.items():
            if prefix is not None:
                metric_name = f"{prefix}_{name}"
            else:
                metric_name = name
            metric_value = value.__getattribute__(statistic)
            self.wandb_log(data={metric_name: metric_value}, step=global_step)
        pass


def parser_to_dict(log: argparse.ArgumentParser) -> dict:
    """
    Transform options to a dict.

    Args:
        log: The options.

    Returns:
        Options dict.
    """
    opts_dict = dict()
    for k, v in vars(log).items():
        if v:
            opts_dict[k] = v
    return opts_dict
