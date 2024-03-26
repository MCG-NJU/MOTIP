# @Author       : Ruopeng Gao
# @Date         : 2022/7/5
# @Description  : Logger will log information.
import os
import json
import argparse
import yaml
import wandb
import time

from tqdm import tqdm
from typing import List, Any
from torch.utils import tensorboard as tb

from log.log import Metrics
from utils.utils import is_main_process


class ProgressLogger:
    def __init__(self, total_len: int, prompt: str = None, only_main: bool = True):
        """
        初始化一个进度日志。

        Args:
            total_len:
            prompt:
            only_main: only for the main process.
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
    def __init__(self, logdir: str, use_tensorboard: bool = True, use_wandb: bool = True, only_main: bool = True,
                 config: dict = None):
        """
        Create a log.

        Args:
            logdir (str): Logger outputs path.
            use_tensorboard: Whether output tensorboard files.
            use_wandb: Whether output WandB files.
            only_main: Only in the main process.
            config: Runtime config.
        """
        self.only_main = only_main
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        self.is_activate = (self.only_main and is_main_process()) or (self.only_main is False)

        self.logdir = None
        self.tensorboard_writer = None
        self.wandb_run = None
        if self.is_activate:
            # init the logdir.
            self.logdir = logdir
            os.makedirs(self.logdir, exist_ok=True)
            if self.use_tensorboard:    # init the tensorboard writer (SummaryWriter):
                tensorboard_dir = os.path.join(self.logdir, "tensorboard")
                os.makedirs(tensorboard_dir, exist_ok=True)
                self.tensorboard_writer = tb.SummaryWriter(log_dir=tensorboard_dir)
            if self.use_wandb:
                # wandb_dir = os.path.join(self.logdir, "wandb")
                # os.makedirs(wandb_dir, exist_ok=True)
                assert config["GIT_VERSION"] is not None, f"For wandb logging, git version should not be None."
                wandb_dir = self.logdir
                timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))
                # exp_name = f"{config['EXP_NAME']}_{timestamp}" if config["EXP_NAME"] is not None else f"{timestamp}"
                exp_name = f"{config['EXP_NAME']}" if config["EXP_NAME"] is not None else f"{timestamp}"
                self.wandb_run = wandb.init(
                    dir=wandb_dir,
                    project=config["PROJECT_NAME"],
                    group=config["EXP_GROUP"],
                    entity=config["EXP_OWNER"],
                    name=exp_name,
                    config=config,
                )   # for more details, see https://docs.wandb.ai/ref/python/init

        return

    def print_config(self, config: dict, prompt: str = ""):
        if self.is_activate:
            print(prompt, end="")
            for _ in config:
                print(f"{_.lower()}: {config[_]}; ", end="")
            print("")

    def print(self, log: str):
        if self.is_activate:
            print(log)

    def save_config(self, config: dict, filename: str):
        if self.is_activate:
            self._write_dict_to_yaml(x=config, filename=filename, mode="w")
        return

    def print_metrics(self, metrics: Metrics, prompt: str = "", fmt: str = "{average:.4f} ({global_average:.4f})"):
        if self.is_activate:
            print(prompt, end="")
            print(metrics.fmt(fmt=fmt))
        return

    def save_metrics_to_file(self, metrics: Metrics, prompt: str = "",
                             fmt: str = "{average:.4f} ({global_average:.4f})",
                             filename: str = "log.txt", mode: str = "a"):
        if self.is_activate:
            log = f"{prompt}{metrics.fmt(fmt=fmt)}\n"
            self.save_log_to_file(log=log, filename=filename, mode=mode)
        return

    def save_log_to_file(self, log: str, filename: str = "log.txt", mode: str = "a"):
        if self.is_activate:
            with open(os.path.join(self.logdir, filename), mode=mode) as f:
                f.write(log)
        return

    def save_metrics(self, metrics: Metrics, prompt: str = "",
                     fmt: None | str = "{average:.4f} ({global_average:.4f})",
                     statistic: None | str = "average", global_step: int = 0, prefix: None | str = None,
                     x_axis_step: None | int = None, x_axis_name: None | str = None,
                     filename: str = "log.txt", file_mode: str = "a"):
        """
        Save the metrics into .txt/tensorboard/wandb.

        Args:
            metrics: The metrics to save.
            prompt: Prompt of logging metrics.
            fmt: Format for Metric Value. If fmt is None, we will not output these into the log.txt file.
            statistic: Which statistic is output to tensorboard and wandb.
                       If is None, we will not output these to tensorboard nor wandb.
            global_step: Global step of metrics records, generally, we set it to total iter of model training.
            prefix: Prefix of all metrics.
                    If your metric name is "loss", prefix is "epoch", the final name of this metric is "epoch_loss".
            x_axis_step: A different X-axis value from global step.
            x_axis_name: Name of X-axis.
            filename: The filename for saving metrics log.
            file_mode: The file mode for saving, like "w" or "a".
        Returns:

        """
        if fmt is not None:
            self.save_metrics_to_file(metrics=metrics, prompt=prompt, fmt=fmt, filename=filename, mode=file_mode)
        if statistic is not None:
            if self.use_tensorboard:
                self.save_metrics_to_tensorboard(
                    metrics=metrics,
                    statistic=statistic,
                    global_step=global_step if x_axis_step is None else x_axis_step,
                    prefix=prefix
                )
            if self.use_wandb:
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

    # def tb_add_scalars(self, main_tag: str, tag_scalar_dict: dict, global_step: int):
    #     if (self.only_main and is_main_process()) or (self.only_main is False):
    #         self.tb_logger.add_scalars(
    #             main_tag=main_tag,
    #             tag_scalar_dict=tag_scalar_dict,
    #             global_step=global_step
    #         )
    #     else:
    #         pass
    #     return

    def tensorboard_add_scalar(self, name: str, value: float, global_step: int):
        if self.is_activate:
            if self.use_tensorboard:
                self.tensorboard_writer.add_scalar(
                    tag=name,
                    scalar_value=value,
                    global_step=global_step
                )
            else:
                pass
        else:
            pass
        return

    def wandb_log(self, data: dict, step: int):
        if self.is_activate:
            if self.use_wandb:
                wandb.log(data=data, step=step)
        return
    # def name_value_to_statistic(self, name: str, value: float, global_step: int):
    #     if self.use_tensorboard:
    #         self.tensorboard_add_scalar(name=name, value=value, global_step=global_step)
    #     if self.use_wandb:

    def save_metrics_to_tensorboard(self, metrics: Metrics, statistic: str = "average",
                                    global_step: int = 0, prefix: None | str = None):
        for name, value in metrics.metrics.items():
            if prefix is not None:
                metric_name = f"{prefix}_{name}"
            else:
                metric_name = name
            metric_value = value.__getattribute__(statistic)
            self.tensorboard_add_scalar(
                name=metric_name, value=metric_value, global_step=global_step
            )
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
