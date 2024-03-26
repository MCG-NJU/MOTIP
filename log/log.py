# @Author       : Ruopeng Gao
# @Date         : 2022/7/13
# @Description  :
import torch.distributed
import time

from typing import List, Any
from utils.utils import is_distributed, distributed_world_size
from collections import deque, defaultdict


class Metrics:
    def __init__(self):
        self.metrics = defaultdict(Value)

    def update(self, name: str, value: float):
        if isinstance(value, torch.Tensor):
            value = value.item()
        self.metrics[name].update(value)
        return

    def sync(self):
        for name, value in self.metrics.items():
            value.sync()
        return

    def __getitem__(self, item):
        return self.metrics[item]

    # Not suitable for PyCharm Debug:
    # def __getattr__(self, item):
    #     return self.metrics[item]

    def __str__(self):
        s = str()
        for name, value in self.metrics.items():
            s += f"{name} = {value.average:.4f} ({value.global_average:.4f}); "
        return s

    def fmt(self, fmt):
        s = str()
        for name, value in self.metrics.items():
            s += f"{name} = {value.fmt(fmt=fmt)}; "
        return s


class TPS:
    """
    Time Per Step.
    """
    def __init__(self, windows_size: int = 50):
        self.tps_deque = deque(maxlen=windows_size)     # time per step.

    def update(self, tps: float):
        self.tps_deque.append(tps)

    @property
    def average(self):
        tps_list = list(self.tps_deque)
        return sum(tps_list) / len(tps_list)

    def eta(self, total_steps: int, current_steps: int):
        return self.average * (total_steps - current_steps)

    @classmethod
    def timestamp(cls):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()

    @classmethod
    def format(cls, seconds: float):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{int(h)}:{int(m)}:{int(s)}"


class Value:
    def __init__(self, window_size: int = 100):
        self.value_deque = deque(maxlen=window_size)
        self.total_value = 0.0
        self.total_count = 0

        self.value_sync: None | torch.Tensor = None
        self.total_value_sync = None
        self.total_count_sync = None

    def update(self, value):
        self.value_deque.append(value)
        self.total_value += value
        self.total_count += 1

    def sync(self):
        if is_distributed():
            torch.distributed.barrier()
            value_gather = [None] * distributed_world_size()
            total_gather = [None] * distributed_world_size()
            torch.distributed.all_gather_object(value_gather, list(self.value_deque))
            torch.distributed.all_gather_object(total_gather, [self.total_value, self.total_count])
            values = [v for v_list in value_gather for v in v_list]
            self.value_sync = torch.as_tensor(values)
            self.total_value_sync = sum([_[0] for _ in total_gather])
            self.total_count_sync = sum([_[1] for _ in total_gather])
        else:                   # only one gpu.
            self.value_sync = torch.as_tensor(list(self.value_deque))
            self.total_value_sync = self.total_value
            self.total_count_sync = self.total_count
        return

    def clear(self):
        self.value_deque.clear()
        self.total_value = 0.0
        self.total_count = 0

        self.value_sync: None | torch.Tensor = None
        self.total_value_sync = None
        self.total_count_sync = None

    def _check_sync(self):
        if self.value_sync is None:
            raise RuntimeError(f"Be sure to use .sync() before metric statistic.")
        return

    @property
    def average(self):
        self._check_sync()
        return self.value_sync.mean().item()

    @property
    def global_average(self):
        self._check_sync()
        return self.total_value_sync / self.total_count_sync

    @property
    def median(self):
        self._check_sync()
        return self.value_sync.median().item()

    def fmt(self, fmt):
        return fmt.format(
            median=self.median,
            average=self.average,
            global_average=self.global_average
        )
