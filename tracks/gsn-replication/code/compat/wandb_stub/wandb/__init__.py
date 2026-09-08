"""wandb 最小 stub：官方代码 train_test_funcs.py 顶层 `import wandb`，
但 `--wandb False` 时所有调用都被跳过。本地冒烟避免安装真 wandb。"""
from __future__ import annotations


class _Run:
    def __init__(self):
        self.summary: dict = {}


_run = _Run()


def run() -> _Run:
    return _run


def init(*args, **kwargs) -> _Run:
    return _run


def log(*args, **kwargs) -> None:
    pass


def watch(*args, **kwargs) -> None:
    pass
