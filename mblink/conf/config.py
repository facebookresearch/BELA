# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from dataclasses import dataclass, field
from typing import List, Any

# @manual "//github/facebookresearch/hydra:hydra"
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

defaults = [
    "_self_",
    {"task": "blink_task"},
    {"checkpoint_callback": "default"},
]


@dataclass
class MainConfig:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    task: Any = MISSING
    datamodule: Any = MISSING
    trainer: Any = MISSING
    test_only: bool = False
    checkpoint_callback: Any = MISSING

cs = ConfigStore.instance()

cs.store(name="config", node=MainConfig)
