# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import List, Any

# @manual "//github/facebookresearch/hydra:hydra"
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

defaults = [
    "_self_",
    {"task": "joint_el_task"},
    # Model
    # {"task/model": "xlmr"},
    # Transform
    # {"task/transform": "joint_el_xlmr_transform"},
    # Optim
    # {"task/optim": "adamw"},
    # Data
    # {"datamodule": "joint_el_datamodule"},
    # Trainer
    # {"trainer": "gpu_1_host"},
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
