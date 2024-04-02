from __future__ import annotations
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig

import functools
from dataclasses import dataclass, field
from typing import Literal, Optional, Type, Union

import torch
import tyro
from jaxtyping import Float, Int
from torch import Tensor, nn
from typing_extensions import assert_never

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.lie_groups import exp_map_SE3, exp_map_SO3xR3
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.engine.optimizers import OptimizerConfig
from nerfstudio.engine.schedulers import SchedulerConfig
from nerfstudio.utils import poses as pose_utils

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle


@dataclass
class TrackerConfig(CameraOptimizerConfig):

    _target: Type = field(default_factory=lambda: Tracker)
    mode: Literal["off", "SO3xR3", "SE3"] = "SE3"


class Tracker(CameraOptimizer):

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.pose_adjustment.requires_grad = True

    def apply_to_camera2(self, camera: Cameras):
        camera.camera_to_worlds = torch.ones((1, 3, 4), device=camera.device)
        super().apply_to_camera(camera)

    def get_param_groups(self, param_groups: dict = {}) -> None:
        param_groups["camera_opt"] = [self.pose_adjustment]
        return param_groups
