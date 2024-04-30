from __future__ import annotations
import torch
from dataclasses import dataclass, field
from typing import Literal, Type
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig

@dataclass
class TrackerConfig(CameraOptimizerConfig):
    """
    Configuration class for the Tracker.

    Attributes:
        mode (Literal["off", "SO3xR3", "SE3"]): The mode of the tracker.
    """

    _target: Type = field(default_factory=lambda: Tracker)
    mode: Literal["off", "SO3xR3", "SE3"] = "SE3"


class Tracker(CameraOptimizer):
    """
    A class that represents a tracker.

    Inherits from CameraOptimizer.

    Attributes:
        pose_adjustment (Tensor): The pose adjustment tensor.
    """

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.pose_adjustment.requires_grad = True

    def apply_to_camera2(self, camera: Cameras):
        """
        Applies the tracker to the given camera.

        Args:
            camera (Cameras): The camera to apply the tracker to.
        """
        camera.camera_to_worlds = torch.ones((1, 3, 4), device=camera.device) * 0.5
        super().apply_to_camera(camera)

    def get_param_groups(self, param_groups: dict = {}) -> None:
        """
        Gets the parameter groups for optimization.

        Args:
            param_groups (dict): The parameter groups dictionary.

        Returns:
            dict: The updated parameter groups dictionary.
        """
        param_groups["camera_opt"] = [self.pose_adjustment]
        return param_groups
