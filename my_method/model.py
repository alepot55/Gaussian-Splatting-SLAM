from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from gsplat._torch_impl import quat_to_rotmat
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from gsplat.sh import num_sh_bases, spherical_harmonics
from pytorch_msssim import SSIM
from torch.nn import Parameter
from typing_extensions import Literal

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers

from nerfstudio.model_components import renderers
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.rich_utils import CONSOLE

from my_method.tracker import Tracker, TrackerConfig
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig, projection_matrix


@dataclass
class MyModelConfig(SplatfactoModelConfig):

    _target: Type = field(default_factory=lambda: MyModel)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)


class MyModel(SplatfactoModel):

    def populate_modules(self):
        super().populate_modules()
        self.mode = "mapping" # inizio con mapping perchè devo prima mettere i gaussiani e poi magari i callbacks capitano sui pari
        self.tracker = self.config.tracker.setup(num_cameras=self.num_train_data, device="cpu")

    def update(self, step):
        self.mode = "tracking" if self.mode == "mapping" else "mapping"

    def get_gaussian_param_groups(self):
        if self.mode == "tracking":
            return {}
        return super().get_gaussian_param_groups()

    def get_param_groups(self):
        if self.mode == "tracking":
            return self.tracker.get_param_groups()
        return super().get_param_groups()
    
    def refinement_after(self, optimizers, step):
        if self.mode == "mapping":
            super().refinement_after(optimizers, step)
    
    def get_outputs(self, camera: Cameras):
        if self.training:
            self.tracker.apply_to_camera(camera)
        return super().get_outputs(camera)
    
    def get_training_callbacks(self, training_callback_attributes):
        cbs = super().get_training_callbacks(training_callback_attributes)
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                self.update,
            )
        )
        return cbs
    
    def covisibility(self, camera_1, camera_2):
        #import logging
        #logging.basicConfig(level=logging.NOTSET)
        #self.logger = logging.getLogger("model")

        indices = []

        for camera in [camera_1, camera_2]:    
        
            camera_downscale = self._get_downscale_factor()
            camera.rescale_output_resolution(1 / camera_downscale)

            # shift the camera to center of scene looking at center
            R = camera.camera_to_worlds[0, :3, :3]  # 3 x 3
            T = camera.camera_to_worlds[0, :3, 3:4]  # 3 x 1

            # flip the z and y axes to align with gsplat conventions
            R_edit = torch.diag(torch.tensor([1, -1, -1], device=self.device, dtype=R.dtype))
            R = R @ R_edit

            # analytic matrix inverse to get world2camera matrix
            R_inv = R.T
            T_inv = -R_inv @ T
            viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
            viewmat[:3, :3] = R_inv
            viewmat[:3, 3:4] = T_inv

            # calculate the FOV of the camera given fx and fy, width and height
            fovx = 2 * math.atan(camera.width / (2 * camera.fx))
            fovy = 2 * math.atan(camera.height / (2 * camera.fy))
            W, H = int(camera.width.item()), int(camera.height.item())
            projmat = projection_matrix(0.001, 1000, fovx, fovy, device=self.device)
            BLOCK_X, BLOCK_Y = 16, 16
            tile_bounds = (
                int((W + BLOCK_X - 1) // BLOCK_X),
                int((H + BLOCK_Y - 1) // BLOCK_Y),
                1,
            )

            xys, depths, radii, conics, num_tiles_hit, cov3d = project_gaussians(  # type: ignore
                self.means,
                torch.exp(self.scales),
                1,
                self.quats / self.quats.norm(dim=-1, keepdim=True),
                viewmat.squeeze()[:3, :],
                projmat.squeeze() @ viewmat.squeeze(),
                camera.fx.item(),
                camera.fy.item(),
                camera.cx.item(),
                camera.cy.item(),
                H,
                W,
                tile_bounds,
            )  
           
            depth_mean = torch.mean(depths)
            indices.append(torch.where((depths < depth_mean) & (radii > 0))[0])
            #self.logger.debug(f"Ho trovato {len(indices[-1])} punti")
                
        unione = len(torch.unique(torch.cat((indices[0], indices[1]))))
        intersezione = (len(indices[0]) + len(indices[1]) - unione) / 2
        if unione == 0:
            return 0
        #self.logger.debug(f"Risultato: {intersezione / unione}")
        return intersezione / unione
        
        


