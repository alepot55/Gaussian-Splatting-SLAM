import math
import torch
from dataclasses import dataclass, field
from typing import Type
from gsplat.project_gaussians import project_gaussians
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackLocation
from my_method.tracker import TrackerConfig
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig, projection_matrix

@dataclass
class MyModelConfig(SplatfactoModelConfig):
    """
    Configuration class for MyModel.

    Attributes:
        tracker (Tracker): The tracker object used for tracking in the "tracking" mode.
    """

    _target: Type = field(default_factory=lambda: MyModel)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)


class MyModel(SplatfactoModel):
    """
    MyModel class represents a custom model that extends the SplatfactoModel class.
    It provides additional functionality for mapping and tracking.

    Attributes:
        mode (str): The current mode of the model, either "mapping" or "tracking".
        config (Config): The configuration object.

    Methods:
        populate_modules(): Populates the modules of the model.
        update(step): Updates the mode of the model based on the current step.
        get_gaussian_param_groups(): Returns the parameter groups for the Gaussian modules.
        get_param_groups(): Returns the parameter groups based on the current mode.
        refinement_after(optimizers, step): Performs refinement after optimization based on the current mode.
        get_outputs(camera): Returns the outputs of the model for a given camera.
        get_training_callbacks(training_callback_attributes): Returns the training callbacks for the model.
        covisibility(camera_1, camera_2): Calculates the covisibility between two cameras.
    """

    def populate_modules(self):
        """
        Populates the modules required for the model.

        This method calls the base class's `populate_modules` method and then sets the mode to "mapping" and initializes the tracker.

        Args:
            None

        Returns:
            None
        """
        super().populate_modules()
        self.mode = "mapping" 
        self.tracker = self.config.tracker.setup(num_cameras=self.num_train_data, device="cpu")

    def update(self, step):
        """
        Updates the mode of the object based on the current mode.

        Parameters:
        - step: The current step of the update process.

        Returns:
        None
        """
        self.mode = "tracking" if self.mode == "mapping" else "mapping"

    def get_gaussian_param_groups(self):
        """
        Returns the parameter groups for Gaussian distribution.

        If the mode is "tracking", an empty dictionary is returned.
        Otherwise, the parameter groups are obtained from the superclass.

        Returns:
            dict: The parameter groups for Gaussian distribution.
        """
        if self.mode == "tracking":
            return {}
        return super().get_gaussian_param_groups()

    def get_param_groups(self):
        """
        Returns the parameter groups for the model.

        If the mode is set to "tracking", it returns the parameter groups
        obtained from the tracker. Otherwise, it calls the base class's
        get_param_groups method.

        Returns:
            list: A list of parameter groups.
        """
        if self.mode == "tracking":
            return self.tracker.get_param_groups()
        return super().get_param_groups()
    
    def refinement_after(self, optimizers, step):
        """
        Perform refinement steps after each optimization step.

        Args:
            optimizers (list): List of optimizers used for optimization.
            step (int): Current optimization step.

        Returns:
            None
        """
        if self.mode == "mapping":
            super().refinement_after(optimizers, step)
    
    def get_outputs(self, camera: Cameras):
        """
        Get the outputs from the model for a given camera.

        Args:
            camera (Cameras): The camera object for which to get the outputs.

        Returns:
            The outputs from the model for the given camera.
        """
        if self.training:
            self.tracker.apply_to_camera(camera)
        return super().get_outputs(camera)
    
    def get_training_callbacks(self, training_callback_attributes):
            """
            Returns a list of training callbacks for the model.

            Args:
                training_callback_attributes (list): A list of training callback attributes.

            Returns:
                list: A list of training callbacks.

            """
            # Get the training callbacks from the base class
            cbs = super().get_training_callbacks(training_callback_attributes)

            # Append a new training callback for updating the mode
            cbs.append(
                TrainingCallback(
                    [TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    self.update,
                )
            )

            # Return the updated list of training callbacks
            return cbs
    
    def covisibility(self, camera_1, camera_2):
        """
        Calculates the covisibility between two cameras.

        Args:
            camera_1 (Cameras): The first camera.
            camera_2 (Cameras): The second camera.

        Returns:
            float: The covisibility value between the two cameras.
        """
        indices = []

        # Iterate over the two cameras
        for camera in [camera_1, camera_2]:    
        
            # Downscale the camera resolution
            camera_downscale = self._get_downscale_factor()
            camera.rescale_output_resolution(1 / camera_downscale)

            # Shift the camera to center of scene looking at center
            R = camera.camera_to_worlds[0, :3, :3]  # 3 x 3
            T = camera.camera_to_worlds[0, :3, 3:4]  # 3 x 1

            # Flip the z and y axes to align with gsplat conventions
            R_edit = torch.diag(torch.tensor([1, -1, -1], device=self.device, dtype=R.dtype))
            R = R @ R_edit

            # Analytic matrix inverse to get world2camera matrix
            R_inv = R.T
            T_inv = -R_inv @ T
            viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
            viewmat[:3, :3] = R_inv
            viewmat[:3, 3:4] = T_inv

            # Calculate the FOV of the camera given fx and fy, width and height
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

            # Project the Gaussians onto the camera
            xys, depths, radii, conics, num_tiles_hit, cov3d = project_gaussians(
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
           
            # Calculate the mean depth
            depth_mean = torch.mean(depths)

            # Filter the indices based on depth and radius conditions
            indices.append(torch.where((depths < depth_mean) & (radii > 0))[0])
            
        # Calculate the union and intersection of the indices
        union = len(torch.unique(torch.cat((indices[0], indices[1]))))
        intersection = (len(indices[0]) + len(indices[1]) - union) / 2

        # Handle the case when the union is empty
        if union == 0:
            return 0

        # Calculate the covisibility value
        return intersection / union
        
        


