from __future__ import annotations
from dataclasses import dataclass, field
from typing import Type
from nerfstudio.utils import profiler
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig

@dataclass
class MyPipelineConfig(VanillaPipelineConfig):
    """
    Configuration class for MyPipeline.
    
    Attributes:
        max_window (int): The maximum window size.
        covisibility_threshold (float): The threshold for covisibility.
    """
    
    _target: Type = field(default_factory=lambda: MyPipeline)
    max_window: int = 10
    covisibility_threshold: float = 0.6


class MyPipeline(VanillaPipeline):
    """
    A custom pipeline class that extends the VanillaPipeline class.

    Attributes:
        current_window (int): The current window index.
        window (list): The list of camera objects in the window.

    Methods:
        get_data(step: int) -> Tuple[Camera, Batch]: Retrieves the camera and batch data for a given step.
        get_train_loss_dict(step: int) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]: Computes the model outputs, loss dictionary, and metrics dictionary for training.

    """

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.current_window = 0
        self.window = []

    def get_data(self, step: int):
        """
        Retrieves the camera and batch data for a given step.

        Args:
            step (int): The step index.

        Returns:
            Tuple[Camera, Batch]: The camera and batch data.

        """
        if self.config.max_window is None or self.current_window < self.config.max_window:

            # If the maximum window size is not set or the current window index is less than the maximum window size
            self.current_window += 1
            camera, batch = self.datamanager.next_train(step)
            self.window.append(camera)
            return camera, batch
        else:
            # Initialize the maximum covisibility with the threshold value
            max_covisibility = self.config.covisibility_threshold

            # Loop until the maximum covisibility is less than the threshold
            while max_covisibility >= self.config.covisibility_threshold:

                # Get the next camera and batch data
                max_covisibility = 0
                new_camera, batch = self.datamanager.next_train(step)

                # Iterate through the cameras in the window
                for camera in self.window:

                    # Calculate the covisibility between the new camera and each camera in the window
                    covisibility = self.model.covisibility(new_camera, camera)

                    # Update the maximum covisibility if a higher value is found
                    max_covisibility = max(max_covisibility, covisibility)

            # Remove the oldest camera from the window
            self.window.pop(0)

            # Add the new camera to the window
            self.window.append(new_camera)
            return new_camera, batch

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """
        Computes the model outputs, loss dictionary, and metrics dictionary for training.

        Args:
            step (int): The step index.

        Returns:
            Tuple[Any, Dict[str, Any], Dict[str, Any]]: The model outputs, loss dictionary, and metrics dictionary.

        """
        camera, batch = self.get_data(step)
        model_outputs = self._model(camera)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        return model_outputs, loss_dict, metrics_dict
    