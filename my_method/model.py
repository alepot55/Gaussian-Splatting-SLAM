from dataclasses import dataclass, field
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackLocation
from my_method.tracker import Tracker, TrackerConfig
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig
from nerfstudio.cameras.cameras import Cameras
from typing import Any, Dict, List, Optional, Tuple, Type, Union


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


