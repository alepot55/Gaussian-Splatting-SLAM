from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Type, Union, cast
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from torch.nn import Parameter
from .model import MyModelConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.utils import profiler


@dataclass
class MyPipelineConfig(VanillaPipelineConfig):
    
    num_traking_iter: int = 50
    # num_window: int = 10

    _target: Type = field(default_factory=lambda: MyPipeline)

    datamanager: FullImageDatamanagerConfig = field(default_factory=FullImageDatamanagerConfig)

    model: MyModelConfig = field(default_factory=MyModelConfig)


class MyPipeline(VanillaPipeline):

    mode: Literal["tracking", "mapping"]
    current_data: Tuple[Cameras, Dict]
    # window: List[Tuple[Cameras, Dict]]

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.mode = "tracking"

    def _get_data(self, step):
        if self.mode == "tracking":
            self.current_data = self.datamanager.next_train(step)
        return self.current_data

    def _new_iter(self):
        self.mode = "mapping" if self.mode == "tracking" else "tracking"

    @profiler.time_function
    def get_train_loss_dict(self, step: int):

        cameras, batch = self._get_data(step)
        model_outputs = self._model(cameras)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        self._new_iter()

        return model_outputs, loss_dict, metrics_dict
    
    def get_param_groups(self):
        datamanager_params = self.datamanager.get_param_groups()
        model_params = self.model.get_param_groups(self.mode)
        return {**datamanager_params, **model_params}
