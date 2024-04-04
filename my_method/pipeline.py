from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Type, Union, cast
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from torch.nn import Parameter
from .model import MyModelConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.utils import profiler
import logging


@dataclass
class MyPipelineConfig(VanillaPipelineConfig):
    
    max_window: int = 10
    covisibility_threshold: float = 0.7

    _target: Type = field(default_factory=lambda: MyPipeline)


class MyPipeline(VanillaPipeline):

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.mode = "tracking"
        self.current_window = 0
        self.window = []

    def get_data(self, step: int):
        if self.config.max_window == None or self.current_window < self.config.max_window:
            #self.logger.debug(f"lunghezza: {self.current_window}")
            self.current_window += 1
            camera, batch = self.datamanager.next_train(step)
            self.window.append(camera)
            return camera, batch
        else:
            max_covisibility = self.config.covisibility_threshold
            #self.logger.debug(f"lunghezza piena: {self.current_window}, covisibility: {max_covisibility}")
            while max_covisibility >= self.config.covisibility_threshold:
                max_covisibility = 0
                #self.logger.debug(f"Non va bene, nuova immagine")
                new_camera, batch = self.datamanager.next_train(step)
                for camera in self.window:
                    covisibility = self.model.covisibility(new_camera, camera)
                    #self.logger.debug(f"covisibility: {covisibility}")
                    max_covisibility = max(max_covisibility, covisibility)
            self.window.pop(0)
            self.window.append(new_camera)
            return new_camera, batch

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        logging.basicConfig(level=logging.NOTSET)
        self.logger = logging.getLogger("pipeline")
        camera, batch = self.get_data(step)
        #self.logger.debug(f"camera idx: {camera.metadata["cam_idx"]}")
        model_outputs = self._model(camera)  # train distributed data parallel model if world_size > 1
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        return model_outputs, loss_dict, metrics_dict
    