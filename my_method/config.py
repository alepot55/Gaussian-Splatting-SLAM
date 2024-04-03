from __future__ import annotations
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification


from collections import OrderedDict
from typing import Dict, Union

import tyro

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.configs.external_methods import ExternalMethodDummyTrainerConfig, get_external_methods
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManagerConfig
from nerfstudio.data.datamanagers.random_cameras_datamanager import RandomCamerasDataManagerConfig
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.dataparsers.dnerf_dataparser import DNeRFDataParserConfig
from nerfstudio.data.dataparsers.instant_ngp_dataparser import InstantNGPDataParserConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.dataparsers.phototourism_dataparser import PhototourismDataParserConfig
from nerfstudio.data.dataparsers.sdfstudio_dataparser import SDFStudioDataParserConfig
from nerfstudio.data.dataparsers.sitcoms3d_dataparser import Sitcoms3DDataParserConfig
from nerfstudio.data.datasets.depth_dataset import DepthDataset
from nerfstudio.data.datasets.sdf_dataset import SDFDataset
from nerfstudio.data.datasets.semantic_dataset import SemanticDataset
from nerfstudio.data.pixel_samplers import PairPixelSamplerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    CosineDecaySchedulerConfig,
    ExponentialDecaySchedulerConfig,
    MultiStepSchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.field_components.temporal_distortions import TemporalDistortionKind
from nerfstudio.fields.sdf_field import SDFFieldConfig
from nerfstudio.models.depth_nerfacto import DepthNerfactoModelConfig
from nerfstudio.models.generfacto import GenerfactoModelConfig
from nerfstudio.models.instant_ngp import InstantNGPModelConfig
from nerfstudio.models.mipnerf import MipNerfModel
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.models.neus import NeuSModelConfig
from nerfstudio.models.neus_facto import NeuSFactoModelConfig
from nerfstudio.models.semantic_nerfw import SemanticNerfWModelConfig
from nerfstudio.models.splatfacto import SplatfactoModelConfig
from nerfstudio.models.tensorf import TensoRFModelConfig
from nerfstudio.models.vanilla_nerf import NeRFModel, VanillaModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig, VanillaPipeline
from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipelineConfig
from nerfstudio.plugins.registry import discover_methods

from my_method.model import MyModelConfig, MyModel
from my_method.pipeline import MyPipelineConfig, MyPipeline
from my_method.tracker import TrackerConfig, Tracker

Method = MethodSpecification(
    config=TrainerConfig(
    method_name="my_method",
    steps_per_eval_image=100,
    steps_per_eval_batch=0,
    steps_per_save=2000,
    steps_per_eval_all_images=1000,
    max_num_iterations=60000,
    mixed_precision=False,
    pipeline=VanillaPipelineConfig(
        _target=VanillaPipeline,
        datamanager=FullImageDatamanagerConfig(
            dataparser=NerfstudioDataParserConfig(load_3D_points=True),
            cache_images_type="uint8",
        ),
        model=MyModelConfig(
            _target=MyModel, 
            tracker=TrackerConfig(
                _target=Tracker, 
                mode="SE3",
                trans_l2_penalty=0.0,
                rot_l2_penalty=0.0,
                )
            ),
    ),
    optimizers={
        "xyz": {
            "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1.6e-6,
                max_steps=30000,
            ),
        },
        "features_dc": {
            "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
            "scheduler": None,
        },
        "features_rest": {
            "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
            "scheduler": None,
        },
        "opacity": {
            "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
            "scheduler": None,
        },
        "scaling": {
            "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
            "scheduler": None,
        },
        "rotation": {
            "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), 
            "scheduler": None
        },
        "camera_opt": { # senza questo il rersto fa schifo
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=40000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
),
    description="Nuovo metodo di slam per 3dgs",
)
