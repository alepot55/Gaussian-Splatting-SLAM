from __future__ import annotations
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from gaussian_splatting_slam.model import MyModelConfig, MyModel
from gaussian_splatting_slam.pipeline import MyPipelineConfig, MyPipeline
from gaussian_splatting_slam.tracker import TrackerConfig, Tracker

Method = MethodSpecification(
    config=TrainerConfig(
        method_name="gaussian_splatting_slam",
        steps_per_eval_image=199, # da 100
        steps_per_eval_batch=0,
        steps_per_save= 3999, # da 2000
        steps_per_eval_all_images=1999, # da 1000
        max_num_iterations=60000, 
        mixed_precision=False,

        pipeline=MyPipelineConfig(
            _target=MyPipeline,
            max_window=10,
            covisibility_threshold=0.8,

            datamanager=FullImageDatamanagerConfig(
                dataparser=NerfstudioDataParserConfig(load_3D_points=True),
                cache_images_type="uint8",
            ),

            model=MyModelConfig(

                # In Spatfacto
                _target=MyModel, 
                refine_every=199, # da 100
                warmup_length=1000, # da 500
                reset_alpha_every=59, # da 30
                loss_coefficients={'rgb_loss_coarse': 1.0, 'rgb_loss_fine': 1.0},

                # Tracker
                tracker=TrackerConfig(
                        _target=Tracker, 
                        mode="SE3",
                        trans_l2_penalty=0.0, # messo io
                        rot_l2_penalty=0.0, # messo io
                    )
                ),
        ),
        optimizers={
            "xyz": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-5, eps=1e-15),
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
            "camera_opt": { # messo io
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=20000), # da 40000
            },
        },

        viewer=ViewerConfig(num_rays_per_chunk=1 << 15), 
        vis="viewer", 
    ),
    description="Gaussian Splatting for SLAM.",
)