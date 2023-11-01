import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models.base import BaseModel
from thre3d_atom.rendering.volumetric.utils.misc import (
    compute_expected_density_scale_for_relu_field_grid,
)
from thre3d_atom.modules.volumetric_model import VolumetricModel
from thre3d_atom.thre3d_reprs.voxels import VoxelGrid, VoxelSize, VoxelGridLocation
from thre3d_atom.thre3d_reprs.renderers import (
    render_sh_voxel_grid,
    SHVoxGridRenderConfig,
)


@models.register('relu_fields')
class ReluModel(nn.Module):
    def __init__(self,config):
        super(ReluModel, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vox_grid_density_activations_dict = {
            "density_preactivation": torch.nn.Identity(),
            "density_postactivation": torch.nn.ReLU(),
            # note this expected density value :)
            "expected_density_scale": compute_expected_density_scale_for_relu_field_grid(
                (3,3,3)
            ),
        }
        self.densities = torch.empty((*config.grid_dims, 1), dtype=torch.float32, device=device)
        torch.nn.init.uniform_(self.densities, -1.0, 1.0)

        num_sh_features = 3 * ((config.sh_degree + 1) ** 2)
        features = torch.empty((*config.grid_dims, num_sh_features), dtype=torch.float32, device=device)
        torch.nn.init.uniform_(features, -1.0, 1.0)
        self.voxel_size = VoxelSize(*[dim_size / grid_dim for dim_size, grid_dim
                                in zip(config.grid_world_size, config.grid_dims)])
        self.voxel_grid = VoxelGrid(
            densities=self.densities,
            features=features,
            voxel_size=self.voxel_size,
            grid_location=VoxelGridLocation(*config.grid_location),
            **self.vox_grid_density_activations_dict,
            tunable=True,
        )
        # fmt: on

        # set up a volumetricModel using the previously created voxel-grid
        # noinspection PyTypeChecker
        self.vox_grid_vol_mod = VolumetricModel(
            thre3d_repr=self.voxel_grid,
            render_procedure=render_sh_voxel_grid,
            render_config=SHVoxGridRenderConfig(
                num_samples_per_ray=config.train_num_samples_per_ray,
                camera_bounds=(2,6),
                white_bkgd=config.white_bkgd,
                render_num_samples_per_ray=config.render_num_samples_per_ray,
                parallel_rays_chunk_size=config.parallel_rays_chunk_size,
            ),
            device=device,
        )



    