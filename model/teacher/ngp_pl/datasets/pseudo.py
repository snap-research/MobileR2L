from torch.utils.data import Dataset
import numpy as np

from .ray_utils import (
    get_ray_directions,
    get_rays,
    rand_360_poses,
    rand_FF_poses
)

class PseudoDataset(Dataset):
    
    def __init__(
            self,
            W,
            H,
            K, 
            mean_radius, 
            min_radius, 
            max_radius,
            min_theta,
            max_theta,
            sr_downscale=8, 
            n_pseudo_data=10000, 
            ff=False,
            centered_poses=None
    ):

        if ff and (centered_poses.any()) is None:
            raise ValueError('for ff dataset, centered poses is needed.')

        self.n_pseudo_data = n_pseudo_data
        self.ff = ff
        self.centered_poses = centered_poses
        self.H = H
        self.W = W
        self.K = K
        self.od_H = H // sr_downscale
        self.od_W = W // sr_downscale
        self.K_downscaled = K / sr_downscale
        self.mean_radius = mean_radius
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.min_theta = min_theta
        self.max_theta = max_theta
        self.direction = get_ray_directions(self.H, self.W, self.K)
        self.sr_direction = get_ray_directions(self.od_H, self.od_W, self.K_downscaled)

        
    def __len__(self):
        return self.n_pseudo_data
    
    def __getitem__(self, idx):
        # get random poses
        if self.ff:
            c2w = rand_FF_poses(self.centered_poses)
        else:
            c2w = rand_360_poses(
                radius=[self.min_radius, self.max_radius],
                theta_range=[self.min_theta , self.max_theta]
            )
        rays_o, rays_d = get_rays(self.direction, c2w.clone())
        return {'rays_o': rays_o,
                'rays_d': rays_d,
                'pose': c2w[:3, 3] if not self.ff else c2w[:3, :4]
                }
        
        
        