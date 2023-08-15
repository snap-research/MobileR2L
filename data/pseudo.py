from ast import Str
import os
from os.path import join 
import torch
from torch.utils.data import Dataset
import numpy as np
import json
from tqdm import tqdm
from glob import glob

from . import poses_utils
from . import rays_utils
from annotation import *

class PseudoDataset(Dataset):
    """Dataset for distilled pseudo data from the teacher 
    """
    def __init__(
        self,
        root_dir : str,
        camera_convention : str='openGL',
        sc : Optional[float]=None
    ) -> None:
        rgb_dir = glob(f'{root_dir}/*rgb')[0]
        poses_dir = join(root_dir, 'poses')

        self.camera_convention = camera_convention
        self._read_intrinscs(root_dir)
        self.dataset_info.update({'sc': sc})
        
        print('Loading pseudo dataset:')
        self.pseudo_data = [
            {
                'rgb': join(rgb_dir, f'rendered_rgb_{str(idx).zfill(5)}.npy'),
                'pose': join(poses_dir, f'c2w_{str(idx).zfill(5)}.npy')
            } 
            for idx in tqdm(range(len(os.listdir(rgb_dir))))
        ]
        print(f'Loaded {len(self.pseudo_data)} pseudo data.')
        
        self.directions = rays_utils.get_ray_directions(
            self.dataset_info['downscaled_height'],
            self.dataset_info['downscaled_width'],
            self.dataset_info['downscaled_focal'],
        )
        print(  self.dataset_info)
        
    def _read_intrinscs(self, root_dir : str):
        """dataset_info keys:
            H,
            W,
            focal,
            scale,
            max_radius,
            dataset_type
            
        """
        with open(join(root_dir, 'hwf/dataset_info.json'), 'r') as f:
            self.dataset_info = json.load(f)

    def __len__(self) -> int:
        return len(self.pseudo_data)

    def __getitem__(self, index : int):
        rgb = np.load(
            self.pseudo_data[index]['rgb']
        )
        pose = np.load(
            self.pseudo_data[index]['pose']
        )
        #TODO change to colmap
        if self.dataset_info['dataset_type'] == 'colmap':
            if self.dataset_info['ff']:
                assert self.camera_convention == 'openGL', \
                    'forward-facing needs openGL convention'
                c2w = poses_utils.openCV2openGL(pose)
                # ngp_pl sacles the cam centers with 'scale'.
                # Here we scale it back and re-scale it in NeRF way
                c2w[:, -1] *= (self.dataset_info['scale'] * self.dataset_info['sc'])
                c2w = torch.FloatTensor(c2w)
            else:
                cam_pos = ( 
                    (torch.FloatTensor(pose) / self.dataset_info['max_radius'])
                    .view(1, 3)
                )
                c2w = poses_utils.lookAt(cam_pos, self.camera_convention)
        else: 
            # npg_pl scales down the center of blender scenes by (4.031128857175551 / 1.5)
            # We scale it back here. 
            cam_pos = (
               (torch.FloatTensor(pose) * (4.031128857175551 / 1.5))
                .view(1, 3)
            )
            c2w = poses_utils.lookAt(cam_pos, self.camera_convention)
            
        rays_o, rays_d = rays_utils.get_rays(self.directions, c2w)   
        return (
                    rays_o,
                    rays_d,
                    rgb
            )
            
            
            
            
            
            

