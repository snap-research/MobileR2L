from .colmap import load_colmap_data
from .blender import load_blender_data
from .pseudo import PseudoDataset
from annotation import *
from utils import (
    get_rank,
    get_world_size
)

import numpy as np
import torch
from easydict import EasyDict as edict
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler


def select_and_load_dataset(
    basedir: str,
    dataset_type: str,
    input_height : int,
    input_width : int,
    output_height : int,
    output_width : int,
    scene : str,
    test_skip : int = 8,
    factor : int = 8,
    bd_factor : float = 0.75,
    llffhold : int = 8,
    ff : bool = False,
    use_sr_module : bool = True,
    camera_convention : str='openGL',
    ndc : Optional[bool] = False,
    device : int = 0,
    n_sample_per_ray : int = 8
) -> Dict[str, Any]:
    if dataset_type == 'Colmap':
        dataset_info = load_colmap_data(
            basedir, 
            input_height,
            input_width,
            output_height,
            output_width,
            scene,
            factor,
            bd_factor,
            llffhold,
            ff,
            use_sr_module,
            camera_convention,
            device
        )
        dataset_info = edict(dataset_info)
        dataset_info.near = 0 if ndc else dataset_info.bds.cpu().min() * .9
        dataset_info.far = 1 if ndc else dataset_info.bds.cpu().max() * 1.
        dataset_info.ndc = ndc
        dataset_info.device = device
        dataset_info.camera_convention = camera_convention
        dataset_info.n_sample_per_ray = n_sample_per_ray
        dataset_info.radius = (
            dataset_info
            .poses[dataset_info.i_split.i_train][:, :3, 3]
            .norm(dim=-1)
            .mean(0)
        )   
    elif dataset_type == 'Blender':
        dataset_info = load_blender_data(
            basedir,
            input_height,
            input_width,
            output_height,
            output_width ,
            camera_convention,
            use_sr_module,
            test_skip,
            device
        )
        dataset_info = edict(dataset_info)
        dataset_info.near = 2
        dataset_info.far = 6
        dataset_info.ndc = False
        dataset_info.ff = False
        dataset_info.camera_convention = camera_convention
        # white bg
        dataset_info.update(
            {
                'images' : dataset_info.images[...,:3] * dataset_info.images[...,-1:] \
                    + (1.-dataset_info.images[...,-1:])
            }
        )
        dataset_info.device = device
        dataset_info.n_sample_per_ray = n_sample_per_ray
        dataset_info.sc = None
        dataset_info.radius = (
            dataset_info
            .poses[dataset_info.i_split.i_train][:, :3, 3]
            .norm(dim=-1)
            .mean(0)
        )   
    else:
        raise NotImplementedError
    return dataset_info


def get_pseduo_dataloader(
    pseudo_dir : str,
    batch_size : int,
    num_workers : int,
    camera_convention : str,
    sc : float
):
    trainset = PseudoDataset(pseudo_dir, camera_convention, sc)    
    trainloader = torch.utils.data.DataLoader(
        dataset=trainset, 
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        # sampler=InfiniteSamplerWrapper(len(trainset)), # DP training
        sampler=DistributedSampler(
            trainset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=True
        ),
        drop_last=True
    )        
    return trainloader, len(trainset)


class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples
        
    def __iter__(self):
        return iter(self._InfiniteSampler(self.num_samples))
    
    def __len__(self):
        return 2 ** 31
    
    def _InfiniteSampler(self, n):
        order = np.random.permutation(n)
        i = 0
        while True:
            yield order[i]
            i += 1
            if i == n:
                order = np.random.permutation(n)
                i = 0
