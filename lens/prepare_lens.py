import torch.nn as nn
import numpy as np
import torch
import onnx
import onnxruntime
import json
import click

def read_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f, indent=2)
    
class Embedder(nn.Module):
    def __init__(self, weights, MULTRIES):
        super(Embedder, self).__init__()
        self.weights = weights # [1, M,  H*W, n_sample * 3]
        self.MULTRIES = MULTRIES

    def forward(self, pts):
        """
        pts: output from samlping, shape [1, 1, H * W, n_sample * 3]
        """
        # broadcast pts([1, 1, H * W, n_sample * 3]) to [1, M,  H*W, n_sample * 3]
        pts_b = torch.cat([pts] * self.MULTRIES , dim=1)
        y = pts_b * self.weights # [1, M,  H*W, n_sample * 3]
        y = torch.cat([torch.sin(y), torch.cos(y), pts], dim=1) # [1, 2M + 1, H * W, n_sample * 3]
        return y
    
class Sampler(nn.Module):
    def __init__(self, z_val, dir, H, W, n_sample):
        super(Sampler, self).__init__()
        self.z_val = z_val #[1, n_sampe, 1]
        self.dir = dir # [H*W, 3, 3]
        self.H = H
        self.W = W
        self.n_sample = n_sample

    def forward(self, c2w_33, c2w_13):
        c2w_b = c2w_33
        c2w_b = torch.cat([c2w_b] * (self.H * self.W), dim=1) # [1, h * w , 3 , 3]
        tmp = self.dir.reshape((1, self.H * self.W, 3, 3)) * c2w_b
        tmp = torch.permute(tmp, (0, 3, 1, 2))
        rays_d =  torch.sum(tmp, dim=1)
        rays_d = rays_d.reshape((1, self.H, self.W, 3)) # [H, W, 3]
        
        c2w = c2w_13.reshape((1, 3))
        rays_o = torch.cat([c2w] * (self.H * self.W), dim=0) # [ H * W, 3]
        rays_o = rays_o.reshape((1, self.H * self.W, 1, 3))
        rays_o = torch.cat([rays_o] * self.n_sample, dim=2) # [1, H * W, N_SAMPLE, 3]

        rays_d = rays_d.reshape((1, self.H * self.W, 1, 3))
        rays_d = torch.cat([rays_d] * self.n_sample, dim=2) # [H * W, N_SAMPLE, 3]

        self.z_val = torch.cat([self.z_val] * (self.H * self.W), dim= 0).reshape((1, self.H * self.W, self.n_sample, 1)) # [H * W, n_sample, 1]
        self.z_val = torch.cat([self.z_val] * 3, dim= 3) # [H * W, n_sample, 3]

        pts = rays_o + rays_d * self.z_val# [1, H*W, n_sample, 3]
        return pts.reshape((1, 1, self.H * self.W, self.n_sample * 3))

@click.command
@click.option('--project_path', type=str)
def main(project_path):
    cam_info = read_json(f'{project_path}/intrinsics.json')
    H = cam_info['H']
    W = cam_info['W']
    n_sample = cam_info['n_sample']
    multires = cam_info['multires']
    near = cam_info['near']
    far = cam_info['far']
    focal = cam_info['focal']
    
    # embedder
    weights = 2 ** torch.linspace(0, multires-1, steps=multires).expand(1, H * W, n_sample * 3, multires).permute((0, 3, 1, 2)) # [1, M, H*W, n_sample * 3]
    emb = Embedder(weights, multires)
    args = (torch.randn(1, 1, H * W, n_sample * 3))
    torch.onnx.export(
        emb.cpu(),
        args,
        "./Embedder.onnx",
        export_params=True,
        do_constant_folding=True,
        keep_initializers_as_inputs=False,
        opset_version=11,
        verbose=True,
        input_names = ['rays'],   # the model's input names
        output_names = ['embbrays'], # the model's output names
        dynamic_axes={
                        'rays' : {0 : 'batch_size'},
                        'embbrays' : {0 : 'batch_size'}
                    }
    )
    del emb
    
    
    #sampler
    shift = 0.5
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1,H))
    i, j = i.t(), j.t()
    dirs = torch.stack([(i - W*.5 + shift)/focal, -(j -H*.5 + shift)/focal, -torch.ones_like(i)], dim=-1) # [H, W, 3]
    t_vals = torch.linspace(0., 1., steps=n_sample) # [n_sample]
    z_vals = near * (1 - t_vals) + far * (t_vals) # [n_sample]

    z_vals = z_vals.reshape((1, n_sample, 1))
    dirs = dirs.reshape((1, H * W, 1, 3))
    dirs = torch.cat([dirs] * 3, dim=2)

    sampler = Sampler(z_vals, dirs, H, W, n_sample)
    args = (torch.randn(1, 1, 3, 3), torch.randn(1, 1, 1, 3))
    torch.onnx.export(
        sampler.cpu(),               
        args,                     
        "./Sampler.onnx",  
        export_params=True,
        do_constant_folding=True,
        keep_initializers_as_inputs=False,
        opset_version=11,
        verbose=True,
        input_names = ['c2w33', "c2w13"],   
        output_names = ['pts'], 
        dynamic_axes={
                        'c2w33' : {0 : 'batch_size'},   
                        'c2w13' : {0 : 'batch_size'},
                        'pts' : {0 : 'batch_size'}
                    }
    )
    del sampler
    
    
if __name__ == '__main__':
    main()