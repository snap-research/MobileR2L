import torch
from kornia import create_meshgrid
from einops import rearrange
from annotation import *

@torch.cuda.amp.autocast(enabled=False)
class PointSampler:
    def __init__(self, dataset_info : dict):
        """_summary_

        Args:
            dataset_info (dict):
                H
                W
                focal
                device
                cam_convention
                near
                far
                n_sample
                ndc
                ff
        """
        self.dataset_info = dataset_info
        self.direction = get_ray_directions(
            self.dataset_info['H'],
            self.dataset_info['W'],
            self.dataset_info['focal'],
            self.dataset_info['device'],
            self.dataset_info['camera_convention']
        )
        self.t = (
            torch.linspace(0., 1., steps=self.dataset_info['n_sample_per_ray'])
            .to(self.dataset_info['device'])
        )
        z = self.dataset_info['near'] * (1 - self.t) + self.dataset_info['far'] * self.t
        self.z = (
            z[None, :]
            .expand(self.dataset_info['H'] * self.dataset_info['W'], self.dataset_info['n_sample_per_ray'])
        )
    
    def sample(
        self,
        rays_o : Optional[Float[Tensor, 'N 3']]=None,
        rays_d : Optional[Float[Tensor, 'N 3']]=None,
        c2w : Optional[Union[Float[Tensor, '3 4'], Float[Tensor, 'N 3 4']]]=None,
        perturb : bool=True
    ):
        
        if c2w is not None:
            # during test phase
            rays_o, rays_d = get_rays(self.direction, c2w)
            #todo: confirm this is the behaviour of orignal code
            perturb = False # don't perturb during inference 
        else:
            # during training phase
            assert rays_o is not None and rays_d is not None
        
        if perturb:
            mids = .5 * (self.z[..., 1:] + self.z[..., :-1])
            upper = torch.cat([mids, self.z[..., -1:]], dim=-1)
            lower = torch.cat([self.z[..., :1], mids], dim=-1)
            t_rand = torch.rand(self.z.shape).to(self.dataset_info['device'])  # [n_ray, n_sample]
            z = lower + (upper - lower) * t_rand
        else:
            z = self.z
        if self.dataset_info['ff'] and self.dataset_info['ndc']:
            # use ndc space for ff
            rays_o, rays_d = self._to_ndc(rays_o, rays_d)
        # (H*W, n_sample, 3)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z[..., :, None]
        
        #todo: check if .view is needed
        return pts.view(pts.shape[0], -1)

    def _to_ndc(
        self,
        rays_o : Float[Tensor, 'N 3'], 
        rays_d : Float[Tensor, 'N 3'],
        near : float = 1
    ) -> Tuple[Float[Tensor, 'N 3'], Float[Tensor, 'N 3']]:
        
        t = -(near + rays_o[...,2]) / rays_d[...,2]
        rays_o = rays_o + t[...,None] * rays_d
        
        # Projection
        o0 = (
                -1./(self.dataset_info['W']/(2.*self.dataset_info['focal'])) * rays_o[...,0] / rays_o[...,2]
        )
        o1 = (
                -1./(self.dataset_info['H']/(2.*self.dataset_info['focal'])) * rays_o[...,1] / rays_o[...,2]
        )
        o2 = 1. + 2. * near / rays_o[...,2]

        d0 = (
                -1./(self.dataset_info['W']/(2.*self.dataset_info['focal'])) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
        )
        d1 = (
                -1./(self.dataset_info['H']/(2.*self.dataset_info['focal'])) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
        )
        d2 = -2. * near / rays_o[...,2]
        
        rays_o = torch.stack([o0,o1,o2], -1)
        rays_d = torch.stack([d0,d1,d2], -1)
        
        return rays_o, rays_d

        
@torch.cuda.amp.autocast(enabled=False)
class PositionalEmbedder:
    def __init__(
        self,
        L : int,
        device : torch.device,
        include_input : bool=True
    ):
        self.weights = 2**torch.linspace(0, L - 1, steps=L).to(device)  # [L]
        self.include_input = include_input
        self.embed_dim = 2 * L + 1 if include_input else 2 * L

    def __call__(self, x):
        y = x[..., None] * self.weights  # [n_ray, dim_pts, 1] * [L] -> [n_ray, dim_pts, L]
        y = torch.cat([torch.sin(y), torch.cos(y)], dim=-1) 
        if self.include_input:
            y = torch.cat([y, x.unsqueeze(dim=-1)], dim=-1) 

        return y.view(y.shape[0], -1)  # [n_ray, dim_pts*(2L+1)]

# TODO: disable?
@torch.cuda.amp.autocast(dtype=torch.float32)
def get_ray_directions(
    H : int,
    W : int,
    focal : float,
    device : torch.device = torch.device('cpu'),
    camera_convention : str = 'openGL'
) -> Float[Tensor, 'N 3']:
    # (H, W, 2)
    grid = create_meshgrid(H, W, False, device=device)[0]
    u, v = grid.unbind(-1)
    
    if camera_convention == 'openGL':
        directions = torch.stack(
            [
                (u - 0.5 * W + 0.5) / focal, 
                -(v - 0.5 * H + 0.5) / focal, 
                -torch.ones_like(u)
            ], -1
        )
    elif camera_convention == 'openCV':
        directions = torch.stack(
            [
                (u - 0.5 * W + 0.5) / focal, 
                (v - 0.5 * H + 0.5) / focal, 
                torch.ones_like(u)
            ], -1
        )
    return directions.reshape(-1, 3)


@torch.cuda.amp.autocast(dtype=torch.float32)
def get_rays(
    directions : Float[Tensor, 'N 3'],
    c2w : Union[Float[Tensor, '3 4'], Float[Tensor, 'N 3 4']]
)-> Tuple[Float[Tensor, 'N 3'], Float[Tensor, 'N 3']]:
    """
        refer to ngp_pl implementation
    """
    if c2w.ndim==2:
        # Rotate ray directions from camera coordinate to the world coordinate
        rays_d = directions @ c2w[:, :3].T
    else:
        rays_d = (
            rearrange(directions, 'n c -> n 1 c') 
                @ rearrange(c2w[..., :3], 'n a b -> n b a')
        )
        rays_d = rearrange(rays_d, 'n 1 c -> n c')
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[..., 3].expand_as(rays_d).clone()
    return rays_o, rays_d
        