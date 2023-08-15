import os
import numpy as np
import torch
import imageio
import json
import cv2
from utils import to_tensor
from tqdm import tqdm

def load_blender_data(
    basedir : str,
    input_height : int = 100,
    input_width : int = 100,
    output_height : int = 800,
    output_width : int = 800,
    camera_convention : str = 'openGL',
    use_sr_module : bool = True,
    testskip : int = 1,
    device : torch.device=torch.device('cuda')
):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
 
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s == 'train' or testskip == 0:
            skip = 1
        else:
            skip = testskip

        for frame in tqdm(meta['frames'][::skip]):
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            pose_mat = np.array(frame['transform_matrix'])[:3, :4]
            poses += [pose_mat]
        imgs = (np.array(imgs) / 255.).astype(np.float32)  # keep all 4 channels (RGBA)
        num_channels = imgs[-1].shape[2] 
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        
        all_imgs.append(imgs)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    assert ('camera_angle_x' in metas['test'])
    camera_angle_x = float(metas['test']['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    if use_sr_module:
        if (
                input_height < 0 
                or input_width < 0 
                or output_height < 0 
                or output_width < 0
        ):
            raise ValueError('input and output height and width should be positive. Check if you \
                            set the the values correctly!'
            )
        focal = focal * (input_height / H)
        out_H = output_height
        out_W = output_width

        if not (out_H == H and out_W == W):
            imgs_scaled= np.zeros((imgs.shape[0], out_H, out_W, 3))
            for i, img in enumerate(imgs):
                imgs_scaled[i] = cv2.resize(
                    img, (out_W, out_H), interpolation=cv2.INTER_AREA
                ) 
            imgs = imgs_scaled
            
        H = input_height
        W = input_width
    return {
                'images': to_tensor(imgs, device),
                'poses': to_tensor(poses, device),
                'H' : int(H),
                'W' : int(W),
                'focal' : focal,
                'i_split': {'i_train' : i_split[0], 'i_val': i_split[1], 'i_test': i_split[2]},
                'dataset_type': 'Blender',
                'render_poses': get_novel_poses(120, to_tensor(poses[i_split[0]], device), camera_convention)
            }

def normalize(vectors):
    if isinstance(vectors, torch.Tensor):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)
    # numpy array
    return vectors / (np.linalg.norm(vectors, axis=-1, keepdims=True) + 1e-10)

@torch.cuda.amp.autocast(enabled=False)
def get_novel_poses(n_pose, train_poses, camera_convention, theta_range=[0, 2 * np.pi]):
    '''Get circular poses'''

    phis = torch.Tensor(np.linspace( theta_range[0],  theta_range[1], n_pose + 1)[:-1]).cuda()
    thetas = torch.Tensor([np.pi/4] * len(phis)).cuda()
    radius = train_poses[:, :3, 3].norm(dim=-1).mean(0) # ~ 4.0 for Blender 
    
    centers = torch.stack(
        [radius * torch.sin(thetas) * torch.cos(phis),
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas)],
        dim=-1
    ) # [B, 3]

    if camera_convention == 'openCV':
        forward_vector = -normalize(centers)
        up_vector = torch.FloatTensor([0, 0, -1]).unsqueeze(0).repeat(n_pose, 1).cuda()
        right_vector = normalize(torch.cross(up_vector, forward_vector, dim=-1))
        up_vector = normalize(torch.cross(forward_vector, right_vector, dim=-1))
    elif camera_convention == 'openGL':
        forward_vector = normalize(centers)
        up_vector = torch.FloatTensor([0, 0, -1]).unsqueeze(0).repeat(n_pose, 1).cuda()
        right_vector = normalize(torch.cross(forward_vector, up_vector, dim=-1))
        up_vector = normalize(torch.cross(forward_vector, right_vector, dim=-1))

    # # lookat
    # forward_vector = normalize(centers) # View direction is -z in openGL
    # up_vector = torch.FloatTensor([0, 0, -1]).to('cuda').unsqueeze(0).repeat(n_pose, 1)
    # right_vector = normalize(torch.cross(forward_vector,up_vector, dim=-1)) # openGL
    # up_vector = normalize(torch.cross(forward_vector, right_vector, dim=-1))
    
    poses = torch.eye(4, dtype=torch.float, device='cuda').unsqueeze(0).repeat(n_pose, 1, 1)
    poses[:,:3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers
    return poses[:, :3, :4]

