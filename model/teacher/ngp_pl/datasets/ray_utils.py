import numpy as np
import torch
from einops import rearrange
from kornia import create_meshgrid


@torch.cuda.amp.autocast(dtype=torch.float32)
def get_ray_directions(H, W, K, device='cpu', random=False, return_uv=False, flatten=True):
    """
    Get ray directions for all pixels in camera coordinate [right down front].
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W: image height and width
        K: (3, 3) camera intrinsics
        random: whether the ray passes randomly inside the pixel
        return_uv: whether to return uv image coordinates

    Outputs: (shape depends on @flatten)
        directions: (H, W, 3) or (H*W, 3), the direction of the rays in camera coordinate
        uv: (H, W, 2) or (H*W, 2) image coordinates
    """
    grid = create_meshgrid(H, W, False, device=device)[0] # (H, W, 2)
    u, v = grid.unbind(-1)
    print("k", H, W, K)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if random:
        directions = \
            torch.stack([(u-cx+torch.rand_like(u))/fx,
                         (v-cy+torch.rand_like(v))/fy,
                         torch.ones_like(u)], -1)
    else: # pass by the center
        directions = \
            torch.stack([(u-cx+0.5)/fx, (v-cy+0.5)/fy, torch.ones_like(u)], -1)
    if flatten:
        directions = directions.reshape(-1, 3)
        grid = grid.reshape(-1, 2)

    if return_uv:
        return directions, grid
    # print('d', directions.shape)
    return directions


@torch.cuda.amp.autocast(dtype=torch.float32)
def get_rays(directions, c2w):
    """
    Get ray origin and directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (N, 3) ray directions in camera coordinate
        c2w: (3, 4) or (N, 3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (N, 3), the origin of the rays in world coordinate
        rays_d: (N, 3), the direction of the rays in world coordinate
    """
    if c2w.ndim==2:
        # Rotate ray directions from camera coordinate to the world coordinate
        rays_d = directions @ c2w[:, :3].T
    else:
        rays_d = rearrange(directions, 'n c -> n 1 c') @ \
                 rearrange(c2w[..., :3], 'n a b -> n b a')
        rays_d = rearrange(rays_d, 'n 1 c -> n c')
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[..., 3].expand_as(rays_d).clone()
    # print('c2w', c2w.shape)
    return rays_o, rays_d


@torch.cuda.amp.autocast(dtype=torch.float32)
def axisangle_to_R(v):
    """
    Convert an axis-angle vector to rotation matrix
    from https://github.com/ActiveVisionLab/nerfmm/blob/main/utils/lie_group_helper.py#L47

    Inputs:
        v: (3) or (B, 3)
    
    Outputs:
        R: (3, 3) or (B, 3, 3)
    """
    v_ndim = v.ndim
    if v_ndim==1:
        v = rearrange(v, 'c -> 1 c')
    zero = torch.zeros_like(v[:, :1]) # (B, 1)
    skew_v0 = torch.cat([zero, -v[:, 2:3], v[:, 1:2]], 1) # (B, 3)
    skew_v1 = torch.cat([v[:, 2:3], zero, -v[:, 0:1]], 1)
    skew_v2 = torch.cat([-v[:, 1:2], v[:, 0:1], zero], 1)
    skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=1) # (B, 3, 3)

    norm_v = rearrange(torch.norm(v, dim=1)+1e-7, 'b -> b 1 1')
    eye = torch.eye(3, device=v.device)
    R = eye + (torch.sin(norm_v)/norm_v)*skew_v + \
        ((1-torch.cos(norm_v))/norm_v**2)*(skew_v@skew_v)
    if v_ndim==1:
        R = rearrange(R, '1 c d -> c d')
    return R


def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)


def average_poses(poses, pts3d=None):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of 3d point cloud (if None, center of cameras).
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.
    
    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
        pts3d: (N, 3)

    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    if pts3d is not None:
        center = pts3d.mean(0)
        print("center: ", center)
    else:
        center = poses[..., 3].mean(0)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0)) # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0) # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z)) # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x) # (3)

    pose_avg = np.stack([x, y, z, center], 1) # (3, 4)
    print("pose avg: ", pose_avg)
    return pose_avg


def focus_point_fn(poses: np.ndarray) -> np.ndarray:
    """Calculate nearest point to all focal axes in poses.
       refer to: https://github.com/google-research/multinerf/blob/5d4c82831a9b94a87efada2eee6a993d530c4226/internal/camera_utils.py#L144
    """
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt


def center_poses_nerf(poses):
    p34_to_44 = lambda p: np.concatenate(
        [p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1
    )

    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv(
            (np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)
    # pt_mindist = focus_point_fn(poses)

    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1, .2, .3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(
        poses[:, :3, :4])
    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:,:3,3]), -1)))
    print("radius: ", rad)
    return poses_reset, c2w, rad

def center_poses_mipnerf(poses: np.ndarray):
    """Transforms poses so principal components lie on XYZ axes.
       refer to: https://github.com/google-research/multinerf/blob/5d4c82831a9b94a87efada2eee6a993d530c4226/internal/camera_utils.py#L191
    Args:
      poses: a (N, 3, 4) array containing the cameras' camera to world transforms.
    Returns:
      A tuple (poses, transform), with the transformed poses and the applied
      camera_to_world transforms.
    """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds] # W_new 2 W_old: xy plane // ground. [new basis in old world coord]
    rot = eigvec.T # W_o 2 W_n
    if np.linalg.det(rot) < 0: # y x z--> det < 0, x y z -> det > 0
        rot = np.diag(np.array([1, 1, -1])) @ rot
        
    t_mean = focus_point_fn(poses)
    
    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses)) # W_old 2 W_new @ cam 2 W_old ==> cam-->W_new
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] > 0:
        # if cam y is not align with world Z. e.g y axix in cam in world is [0, 0, -1]: means y and world z are flipped. 
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    return poses_recentered, transform


def pad_poses(p: np.ndarray) -> np.ndarray:
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.0], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p: np.ndarray) -> np.ndarray:
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]


def center_poses(poses, pts3d=None):
    """
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
        pts3d: (N, 3) reconstructed point cloud

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pts3d_centered: (N, 3) centered point cloud
    """

    pose_avg = average_poses(poses, pts3d) # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg # convert to homogeneous coordinate for faster computation
                                # by simply adding 0, 0, 0, 1 as the last row
    pose_avg_inv = np.linalg.inv(pose_avg_homo)
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1) # (N_images, 4, 4) homogeneous coordinate

    poses_centered = pose_avg_inv @ poses_homo # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3] # (N_images, 3, 4)

    if pts3d is not None:
        pts3d_centered = pts3d @ pose_avg_inv[:, :3].T + pose_avg_inv[:, 3:].T
        return poses_centered, pts3d_centered,  pose_avg_inv
    return poses_centered, None, pose_avg_inv




def spheric_pose(theta, phi, radius, h):
    trans_t = lambda t : np.array([
        [1,0,0,0],
        [0,1,0,3 * h],
        [0,0,1,-t]
    ])

    rot_phi = lambda phi : np.array([
        [1,0,0],
        [0,np.cos(phi),-np.sin(phi)],
        [0,np.sin(phi), np.cos(phi)]
    ])

    rot_theta = lambda th : np.array([
        [np.cos(th),0,-np.sin(th)],
        [0,1,0],
        [np.sin(th),0, np.cos(th)]
    ])

    c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
    c2w = np.array([[-1,0,0],[0,0,1],[0,1,0]]) @ c2w
    return c2w


def create_spheric_poses(radius, mean_h, n_poses=120):
    """
    Create circular poses around z axis. world coord sys: z is up
    Inputs:
        radius: the (negative) height and the radius of the circle.
        mean_h: mean camera height
    Outputs:
        spheric_poses: (n_poses, 3, 4) the poses in the circular path
    """


    spheric_poses = []
    for th in np.linspace(0, 2*np.pi, n_poses+1)[:-1]:
        spheric_poses += [spheric_pose(th, -np.pi/12, radius,mean_h)]
    return np.stack(spheric_poses, 0)


def normalize(vectors):
    if isinstance(vectors, torch.Tensor):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)
    # numpy
    return vectors / (np.linalg.norm(vectors, axis=-1, keepdims=True) + 1e-10)


@torch.cuda.amp.autocast(dtype=torch.float32)
def rand_360_poses(
    size=1, 
    device='cpu', 
    radius=[], 
    theta_range=[0, 80 * np.pi/180], 
    phi_range=[0, 2*np.pi]
):
    ''' generate random poses from an orbit camera
        refer to: https://github.com/ashawkey/torch-ngp/blob/b6e080468925f0bb44827b4f8f0ed08291dcf8a9/nerf/provider.py#L57
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, \pi]
        phi_range: [min, max], should be in [0, 2\pi]
    Return:
        poses: [size, 4, 4]
    '''
    phi_range = [-178 * np.pi / 180, 178 * np.pi / 180]

    min_theta, max_theta = theta_range[0], theta_range[1] 
    mu = 0.5 * (min_theta + max_theta)
    std = 0.5 * (max_theta - min_theta)
    thetas = np.random.normal(mu, std) 
    low, high = min_theta, max_theta
    # low = max(min_theta - np.radians(10), np.radians(0.00001))
    # high = min(max_theta + np.radians(10), np.radians(89))

    is_out_range = (thetas < low) | (thetas > high)
    # if out of bounds, generate the theta unifromly ~ Uniform(min_theta, max_theta)
    thetas = np.where(
        is_out_range,
        np.random.uniform(min_theta, max_theta, 1),
        thetas
    )
    thetas = torch.FloatTensor(thetas)
    
    phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]    
    radius =  torch.rand(size, device=device) * (radius[1] - radius[0]) + radius[0]
    # print(thetas *  180 / np.pi, phis  *  180 / np.pi, "theta and phi")

    centers = torch.stack(
        [
            radius * torch.sin(thetas) * torch.cos(phis),
            radius * torch.sin(thetas) * torch.sin(phis),
            radius * torch.cos(thetas)
        ],
        dim=-1
    ) # [B, 3]

    # lookat
    forward_vector = -normalize(centers)
    up_vector = torch.FloatTensor([0, 0, -1]).to(device).unsqueeze(0).repeat(size, 1) # confused at the coordinate system...
    right_vector = normalize(torch.cross( up_vector, forward_vector, dim=-1))
    up_vector = normalize(torch.cross(forward_vector, right_vector, dim=-1))

    # poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses = torch.zeros((3, 4))
    poses[:3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:3, 3] = centers
        
    return poses


def cart_to_polar(poses):
    ''' Cartisian to polar coordinate
        poses: [N, 3, 4]
    '''
    
    rad = np.linalg.norm(poses[:, :3, 3], axis=-1)
    z = poses[:, 2, 3]
    theta = np.arccos(z / rad)
    x = poses[:, 0, 3]
    y = poses[:, 1, 3]
    
    phi = np.arctan2(y, x)
    return (np.min(theta), np.max(theta), np.min(phi), np.max(phi))


def rand_FF_poses(centered_poses):
    """
    Generate random poses for Forward-Facing sences

    Args:
        centered_poses : [N, 3, 4]. 
        avg_poses: [3, 4]
    """   
    cam_pos = centered_poses[..., 3] 
    z = centered_poses[..., 2]
    up = normalize(centered_poses[..., 1].mean(axis=0))

    # find the bounds for origin and view directions
    mins_o, maxs_o = get_bbox(cam_pos)
    mins_d, maxs_d = get_bbox((z))

    # randomly sample within the bounds
    c =  np.array(
        [
            rand_uniform(mins_o[0], maxs_o[0], scale=1.1),
            rand_uniform(mins_o[1], maxs_o[1], scale=1.1),
            rand_uniform(mins_o[2], maxs_o[2], scale=1.1)
        ]
    )
    z = np.array(
        [
            rand_uniform(mins_d[0], maxs_d[0], scale=1.1),
            rand_uniform(mins_d[1], maxs_d[1], scale=1.1),
            rand_uniform(mins_d[2], maxs_d[2], scale=1.1)
        ]
    )
    # calculate the poses
    return torch.FloatTensor(viewmatrix(z, up, c))
    


def get_bbox(array):
    '''get the bounding box of a bunch of points in 3d space'''
    array = np.array(array)
    assert len(array.shape) == 2 and array.shape[1] == 3  # shape should be [N, 3]
    mins, maxs = np.min(array, axis=0), np.max(array, axis=0)
    return mins, maxs


def rand_uniform(left, right, scale=1.):
    assert right > left
    middle = (left + right) * 0.5
    left = middle - (right - left) * scale * 0.5
    right = 2 * middle - left
    return np.random.rand() * (right - left) + left


def viewmatrix(z, up, pos):    
    z = normalize(z) 
    x = normalize(np.cross(up, z))  
    y = normalize(np.cross(z, x))
    pose = np.stack([x, y, z, pos], 1)
    return pose
