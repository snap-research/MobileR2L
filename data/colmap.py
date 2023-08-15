import numpy as np
import os, imageio
import cv2
from subprocess import check_output
import numpy as np
from dataclasses import dataclass

from utils import *

@dataclass
class SceneCenter:
    '''Scene center pre-calculated from the mean of the points in points3D.bin.
    Used in recentering the poses.
    '''
    fern: tuple = (-0.85707248, -0.15615284, 29.99040892)
    flower: tuple = (-2.82863346, 6.63070772, 70.20097372)
    room: tuple = (4.1845688, -1.537188, 32.04525597)
    orchids: tuple = (-0.17837235, 0.07818467, 32.35640172)
    horns: tuple = (1.27204619, 1.52593104, 39.45094112)
    trex: tuple = (-0.07751921, 1.63867973, 52.46994463)
    fortress: tuple = (-4.36502165e-02, -4.46507414e-03, 2.22926420e+01)
    leaves: tuple = (5.40735207, -2.33496984, 136.81563356)
scene_center = SceneCenter()

def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir
    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100. / r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue

        print('Minifying', r, basedir)

        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)

        ext = imgs[0].split('.')[-1]
        args = ' '.join(
            ['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)]
        )
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)

        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')


def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):

    poses_arr = np.load(
        os.path.join(basedir, 'poses_bounds.npy')
    )  # shape [20, 17]
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])  # shape [3, 5, 20]
    bds = poses_arr[:, -2:].transpose([1, 0]) 

    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png') or f.endswith('PNG')][0]
    sh = imageio.imread(img0).shape

    sfx = ''

    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1

    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print(imgdir, 'does not exist, returning')
        return

    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) \
                if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png') or f.endswith('PNG')]
    if poses.shape[-1] != len(imgfiles):
        print('Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]))
        return

    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1. / factor

    if not load_imgs:
        return poses, bds

    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    imgs = imgs = [imread(f)[..., :3] / 255. for f in imgfiles]
    imgs = np.stack(imgs, -1)

    print('Loaded image data', imgs.shape, poses[:, -1, 0])  # imgs.shape (H, W, 3, 20)
    return poses, bds, imgs


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec0 = normalize(
        np.cross(up, vec2)
    )  
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3, :3].T, (pts - c2w[:3, 3])[..., np.newaxis])[..., 0]
    return tt


def poses_avg(poses, scene):
    hwf = poses[0, :3, -1:]
    if scene_center.__dict__.get(scene, None):
        # ngp_pl doesn't work for ff scenes if centering by avg camera centers?
        center = np.array(scene_center.__dict__[scene])
    else: 
        center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    return c2w


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    '''@mst: zrate = 0.5, rots = 2'''
    render_poses = []
    rads = np.array(list(rads) + [1.])  #
    hwf = c2w[:, 4:5]
    for theta in np.linspace(0., 2. * np.pi * rots, N + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array(
                [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) *
            rads)
        # @mst: above is equivalent to matrix mul: [3, 4] @ [4, 1]
        z = normalize(
            c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.]))
        )  # @mst: why use extra focal instead of the focal in poses?
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses


def focus_point_fn(poses: np.ndarray) -> np.ndarray:
    """Calculate nearest point to all focal axes in poses.
    Reference: https://github.com/google-research/multinerf/blob/5d4c82831a9b94a87efada2eee6a993d530c4226/internal/camera_utils.py#L144
    """
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt


def spherify_poses(poses, bds, camera_convention):
    """ Reference: https://github.com/google-research/multinerf/blob/5d4c82831a9b94a87efada2eee6a993d530c4226/internal/camera_utils.py#L191
    
    Transforms poses so principal components lie on XYZ axes.
    Args:
        poses: a (N, 3, 4) array containing the cameras' camera to world transforms.
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
    # Flip coordinate system if z component of y-axis is negative
    if camera_convention == 'openGL':
        not_yz_align = (poses_recentered.mean(axis=0)[2, 1] < 0)
    elif camera_convention == 'openCV':
        not_yz_align = (poses_recentered.mean(axis=0)[2, 1] > 0)
        
    if not_yz_align: # if cam y is not align with world Z. e.g y axix in cam in world is [0, 0, -1]: means y and world z are flipped. 
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
    scale_factor = 1.0 / np.linalg.norm(poses_recentered[:, :3, 3], axis=-1).max()
    poses_recentered[:, :3, 3] *= scale_factor
    bds *= scale_factor
    rad = np.linalg.norm(poses_recentered[:, :3, 3], axis=-1).mean()
    centroid = np.mean(poses_recentered[:, :3, 3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2 - zh**2)
    new_poses = []

    for th in np.linspace(0., 2. * np.pi, 120):
        camorigin = np.array(
            [radcircle * np.cos(th), radcircle * np.sin(th), zh]
        )
        up = np.array([0, 0, -1.])
        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)
    new_poses = np.concatenate(
        [new_poses, np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)], -1
    )
    poses_recentered  = np.concatenate(
        [poses_recentered[:, :3, :4], np.broadcast_to(poses[0, :3, -1:], poses_recentered[:, :3, -1:].shape)], -1
    )
    
    return poses_recentered, new_poses, bds


def pad_poses(p: np.ndarray) -> np.ndarray:
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.0], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p: np.ndarray) -> np.ndarray:
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]

def recenter_poses(poses, scene):
    # poses shape: [n_img, 3, 5]
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])

    # get the average c2w
    c2w = poses_avg(poses, scene)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)  # make it to 4x4
    bottom = np.tile(
        np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1]
    )  # [n_img, 1, 4]
    poses = np.concatenate(
        [poses[:, :3, :4], bottom], -2
    )  # [n_img, 4, 4]

    # rotate: camera->world->average pose
    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4] 
    poses = poses_
    return poses

def load_colmap_data(
    basedir : str,
    input_height : int,
    input_width : int,
    output_height : int,
    output_width : int,
    scene : str,
    factor : int = 4,
    bd_factor : float = .75,
    llffhold : int = 8,
    ff : bool = False,
    use_sr_module : bool = True,
    camera_convention : str = 'openGL',
    device : torch.device = torch.device('cuda')
):
    
    poses, bds, imgs = _load_data(basedir, factor=factor)  
    print('Loaded', basedir, bds.min(), bds.max())

    # Correct rotation matrix ordering and move variable dim to axis 0
    if camera_convention == 'openGL': # from (-u, r, b) to (r, u, b)
        poses = np.concatenate(
            [poses[:, 1:2, :], 
            -poses[:, 0:1, :], 
            poses[:, 2:, :]], 
            1
        )
    elif camera_convention == 'openCV': # from (-u, r, b) to (r, -u, -b)
        poses = np.concatenate(
            [poses[:, 1:2, :], 
            poses[:, 0:1, :], 
            -poses[:, 2:3, :], 
            poses[:, 3:, :]],
            1
        )
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    if ff: # forward-facing
        poses = recenter_poses(poses, scene)
        sc = 1. if bd_factor is None else 1. / (bds.min() * bd_factor)
        print("Bound range: ", bds.min(), bds.min() * bd_factor)
        poses[:, :3, 3] *= sc
        bds *= sc
    else: # 360 degree scenes
        poses, render_poses, bds = spherify_poses(poses, bds, camera_convention)
    # poses, render_poses, bds = center_poses_mipnerf(poses, bds)

    c2w = poses_avg(poses, scene)
    print('Data:')
    print(poses.shape, images.shape,  bds.shape)  # (20, 3, 5) (20, 378, 504, 3) (20, 2)

    dists = np.sum(
        np.square(c2w[:3, 3] - poses[:, :3, 3]), -1
    )
    i_test = np.argmin(dists)
    images = images.astype(np.float32)
    poses = poses.astype(np.float32)
    num_frame = images.shape[0]
    
    if llffhold > 0:
        i_test = np.arange(num_frame)[::llffhold]
    i_val = i_test
    is_train = lambda i: i not in i_test and i not in i_val
    i_train = np.array([i for i in np.arange(num_frame) if is_train(i)])
        
    H, W, focal = poses[0,:3,-1]
    if use_sr_module:
        if (
            input_height < 0 or input_width < 0 
            or output_height < 0 or output_width < 0
        ):
            raise ValueError(
                'input and output height and width should be positive. Check if you \
                            set the the values correctly!'
            )
        focal = focal * (input_height / H)
        out_H = output_height
        out_W = output_width
        print(H, W, output_height, output_width, '----')
        if not (out_H == H and out_W == W):
            imgs_scaled = []
            for i, img in enumerate(images):
                imgs_scaled += [cv2.resize(img, (out_W, out_H), interpolation=cv2.INTER_AREA)]
                
            images = np.stack(imgs_scaled, axis=0)
            print(images.shape, ">.......")
            
        H = input_height
        W = input_width
    # breakpoint()
    return {
                'images': to_tensor(images, device),
                'poses': to_tensor(poses[:, :3, :4], device),
                'bds': to_tensor(bds, device),
                'i_split': {'i_train': i_train, 'i_val': i_val, 'i_test': i_test},
                'H' : int(H),
                'W' : int(W),
                'focal' : focal,
                'ff': True if ff else False,
                'dataset_type': 'Colmap',
                'sc': sc if ff else None,
                'render_poses': render_poses.to(device) if not ff else None
            }
    # return (to_tensor(imgs), to_tensor(poses), to_tensor(bds),
    #         to_tensor(render_poses), i_test, [H, W, focal], poses)
    