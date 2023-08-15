import numpy as np
import torch
from annotation import *


def openCV2openGL(pose : Float[ndarray, "3 4"]) ->  Float[ndarray, "3 4"]:
    """Convert forward-facing poses from openCV to openGL convention.
    I = np.diag((1, -1, -1, 1))
    # OpenCV c2w to OpenGL
    c2w = np.linalg.inv(I) @ c2w @ I

    equivaltent to:

    c2w *  np.array(
        [[1, -1, -1, 1],
        [-1, 1, 1, -1],
        [-1, 1, 1, -1]]
    )
    Args:
        poses : (3, 4)
    """
    assert pose.shape == (3, 4), \
                'Shape of the pose should be (3, 4)'
    mask = np.array(
        [[1, -1, -1, 1],
        [-1, 1, 1, -1],
        [-1, 1, 1, -1]]
    )
    return pose * mask

def lookAt(
        centers : Float[Tensor, "1 3"],
        camera_convention : str = 'openGL'
    ) -> Float[Tensor, "3 4"]:
    """Create c2w matrix from a camera center
       refer to: https://github.com/google-research/multinerf/blob/5d4c82831a9b94a87efada2eee6a993d530c4226/internal/camera_utils.py#L191
    """
    if camera_convention == 'openCV':
        forward_vector = -normalize(centers)
        up_vector = torch.FloatTensor([0, 0, -1]).unsqueeze(0)
        right_vector = normalize(torch.cross(up_vector, forward_vector, dim=-1))
        up_vector = normalize(torch.cross(forward_vector, right_vector, dim=-1))
    elif camera_convention == 'openGL':
        forward_vector = normalize(centers)
        up_vector = torch.FloatTensor([0, 0, -1]).unsqueeze(0)
        right_vector = normalize(torch.cross(forward_vector, up_vector, dim=-1))
        up_vector = normalize(torch.cross(forward_vector, right_vector, dim=-1))

    poses = torch.zeros((3, 4))
    poses[:3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:3, 3] = centers
    return poses
    
def normalize(
        vectors: Union[Float[ndarray, "..."], Float[Tensor, "..."]]
    ):
    if isinstance(vectors, torch.Tensor):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)
    # numpy array
    return vectors / (np.linalg.norm(vectors, axis=-1, keepdims=True) + 1e-10)
