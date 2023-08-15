import os
import torch
import torch.nn as nn
import numpy as np
import copy
from collections import OrderedDict
import torch.distributed as dist


def to_tensor(arr, device):
    if isinstance(arr, torch.Tensor):
        return arr.to(device)
    else:
        return torch.FloatTensor(arr).to(device)

def to_ndarray(arr):
    if isinstance(arr, np.ndarray):
        return arr
    else:
        return arr.data.cpu().numpy()

def to_8bit(x):
    return (255 * np.clip(to_ndarray(x), 0, 1)).astype(np.uint8)

def img2mse(x, y):
    return torch.mean((x - y)**2)

def mse2psnr(x):
    return  -10. * torch.log(x) / torch.log(to_tensor([10.], get_rank()))

def save_onnx(model, onnx_path, dummy_input):
    model = copy.deepcopy(model)
    model.eval()
    if hasattr(model, 'module'): 
        model = model.module
        model.eval()
    torch.onnx.export(
        model.cpu(),
        dummy_input.cpu(),
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        keep_initializers_as_inputs=False,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    del model
    
def save_ml(model, onnx_path, dummy_input):
    import coremltools as ct
    model = copy.deepcopy(model)
    if hasattr(model, 'module'): 
        model = model.module
    model.eval()
    traced_model = torch.jit.trace(model, dummy_input)
    out = traced_model(dummy_input)   
    model = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[ct.TensorType(shape=dummy_input.shape)]
    )
    model.save(onnx_path)
    del model

def save_tflite(model, onnx_path, dummy_input):
    from tinynn.converter import TFLiteConverter
    model = copy.deepcopy(model)
    model.cpu()
    if hasattr(model, 'module'): 
        model = model.module
    model.eval()
    converter = TFLiteConverter(model, dummy_input.cpu(), onnx_path)
    converter.convert()
    del model

def check_onnx(model, onnx_path, dummy_input):
    r"""Refer to https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
    """
    import onnx, onnxruntime
    model = copy.deepcopy(model)
    model.eval()
    if hasattr(model, 'module'): 
        model = model.module
        model.eval()
    model, dummy_input = model.cpu(), dummy_input.cpu()
    torch_out = model(dummy_input)[0]
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(
        onnx_path,
        providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_ndarray(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)
    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(
        to_ndarray(torch_out),
        ort_outs[0],
        rtol=1e-03,
        atol=1e-05
    )
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

def undataparallel(input):
    '''remove the module. prefix caused by nn.DataParallel'''
    if isinstance(input, nn.Module):
        model = input
        if hasattr(model, 'module'):
            model = model.module
        return model
    elif isinstance(input, OrderedDict):
        state_dict = input
        w = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                assert len(k.split('module.')) == 2
                k = k.split('module.')[1]
            w[k] = v
        return w
    else:
        raise NotImplementedError

def mkdirs(*dirs, exist_ok=True):
    for d in dirs:
        os.makedirs(d,  mode=0o777, exist_ok=exist_ok)

def cache_code(root_dir, dest_dir):
    cmd = f"rsync -az --exclude='logs/*' --exclude='model/teacher/*' --exclude='dataset/*' {root_dir} {dest_dir}"
    os.system(cmd)
    
    
    
#------------------ ddp utils ------------------#

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        print( args.gpu, '.....')
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print(
        '| distributed init (rank {})'.format(args.rank), flush=True
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        world_size=args.world_size,
        rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def main_process(func):
    def wrapper(*args, **kwargs):
        if is_main_process():
            return func(*args, **kwargs)
        else :
            pass
    return wrapper

def set_epoch_num(global_step, num_iters, batch_size, num_pseudo, world_size):
    steps_per_epoch = num_pseudo // (world_size * batch_size)
    iters_to_run = num_iters - global_step
    num_epochs = iters_to_run // steps_per_epoch
    return num_epochs