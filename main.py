try:
    import pretty_traceback
    pretty_traceback.install()
except ImportError:
    pass  
from os.path import join
import click
import torch
import logging
from tqdm import tqdm
from pprint import pprint
import torch.distributed as dist
from easydict import EasyDict as edict
from tqdm.contrib.logging import logging_redirect_tqdm

from data import (
    select_and_load_dataset,
    get_pseduo_dataloader
)
from utils import (
    is_main_process,
    set_epoch_num,
    get_rank,
    init_distributed_mode,
    get_world_size
)
from model import R2LEngine

@click.command()
@click.option('--project_name', type=str)
# dataset 
@click.option('--root_dir', type=str)
@click.option('--dataset_type', type=click.Choice(['Blender', 'Colmap'], case_sensitive=True))
@click.option('--pseudo_dir', type=str)
@click.option('--ff', is_flag=True, default=False, help='Whether the scene is forward-facing')
@click.option('--ndc', is_flag=True, default=False)
@click.option('--scene', type=str)
@click.option('--testskip', type=int, default=8)
@click.option('--factor', type=int, default=4)
@click.option('--llffhold', type=int, default=8)
@click.option('--bd_factor', type=float, default=0.75)
@click.option('--camera_convention', type=str, default='openGL')
# train/test 
@click.option('--run_train', is_flag=True)
@click.option('--run_render', is_flag=True)
@click.option('--render_test', is_flag=True, help='Render the testset.')
@click.option('--finetune', is_flag=True)
@click.option('--amp', is_flag=True, default=False)
@click.option('--resume', is_flag=True, default=False)
@click.option('--perturb', is_flag=True, default=True)
@click.option('--num_workers', type=int)
@click.option('--batch_size', type=int)
@click.option('--num_epochs', type=int, default=500000)
@click.option('--num_iters', type=int, default=500000)
@click.option('--ckpt_dir', type=str)
@click.option('--lrate', type=float, default=0.0005)
@click.option('--lr_scale', type=float, default=1.0)
@click.option('--lrate_decay', type=int, default=500)
@click.option('--warmup_lr', type=str, default='0.0001,200')
@click.option('--lpips_net', type=str, default='alex')
@click.option('--export_onnx', is_flag=True, default=False)
@click.option('--debug', is_flag=True, default=False)
@click.option('--no_cache', is_flag=True, default=False)
@click.option('--convert_snap_gelu', type=bool, default=True)
# model
@click.option('--input_height', type=int)
@click.option('--input_width', type=int)
@click.option('--output_height', type=int)
@click.option('--output_width', type=int)
@click.option('--n_sample_per_ray', type=int, default=8)
@click.option('--multires', type=int, default=6)
@click.option('--num_sr_blocks', type=int, default=3)
@click.option('--num_conv_layers', type=int, default=2)
@click.option('--sr_kernel', type=(int, int, int), default=(64, 64, 16))
@click.option('--netdepth', type=int, default=60)
@click.option('--netwidth', type=int, default=256)
@click.option('--activation_fn', type=str, default='gelu')
@click.option('--use_sr_module', is_flag=True, default=True)
# logging/saving options
@click.option("--i_print", type=int, default=10000, help='frequency of console printout and metric loggin')
@click.option("--train_image_log_step", type=int, default=5000, help='frequency of tensorboard image logging')
@click.option("--i_weights", type=int, default=10000, help='frequency of weight ckpt saving')
@click.option("--i_save_rendering", type=int, default=10000, help='frequency of weight ckpt saving')
@click.option("--i_testset", type=int, default=10000, help='frequency of testset saving')
@click.option("--i_video", type=int, default=100000, help='frequency of render_poses video saving')
def main(**kwargs):
    torch.backends.cudnn.benchmark = True
    # setup args
    args = edict(kwargs)
    init_distributed_mode(args)
    device = get_rank()
    world_size = get_world_size()
    
    # load data    
    dataset_info = select_and_load_dataset(
        basedir=join(args.root_dir, args.scene),
        dataset_type=args.dataset_type,
        input_height=args.input_height,
        input_width=args.input_width,
        output_height=args.output_height,
        output_width=args.output_width,
        scene=args.scene,
        test_skip=args.testskip,
        factor=args.factor,
        bd_factor=args.bd_factor,
        llffhold=args.llffhold,
        ff=args.ff,
        use_sr_module=args.use_sr_module,
        camera_convention=args.camera_convention,
        ndc=args.ndc,
        device=device,
        n_sample_per_ray=args.n_sample_per_ray
    )
    i_test = dataset_info.i_split.i_test
    test_images = dataset_info.images[i_test]
    test_poses = dataset_info.poses[i_test]
    video_poses = dataset_info.render_poses
   
    logger = logging.getLogger(__name__)
    pprint(args)    
    # model
    engine = R2LEngine(dataset_info, logger, args)
    if args.export_onnx:
        engine.export_onnx()
        exit(0)
        
    if args.run_render:
        logger.info('Starting rendering. \n')
        # render testset
        engine.render(
            c2ws=test_poses,
            gt_imgs=test_images,
            global_step=0,
            save_rendering=True
        )
        # render videos
        if video_poses is not None:
            engine.render(
                c2ws=video_poses,
                gt_imgs=None,
                global_step=0,
                save_video=True
            )
        engine.export_onnx()
        
    if args.run_train:
        pseudo_dataloader, num_pseudo = get_pseduo_dataloader(
            args.pseudo_dir,
            args.batch_size,
            args.num_workers,
            args.camera_convention,
            dataset_info.sc
        )
        global_step = engine.buffer.start
        with tqdm(
            range(
                set_epoch_num(
                    global_step,
                    args.num_iters,
                    args.batch_size,
                    num_pseudo,
                    world_size
                )
            ),
            ascii=True,
            ncols=120,
            disable=not is_main_process()
        ) as pbar:
            for i in pbar:
                pseudo_dataloader.sampler.set_epoch(i)
                for _, (rays_o, rays_d, target_rgb) in enumerate(pseudo_dataloader):
                    global_step += 1
                    loss, psnr, best_psnr = engine.train_step(
                        rays_o=rays_o.to(device),
                        rays_d=rays_d.to(device),
                        target_rgb=target_rgb.to(device),
                        global_step=global_step
                    )
                    if global_step % args.i_video == 0 and video_poses is not None:
                        engine.render(
                            c2ws=video_poses,
                            gt_imgs=None,
                            global_step=global_step,
                            save_video=True
                        )
                    
                    if global_step % args.i_testset == 0:
                        engine.render(
                            c2ws=test_poses,
                            gt_imgs=test_images,
                            global_step=global_step,
                            save_rendering=(global_step % args.i_save_rendering == 0)
                        )  
                    pbar.set_postfix(iter=global_step, loss=loss.item(), psnr=psnr, best_psnr=best_psnr) 
                    dist.barrier() 
        engine.export_onnx()
        
        
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    with logging_redirect_tqdm():
        main()
