
import os 
import json
import torch
import torch.nn as nn
import torchvision
from os.path import join
import numpy as np
from tqdm import tqdm
import pytz
import lpips 
import imageio
from datetime import datetime
from easydict import EasyDict as edict
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from annotation import *
from .R2L import R2L
from lens import (
    convertModel,
    Sampler,
    Embedder
)
from data import (
    PointSampler,
    PositionalEmbedder
)
from utils import (
    is_main_process,
    undataparallel,
    to_8bit,
    mse2psnr,
    img2mse,
    mkdirs,
    cache_code,
    save_onnx,
    save_ml,
    save_tflite,
    main_process,
)
from metrics import (
    ssim_,
    FLIP
)

class R2LEngine():
    def __init__(
        self,
        dataset_info,
        logger,
        args
):
        self.args = args
        self.logger = logger
        self.dataset_info = dataset_info
        self.sampler = PointSampler(dataset_info)
        self.embedder = PositionalEmbedder(
            args.multires,
            dataset_info.device,
            True
        )
        self._setup_system()
        
        # metrics
        self.ssim = ssim_
        self.flip = FLIP()
        self.lpips = (
            lpips.LPIPS(net=self.args.lpips_net)
            .to("cpu")
        )
        
        
    def _setup_system(self):
        #todo: experiment eps
        self._prepare_experiment()
        self.engine = R2L(
            self.args,
            3*self.args.n_sample_per_ray*self.embedder.embed_dim,
            3
        ).to(self.dataset_info.device)
        #Todo: opt before or after DDP?
        if self.args.run_train:
            self.engine = nn.SyncBatchNorm.convert_sync_batchnorm(self.engine)     
            self.engine = DDP(self.engine)
            if hasattr(self.engine.module, 'input_dim'):
                self.engine.input_dim = self.engine.module.input_dim
            self.logger.info(f'[Rank {self.dataset_info.device}]: Using Distributed Data Parallel.')
            
        self.optimizer = torch.optim.AdamW(
            params=list(self.engine.parameters()),
            lr=self.args.lrate,
            betas=(0.9, 0.999),
        )
        # self.optimizer = torch.optim.Adam(
        #     params=list(self.engine.parameters()),
        #     lr=self.args.lrate,
        #     # betas=(0.9, 0.999),
        # )  
        self._register(
            {
                'loss': 0,
                'start': 1,
                'psnr': 0,
                'best_psnr': 0,
                'best_psnr_step': 0,
                'input_dim': self.engine.input_dim,
            }
        )
        if self.args.ckpt_dir:
            ckpt = torch.load(
                self.args.ckpt_dir,
                map_location={'cuda:%d' % 0: 'cuda:%d' % self.dataset_info.device} 
            )
            self._load_ckpt(ckpt)
            self.logger.info(f'[Rank {self.dataset_info.device}]: Loadded checkpoint {self.args.ckpt_dir}')
            
            if self.args.resume:
                self._register('start', ckpt.get('global_step', 0))
                self._register('best_psnr', ckpt.get('best_psnr', 0))
                self._register('best_psnr_step', ckpt.get('best_psnr_step', 0))
                self.optimizer.load_state_dict(ckpt.get('optimizer_state_dict', None))
                if self.args.amp:
                    self.scaler.load_state_dict(ckpt.get('scaler', None))
                self.logger.info(f'[Rank {self.dataset_info.device}]:Resume optimizer successfully.')
                
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.args.amp)  
        # Save intrinsics for lens
        self._save_instrinsics()
        
        
    @main_process
    def _save_instrinsics(self):
        with open(f'{self.buffer.exp_dir}/intrinsics.json', 'w') as f:
            json.dump(
                {
                    'H' : self.dataset_info.H,
                    'W' : self.dataset_info.W,
                    'focal': self.dataset_info.focal.item(),
                    'radius': self.dataset_info.radius.item(),
                    'near' : self.dataset_info.near,
                    'far': self.dataset_info.far,
                    'n_sample': self.dataset_info.n_sample_per_ray,
                    'numtires': self.args.multires
                },
                f,
                indent=2
            )
    
    
    def _register(self, key, value=None):
        if not hasattr(self, 'buffer'):
            self.buffer = edict(dict())
        if isinstance(key, dict):
            self.buffer.update(key)
        elif isinstance(key, str):
            self.buffer.update({key: value})
        else:
            raise ValueError
    
    
    def _load_ckpt(self, ckpt):
        model_dataparallel = False
        for name, module in self.engine.named_modules():
            if name.startswith('module.'):
                model_dataparallel = True
                break

        state_dict = ckpt['network_fn_state_dict']
        weights_dataparallel = False
        for k, v in state_dict.items():
            if k.startswith('module.'):
                weights_dataparallel = True
                break
        if model_dataparallel and weights_dataparallel or (
                not model_dataparallel) and (not weights_dataparallel):
            self.engine.load_state_dict(state_dict)
        else:
            raise NotImplementedError
    
    
    @main_process
    def _save_ckpt(self, file_name, best_psnr, best_psnr_step, global_step):
        path = join(f'{self.buffer.exp_dir}/weights', file_name)
        to_save = {
            'global_step': global_step,
            'best_psnr': best_psnr,
            'best_psnr_step': best_psnr_step,
            'network_fn_state_dict': undataparallel(self.engine.state_dict()),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict() if self.args.amp else None
        }
        to_save['network_fn'] = undataparallel(self.engine)
        torch.save(to_save, path)
        self.logger.info(f'[Rank {self.dataset_info.device}]: Save checkpoint to {path}.')
        
        
    @main_process
    def _prepare_experiment(self):
        """Create the path for experiments and tensorboad logs
        """
        tz = pytz.timezone('US/pacific')
        today = datetime.now(tz).strftime("%Y-%m-%d@%H:%M:%S")
        debug = '_Debug' if self.args.debug else ''
        render = '_Render' if self.args.run_render else ''
        train = '_Train' if self.args.run_train else ''
        onnx = '_Export_Onnx' if self.args.export_onnx else ''
        exp_id = f'{self.args.project_name}{debug}{train}{render}{onnx}-[{today}]'
        exp_dir = f'./logs/Experiments/{exp_id}'
        mkdirs(
            join(exp_dir, 'weights'),
            join(exp_dir, 'config'),
            join(exp_dir, 'gen_images'),
            join(exp_dir, 'caches')
        )
        if not self.args.no_cache:
            cache_code('./',   join(exp_dir, 'caches'))
            self.logger.info(f"Backup the code to caches.")
        
        with open(join(exp_dir, 'config/config.json'), 'w') as f:
            json.dump(self.args, f, indent=2) 
        if self.args.run_train:
            os.makedirs(f'./logs/tb_logs/{exp_id}')
        
        self._register(
            {
                'exp_id': exp_id,
                'exp_dir': exp_dir,
                'weight_dir': join(exp_dir, 'weights'),
                'gen_images_dir':  join(exp_dir, 'gen_images'),
                'tb_dir': f'./logs/tb_logs/{exp_id}'
            }
        )
        self.writer = SummaryWriter(
            f"{self.buffer.tb_dir}/{self.args.project_name}/{self.buffer.exp_id}"
        )
        
        
    def train_step(
        self,
        rays_o : Float[Tensor, 'N 3'],
        rays_d : Float[Tensor, 'N 3'],
        target_rgb : Float[Tensor, 'N 3'],
        global_step : int,
        perturb : bool = True
    ):
        self._scheduel_lr(global_step)
        pts = self.sampler.sample(rays_o, rays_d, perturb=perturb)
        pts = (
            self.embedder(pts)
                .view((-1, self.args.input_height, self.args.input_width, self.buffer.input_dim))
        )
        pts = torch.permute(pts, (0, 3, 1, 2))
        with torch.cuda.amp.autocast(enabled=self.args.amp):
            rgb = self.engine(pts)
            dim = rgb.shape[1]
            rgb = (
                torch.permute(rgb, (0, 2, 3, 1))
                .reshape((-1, dim))
            )
            self._register(
                'loss',
                img2mse(rgb, target_rgb.view(-1, 3))
            )
            self._register(
                'psnr',
                mse2psnr(self.buffer.loss).item()
            )
        self.scaler.scale(self.buffer.loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        
        if is_main_process():
            self.writer.add_scalar("Loss/train_psnr", self.buffer.psnr, global_step)
            self.writer.add_scalar("Loss/train_mse", self.buffer.loss.item(), global_step)
            self.writer.add_scalar("LR", self.buffer.lr, global_step)

            if global_step % self.args.train_image_log_step == 0:
                rgb_plot =  (
                    rgb.reshape(-1, self.args.output_height, self.args.output_width, 3)[:4]
                    .permute(0, 3, 1, 2)
                    .detach()
                    .cpu()
                )
                target_rgb_plot = (
                    target_rgb.reshape(-1, self.args.output_height, self.args.output_width, 3)[:4]
                    .permute(0, 3, 1, 2)
                    .detach()
                    .cpu()
                )
                self.writer.add_image(
                    "images/train", torchvision.utils.make_grid(rgb_plot, nrow=2), global_step
                )
                self.writer.add_image(
                    "images/train_gt",
                    torchvision.utils.make_grid(target_rgb_plot, nrow=2),
                    global_step,
                )
            if global_step % self.args.i_weights == 0:
                self._save_ckpt(
                    'ckpt.tar',
                    self.buffer.best_psnr,
                    self.buffer.best_psnr_step,
                    global_step
                )
        return (
                    self.buffer.loss,
                    self.buffer.psnr,
                    self.buffer.best_psnr
                )
        
        
    @main_process
    @torch.no_grad()
    def render(
        self,
        c2ws : Float[Tensor, "N 3 4"],
        gt_imgs : Optional[Float[Tensor, "N H W 3"]] = None,
        global_step : Optional[int] = 0,
        save_rendering : bool = False,
        save_video : bool = False
        # render_poses : Float[Tensor, "N 3 4"] = None
    ):
        rgbs = []
        psnr = []
        ssim = []
        engine = self.engine.module if hasattr(self.engine, 'module') else self.engine
        engine.eval()
        self.logger.info(f"Rank [{self.dataset_info.device}]: Start rendering.\n")
        for i, c2w in tqdm(enumerate(c2ws)):
            pts = self.sampler.sample(c2w=c2w[:3, :4], perturb=False)
            pts = (
                self.embedder(pts)
                .transpose(1, 0)
                .view(1, self.buffer.input_dim, self.args.input_height, self.args.input_width)
            )
            rgb = (
                engine(pts)
                .squeeze()
                .permute(1, 2, 0)
            )
            
            rgbs += [rgb]
            if gt_imgs is not None:
                psnr += [mse2psnr(img2mse(rgb, gt_imgs[i]))]
                ssim += [self.ssim(rgb, gt_imgs[i])]      
                if save_rendering:
                    fn = join(self.buffer.exp_dir, 'gen_images', '{:03d}.png'.format(i))
                    imageio.imwrite(fn, to_8bit(rgbs[-1]))
                    imageio.imwrite(
                        fn.replace('.png', '_gt.png'),
                        to_8bit(gt_imgs[i])
                    ) 
                    
        rgbs = torch.stack(rgbs, dim=0)
        if save_video:
            video_path = f'{self.buffer.gen_images_dir}/video_{self.buffer.exp_id}_iter{global_step}.mp4'
            imageio.mimwrite(
                video_path,
                to_8bit(rgbs),
                fps=30,
                quality=8
            )
            self.logger.info(f'[VIDEO] Rendering done. Save video: "{video_path}"')
        if gt_imgs is not None:
            self._get_metrics(rgbs, gt_imgs, psnr, ssim)
            if self.args.run_train:
                # update tensorboard
                self.writer.add_image(
                    "images/testset_0", rgbs[0], global_step, dataformats="HWC"
                )
                self.writer.add_image(
                    "images/testset_2", rgbs[2], global_step, dataformats="HWC"
                )
                self.writer.add_image(
                    "images/testset_0_gt", gt_imgs[0], global_step, dataformats="HWC"
                )
                self.writer.add_image(
                    "images/testset_0_gt", gt_imgs[2], global_step, dataformats="HWC"
                )
                if not self.args.ff: # add more images
                    self.writer.add_image(
                        "images/testset_4", rgbs[4], global_step, dataformats="HWC"
                    )
                    self.writer.add_image(
                        "images/testset_9", rgbs[9], global_step, dataformats="HWC"
                    )
                    self.writer.add_image(
                        "images/testset_14", rgbs[14], global_step, dataformats="HWC"
                    )

                    self.writer.add_image(
                        "images/testset_4_gt", gt_imgs[4], global_step, dataformats="HWC"
                    )
                    self.writer.add_image(
                        "images/testset_9_gt", gt_imgs[9], global_step, dataformats="HWC"
                    )
                    self.writer.add_image(
                        "images/testset_14_gt", gt_imgs[14], global_step, dataformats="HWC"
                    )
                    
                if self.buffer.test_psnr > self.buffer.best_psnr:
                    self._register('best_psnr', self.buffer.test_psnr)
                    self._register('best_psnr_step', global_step)
                    self._save_ckpt(
                        f'best_ckpt.tar',
                        self.buffer.best_psnr,
                        self.buffer.best_psnr_step,
                        global_step
                    )
                    with open(os.path.join(self.buffer.weight_dir, "psnr.json"), 'a') as fn:
                        psnr = {
                            "best_psnr": self.buffer.best_psnr,
                            "best_psnr_step":  self.buffer.best_psnr_step,
                            "train_psnr": self.buffer.psnr
                        }
                        json.dump(psnr, fn, indent=2)
                        
            self.logger.info(
                    f"[TEST] Iter {global_step} TestPSNR {self.buffer.test_psnr:.4f} \
                    BestPSNRv2 {self.buffer.best_psnr:.4f} (Iter {self.buffer.best_psnr_step}) \
                    TestSSIM {self.buffer.test_ssim:.4f} TestLPIPS {self.buffer.test_lpips:.4f} \
                    TestFLIP {self.buffer.test_flip:.4f} TrainHistPSNR {self.buffer.psnr:.4f}"
            )
        engine.train()
        torch.cuda.empty_cache()


    def _get_metrics(self, rgbs, gt_imgs, psnrs, ssims):
        # https://github.com/richzhang/PerceptualSimilarity
        # LPIPS demands input shape [N, 3, H, W] and in range [-1, 1]
        rec = rgbs.permute(0, 3, 1, 2).cpu() # [N, 3, H, W]
        ref = gt_imgs.permute(0, 3, 1, 2).cpu() # [N, 3, H, W]
        rescale = lambda x, ymin, ymax: (ymax - ymin) / (x.max() - x.min()) * (x - x.min()) + ymin
        lpipses = self.lpips(rescale(rec, -1, 1), rescale(ref, -1, 1)) 

        # -- get FLIP loss
        # flip standard values
        monitor_distance = 0.7
        monitor_width = 0.7
        monitor_resolution_x = 3840
        pixels_per_degree = monitor_distance * (monitor_resolution_x / monitor_width) * (np.pi / 180)
        flips = self.flip.compute_flip(rec, ref, pixels_per_degree) # shape [N, 1, H, W]
        # --
        psnrs = torch.stack(psnrs, dim=0)
        ssims = torch.stack(ssims, dim=0)
        test_loss = img2mse(rgbs, gt_imgs)

        self._register(
            {
                'test_loss': test_loss.item(),
                'test_psnr': psnrs.mean().item(),
                'test_ssim': ssims.mean().item(),
                'test_lpips': lpipses.mean().item(),
                'test_flip': flips.mean().item()
            }
        )

    @main_process
    def export_onnx(self):
        if self.args.ckpt_dir:
            onnx_path = self.args.ckpt_dir.replace('.tar', '.onnx')         
        else:
            onnx_path = f'{self.buffer.weight_dir}/{self.args.project_name}.onnx'
            
        dummy_input = torch.randn(
            1,
            self.buffer.input_dim, 
            self.args.input_height, 
            self.args.input_width,
        ).to(self.dataset_info.device)
        ml_path = onnx_path.replace('.onnx', '.mlpackage')
        save_ml(self.engine, ml_path, dummy_input)
        
        if self.args.activation_fn == 'gelu':
            # Export to relu first then convert to SnapGelu that 
            # is compatiable with SnapML
            self.args.activation_fn = 'relu'
            self.engine = R2L(
                self.args,
                3*self.args.n_sample_per_ray*self.embedder.embed_dim,
                3
            ).to(self.dataset_info.device)
            ckpt = torch.load(
                self.args.ckpt_dir if self.args.ckpt_dir else \
                        join(f'{self.buffer.exp_dir}/weights/ckpt.tar'),
                map_location={'cuda:%d' % 0: 'cuda:%d' % self.dataset_info.device} 
            )
            self._load_ckpt(ckpt)
            self.logger.info(f'[Rank {self.dataset_info.device}]: Re-Loadded checkpoint {self.args.ckpt_dir}')
        
        save_onnx(self.engine, onnx_path, dummy_input)
        
        if self.args.convert_snap_gelu:
            convertModel(
                onnx_path,
                onnx_path.replace(
                    '.onnx',
                    '_SnapGELU.onnx'
                )
            )
            
        # Optional: exporting tflite
        #pip install git+https://github.com/alibaba/TinyNeuralNetwork.git

        # tf_path = onnx_path.replace('.onnx', '.tflite')
        # save_tflite(self.engine, tf_path, dummy_input)

        # export Sampler and Embedder
        if self.args.ff:
            self.logger.info(
                f'Convert to onnx done. Saved at "{onnx_path}". Sampler and Embedder are not exported for ff scenes.'
            )
            return
        
        H = self.dataset_info.H
        W = self.dataset_info.W
        n_sample = self.dataset_info.n_sample_per_ray
        multires = self.args.multires
        near = self.dataset_info.near
        far = self.dataset_info.far
        focal = self.dataset_info.focal.item()
        
        # embedder
        weights = (
            2 ** torch.linspace(0, multires-1, steps=multires)
            .expand(1, H * W, n_sample * 3, multires).permute((0, 3, 1, 2)) # [1, M, H*W, n_sample * 3]
        )
        emb = Embedder(weights, multires)
        args = (torch.randn(1, 1, H * W, n_sample * 3))
        torch.onnx.export(
            emb.cpu(),
            args,
            '/'.join(onnx_path.split('/')[:-1] + ['Embedder.onnx']),
            export_params=True,
            do_constant_folding=True,
            keep_initializers_as_inputs=False,
            opset_version=11,
            verbose=True,
            input_names = ['rays'],   # the model's input names
            output_names = ['embbrays'], # the model's output names
            dynamic_axes={
                            'rays' : {0: 'batch_size'},
                            'embbrays' : {0: 'batch_size'}
                        }
        )
        del emb
        
        #sampler
        shift = 0.5
        i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1,H))
        i, j = i.t(), j.t()
        dirs = torch.stack([(i - W*.5 + shift)/focal, -(j - H*.5 + shift)/focal, -torch.ones_like(i)], dim=-1) # [H, W, 3]
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
            '/'.join(onnx_path.split('/')[:-1] + ['Sampler.onnx']), 
            export_params=True,
            do_constant_folding=True,
            keep_initializers_as_inputs=False,
            opset_version=11,
            verbose=True,
            input_names = ['c2w33', "c2w13"],   
            output_names = ['pts'], 
            dynamic_axes={
                            'c2w33' : {0: 'batch_size'},   
                            'c2w13' : {0: 'batch_size'},
                            'pts' : {0: 'batch_size'}
                        }
        )
        del sampler
        
        self.logger.info(f'Convert to onnx done. Saved at "{onnx_path}"') 
        
        
    def _scheduel_lr(self, global_step):
        decay_rate = 0.1
        decay_steps = self.args.lrate_decay * 1000
        if self.args.warmup_lr: # @mst: example '0.0001,2000'
            start_lr, end_iter = [float(x) for x in self.args.warmup_lr.split(',')]
            if global_step < end_iter: # increase lr until args.lrate
                new_lrate = (self.args.lrate - start_lr) / end_iter * global_step + start_lr
            else: # decrease lr as before
                new_lrate = self.args.lrate * (decay_rate ** ((global_step - end_iter) / decay_steps))
        else:
            new_lrate = self.args.lrate * (decay_rate ** (global_step / decay_steps))
        new_lrate *= self.args.lr_scale
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lrate
        self._register('lr', new_lrate)
        


