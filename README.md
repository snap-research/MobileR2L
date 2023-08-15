# Real-Time Neural Light Field on Mobile Devices


### [Project](https://snap-research.github.io/MobileR2L/) | [ArXiv](https://arxiv.org/abs/2212.08057) | [PDF](https://arxiv.org/pdf/2212.08057.pdf) 

<div align="center">
    <a><img src="figs/snap.svg"  height="120px" ></a>
   
</div>

This repository is for the real-time neural rendering introduced in the following CVPR'23 paper:
> **[Real-Time Neural Light Field on Mobile Devices](https://snap-research.github.io/R2L/)** \
> Junli Cao <sup>1</sup>, [Huan Wang](http://huanwang.tech/) <sup>2</sup>, Pavlo Chemerys<sup>1</sup>, Vladislav Shakhrai<sup>1</sup>, Ju Hu<sup>1</sup>,  [Yun Fu](http://www1.ece.neu.edu/~yunfu/) <sup>2</sup>, Denys Makoviichuk<sup>1</sup>,  [Sergey Tulyakov](http://www.stulyakov.com/) <sup>1</sup>, [Jian Ren](https://alanspike.github.io/) <sup>1</sup> 
>
> <sup>1</sup> Snap Inc. <sup>2</sup> Northeastern University 



<details>
  <summary>
  <font size="+1">Abstract</font>
  </summary>

Recent efforts in Neural Rendering Fields (NeRF) have shown impressive results on novel view synthesis by utilizing implicit neural representation to represent 3D scenes. Due to the process of volumetric rendering, the inference speed for NeRF is extremely slow, limiting the application scenarios of utilizing NeRF on resource-constrained hardware, such as mobile devices. Many works have been conducted to reduce the latency of running NeRF models. However, most of them still require high-end GPU for acceleration or extra storage memory, which is all unavailable on mobile devices. Another emerging direction utilizes the neural light field (NeLF) for speedup, as only one forward pass is performed on a ray to predict the pixel color. Nevertheless, to reach a similar rendering quality as NeRF, the network in NeLF is designed with intensive computation, which is not mobile-friendly. In this work, we propose an efficient network that runs in real-time on mobile devices for neural rendering. We follow the setting of NeLF to train our network. Unlike existing works, we introduce a novel network architecture that runs efficiently on mobile devices with low latency and small size, i.e., saving 15x ~ 24x storage compared with MobileNeRF. Our model achieves high-resolution generation while maintaining real-time inference for both synthetic and real-world scenes on mobile devices, e.g., 18.04ms (iPhone 13) for rendering one 1008x756 image of real 3D scenes. Additionally, we achieve similar image quality as NeRF and better quality than MobileNeRF (PSNR 26.15 vs. 25.91 on the real-world forward-facing dataset)

</details>


<div align="center">
<img src="figs/Lego-Tracking.gif" width="200" height="400" />
<img src="figs/blue-.gif" width="200" height="400" />
<img src="figs/shoe_1.gif" width="200" height="400" />
</div>

# Overview
This repo contains the codebases for both the teacher and student models. We use the public repo [ngp_pl](https://github.com/kwea123/ngp_pl) as the teacher for more efficient pseudo data distillation(instead of NeRF and MipNeRF as discussed in the paper).

Observed differences between `ngp` and `NeRF` teacher:
1. the training with `ngp_pl` should be less than 15 mins with 4 GPUs and pseudo data distillation for 10k images is less than 2 hours with single GPU. 
2. `ngp` renders high quality synthetic scenes than `NeRF`
3. no space contraction techniques were employed in `ngp`, thus having a inferior performance on real-world scenes

# Installation
`conda` virtual environment is recommended. The experiments were conducted on 4 Nvidia V100 GPUs. Training on one GPU should work but takes longer to converge.
## MobileR2L

```
git clone https://github.com/snap-research/MobileR2L.git

cd MobileR2L

conda create -n r2l python==3.9
conda activate r2l
conda install pip

pip install torch torchvision torchaudio
pip install -r requirements.txt 

conda deactivate
```

## NGP_PL
```
cd model/teacher/ngp_pl

# create the conda env
conda create -n ngp_pl python==3.9
conda activate ngp_pl
conda install pip

# install torch with cuda 116
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

# install tiny-cuda-nn
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# install torch scatter
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+${cu116}.html

# ---install apex---
git clone https://github.com/NVIDIA/apex
cd apex
# denpency for apex
pip install packaging

## if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key... 
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
## otherwise
# pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# ---end installing apex---


cd ../
# install other requirements
pip install -r requirements.txt

# build
pip install models/csrc/

# go to root
cd ../../../
```

# Dataset
Download the example data: `lego` and `fern`
```
sh script/download_example_data.sh
```

# Training the Teacher

```
cd model/teacher/ngp_pl

export ROOT_DIR=../../../dataset/nerf_synthetic/
python3 train.py \
     --root_dir $ROOT_DIR/lego \
     --exp_name lego\
     --num_epochs 30 --batch_size 16384 --lr 2e-2 --eval_lpips --num_gpu 4 
```
or running the bash script
```
sh benchmarking/benchmark_synthetic_nerf.sh lego
```

Once we have the teacher trained(checkpoints saved already), we can start to generate the pseudo data for MobileR2L. Depending your disk storage, the number of pseudo images could range from 2,000 to 10,000(performance varies!). Here, we set the number to 5000.

```
export ROOT_DIR=../../../dataset/nerf_synthetic/
python3 train.py \
    --root_dir $ROOT_DIR/lego \
    --exp_name Lego_Pseudo  \
    --save_pseudo_data \
    --n_pseudo_data 5000 --weight_path ckpts/nerf/lego/epoch=29_slim.ckpt \
    --save_pseudo_path Pseudo/lego --num_gpu 1
```
or running the bash script

```
sh benchmarking/distill_nerf.sh lego
```

# Training MobileR2L

```
# go to the MobileR2L directory
cd ../../../MobileR2L

conda activate r2l

# use 4 gpus for training: NeRF
sh script/benchmarking_nerf.sh 4 lego

# use 4 gpus for training: LLFF
sh script/benchmarking_llff.sh 4 orchids

```
The model will be running a day or two depending on you GPUs. When the model converges, it will automatically export the onnx files to the `Experiment/Lego_**` folder. There should be three onnx files: `Sampler.onnx`, `Embedder.onnx` and `*_SnapGELU.onnx`.

Alternatively, you can export the onnx manully by running the flowing script with `--ckpt_dir` replaced by the trained model:

```
sh script/export_onnx_nerf.sh lego path/to/ckpt

```

# Run AR lens in Snapchat
We provide the snapcodes for the AR lens in Snapchat. Scan it with Snapchat and try it out! Note: full-resolution lens need iPhone 13 or above to run smoothly in Snapchat. Try to reduce to a smaller resolution for other phones.

<div align="center">
<img src="figs/Lego.png" width="200" height="200" />
<img src="figs/Hotdog.png" width="200" height="200" />
<img src="figs/Mic.png" width="200" height="200" />
</div>
<div align="center">
<img src="figs/Hotdog-surface.png" width="200" height="200" />
<img src="figs/lego-surface.png" width="200" height="200" />
<img src="figs/mic-surface.png" width="200" height="200" />
</div>

# Acknowledgement

In this code we refer to the following implementations: [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch), [R2L](https://github.com/snap-research/R2L) and [ngp_pl](https://github.com/kwea123/ngp_pl). We also refer to some great implementation from [torch-ngp](https://github.com/ashawkey/torch-ngp/tree/main) and [MipNeRF](https://github.com/google-research/multinerf/blob/main/internal/camera_utils.py#L144). Great thanks to them! Our code is largely built upon their wonderful implementation.  We also greatly thank the anounymous CVPR'23 reviewers for the constructive comments to help us improve the paper.

# Reference

If our work or code helps you, please consider to cite our paper. Thank you!
```BibTeX
@inproceedings{cao2023real,
  title={Real-Time Neural Light Field on Mobile Devices},
  author={Cao, Junli and Wang, Huan and Chemerys, Pavlo and Shakhrai, Vladislav and Hu, Ju and Fu, Yun and Makoviichuk, Denys and Tulyakov, Sergey and Ren, Jian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8328--8337},
  year={2023}
}
```
