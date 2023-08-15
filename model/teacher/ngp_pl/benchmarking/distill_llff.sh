#!/bin/bash

export ROOT_DIR=../../../dataset/nerf_llff_data
scene=$1

python3 train.py \
    --root_dir $ROOT_DIR/$scene \
    --exp_name Pseudo_$scene --dataset_name colmap\
    --scale 4.0 --downsample 0.25 --ff \
    --save_pseudo_data \
    --n_pseudo_data 8000 --weight_path ckpts/colmap/$scene/epoch=19_slim.ckpt \
    --save_pseudo_path Pseudo/$scene --num_gpu 1 --sr_downscale 12
