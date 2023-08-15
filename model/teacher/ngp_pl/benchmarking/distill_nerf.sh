#!/bin/bash

export ROOT_DIR=../../../dataset/nerf_synthetic/
scene=$1

python3 train.py \
    --root_dir $ROOT_DIR/$scene \
    --exp_name Pseudo_$scene  \
    --save_pseudo_data \
    --n_pseudo_data 10000 --weight_path ckpts/nerf/$scene/epoch=29_slim.ckpt  \
    --save_pseudo_path Pseudo/$scene --num_gpu 1

