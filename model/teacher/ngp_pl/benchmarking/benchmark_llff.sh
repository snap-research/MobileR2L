#!/bin/bash

export ROOT_DIR=../../../dataset/nerf_llff_data
scene=$1

python3 train.py \
    --root_dir $ROOT_DIR/$scene \
    --exp_name $scene  --dataset_name colmap\
    --num_epochs 20 --scale 4.0 --downsample 0.25  --lr 2e-2 --ff
