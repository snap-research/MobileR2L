#!/bin/bash

export ROOT_DIR=../../../dataset/nerf_synthetic/
scene=$1

python3 train.py \
    --root_dir $ROOT_DIR/$scene \
    --exp_name $scene  \
    --num_epochs 30 --batch_size 16384 --lr 2e-2 --eval_lpips

