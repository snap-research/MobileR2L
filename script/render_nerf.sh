scene=$1
ckpt=$2

python3 -m torch.distributed.launch --nproc_per_node=1 --use_env main.py \
    --project_name $scene \
    --dataset_type Blender \
    --pseudo_dir model/teacher/ngp_pl/Pseudo/$scene\
    --root_dir dataset/nerf_synthetic \
    --run_render \
    --input_height 100 \
    --input_width 100 \
    --output_height 800 \
    --output_width 800 \
    --scene $scene \
    --amp \
    --ckpt_dir $ckpt \
    --testskip 1


