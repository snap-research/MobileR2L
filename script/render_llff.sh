scene=$1
ckpt=$2

python3 -m torch.distributed.launch --nproc_per_node=1 --use_env  main.py \
    --project_name $scene \
    --dataset_type Colmap \
    --pseudo_dir model/teacher/ngp_pl/Pseudo/$scene \
    --root_dir dataset/nerf_llff_data \
    --run_render \
    --input_height 63 \
    --input_width 84 \
    --output_height 756 \
    --output_width 1008 \
    --scene $scene \
    --ff \
    --ndc \
    --amp \
    --factor 4 \
    --ckpt_dir $ckpt

