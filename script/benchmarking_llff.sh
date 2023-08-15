nGPU=$1
scene=$2

python3 -m torch.distributed.launch --nproc_per_node=$nGPU --use_env  main.py \
    --project_name $scene \
    --dataset_type Colmap \
    --pseudo_dir model/teacher/ngp_pl/Pseudo/$scene \
    --root_dir dataset/nerf_llff_data \
    --run_train \
    --num_workers 12 \
    --batch_size 10 \
    --num_iters 600000 \
    --input_height 63 \
    --input_width 84 \
    --output_height 756 \
    --output_width 1008 \
    --scene $scene \
    --ff \
    --ndc \
    --factor 4 \
    --amp \
    --i_testset 10000 \
    --lrate 0.0005
