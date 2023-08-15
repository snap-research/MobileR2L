nGPU=$1
scene=$2

python3 -m torch.distributed.launch --nproc_per_node=$nGPU --use_env main.py \
    --project_name $scene \
    --dataset_type Blender \
    --pseudo_dir model/teacher/ngp_pl/Pseudo/$scene  \
    --root_dir dataset/nerf_synthetic \
    --run_train \
    --num_workers 12 \
    --batch_size 10 \
    --num_iters 600000 \
    --input_height 100 \
    --input_width 100 \
    --output_height 800 \
    --output_width 800 \
    --scene $scene \
    --i_testset 10000 \
    --amp \
    --lrate 0.0005 
