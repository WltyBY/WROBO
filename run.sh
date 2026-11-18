PYTHONPATH=. python wrobo/methods/PI0/train.py \
    --dataset_dir \
        ./LIBERO/libero_spatial \
        ./LIBERO/libero_object \
        ./LIBERO/libero_goal \
        ./LIBERO/libero_10 \
    --log_dir ./Logs \
    --config_file ./Config/PI0_LIBERO_VLAP.yaml \
    --method_name PI0_LIBERO_VLAP \
    --gpu 0 \
    --fold all \
    --num_epoch 120 \
    --warmup_epoch 4 \
    --batch_size 16 \
    --gradient_accumulation_steps 2 \
    --merge_norm_stats \
    --compile


####  DDP  ####
PYTHONPATH=. torchrun --nproc_per_node=2 wrobo/methods/PI0/train.py \
    --dataset_dir \
        ./LIBERO/libero_spatial \
        ./LIBERO/libero_object \
        ./LIBERO/libero_goal \
        ./LIBERO/libero_10 \
    --log_dir ./Logs \
    --config_file ./Config/PI0_LIBERO_VLAP.yaml \
    --method_name PI0_LIBERO_VLAP \
    --gpu 0,1 \
    --fold all \
    --num_epoch 120 \
    --warmup_epoch 4 \
    --batch_size 32 \
    --gradient_accumulation_steps 1 \
    --merge_norm_stats \
    --compile