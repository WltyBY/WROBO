PYTHONPATH=. python wrobo/methods/ACT/train.py \
    --dataset_dir ./Dataset/ACT_wrobo/sim_transfer_stack_cube \
    --log_dir ./Logs \
    --config_file ./Config/ACT.yaml \
    --method_name transfer_stack_top_angle_w_PE \
    -f 0 \
    --num_epoch 300 \
    --batch_size 8
PYTHONPATH=. python wrobo/envs/Aloha/eval.py \
    --log_dir ./Logs/transfer_stack_top_angle_w_PE/BS_8_EPOCH_300_SEED_319_PRETRAINED_False/w_KL_10.0/fold_0 \
    --config_file ACT.yaml \
    --task_name sim_transfer_stack_cube \
    --num_rollouts 50 \
    --gpu 0 \
    --temporal_agg
PYTHONPATH=. python wrobo/envs/Aloha/eval.py \
    --log_dir ./Logs/transfer_stack_top_angle_w_PE/BS_8_EPOCH_300_SEED_319_PRETRAINED_False/w_KL_10.0/fold_0 \
    --config_file ACT.yaml \
    --task_name sim_transfer_stack_cube \
    --num_rollouts 50 \
    --gpu 0
PYTHONPATH=. python wrobo/envs/Aloha/eval.py \
    --log_dir ./Logs/transfer_stack_top_angle_w_PE/BS_8_EPOCH_300_SEED_319_PRETRAINED_False/w_KL_10.0/fold_0 \
    --config_file ACT.yaml \
    --task_name sim_transfer_stack_cube \
    --num_rollouts 50 \
    --gpu 0 \
    --temporal_agg \
    --ckpt_name checkpoint_final.pth
PYTHONPATH=. python wrobo/envs/Aloha/eval.py \
    --log_dir ./Logs/transfer_stack_top_angle_w_PE/BS_8_EPOCH_300_SEED_319_PRETRAINED_False/w_KL_10.0/fold_0 \
    --config_file ACT.yaml \
    --task_name sim_transfer_stack_cube \
    --num_rollouts 50 \
    --gpu 0 \
    --ckpt_name checkpoint_final.pth


####  DDP  ####
# PYTHONPATH=. torchrun --nproc_per_node=2 wrobo/methods/ACT/train.py \
#     --dataset_dir ./Dataset/ACT_wrobo/sim_insertion \
#     --log_dir ./Logs \
#     --config_file ./Config/ACT.yaml \
#     --method_name insertion_top_angle_rl_wrist \
#     -f 0 \
#     --num_epoch 300 \
#     --batch_size 64 \
#     --gpu 0,1 \
#     --do_compile