# PYTHONPATH=. python wrobo/envs/Aloha/record_episodes.py \
#     --dataset_dir ./Dataset/ACT_wrobo/sim_transfer_cube \
#     --task_name sim_transfer_cube \
#     --max_timesteps 400 \
#     --camera_names image_angle image_top image_left_wrist image_right_wrist\
#     --num_episodes 50 \
#     --skip_failure


# PYTHONPATH=. python wrobo/envs/Aloha/record_episodes.py \
#     --dataset_dir ./Dataset/ACT_wrobo/sim_insertion \
#     --task_name sim_insertion \
#     --max_timesteps 400 \
#     --camera_names image_angle image_top image_left_wrist image_right_wrist\
#     --num_episodes 50 \
#     --skip_failure


PYTHONPATH=. python wrobo/envs/Aloha/record_episodes.py \
    --dataset_dir ./Dataset/ACT_wrobo/sim_transfer_stack_cube \
    --task_name sim_transfer_stack_cube \
    --max_timesteps 2400 \
    --camera_names image_angle image_top image_left_wrist image_right_wrist\
    --num_episodes 50 \
    --skip_failure 