PYTHONPATH=. python wrobo/envs/Aloha/record_episodes.py \
    --dataset_dir ./Dataset/ACT_wrobo/sim_transfer_cube \
    --task_name sim_transfer_cube \
    --episode_len 400 \
    --camera_names image_angle image_top image_left_wrist image_right_wrist\
    --num_episodes 50 \
    --skip_failure


PYTHONPATH=. python wrobo/envs/Aloha/record_episodes.py \
    --dataset_dir ./Dataset/ACT_wrobo/sim_insertion \
    --task_name sim_insertion \
    --episode_len 400 \
    --camera_names image_angle image_top image_left_wrist image_right_wrist\
    --num_episodes 50 \
    --skip_failure