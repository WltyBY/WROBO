#!/usr/bin/env bash
# 使用 sh 启动时，先切换到 Bash，再执行下面的 Bash 语法。
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash -- "$0" "$@"
fi
set -euo pipefail

# 直接编辑下面的训练和评估命令；多组实验按相同顺序继续往下写。
if (($#)); then
    printf '请直接编辑脚本中的训练和评估命令；此脚本不接收命令行参数。\n' >&2
    exit 2
fi
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate wrobo
export MUJOCO_GL=egl

# 修改这里选择显卡，训练和评估命令通过 --gpu 使用此变量。
GPU_ID=0

# 实验 1：SmolVLA，四套件联合训练。
PYTHONPATH=. python wrobo/methods/SmolVLA/train.py \
    --dataset_dir \
        /media/data/wlty/RoBoDataset/wrobo/LIBERO/libero_spatial \
        /media/data/wlty/RoBoDataset/wrobo/LIBERO/libero_object \
        /media/data/wlty/RoBoDataset/wrobo/LIBERO/libero_goal \
        /media/data/wlty/RoBoDataset/wrobo/LIBERO/libero_10 \
    --log_dir ./Logs \
    --config_file ./Config/SmolVLA_LIBERO.yaml \
    --method_name SmolVLA_LIBERO_4suites \
    --fold all \
    --seed 319 \
    --num_epoch 120 \
    --warmup_epoch 4 \
    --batch_size 1 \
    --gradient_accumulation_steps 32 \
    --gpu "${GPU_ID}" \
    --merge_norm_stats

PYTHONPATH=. python wrobo/sim_envs/LIBERO/run_eval.py \
    --log-dir ./Logs/SmolVLA_LIBERO_4suites/SmolVLATrainer__BS_1__GA_32__EPOCH_120__SEED_319__PRETRAINED_True__SAMPLE_uniform/default/fold_all \
    --method SMOLVLA \
    --libero-env libero \
    --gpu "${GPU_ID}" \
    --suites libero_spatial libero_object libero_goal libero_10 \
    --num-trials-per-task 5 \
    --replan-steps 5 \
    --seed 319 \
    --save-video \
    --output-dir ./Eval/SmolVLA_LIBERO_4suites

# 实验 2、3……：在下面继续写各自的训练命令，再接 run_eval.py。
# 每组修改 --method_name、评估 --log-dir 和 --output-dir，避免混用实验产物。
# 修改训练参数后，同步更新评估的完整 --log-dir，以训练打印的输出目录为准。
