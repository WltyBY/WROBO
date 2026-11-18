# PI0 in wrobo

Continuous-action **pi0**, not pi0_fast or pi05. The model core is vendored from
this workspace's LeRobot source. It does not import the LeRobot runtime.
Upstream Apache-2.0 notices are retained in the model files.

## Training responsibilities

All methods enter through their own `train.py`: parse shared running arguments
and method-specific options, construct the Trainer, and call `run_training()`.
`ABCTrainer` owns the epoch loop, validation, logging, EMA-based best selection,
checkpoint lifecycle, dataset splits, normalization statistics, gradient accumulation
and optimizer/scaler updates.
`DDPABCTrainer` supplies device/process setup, reduction, barriers and wrapping.

| Method | Trainer | Model |
| --- | --- | --- |
| ACT | Dataset/transforms, optimizer, padding-aware L1 plus KL loss | Image/state encoding, variational latent, transformer action-chunk prediction |
| SmolVLA | Tokenization, image conversion, flow loss, LoRA and checkpoint integration | Official SmolVLM/action-expert flow matching, preprocessing and Euler sampling |
| wVLA | Assemble configured encoders/expert, tokenization and corresponding regression/flow loss | Modular vision/language encoders, fusion and action expert |
| PI0 | Same wrobo lifecycle; PaliGemma tokenization, masked flow loss, selectable LoRA parts | Official PaliGemma + Gemma expert, joint attention, state/action projections and Euler sampling |

PI0 files:

- `train.py`: CLI. Shares `--instruction_mode` (one per dataset), GPU/DDP, resume,
  dataset and logging arguments with the other methods.
- `PI0Trainer.py`: datasets/collation, AdamW, epoch-based cosine warmup/decay,
  training/validation, full/adapter/merged checkpoints, evaluator hooks.
- `models/policy.py`: translates wrobo inputs/outputs; official safetensors
  initialization; tokenizer and preprocessing.
- `models/modeling_pi0.py`, `pi_gemma.py`: official model computation and names.
- `models/checkpoint.py`: strict official key/shape checking and shard loading.
- `models/configuration.py`: fixed pi0_base architecture plus tuning controls.
- `Config/PI0_SO101.yaml`: SO101 data/model/training settings.

## LIBERO 四套件实验

项目根目录的 [train_pi0.sh](../../../train_pi0.sh) 和
[eval_pi0.sh](../../../eval_pi0.sh) 分别列出训练与评估命令，不使用函数包装训练入口。
训练脚本不会启动评估；两阶段分别运行：

```bash
bash ./train_pi0.sh
bash ./eval_pi0.sh
```

训练和评估脚本各自在顶部设置 `GPU_ID=0`，每条 Python 命令都通过
`--gpu "${GPU_ID}"` 引用该变量。换到物理 GPU 1 时，把要运行脚本中的
`GPU_ID` 改成 `1`；两个独立脚本各自选择设备。shell 不设置
`export CUDA_VISIBLE_DEVICES=0`。
评估入口只加载一份策略：`--gpu 1` 在模型加载和随机种子初始化前设置可见设备，
模型内部使用 `cuda:0`；它不表示多 GPU/DDP 推理，不能传逗号列表或 `all`。
`--gpu -1` 让策略在 CPU 上运行，并保留仿真渲染所需的 GPU 环境。
评估也可显式使用 `--device cpu`、`--device mps` 或 `--device cuda:N`；此时沿用
已有可见设备环境，CUDA 编号为其中的逻辑编号。`--device` 与 `--gpu` 不能同时指定。
更多设备说明见 [LIBERO 评估文档](../../sim_envs/LIBERO/README.md)。

脚本包含 10 组实验：普通 VLAP、普通 AP，以及 VLAP/AP 各自的 LoRA
rank/alpha 16、32、64、128。各组使用独立的 `--method_name` 和评估输出目录。
只运行其中一组时，保留对应的直接命令；训练参数变化后，同步修改评估的精确
`--log-dir`，以训练日志打印的实际输出目录为准。

| 配置文件 | 普通微调/LoRA | 训练范围 |
| --- | --- | --- |
| [PI0_LIBERO_VLAP.yaml](../../../Config/PI0_LIBERO_VLAP.yaml) | 普通微调 | 全部动作相关的 Vision、Language、Action、Projection 参数 |
| [PI0_LIBERO_AP.yaml](../../../Config/PI0_LIBERO_AP.yaml) | 普通微调 | Action、Projection；冻结 Vision 和 Language |
| [PI0_LIBERO_VLAP_LoRA.yaml](../../../Config/PI0_LIBERO_VLAP_LoRA.yaml) | LoRA | `--lora_parts all` 选中的四类 adapter |
| [PI0_LIBERO_AP_LoRA.yaml](../../../Config/PI0_LIBERO_AP_LoRA.yaml) | LoRA | 默认 Action、Projection adapter |

四份配置完整加载 `lerobot/pi0_base`，采用 OpenPI 的选择性 BF16 参数存储、gradient checkpointing、两路
`image_top/image_wrist`、8 维 `proprio_ee` 和 7 维 `action_rel`，不使用 SO101 的
`action_abs`。256×256 仿真图像由 PI0 内部缩放到 224×224。状态和动作继续使用
训练数据的 standard 归一化统计，评估后逆归一化为 LIBERO 原生 OSC_POSE 命令。

训练均为 micro-batch 1、梯度累计 32、有效 batch 32，120 epoch，每轮 250 次更新，
合计 30,000 次更新，预热 4 epoch。根据普通微调/LoRA 和适配范围分别设置学习率：

| 实验范围 | 峰值 `base_lr` | 最低 `min_lr` | 设置依据 |
| --- | ---: | ---: | --- |
| 普通 VLAP | `2.5e-5` | `2.5e-6` | PI0 官方基础配置 |
| 普通 AP | `1e-4` | `1e-5` | 只训练动作侧，使用较高学习率的首轮实验起点 |
| LoRA VLAP | `1e-4` | `1e-5` | 提高 adapter 学习率，同时保守调整视觉/语言部分 |
| LoRA AP | `3e-4` | `3e-5` | 动作侧 adapter 的首轮实验起点 |

后三项是工程上的初始实验设置，尚未通过 PI0 LIBERO 训练/评估证明最优，也不是
官方分别给出的四套配方。[LeRobot PEFT 文档](https://huggingface.co/docs/lerobot/peft_training)
给出 LoRA 学习率通常可以高于普通微调的通用建议；具体数值仍需用当前任务结果验证。
同一范围的 r16/r32/r64/r128 保持相同学习率，并保持 `alpha/r=1`，便于比较 rank。
四份 YAML 均保持 `weight_decay=0.01`、`grad_clip_norm=1.0`。
`--fold all` 使用训练目录的全部选中轨迹，验证 loss 不代表独立留出集性能。

PI0、PI05 的配置类及现有 YAML 默认 `dtype: bfloat16`，参数精度按
[OpenPI 的 PyTorch 实现](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models_pytorch/gemma_pytorch.py)
选择：视觉 patch embedding 的 weight/bias、position embedding 的 weight，以及
Gemma 的 `input_layernorm`、`post_attention_layernorm`、`model.norm` 保留 FP32；
core 中其余参数为 BF16，包括视觉 encoder、视觉 projector 和视觉 LayerNorm。
core 外的状态、动作和时间投影仍为 FP32，与
[OpenPI 顶层模型](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models_pytorch/pi0_pytorch.py)
一致。视觉 embedding 输出在进入 encoder 前转换为 encoder 的 dtype；前向不改变参数 dtype。
显式设置 `dtype: float32` 仍可使用全 FP32 基座存储。

梯度与当前 AdamW 的一阶、二阶动量跟随各自参数的 dtype，无额外 FP32 master 权重副本。
LoRA 保留 PEFT 默认行为：**所有可训练 adapter 都存为 FP32**，包括附着在 BF16
线性层上的 adapter；冻结的基座仍按上述规则存储。这是 wrobo 的 PEFT 精度选择，
不是 OpenPI PyTorch 的 LoRA 配方。

| 参数类别 | 参数存储 | `.grad` / AdamW `exp_avg`、`exp_avg_sq` |
| --- | --- | --- |
| 普通微调中 BF16 的可训练基座参数 | BF16 | BF16 |
| 普通微调中保留 FP32 的可训练基座参数 | FP32 | FP32 |
| LoRA 可训练 adapter | FP32 | FP32 |
| 冻结基座参数 | 按所在模块为 BF16 或 FP32 | 不创建梯度和 AdamW 动量 |

AdamW 的 `step` 是独立计步标量，默认 FP32。权重和动量长期保留；梯度在下一组
更新开始时由 `zero_grad(set_to_none=True)` 清空，峰值显存仍需计入它。

**参数存储精度与算子计算精度独立**：公共 CUDA Trainer 在支持 BF16 的设备上使用
BF16 autocast，否则使用 FP16 autocast 和 GradScaler；CPU 不启用 autocast。
loss、norm 等运算仍按算子策略或显式转换使用所需精度。显式 `dtype: float32` 不会
关闭 AMP，autocast 也不会永久改写参数和动量 dtype。
`backward()` 和 optimizer 更新在前向 autocast 上下文之外执行，反向中间算子的
精度跟随对应前向计算；`backward_pass_autocast="off"` 不会强制整个反向为 FP32。
参见 [PyTorch autocast 用法](https://docs.pytorch.org/docs/2.11/amp.html#autocasting)。

基座 metadata 的 dtype 不能用来推断实际训练配方：

| 实现或来源 | 配置与实际训练的区别 |
| --- | --- |
| LeRobot 的 PI0、PI05 配置类及基座 metadata | 默认参数 dtype 为 FP32 |
| LeRobot [PI0](https://huggingface.co/docs/lerobot/pi0) / [PI05](https://huggingface.co/docs/lerobot/pi05) 训练示例 | 命令显式覆盖为 `--policy.dtype=bfloat16` |
| Physical Intelligence [OpenPI 训练配置](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py)及 [PyTorch 训练入口](https://github.com/Physical-Intelligence/openpi/blob/main/scripts/train_pytorch.py) | `pytorch_training_precision` 默认 BF16；入口用它覆盖模型 dtype，并按实现保留部分参数为 FP32 |
| 当前 wrobo PI0/PI05 示例 | 默认采用 OpenPI 的选择性 BF16 基座存储；公共 Trainer 独立开启 autocast，LoRA adapter 保留 FP32 |

这里对齐的是 OpenPI 的参数精度规则；数据处理、学习率调度和 LoRA 目标仍按 wrobo
实现，不意味着复现 OpenPI 的完整训练配方。

| 默认选择性 BF16 显存预算 | 普通 VLAP | 普通 AP |
| --- | ---: | ---: |
| 全部模型权重 | 6.039 GiB | 6.039 GiB |
| 可训练参数的梯度 | 5.836 GiB | 0.592 GiB |
| AdamW 两份动量 | 11.673 GiB | 1.185 GiB |
| 合计，不含激活与临时空间 | **23.548 GiB** | **7.816 GiB** |

该表按完整架构的参数数量及实际 dtype 计算，不包含激活、CUDA runtime、编译、
优化器临时空间和分配器开销。普通 VLAP 已占约 23.548 GiB，24GB 3090 无法给
实际训练留下足够空间；脚本保留完整实验命令，应在显存足够的设备执行。
micro-batch 和梯度累计减少激活需求，不能减少模型参数和优化器状态。
LoRA 只为 FP32 adapter 保存梯度和动量，冻结基座仍占约 6.039 GiB；
各范围均未进行 GPU 峰值显存实测。若显式改回 `dtype: float32`，普通 VLAP/AP
上述四项合计分别约 47.032/15.580 GiB，与当前默认预算不同。

评估脚本对每组加载默认 `checkpoint_final.pth`，四个套件各 10 个任务、每任务
50 次，共 2,000 次，使用 4 个仿真进程并保存双视角拼接视频。模型在 `wrobo`
环境加载，仿真由入口自动在 `libero` 环境运行。种子 319、每次执行 5 步等默认值
不在 shell 中重复填写；评估没有开启额外 AMP 或推理编译。

LoRA parts 按 `language-vision-action-projection` 顺序进入目录。PI0 CLI 的 alpha
类型为 float：省略默认 alpha 时目录为 `alpha16`，显式传入 `--lora_alpha 32`
时目录为 `alpha32.0`，64/128 同理。评估脚本已经按实际参数格式填写，不要只按
显示的整数 rank 猜测目录。

## PI0_SO101.yaml 参数说明

以下说明对应当前实现，而不只是上游参数名称。`Policy` 控制模型构建和
输入输出；`Training_settings` 控制数据读取、每轮迭代次数和优化器。
LoRA、batch size、梯度累积次数、总 epoch 数、warmup epoch 数通过 running args 设置。

### Policy

| 参数（当前值） | 实际作用与设置方式 |
| --- | --- |
| `pretrained_load_mode: full` | `full` 严格加载完整预训练模型；`none` 不加载模型权重。当前只支持这两个值。LoRA 训练必须使用 `full`。 |
| `pretrained_policy_path: lerobot/pi0_base` | 官方权重来源。可以是 Hub 仓库、本地 safetensors 目录或单个 safetensors 文件。wrobo 的 .pth 初始化走 `--pretrained_weight`；断点续训走 `--continue_train`。 |
| `pretrained_revision: null` | Hub 权重版本；null 使用默认 revision。可设为 commit SHA 固定模型版本；已验证的是 `25c379b52ba2ff8788cab921758a3cc3fe3f77f2`。此字段不固定 tokenizer revision。 |
| `local_files_only: false` | false 允许联网获取权重/tokenizer；true 只使用本地路径或已有缓存，缺文件就报错。 |
| `text_tokenizer_name: google/paligemma-3b-pt-224` | 与预训练词表对应的 tokenizer，也可设为本地 tokenizer 目录。不能随意换成其他模型的 tokenizer。 |
| `tokenizer_max_length: 48` | 文本 token 长度上限，包含特殊 token，不是 48 个单词；不足补齐，超长截断。episode 和 frame 合并后的整条指令也受此限制。 |
| `action_chunk_size: 50` | 每次预测未来 50 个数据采样步的动作，也决定训练时动作块长度。不是 50 次去噪，不决定部署时一次执行多少步；对应时长为 50/数据采样频率。 |
| `proprio_dim: 6` | 数据中 proprio_state 的真实维度，必须与数据一致。内部补零到官方 32 维；合法范围 1–32。 |
| `action_dim: 6` | 数据中 action_abs 的真实维度，必须与数据一致。内部补零到 32 维，输出和损失再截取真实维度；合法范围 1–32。 |
| `num_inference_steps: 10` | 推理时 Euler 积分/去噪次数。增大通常增加推理耗时；它不控制训练迭代次数，训练时采样连续时间 t。 |
| `dtype: bfloat16` | 默认按 OpenPI 规则采用 BF16 基座，仅视觉 embedding、Gemma norm 和顶层投影保留 FP32；显式 float32 使用全 FP32 基座。LoRA adapter 始终沿用 PEFT 默认 FP32。它不是 AMP 开关，CUDA autocast 由 Trainer 独立控制。 |
| `gradient_checkpointing: true` | 训练时重算部分激活以节省显存，代价是额外计算。与磁盘 checkpoint 保存无关；不改变可训练参数范围。 |
| `freeze_vision_encoder: False` | 非 LoRA 情况下，True 同时冻结视觉 tower 和视觉 projector，并保持二者 eval 模式；独立于语言冻结开关。 |
| `freeze_language_encoder: True` | 非 LoRA 情况下，True 冻结语言模型，并保持 eval 模式；False 仅开放影响动作的语言参数。不会冻结 action expert。未使用的 lm_head 已移除。 |

`deep_supervision=False`、`history_width=1` 由 Trainer 内部注入，不再在 YAML
暴露。PI0 不使用中间层动作监督，每次只输入一个时间点的图像和状态；
多摄像头由 `data_keys` 指定，与 history_width 无关。

训练编译统一由公共 `--compile` 控制，PI0 模型构造本身不执行编译。
公共 Trainer 先设置 `torch.set_float32_matmul_precision("high")`，再以
`options={"inplace_buffers": False}` 编译网络。`high` 是进程级 FP32 矩阵乘法内部精度
设置，允许 TF32 等算法，不改变参数存储 dtype，也不关闭 AMP；原位 buffer 选项
按网络传入，不修改全局编译器配置。部署编译独立由显式 `ModelOptions.compile`
控制，不会自动沿用训练开关。参见 [公共 Trainer](../../documents/abc_trainer.md)。

非 LoRA 时的冻结组合如下，动作侧（action expert、状态/动作投影和时间 MLP）始终可训练。
表中的语言“可训练”只包括影响动作预测的参数；未使用的语言尾部始终冻结：

| freeze_vision_encoder | freeze_language_encoder | 视觉 tower | 视觉 projector | 语言模型 |
| --- | --- | --- | --- | --- |
| False | False | 可训练 | 可训练 | 可训练 |
| False | True（当前配置） | 可训练 | 可训练 | 冻结 |
| True | False | 冻结 | 冻结 | 可训练 |
| True | True | 冻结 | 冻结 | 冻结 |

当前配置不再是原先的全量微调：它冻结语言部分，保留视觉和动作侧训练。
冻结语言参数不会截断计算图，视觉部分仍可通过语言模型获得梯度。
两个开关同时为 True 等价于上游 `train_expert_only=True`；旧运行配置仍兼容：
旧字段为 True 时映射为两个冻结开关均 True，为 False 且没有新字段时保持原先不冻结语言的行为。

使用 `--use_lora` 时，PEFT 会冻结基座，只开放指定 LoRA 参数：
上述两个 freeze 开关不是“再额外全量训练哪些层”的开关；LoRA 范围由
`--lora_parts` 决定。两个值均 false、且没有启用 LoRA 时，训练全部动作相关基座参数。

### Training_settings

| 参数（当前值） | 实际作用与设置方式 |
| --- | --- |
| `data_keys` | 从 episode 读取的字段清单。当前两路相机为 image_angle/image_wrist，状态为 proprio_state，目标动作为 action_abs，语言为 instruction。相机名称由此生成，顺序沿用列表顺序。 |
| `tr_iterations_per_epoch: 250` | 每个训练 epoch 执行 250 组参数更新，每组读取 `gradient_accumulation_steps` 个 batch；FP16 溢出时可能跳过实际更新。这里的 epoch 是固定更新区间，不保证恰好遍历数据集一次。 |
| `val_iterations_per_epoch: 50` | 每个 epoch 验证 50 个 batch，不反向传播、不更新优化器。影响验证开销和损失估计。 |
| `base_lr: 2.5e-5` | AdamW 的目标峰值学习率，即 0.000025；启用 warmup 后实际初始学习率更低。与 LeRobot PI0 默认峰值相同。 |
| `min_lr: 2.5e-6` | cosine 衰减的目标下限，即 0.0000025。不是 warmup 起点。当前最后一轮使用 epoch=E-1，通常略高于该下限。 |
| `weight_decay: 0.01` | AdamW 的解耦权重衰减系数，不是学习率 decay 的速度。当前没有 bias/norm 免衰减参数组；LoRA 模式下仅优化可训练 adapter。 |
| `grad_clip_norm: 1.0` | 反向传播后、optimizer.step 前裁剪整体梯度范数，FP16 时先 unscale。当前实现始终调用裁剪，不能用 0 表示关闭；0 会把非零梯度压为零。 |

删除 data_keys 中的 instruction 后，数据集不再读取语言标注，
`--instruction_mode` 被忽略；模型仍使用空文本提示及其 tokenizer。
数据中的状态和动作先由数据集按统计量归一化，policy 不重复归一化。

### 优化器和学习率与 LeRobot 是否一致

对照的是本工作区的 LeRobot 源码（不是其他版本或 OpenPI 原生训练配置）：

- `lerobot/src/lerobot/policies/pi0/configuration_pi0.py`
- `lerobot/src/lerobot/optim/optimizers.py`、`schedulers.py`
- `lerobot/src/lerobot/scripts/lerobot_train.py`、`configs/accelerator.py`
- 本目录 `PI0Trainer.py`，以及 wrobo 的 `abc_trainer.py` 和
  `training/lr_scheduler/CosineAnnealingLR.py`。

| 项目 | LeRobot PI0 preset | 当前 wrobo PI0 |
| --- | --- | --- |
| 优化器 | torch.optim.AdamW | 相同 |
| 峰值 LR | 2.5e-5 | YAML base_lr=2.5e-5，相同 |
| betas | (0.9, 0.95) | 相同，固定在 get_optimizers 中 |
| eps | 1e-8 | 相同，固定在 get_optimizers 中 |
| weight decay | 0.01 | YAML weight_decay=0.01，相同 |
| 梯度范数裁剪 | 1.0；非正数关闭 | 默认同为 1.0；不支持用 0 关闭 |
| 最低 LR | 2.5e-6 | YAML min_lr=2.5e-6，相同 |
| warmup | 默认 1,000 个训练 step | running args 的 warmup_epoch；默认 0 |
| cosine 衰减终点 | 默认全局 step 30,000；之后保持下限 | 总训练 epoch 数决定；随 num_epoch 改变 |
| LR 更新频率 | 每个训练 batch 后调用 scheduler.step | 每个 epoch 开始调用 scheduler.step(epoch)；轮内恒定 |
| AMP | accelerator.mixed_precision 默认 no，可设 bf16/fp16 | CUDA 自动选 BF16 或 FP16；CPU 不启用 autocast |
| 梯度累积 | 默认 1，可配置 | 公共 `--gradient_accumulation_steps`，默认 1，正整数 |
| batch size | 每个数据并行进程的 batch size | 每次前后向的全局 batch size，除以 world_size；有效 batch 为全局值乘累积次数 |
| EMA | 可选模型权重 EMA，默认关闭 | best 选择使用验证损失 EMA（默认 decay=0.9），不是权重 EMA |

优化器本身不根据 `--use_lora` 自动切换学习率；本节 SO101 示例沿用上述 AdamW
数值，LIBERO 的四份 YAML 则使用前面的分范围设置。LeRobot 的 PI0 get_optim_params
返回全部参数，wrobo 先过滤 requires_grad；没有梯度的冻结参数不会被 AdamW 更新，不能把“传给优化器的
参数数量”直接当成可训练参数数量。

**学习率公式也不同，不能仅靠 epoch/step 换算达到完全相同。**

设 P 为 base_lr，m 为 min_lr。LeRobot 默认 W=1000、D=30000，
其局部源码使用以下调度（s 是 scheduler 的全局 step 索引）：

```text
0 <= s < W:
    lr = P * [1/(W+1) + (1 - 1/(W+1))*s/W]

s >= W:
    lr = m + (P-m)/2 * [1 + cos(pi * min(s,D)/D)]
```

cosine 使用 s/D，不是 (s-W)/(D-W)。默认第一步 LR 约为
2.4975e-8。训练总步数 S < 30000 时，上游自动缩放为
D=S、W=int(1000*S/30000)；S >= 30000 时保持默认终点，
不会把 cosine 自动拉长到 S。这里按梯度累积=1 比较；启用累积后，
本地 LeRobot 的 step 仍按 micro-batch 计数，不能直接视为 optimizer update 数。

wrobo 设 E=num_epoch、W_e=warmup_epoch、e 为从 0 开始的 epoch：

```text
0 <= e < W_e:
    lr = max(P*e/W_e, 1e-7)

e >= W_e:
    q = min(e-W_e, max(E-W_e, 1))
    lr = m + (P-m)/2 * [1 + cos(pi*q/max(E-W_e,1))]
```

因此启用 warmup 时，第一整个 epoch 使用 1e-7；warmup 后从峰值
重新开始 cosine。不开 warmup 则从峰值起步。正常情况下总更新数
约为 num_epoch * tr_iterations_per_epoch，验证次数不计入。

### 分别怎么设置

如需沿用 LeRobot 默认优化器数值，当前 YAML 已经匹配：

```yaml
Training_settings:
  base_lr: 2.5e-5
  min_lr: 2.5e-6
  weight_decay: 0.01
  grad_clip_norm: 1.0
  tr_iterations_per_epoch: 250
```

wrobo 中保持每轮 250 次更新时：

- 近似对齐 30,000 次训练更新及 1,000 次 warmup：
  `--num_epoch 120 --warmup_epoch 4`。时长匹配，曲线仍不完全相同。
- README 前面的 `--num_epoch 300 --warmup_epoch 4` 实际为
  75,000 次更新、1,000 次 warmup；cosine 被拉长到训练结束，
  并不等价于 LeRobot 默认在 30,000 step 后保持最低 LR。
- 若不传 warmup_epoch，默认 0，即没有 warmup，不能视作上游默认。
- 对短程 S=3,000 的上游训练，默认 warmup 会缩为 100 step。
  每轮 250 step 的 wrobo 无法用整数 warmup_epoch 精确表示这段 warmup。

LeRobot 如使用默认 policy training preset，可通过以下字段明确指定
上述数值（以下是应附加到完整 LeRobot 训练命令的参数）：

```text
--steps=30000
--policy.optimizer_lr=2.5e-5
--policy.optimizer_betas='[0.9,0.95]'
--policy.optimizer_eps=1e-8
--policy.optimizer_weight_decay=0.01
--policy.optimizer_grad_clip_norm=1.0
--policy.scheduler_warmup_steps=1000
--policy.scheduler_decay_steps=30000
--policy.scheduler_decay_lr=2.5e-6
--accelerator.gradient_accumulation.steps=1
--accelerator.mixed_precision=bf16
```

其中 bf16 用于对齐支持 BF16 的 CUDA 上当前 wrobo 的 autocast，
不是 LeRobot 的默认值。如果需要比较全 FP32 参数存储，应在两边显式设置
policy dtype=float32；wrobo 当前默认 bfloat16，本地 LeRobot 与 OpenPI 的
BF16 模式所保留的 FP32 模块也不完全相同，不能只凭 dtype 名称认为参数精度相同。
LeRobot 开启 policy training preset 时，应修改 policy.optimizer_* /
policy.scheduler_*，而不是指望自定义顶层 optimizer/scheduler 不被 preset 覆盖。

例如两张 GPU、LeRobot 每卡 batch_size=4、累积=1，对应 wrobo
全局 `--batch_size 8 --gradient_accumulation_steps 1`。若两边均累积 4 次，
wrobo 使用 `--batch_size 8 --gradient_accumulation_steps 4`，有效 batch 同为 32。
这只对齐批次组成；两边的训练 step 计数与学习率调度仍需分别核对。

若要求严格复现 LeRobot 调度，需要改 wrobo 代码：使用上游的全局 step
公式和短训练自动缩放规则，每个训练 step 后更新，并在 checkpoint 中
保存/恢复对应计数。仅修改现有 YAML/CLI 无法做到；新增任意未被读取的
scheduler_warmup_steps、scheduler_decay_steps 或 optimizer_betas 字段不会生效。
本节列出对比，当前优化器和学习率调度仍保持 wrobo 实现。

## Shared gradient accumulation

`--gradient_accumulation_steps K` is a shared running argument, defaults to `1`, and
must be a positive integer. For one GPU, for example:

```text
--batch_size 2 --gradient_accumulation_steps 16
```

This processes 16 micro-batches before one parameter update, giving an effective batch
of 32. With multiple GPUs, `batch_size` already includes all ranks: per-rank batch is
`batch_size / world_size`, and effective batch is `batch_size * K`.
`tr_iterations_per_epoch=250` still means 250 update groups, so 120 epochs remain
30,000 update groups; the number of forward/backward passes is multiplied by `K`.
Validation remains one batch per iteration and the scheduler still advances by epoch.

PI0 computes its padding-aware flow loss; `ABCTrainer` handles accumulation, backward,
AMP unscaling, gradient clipping and optimizer/scaler updates. Loss normalization uses
the group's non-padded action count across all ranks, so batches with different padding
are weighted correctly. DDP synchronizes gradients only on the final micro-batch.
Only one micro-batch's graph is kept on the GPU; CPU batch storage grows with the
accumulation count. This saves activation memory relative to directly using the effective
batch, but cannot make a model fit if batch 1 already exceeds device memory.

Run names record `GA`, startup logs record micro/effective batch sizes, and full
checkpoint resume validates batch size, world size and accumulation count. Use a new
run when changing those values. PI05 inherits this same implementation; see
[the common Trainer contract](../../documents/abc_trainer.md#gradient-accumulation).

## Setup and training

Use a compatible PyTorch/torchvision environment (tested with PyTorch 2.11 and
Transformers 5.4.0). Install the shared wrobo dependencies:

```bash
python -m pip install -r wrobo/requirements.txt
PYTHONPATH=. python -m wrobo.methods.PI0.train \
  --dataset_dir ./Dataset/lerobot_SO101/2026_09_18 \
  --instruction_mode episode_only \
  --config_file ./Config/PI0_SO101.yaml \
  --log_dir ./Logs --method_name PI0 --fold 0 \
  --num_epoch 300 --warmup_epoch 4 --batch_size 1 --gpu 0 \
  --use_lora --lora_rank 16 --lora_alpha 16 --lora_dropout 0 \
  --lora_save_mode merged
```

For full fine-tuning omit `--use_lora` and set both `freeze_vision_encoder` and
`freeze_language_encoder` to `False`; memory requirements are substantially
higher because gradients and AdamW states cover the full trainable base.
Multi-GPU entry uses `torchrun --nproc_per_node=N -m wrobo.methods.PI0.train ...`.
`--batch_size` is the global batch size. Set `--gpu` consistently with the launch.

Default weights: [lerobot/pi0_base](https://huggingface.co/lerobot/pi0_base).
Default tokenizer: `google/paligemma-3b-pt-224`; Hugging Face access to that
repository may require accepting its license and authenticating. A local tokenizer
directory can be supplied as `Policy.text_tokenizer_name`.

`pretrained_load_mode: full` loads every model tensor strictly, before LoRA is
injected. Missing/unexpected/mismatched weights raise an error; no random fallback.
Use `pretrained_policy_path` for a Hub repo, local safetensors directory or file.
Single-file and indexed/sharded safetensors are supported. Hub revisions can be
pinned with `pretrained_revision`. A resolved Hub commit and load report are
saved with the run. OpenPI JAX checkpoints must first be converted to LeRobot
PyTorch safetensors.

`pretrained_load_mode: none` allows random full-model experiments or reconstruction
for loading a complete wrobo deployment checkpoint. LoRA training requires a
reconstructible full pretrained base. Use `--continue_train` to resume a run;
`--pretrained_weight` loads wrobo full/weights-only checkpoints for non-LoRA training.

## Data and loss

Camera order follows `data_keys`. Frames enter as RGB [0,1], then are resized with
centered padding to 224x224 and mapped to [-1,1]. Missing configured cameras get
masked images. Only history_width=1 is supported.

The dataset already applies state/action mean/std normalization; it is not applied
a second time in the policy. SO101 dimensions stay 6 while the model internally
pads to the official 32 state/action dimensions. Text gets a trailing newline,
right padding and PaliGemma tokenization (48 tokens by default). When data_keys
omits instruction, instruction_mode is ignored and the model uses an empty prompt.

The official Beta timestep sampler, velocity target, joint attention and Euler
inference are retained. Loss is truncated to the real action dimensions.
Unlike LeRobot's unmasked time average, the Trainer excludes wrobo padded action
timesteps and divides by the valid element count. Scheduling remains wrobo's
epoch-based scheduling, rather than LeRobot's step-based scheduler.

## LoRA and trainable parameters

PI0 and PI05 use the same four part names and `--lora_parts` interface as SmolVLA:

| Part | PI0 modules | LoRA targets |
| --- | --- | --- |
| `vision` | PaliGemma `vision_tower` and `multi_modal_projector` | Vision attention Q/V and the visual projector's linear layers |
| `language` | PaliGemma `language_model`, including input embeddings | Earlier language attention Q/V; the last language layer's K/V |
| `action` | `gemma_expert`, including its norms | Every expert layer's attention Q/V |
| `projection` | `state_proj`, `action_in_proj`, `action_out_proj`, `action_time_mlp_in/out` | These five linear layers |

The visual projector converts image tokens to the language width, so it belongs to
Vision. This matches SmolVLA's ownership of its vision-language connector; Projection
here means the separate robot state, action and time projections. PI0's vision freeze
switch controls both its tower and visual projector, regardless of the language switch.

`get_parameter_parts()` verifies that every policy parameter belongs to exactly one
part, including adapters after PEFT wrapping. Startup logs show each part's total,
trainable and frozen counts. Every run also writes `trainable_parameters.json` for
auditing actual trainable tensor names/shapes.

The default `--lora_parts action projection` retains LeRobot's default PI0 targets:
expert Q/V plus the five projections. At rank 16 this is **1,385,984** trainable
parameters in 82 adapter tensors (41 adapted linear layers). Counts for each part,
verified by constructing the full architecture and applying PEFT on the meta device:

| Part | Adapted linear layers | Trainable rank-16 parameters |
| --- | ---: | ---: |
| `language` | 36 | 1,814,528 |
| `vision` | 55 | 2,041,856 |
| `action` | 36 | 1,253,376 |
| `projection` | 5 | 132,608 |
| `all` | 132 | **5,242,368** |

Other ranks change the adapter parameter count proportionally. Select parts with:

```text
--use_lora --lora_parts all
--use_lora --lora_parts vision action projection
```

`all` must be used alone. Parts are normalized to the stable order
`language-vision-action-projection`; duplicate part names do not create duplicate
adapters. The run directory records the selection, for example
`lora_rank16_alpha16.0_dropout0.0_merged_parts_language-vision-action-projection/fold_0`.
PI0 and PI05 both default to `rank=16, alpha=16`. Omitting `--lora_alpha` produces
`alpha16`; explicitly passing `--lora_alpha 16` produces `alpha16.0` because the CLI
parses supplied alpha values as floats. PEFT keeps trainable adapters in FP32 even
when the underlying base layer uses BF16; their gradients and AdamW moments are FP32 too.

PEFT freezes all base parameters, including embeddings, norms, biases and any
non-target linear layers. There are no `modules_to_save` or additionally trained
base biases. Choosing Vision or Language enables that part's adapter training mode
(including LoRA dropout), without unfreezing its base weights. The YAML freeze
switches do not override the selected adapter parts.

Without LoRA, the two freeze switches follow the independent part rules above.
With Language enabled, its final Q/O/MLP, post-attention norm and final language norm
remain frozen: the action path consumes the last layer's K/V and input norm, not
its final language output. These **109,056,000** registered parameters have no
action-loss gradient path. They remain in checkpoints; freezing them changes the
optimizer scope, not the model layout or action computation. The last input norm
and K/V stay trainable. Unused vocabulary-output heads remain removed as below.

### Parameter counts and unused vocabulary heads

The PI0 action path consumes transformer hidden states, not vocabulary logits.
Both `paligemma.lm_head` and `gemma_expert.lm_head` are therefore removed before
weight loading, device transfer, and optimizer construction. They are absent from
new weights-only, retrainable and merged-LoRA checkpoints. The action expert's
unused input embedding remains removed as before.

| Architecture | Registered parameters |
| --- | ---: |
| Previous implementation, without LoRA | 4,028,019,472 |
| Removed PaliGemma vocabulary head | 526,647,296 |
| Removed expert vocabulary head | 263,323,648 |
| Current implementation, without LoRA | 3,238,048,528 |
| Current implementation, rank-16 LoRA, default Action + Projection | 3,239,434,512 |
| Trainable rank-16 adapters, default Action + Projection | 1,385,984 |

With both freeze switches false and LoRA disabled, **3,128,992,528** parameters
are trainable; the 109,056,000 unused language-tail parameters stay frozen. Other
freeze selections reduce the trainable count further.

These heads were genuinely separate parameters in the tested Transformers 5.4.0
construction, not just duplicate state-dict names. Removing them saves about
1.47 GiB of BF16 resident parameter storage, or 2.94 GiB with explicit FP32 storage.
The upstream constructors can still
allocate them temporarily before removal; this change does not eliminate every
initialization-time memory peak.

Official safetensors are inspected before tensor loading. Unused output heads are
skipped, except that a PaliGemma head stored in place of the tied input embedding
is read and mapped to that required embedding. Other model keys/shapes remain
strictly checked. Entire Hub files may still need downloading: selective tensor
loading is not selective network downloading.

Full/merged model weight layouts are unchanged by the new freezing rules. Full
training checkpoints also contain optimizer parameter groups, so runs with a different
trainable scope cannot be continued with the same optimizer state. Use
`--pretrained_weight CHECKPOINT` to initialize a non-LoRA run from model weights
with a fresh optimizer when changing the freezing scope.

SmolVLA is not completely head-free: its expert uses `AutoModel` without a vocabulary
head and drops `embed_tokens`, but its VLM still retains `lm_head`.

## Checkpoints and evaluation

Without LoRA, wrobo saves normal full retrainable and weights-only checkpoints.
With LoRA, latest/best/final retrainable checkpoints contain adapters, optimizer,
scheduler, scaler, epoch, logs, best EMA and base/LoRA metadata. The base weights
must remain accessible for resume. Changing parts/rank/alpha/dropout on resume is
rejected. Adapter tensor names and shapes must also match the selected targets.

- `--lora_save_mode merged`: best/final deployment .pth files contain standalone
  merged weights, loadable without fetching the original base model.
- `--lora_save_mode adapter`: best/final deployment .pth files contain only adapters.
  Evaluation reconstructs the recorded base and injects the recorded LoRA settings.

Merged export uses the shared `merged_lora_state_dict` utility to merge into temporary
weight copies. It never merges/unmerges the live policy, so exporting final cannot
round the frozen base and contaminate the subsequent best export. Export preserves
the configured parameter dtypes, PEFT safe-merge rounding, checkpoint keys and adapter-only layout;
BF16 deployment weights still have their inherent merge rounding. This change does
not repair previously exported files.

The tokenizer, PI0 config and LoRA metadata are saved under Config_and_Trainer.
`pi0_lora_config.json` records the canonical `parts` list; retrainable checkpoints
carry the same selection. Adapter evaluation reconstructs the selected parts from
this metadata. Merged deployment weights reconstruct a plain model without adapters.
The method registry resolves PI0Trainer for the generic evaluator.

## Migration provenance

Source: workspace `lerobot/src/lerobot`, version 0.6.2. License: Apache-2.0;
original notices remain in the vendored Python files.

Original source SHA256 values (paths relative to `lerobot/src/lerobot`):

| Source file | SHA256 |
| --- | --- |
| `policies/pi0/modeling_pi0.py` | `d49e0ade33734a0723d82198a4774ddb9745c004d45feb4a8db1aec528b8af77` |
| `policies/pi_gemma.py` | `faec69b28f74b42cfb6217898f9a5ee29a442a5f8275339eb2d4453d450dc412` |
| `policies/common/vla_utils.py` | `88efbc82314a0ffd826f4693ff42fc580a5385424370b271351e2674cf0e33cb` |
| `policies/common/flow_matching.py` | `ce1cff1aab02f971dacfc6f78779c6c6813a404165d2c16bf66c3ad2174049eb` |

Migration adjustments:

- Replace LeRobot imports with local modules and direct Transformers imports.
- Replace PreTrainedPolicy/config/processor infrastructure with wrobo wrappers.
- Preserve RNG state during gradient checkpointing for nonzero LoRA dropout.
- Exclude RTC from the wrobo policy interface.
- Conservatively use float32 instead of float64 on XPU/MPS in the dtype helper.
- Load official safetensors strictly; propagate loading errors.
- Use wrobo's action-padding-aware loss reduction and epoch-based scheduler.
- Expose independent vision/language freezing switches, with legacy config compatibility;
  keep deep supervision and history width as internal Trainer defaults.
- Remove PI0's unused vocabulary heads and selectively read official safetensors;
  preserve legacy model-weight loading and explicitly guard old optimizer groups.

## Verification record

These are historical results, not automatically refreshed checks.
The 778-tensor results below predate vocabulary-head removal; they describe the
previous model layout, not the current number of retained tensors.

Verified on 2026-09-26 with PyTorch 2.11.0+cu130, Transformers 5.4.0,
PEFT 0.21.0 and an NVIDIA GeForce RTX 3090:

- All 10 PI0 integration tests and 21 existing wrobo tests passed.
- Official checkpoint: `lerobot/pi0_base`, commit
  `25c379b52ba2ff8788cab921758a3cc3fe3f77f2`.
- All 778 tensor names/shapes matched, and the complete tensor data was loaded
  strictly with no missing or unexpected keys.
- LoRA backward and action sampling passed using synthetic token IDs.
- Rank 16: 41 adapted linear layers, 82 trainable adapter tensors,
  1,385,984 trainable parameters; targets match the local LeRobot implementation.
- Multi-GPU DDP execution, full-size non-LoRA optimizer training and robot
  rollout were not tested.

The completed `test_model.py` and `test_training.py` scripts were removed on
2026-09-27 after their passing results were recorded here. Separate provenance
and verification JSON files are no longer kept in this directory.

Real-text validation passed on 2026-09-27 after Hugging Face access was updated.
The retry used all four HTTP/HTTPS proxy environment variables pointing to
`http://127.0.0.1:7897`, with the same environment and official checkpoint commit
recorded above. No tokenizer substitution or synthetic-token option was used.

- Official `google/paligemma-3b-pt-224` GemmaTokenizer loaded successfully.
- Prompt `Pick up the blue cube.\n` encoded to
  `[2, 21585, 908, 573, 3868, 28660, 235265, 108]`.
- All 778 official tensors loaded strictly: zero missing or unexpected keys.
- Real text passed through the tokenizer and full PI0 model; the LoRA backward
  pass produced gradients and a finite loss.
- Action sampling returned finite output with shape `[1, 50, 6]` on the RTX 3090.

This was a model-level smoke test using real text and synthetic image/state/action
tensors, not a real dataset training run or robot rollout. The other untested
items listed above remain untested.

The final `verify_official.py` script was removed after this successful check,
as requested. All migration and verification records now live in this README.

Freeze-interface checks passed on 2026-09-27 using an inline test (no test script retained):

- All four freeze combinations: parameter trainability and train/eval modes.
- Tiny-model backward with gradient checkpointing: frozen language parameters had
  no gradients, while vision tower and projector received nonzero gradients.
- Identical state-dict keys/shapes across freeze combinations, legacy config mapping,
  and Trainer injection of deep supervision/history defaults.
- Full-size architecture on the meta device: rank-16 LoRA still has exactly
  1,385,984 trainable parameters, all belonging to adapters.

These checks did not repeat full official-weight loading or full-size non-LoRA training.

Vocabulary-head removal was verified on 2026-09-27 (inline checks, no test script retained):

- Loaded actual tensors from the cached official checkpoint into the head-free model:
  776 retained tensors, no missing/unexpected keys or shape mismatches.
- Old model weights with both head keys loaded strictly; subsequent state dicts
  contained neither head. Single-file and cross-shard selective reading and
  head-to-input-embedding fallback passed.
- Full-size meta-device construction confirmed the parameter counts above and
  unchanged rank-16 LoRA trainability.
- Tiny-model forward, backward with gradient checkpointing, and action sampling
  passed for all four vision/language freeze combinations.
- All 21 existing wrobo tests passed, as did syntax checks. Full-size GPU training,
  multi-GPU execution, and robot rollout were not repeated for this change.
