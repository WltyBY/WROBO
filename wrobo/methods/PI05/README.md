# PI0.5 / PI05 in wrobo

本目录迁移本地 `lerobot/src/lerobot/policies/pi05` 的标准单帧 PI0.5，
不是 PI0、PI0-FAST，也不包含可选 MEM 历史记忆和 RTC 实时分块扩展。
运行时不依赖安装 LeRobot；共享组件来自已迁移的 `wrobo.methods.PI0`。
上游代码采用 Apache-2.0，模型文件保留相应版权声明。

## 文件职责

| 文件 | 职责 | 不负责 |
| --- | --- | --- |
| `train.py` | 共享 running args + LoRA 参数，实例化 PI05Trainer 并启动训练 | 不构建网络、不实现 loss/优化器 |
| `PI05Trainer.py` | PI05 配置校验、quantile 模式、原始指令 transform、策略与 artifact 标识 | 公共构造、加载器、输入搬运、验证和优化器更新由基类负责 |
| `models/configuration.py` | PI05 架构配置、合法性校验、200-token 提示长度 | 不读取数据或计算统计 |
| `models/policy.py` | wrobo 字典 I/O、图像预处理、离散状态提示、分词与官方权重 | 不执行 optimizer.step |
| `models/modeling_pi05.py` | PI05 AdaRMS 时间条件、动作投影、复用 flow matching 和 Euler 采样 | 不解析 CLI、不选择数据划分 |
| 通用 `utils/dataset_statistic.py` | 从选中的训练轨迹统一计算 mean/std/q01/q99 | 与其他 Trainer 共用统计入口 |

PI05Trainer 继承 PI0Trainer 的模型、scheduler、LoRA 保存/恢复钩子；
构造、初始化流程、数据集/加载器、输入字段绑定、验证、梯度累积和优化器/scaler 更新
统一由 ABCTrainer 提供。PI0 的 build_training_policy 钩子处理 artifacts 和 LoRA，
随后基类完成 device、DDP/compile、优化器及 loss 初始化；无需复制构造或验证代码。
PI05 只保留策略构建、quantile 模式、原始指令 transform、配置检查和 artifact 名称。
PI05Pytorch 复用 PI0 的联合注意力、PI-Gemma、flow matching 和 KV-cache 采样，
但独立构建 PI05 模块：不是先构建 PI0 再替换已加载的权重。
继承的内部函数有兼容性的 `state` 参数；PI05 不把它投影成连续状态 token。

## 相比 PI0 的关键差异

| 项目 | PI0 | PI05 |
| --- | --- | --- |
| 状态输入 | 连续 state_proj token | 归一化后离散为数字，放入文本提示 |
| 时间条件 | action_time_mlp，与动作 embedding 融合 | time_mlp，经 AdaRMS 调制动作专家 |
| 专家归一化 | 普通 RMSNorm | 带 scale/shift/gate 的 AdaRMS |
| 文本长度 | 48 | 200（包括任务与状态数字） |
| 数据规范化 | wrobo 的均值/标准差 | q01/q99 分位数仿射变换 |

当前状态采用真实机器人维度，例如 SO101 的 6 维，而不是先补齐 32 维再生成数字。
这与本地 LeRobot 的 `Pi05PrepareStateTokenizerProcessorStep` 一致。
输入动作仍补零到官方 32 维；损失/输出只保留真实动作维度，尾部 padding 帧不计入 loss。

提示格式严格为（末尾有空格）：

```text
Task: pick up the blue cube, State: 0 128 255 100 80 40;
Action: 
```

任务文本执行 `strip().replace("_", " ").replace("\n", " ")`。
状态使用 `np.digitize(state, np.linspace(-1, 1, 257)[:-1]) - 1`；不额外裁剪越界值。
公共 Dataset 工厂使用 PI05 的 get_text_transform 保留 raw string，
公共 prepare_inputs 不改变字符串；batch 后结合当前状态才分词。
因此不能把 PI0 已分词的任务 tensor 直接传给 PI05Policy。
没有 `instruction` 数据键时仍生成空任务 + 状态的提示，不读取 instruction library。

`PI05Policy._prepare_language` 使用 `@torch.compiler.disable`，将状态离散化、
prompt 拼接、tokenizer 及其 Tensor 构造一起排除在编译区域之外；返回的 Tensor
字典交给后续模型计算，神经网络部分仍可通过公共 `--compile` 编译。这样避免
tokenizer 内部 Python 列表的 token 值随 batch 改变而反复触发 Dynamo 重编译。
提示格式、分词结果、梯度检查点和参数精度保持原样；状态的 GPU 到 CPU 搬运仍在
这个方法内进行。此调整处理的是文本前处理的重编译，不保证降低模型反向的显存峰值。

相机顺序优先采用训练配置 `data_keys` 推导出的 camera_names；各相机共享视觉编码器，
按配置顺序拼接图像 tokens，再接文本。模型没有自动识别“腕部/第三人称”的排序规则。
图像 resize/pad 为 224×224，从 [0,1] 转到 [-1,1]。
相机缺失遵循共享 PI0 行为：保留存在的相机并在末尾追加 mask 为 false 的空图；
部署最好保持所有训练相机齐全。

## 归一化与部署

PI05Trainer 使用通用统计入口，在运行目录保存 `norm_stats.json`；统计中
mean/std 是真实样本均值与标准差，q01/q99 是真实分位数，不再把中心和半宽
伪装成 mean/std。`normalization.py` 专用统计模块已移除。
只统计当前 fold 经 `--task_ids` 过滤后的 train episodes；`fold=all` 使用
过滤后的第 0 折 train 与 val 并集，仍不包含 test。
多数据集默认分别统计；`--merge_norm_stats` 对各来源选中的样本联合统计，
不会平均各数据集的分位数。数值每条轨迹最多采样 1000 个时间点，图像最多采样
200 帧；串行、多进程使用一致的随机采样。

PI05 的 Dataset 和部署均通过共享归一化函数使用 q01/q99：

```text
center = (q01 + q99) / 2
scale = max((q99 - q01) / 2, 1e-8)
normalized = (raw - center) / scale
raw = normalized * scale + center
```

不额外 clipping；常量维度映射为 0。图像仍使用通道 mean/std。
新运行要求统计及 `norm_stats_selection.json` 同时存在并与所选训练数据一致；
缺少 q 字段或旧 `quantiles_affine` 统计会明确报错，不做旧格式回退。
参见 [统一数据选择与统计](../../documents/training_data_selection.md)。

直接调用 policy 时，输入 proprio_state/actions 必须已按本次统计归一化；
输出动作也是归一化空间。通用 Aloha evaluator 使用运行目录统计完成正反变换。
多数据集独立统计的部署需选择对应数据集统计；单一机器人可用 `--merge_norm_stats`。

## LIBERO：独立的训练与评估脚本

根目录 [train_pi05.sh](../../../train_pi05.sh) 和
[eval_pi05.sh](../../../eval_pi05.sh) 分别执行训练、评估，不会自动调用对方。
两份脚本直接列出 10 组命令：普通 VLAP、普通 AP，以及各自
rank/alpha 为 16、32、64、128 的 LoRA。需要只跑部分实验时，保留对应命令。
从仓库根目录使用 Bash 运行：

```bash
bash ./train_pi05.sh
bash ./eval_pi05.sh
```

每个脚本顶部的 `GPU_ID=0` 控制该脚本所有命令的 `--gpu`，训练和评估可以分别选择
设备；shell 不设置固定的 `CUDA_VISIBLE_DEVICES`。先完成所需训练，再执行评估。

| 配置 | 微调范围 | 峰值学习率 | 最低学习率 |
| --- | --- | ---: | ---: |
| [PI05_LIBERO_VLAP.yaml](../../../Config/PI05_LIBERO_VLAP.yaml) | 普通 VLAP，开放四类动作相关参数 | `2.5e-5` | `2.5e-6` |
| [PI05_LIBERO_AP.yaml](../../../Config/PI05_LIBERO_AP.yaml) | 普通 AP，冻结视觉和语言 | `1e-4` | `1e-5` |
| [PI05_LIBERO_VLAP_LoRA.yaml](../../../Config/PI05_LIBERO_VLAP_LoRA.yaml) | `--lora_parts all`，仅四类 adapter | `1e-4` | `1e-5` |
| [PI05_LIBERO_AP_LoRA.yaml](../../../Config/PI05_LIBERO_AP_LoRA.yaml) | 默认 AP，仅动作和投影 adapter | `3e-4` | `3e-5` |

这些学习率沿用对应 PI0 实验作为首轮比较起点，尚未通过 PI05 LIBERO 实验证明最优；
尤其 AP/LoRA 设置不是官方验证的独立配方。各 rank 使用相同范围的同一 YAML，
保持 `alpha/r=1`、`weight_decay=0.01`、`grad_clip_norm=1.0`。

四份配置完整加载 `lerobot/pi05_base`，使用两路 `image_top/image_wrist`、8 维
`observations/proprio_ee`、7 维 `action_rel` 和任务 instruction。LIBERO 动作已经是
相对 OSC 命令，不套用 SO101 的 `action_abs` 配置。256×256 图像由模型缩放为
224×224；文本长度 200 包含任务与离散状态提示。PI05Trainer 自动采用 q01/q99
归一化，并在 batch 后将归一化的真实 8 维状态编码进提示；评估使用保存的同一统计
进行状态归一化与动作逆变换。基座采用选择性 BF16，LoRA adapter 保留 FP32。

训练合并四套件统计，`--fold all` 使用全部选中的训练轨迹；验证 loss 不代表独立
留出集性能。micro-batch 1、梯度累计 32，有效 batch 32；120 epoch，每轮
250 次更新，共 30,000 次更新，预热 4 epoch，并开启公共 `--compile`。
脚本沿用默认种子 319。PI05 在 torch 2.14 上拒绝编译训练；torch 2.11 下仍需
按微调范围和 micro-batch 验证编译显存，版本检查通过不代表不会 OOM。

普通 VLAP 仅权重、梯度和 AdamW 两份动量就约 **25.27 GiB**，还不含激活与临时
空间，不能直接在 24GB 显卡上训练。脚本保留这组完整命令，需在显存足够的设备
执行，或编辑脚本只运行当前设备适合的实验；`set -e` 会在任一命令失败后停止后续命令。

评估使用 `--method PI05` 和每组训练的完整 `--log-dir`，默认加载 final checkpoint；
四套件各 10 个任务，每任务默认 50 次，共 2,000 次，使用 4 个仿真进程并保存
双视角拼接视频。需要 best 时添加 `--checkpoint-name checkpoint_best.pth`。
模型在 `wrobo` 环境运行，入口自动启动 `libero` 环境仿真。
修改 batch、梯度累计、epoch、rank/alpha 或 parts 后，应同步评估脚本中的目录：
默认 r16/a16 为 `alpha16`，显式传入 alpha 的 r32/r64/r128 分别为
`alpha32.0`、`alpha64.0`、`alpha128.0`。

## 启动训练

依赖统一放在 `wrobo/requirements.txt`，没有 PI05 私有 requirements。
此前 PI0 的服务器对照中，torch 2.14 + compile 曾显著增加反向显存，torch 2.11
仅在已测配置中通过；PI05 VLAP 的 batch=16、累计=2 在 torch 2.11 编译反向时
也出现过 48GB 显卡 OOM。PI05 对 2.14 的编译训练作保护性拒绝；先用不编译训练
建立对应批量的基线，再验证编译后的峰值显存。
运行脚本不硬编码代理；需要联网时由调用环境提供 HTTP_PROXY/HTTPS_PROXY 等变量。

PI05 与 PI0 的配置类及示例 YAML 均默认 `dtype: bfloat16`，采用
[OpenPI 的参数精度规则](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models_pytorch/gemma_pytorch.py)：
视觉 patch embedding 的 weight/bias、position embedding 的 weight，以及 Gemma 的
`input_layernorm`、`post_attention_layernorm`、`model.norm` 保留 FP32，其余 core
参数为 BF16。PI05 AdaRMS norm 中的条件线性层也属于这些 FP32 模块；core 外的
action/time projections 仍为 FP32。视觉 encoder 与视觉 projector 为 BF16，
embedding 输出进入 encoder 前转换 dtype，前向不改写参数存储精度。
显式设置 `dtype: float32` 仍支持全 FP32 基座。

普通微调的 `.grad` 和 AdamW `exp_avg`、`exp_avg_sq` 跟随对应参数 dtype；
FP32 参数对应 FP32，BF16 参数对应 BF16，`step` 计步标量默认 FP32。
LoRA 保留 PEFT 默认 **FP32 adapter**，因此 adapter 的梯度和动量也为 FP32，
冻结基座没有梯度或优化器动量。没有额外的 FP32 master 权重副本。

参数存储与 AMP 独立：公共 Trainer 在支持的 CUDA 设备上使用 BF16 autocast，
否则使用 FP16 autocast 和 GradScaler；它不永久改写参数 dtype。
反向中间算子的精度跟随对应前向计算，不能由最终 `.grad` 推断整个反向的精度。

LeRobot 的配置类及基座 metadata 默认记录 FP32，但
[PI0](https://huggingface.co/docs/lerobot/pi0) / [PI05](https://huggingface.co/docs/lerobot/pi05)
官方训练示例显式选择 BF16；[OpenPI PyTorch 训练配置](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py)
默认 BF16 并保留上述 FP32 模块。当前 wrobo 对齐的是 OpenPI 的参数精度规则，
不代表数据、调度或 LoRA 目标完全相同。完整对照见 [PI0 精度说明](../PI0/README.md)。

按完整 PI05 架构与实际 dtype 计算，默认模型权重约 6.469 GiB；加上可训练参数的
梯度及 AdamW 两份动量后，普通 VLAP 约 25.268 GiB，普通 AP 约 9.536 GiB。
这些数字不含激活、CUDA runtime、编译和临时空间，普通 VLAP 不能直接在 24GB
单卡上训练；AP 的数字也不是峰值显存保证。LoRA 另外为 FP32 adapter 保存权重、
梯度及动量；尚未进行 GPU 峰值显存实测。

```bash
PYTHONPATH=. python -m wrobo.methods.PI05.train \
  --dataset_dir ./Dataset/lerobot_SO101/2026_09_07 \
  --instruction_mode episode_only \
  --config_file ./Config/PI05_SO101.yaml \
  --log_dir ./Logs --method_name PI05 --fold 0 \
  --num_epoch 80 --warmup_epoch 2 --batch_size 8 --gpu 0 \
  --use_lora --lora_rank 16 --lora_alpha 16 --lora_dropout 0 \
  --lora_save_mode merged
```

每个 dataset_dir 必须有 `episodes/`；`dataset_split.json` 不存在时由训练阶段
自动创建五折划分。语言启用或显式选择整体任务时需要 `instruction_library.json`。
`--task_ids`（如 `0,2 all`）和显式 `--instruction_mode` 列表分别与目录逐一对应；
未传 `--task_ids` 时全部任务参与。任务筛选不拆分帧级标注，也不重新生成划分。
示例 batch size 不是显存保证；根据实际 GPU 调整。每轮 250 组更新时 80 轮为
20,000 组更新；FP16 溢出时可能跳过实际优化器更新。
DDP 可用 `torchrun --nproc_per_node=N -m wrobo.methods.PI05.train ...`；
batch_size 是每次前后向的全局值，需能被 world_size 整除。

梯度累积通过公共 running args `--gradient_accumulation_steps K` 设置，默认 `1`，
必须是正整数。例如单卡 `--batch_size 2 --gradient_accumulation_steps 16`，
每次参数更新累计 16 个 batch，有效 batch 为 32。多卡时有效 batch 仍为
`batch_size * K`，不用再乘 GPU 数；每卡实际 batch 为 `batch_size / world_size`。
`tr_iterations_per_epoch` 仍按更新组计数，不会因为开启累积而减少更新组数；
验证不累积，学习率仍按 epoch 调度。

公共 Trainer 按整组及全部 rank 的有效动作数量归一化 masked flow loss，
只在组末执行梯度裁剪、优化器/scaler 更新；DDP 前面的前后向跳过梯度同步。
GPU 逐批前后向，不保留前面批次的计算图；CPU 需暂存整组 batch。
累积降低给定有效 batch 的激活显存需求，但不会减少基座参数、梯度及优化器状态。
如果实际 batch 1 已无法运行，增加累积次数不能解决。
运行名记录 `GA`，完整断点恢复会校验 batch size、world size 和累积次数；
改变这些值应开启新运行。参见 [公共 Trainer](../../documents/abc_trainer.md#gradient-accumulation)。

`dtype` 选择参数存储精度，不是 AMP 开关；`gradient_checkpointing` 控制激活重算。
`deep_supervision=False`、`history_width=1` 在 Trainer 内部设定，不暴露在示例 YAML。

训练编译统一由公共 `--compile` 控制，PI05 模型构造本身不执行编译。
公共 Trainer 先设置 `torch.set_float32_matmul_precision("high")`，再以
`options={"inplace_buffers": False}` 编译网络。`high` 是进程级 FP32 矩阵乘法内部精度
设置，允许 TF32 等算法，不改变参数存储 dtype，也不关闭 AMP；原位 buffer 选项
按网络传入，不修改全局编译器配置。部署编译独立由显式 `ModelOptions.compile`
控制，不会自动沿用训练开关。参见 [公共 Trainer](../../documents/abc_trainer.md)。

## 配置和优化器

- `pretrained_load_mode`: full 严格加载完整基座；none 随机初始化，仅用于非 LoRA 实验。
- `pretrained_policy_path`: 默认 `lerobot/pi05_base`，也支持本地 safetensors 文件/目录。
- `pretrained_revision`: 可固定 Hub commit；`local_files_only` 禁止下载。
- `text_tokenizer_name`: 官方 PaliGemma tokenizer 或对应本地目录，需要访问授权。
- `tokenizer_max_length`: 包括状态的完整提示长度；过长会截断。
- `action_chunk_size`: 训练和预测动作块长度；`num_inference_steps` 是 Euler 去噪步数。
- `proprio_dim` / `action_dim`: 真实数据维度，均不能超过 32。
- `freeze_vision_encoder` / `freeze_language_encoder`: 非 LoRA 下分别控制 Vision/Language。
  Vision 包括 tower 和视觉 projector；冻结 Vision 时二者一起冻结，与语言开关独立。
  示例 False/True 冻结整个 Language；两个开关均 False 时仍冻结不影响动作的语言尾部。
  开启 LoRA 后只有 `--lora_parts` 选中的 adapter 可训练；两个 freeze 开关不扩大 LoRA 范围。
- `data_keys`: 相机顺序、状态、动作和可选语言输入；统一选择一个
  `observations/proprio_*` 和一个 `action_*` 字段。示例使用 proprio_state/action_abs，
  也可选择转换后的 proprio_joint/action_rel 等字段；维度须匹配 Policy 配置，
  动作数值语义由源字段决定，不会因选择字段而自动变换控制坐标。
- `tr_iterations_per_epoch` / `val_iterations_per_epoch`: 每轮训练更新组数/验证 batch 数，
  不是完整遍历数据集；每个训练更新组处理 `gradient_accumulation_steps` 个 batch。
- `base_lr=2.5e-5`、`weight_decay=0.01`、`grad_clip_norm=1.0`：与本地上游预设一致。
  AdamW betas=(0.9,0.95)、eps=1e-8；只传入 requires_grad 参数。
- `min_lr=2.5e-6`：wrobo epoch-based cosine 的下限。

注意 scheduler 并非 LeRobot 的逐步曲线：本方法沿用 wrobo 的逐 epoch warmup+cosine，
总时长由 num_epoch 决定；warmup_epoch 默认来自 running args。
LeRobot 默认按 step 调度，warmup=1000、decay=30000。把 warmup_epoch 换算成步数
只能近似对齐，不能保证逐步 LR 相同。验证损失 EMA 用于选 best，不是模型权重 EMA。

## 参数归属与冻结

PI05 与 PI0、SmolVLA 使用相同的四类名称。PI05Trainer 继承 PI0Trainer 的分类和
完整性检查，保证基座参数及 adapter 都恰好归属一类，没有重复或遗漏：

| 类别 | PI05 模块 | LoRA 覆盖范围 |
| --- | --- | --- |
| `vision` | PaliGemma `vision_tower`、`multi_modal_projector` | 视觉 attention 的 Q/V，以及视觉 projector 的线性层 |
| `language` | PaliGemma `language_model`，包含输入 embedding | 前面语言层的 Q/V，最后有效语言层的 K/V |
| `action` | `gemma_expert`，包含全部 AdaRMS dense 和 norm | 各专家层 attention 的 Q/V |
| `projection` | `action_in_proj`、`action_out_proj`、`time_mlp_in/out` | 这四个线性层 |

视觉 projector 负责把图像 token 映射到语言宽度，因此归入 Vision，和 SmolVLA 将
vision-language connector 归入 Vision 的方式一致；Projection 专指动作、状态和时间
投影。PI05 状态经过离散提示进入 Language，没有连续的 `state_proj`，不存在漏分类。
启动日志按四类打印总参数量、可训练量和冻结量，并保存 `trainable_parameters.json`。

非 LoRA 时，`freeze_vision_encoder=True` 冻结 tower 和视觉 projector，并保持二者
eval 模式；`freeze_language_encoder=True` 冻结整个 Language。两个开关均 False 时，
最后语言层的 Q/O/MLP、post-attention norm 和最终 language norm 仍保持冻结：动作只
使用该层的输入 norm 和 K/V，它们仍可训练。两种模型的这一语言尾部均为
109,056,000 个参数，保留在权重中，只从优化器的可训练范围移除。

Action Expert 的 AdaRMS dense 属于动作路径，普通微调时可训练；不能套用语言尾部的
冻结规则。LoRA 下它与其他基础 norm、bias、非目标线性层一起冻结，不额外开放。

## LoRA、权重与 checkpoint

`--lora_parts` 与 SmolVLA 一样接受 `language vision action projection` 中的一项或
多项，也支持单独的 `all`。默认 `action projection` 保留本地 LeRobot 标准 PI0.5 的
实际目标：专家 Q/V，以及 action_in_proj/action_out_proj/time_mlp_in/time_mlp_out。
默认范围在 rank=16 时为 **1,352,704** 个可训练参数。完整架构在 meta device 构建并
实际应用 PEFT 后，各部分的参数量如下；其他 rank 的 adapter 参数量按比例变化：

| 类别 | 适配的线性层数 | rank-16 可训练参数 |
| --- | ---: | ---: |
| `language` | 36 | 1,814,528 |
| `vision` | 55 | 2,041,856 |
| `action` | 36 | 1,253,376 |
| `projection` | 4 | 99,328 |
| `all` | 131 | **5,209,088** |

例如，在训练命令中使用 `--use_lora --lora_parts all` 适配四个部分，或者使用
`--use_lora --lora_parts vision action projection` 只适配这三个部分。`all` 不能和其他
名称混用。parts 按 `language-vision-action-projection` 顺序规范化并记录在运行目录，
例如 `lora_rank16_alpha16.0_dropout0.0_merged_parts_language-vision-action-projection/fold_0`。
PI0 和 PI05 的 CLI 默认值统一为 `rank=16, alpha=16`。省略 alpha 参数时目录为
`alpha16`，显式传入 `--lora_alpha 16` 时因浮点解析而为 `alpha16.0`。

PEFT 冻结全部基座，只训练选中的 adapter；无 `modules_to_save`、无额外可训练基础
bias 或 norm。选择 Vision/Language 会为该部分的 adapter 开启训练模式，使 LoRA
dropout 正常工作，但不会解冻对应基座，也不会由 YAML freeze 开关额外扩大适配范围。

基座注册参数 3,353,433,872；两个 freeze 开关均 False、非 LoRA 时，可训练参数为
**3,244,377,872**。默认 Action + Projection 的 rank-16 LoRA 总参数为 3,354,786,576。
两个不用的词表输出头不保留、不写入新 checkpoint。官方若只保存 PaliGemma lm_head
作为共享 embedding，读取后映射到实际输入 embedding。其他缺失/多余键及形状不符报错；
不会用随机初始化掩盖官方权重加载失败。PI0 权重不能作为 PI05 权重加载。

- 非 LoRA：best/final 部署权重及完整 retrainable checkpoint。
- LoRA：retrainable 保存 adapter、优化器、scheduler、scaler、epoch、日志和基座元数据。
- `lora_save_mode=merged`：部署文件合并 adapter，离线加载无需重新下载基座。
- `lora_save_mode=adapter`：部署需要原始官方基座及本次 adapter。
- `--continue_train`：恢复 latest，包括优化器；LoRA parts/rank/alpha/dropout 必须一致，
  adapter 的键名与形状也必须匹配。改变非 LoRA 冻结范围也会改变优化器参数组，
  此时应加载模型权重并使用新优化器，不作原优化器的精确续训。
- 非 LoRA 的 `--pretrained_weight`：只初始化 wrobo .pth 权重，不恢复优化器。

PI05 继承 PI0 的导出流程，通过公共 `merged_lora_state_dict` 只在临时权重副本上合并。
导出不再对原模型执行 merge/unmerge，避免 final 导出引起基础权重舍入漂移，
进而污染随后导出的 best。导出保留当前配置的参数 dtype、PEFT safe_merge 的舍入规则、
checkpoint 键名及 adapter 布局；BF16 合并文件本身仍有固有舍入，已导出的旧文件不会自动修复。

PI05 使用独立 `pi05_config/`、`pi05_lora_config.json` 和断点类型标记；LoRA metadata
及 retrainable checkpoint 都保存规范化的 `parts`。adapter 评估根据记录重建相同范围，
merged 权重则加载普通模型。
注册名称为 PI05Trainer。评估器按日志名解析 Trainer，并调用其 build_policy /
build_policy_inputs；无需在 train.py 内复制推理逻辑。

## 验证记录

环境：torch 2.11.0+cu130、Transformers 5.4.0、PEFT 0.21.0。
官方 `lerobot/pi05_base` commit：`b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba`。
2026-09-29 已完成：

- 真实官方 safetensors 头部核对：812 个源 tensor，映射/去除无用头后目标 811 个 tensor，
  缺失/多余/形状不符均为零。
- 完整架构 meta-device 构建及 LoRA 参数量核对。
- 缩小网络与本地 LeRobot 源码的完整 flow forward / embed_suffix 数值一致。
- 真实 tokenizer、LoRA 梯度、Euler 动作采样、merged/adapter 导出。
- 临时 HDF5 数据集：训练集分位数、无验证泄漏、指令字符串批处理和 Trainer 输入。
- 缩小网络的实际 Trainer 两轮 LoRA 训练、best/final 保存、adapter+优化器恢复，
  merged/adapter 两种评估重建；非 LoRA 单步训练和完整断点恢复。
- 21 项现有 wrobo 回归测试和新旧 PI 系列语法检查通过。

- 完整官方 tensor 数据通过实际 PI05Policy 构建路径加载（CPU、非 meta 替代）：
  811 个保留 tensor 严格匹配，无缺失/多余项。真实 tokenizer 和任务文本配合合成
  图像/状态/动作，前向 loss 有限，1 步 Euler 输出有限 `[1,2,6]` 动作。
  此 smoke test 使用 action_chunk_size=2，并非完整 50 步任务训练。

真机 rollout、全尺寸 GPU 训练、DDP 和 compile 峰值显存尚未验证。
测试辅助脚本不保留在方法目录；正式训练也不应视为已获得任务成功率保证。

迁移时本地上游源文件 SHA256：

| 文件（相对 `lerobot/src/lerobot/policies/pi05/`） | SHA256 |
| --- | --- |
| modeling_pi05.py | 3bbc8480e4a96434275daef50d4341fafcf4bacfc8533d07a2fd16df2f56ae23 |
| processor_pi05.py | d276557b6bc002df714a2011867363b6ec45b687d3af84dc020c3734071409d6 |
