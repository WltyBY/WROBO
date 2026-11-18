# LIBERO 闭环评估

此目录用于评估 wrobo 训练的策略：从 LIBERO 官方初始状态启动仿真，执行模型预测的动作并统计任务成功率。这里不采集演示，不转换数据，也不读取 `dataset_split.json` 的 test 文件列表。

## PI0：分别运行训练与评估

根目录 [train_pi0.sh](../../../train_pi0.sh) 与
[eval_pi0.sh](../../../eval_pi0.sh) 是独立脚本，直接列出各组原生命令。
训练包含普通 VLAP、普通 AP，以及两种范围下 rank/alpha 为 16、32、64、128 的
LoRA，共 10 组；评估脚本与这 10 组的实际输出目录逐一对应。

```bash
bash ./train_pi0.sh
bash ./eval_pi0.sh
```

训练与评估脚本各自在顶部设置 `GPU_ID=0`，每条 Python 命令都通过
`--gpu "${GPU_ID}"` 引用它。需要切换到物理 GPU 1 时，把相应脚本中的 `GPU_ID`
改为 `1`；训练和评估两个脚本可以分别选择设备。shell 不设置
`export CUDA_VISIBLE_DEVICES=0`。

先完成所需训练，再执行对应评估命令；两个脚本不会自动调用对方。需要只跑部分
实验时，保留相应命令，使用各自独立的 `--method_name`、`--log-dir` 和
`--output-dir`。更多说明见 [PI0 文档](../../methods/PI0/README.md)。

四份专用 YAML 位于 `Config/PI0_LIBERO_{VLAP,AP,VLAP_LoRA,AP_LoRA}.yaml`，
使用两路图像、8 维 `observations/proprio_ee`、7 维 `action_rel` 和任务 instruction。
普通 VLAP 开放四类动作相关参数，普通 AP 冻结视觉和语言；LoRA 的基础权重全部冻结，
adapter 范围由 `--lora_parts` 控制。训练均使用 micro-batch 1、梯度累计 32、
120 epoch、每轮 250 次更新，种子默认 319。

| PI0 实验范围 | 峰值学习率 | 最低学习率 |
| --- | ---: | ---: |
| 普通 VLAP | `2.5e-5` | `2.5e-6` |
| 普通 AP | `1e-4` | `1e-5` |
| LoRA VLAP | `1e-4` | `1e-5` |
| LoRA AP | `3e-4` | `3e-5` |

普通 VLAP 沿用官方基础学习率，其余是首轮实验的工程起点，尚未在 PI0 LIBERO 上
验证为最优。各范围内部的四个 rank 保持相同学习率，且 `alpha/r=1`；四份配置的
weight decay 均为 `0.01`，梯度裁剪均为 `1.0`。

当前 PI0/PI05 默认 `dtype: bfloat16`，按 OpenPI 规则仅保留视觉 embedding、
Gemma norm 和顶层状态/动作/时间投影为 FP32，其余基座参数使用 BF16。
普通微调的梯度及 AdamW 两份动量跟随各自参数 dtype；LoRA adapter 保留 PEFT
默认 FP32，其梯度和动量也为 FP32。冻结基座不创建梯度或动量，没有额外 master
权重副本。显式 `dtype: float32` 仍支持全 FP32 基座，公共 Trainer 的 autocast
独立控制算子计算精度，不会永久改变参数存储 dtype。

LeRobot 的配置类默认值与训练示例的 BF16 覆盖需要区分；OpenPI 的选择性 BF16
也不表示所有数值都是 BF16。官方来源、保留清单和 AMP 说明见
[PI0 文档](../../methods/PI0/README.md)。
默认 PI0 权重约 6.039 GiB，计入梯度及 AdamW 两份动量后，普通 VLAP 约
23.548 GiB，普通 AP 约 7.816 GiB，均未包含激活、CUDA runtime、编译和临时空间。
普通 VLAP 在 24GB 3090 上仍没有足够的实际训练空间；脚本保留完整实验命令，需
使用显存足够的设备。小 micro-batch 不会减少参数和优化器状态，这些预算不是
峰值显存实测。若显式改回 FP32，VLAP/AP 对应预算分别约 47.032/15.580 GiB。

PI0 评估使用 `--method PI0`，默认读取 final，每任务 50 次、四套件共 2,000 次，
4 个仿真进程并保存双视角拼接视频。仿真图像仍为旋转后的 256×256 RGB，PI0 模型
内部缩放到 224×224。模型加载和仿真环境沿用下文的 `wrobo`/`libero` 分工。
LoRA 目录里的 alpha 按 CLI 实际类型生成：省略默认值是 `alpha16`，显式的
32/64/128 分别是 `alpha32.0`、`alpha64.0`、`alpha128.0`；脚本已填写对应路径。

## PI05：分别运行训练与评估

根目录 [train_pi05.sh](../../../train_pi05.sh) 与
[eval_pi05.sh](../../../eval_pi05.sh) 与 PI0 脚本对应，同样直接列出普通 VLAP/AP
及两种范围的 r16/r32/r64/r128 LoRA，共 10 组实验；训练和评估分开执行：

```bash
bash ./train_pi05.sh
bash ./eval_pi05.sh
```

四份配置为 `Config/PI05_LIBERO_{VLAP,AP,VLAP_LoRA,AP_LoRA}.yaml`，加载
`lerobot/pi05_base`，两路图像、8 维状态和 7 维 `action_rel` 与 PI0 保持一致。
PI05 使用 200-token 提示，将 q01/q99 归一化后的状态离散编码进文本；训练由
PI05Trainer、部署由公共加载器按 PI05 方法选择 quantile 模式，评估使用该运行保存的合并统计。
不使用 SO101 的状态/动作字段，也不把 PI0 的均值/标准差变换直接套到 PI05。

微调范围、学习率、weight decay、BS1×GA32、120 epoch、预热 4 epoch、每轮
250 次更新和 `--compile` 均与对应 PI0 实验一致。学习率是首轮实验起点，并非
已验证最优的 PI05 配方。基座采用选择性 BF16，LoRA adapter 保留 FP32。
每份脚本顶部的 `GPU_ID` 通过各条命令的 `--gpu` 选择设备；默认 seed 319。

评估使用 `--method PI05`、各自完整训练目录和独立的 `Eval/PI05_*` 输出目录，
默认 final checkpoint、4 个仿真进程、四套件每任务 50 次，并保存拼接视频。
修改训练的 batch/梯度累计/epoch 或 LoRA 参数时，需同步评估目录。
普通 PI05 VLAP 仅参数、梯度与动量已约 **25.27 GiB**，无法直接在 24GB 单卡
运行；应选择足够显存的设备，或编辑脚本保留适用实验。脚本 `set -e` 会在首个
失败命令处停止。配置范围及参数详情见 [PI05 文档](../../methods/PI05/README.md)。

## 评估设备选择

`run_eval.py` 支持与训练入口相同写法的单设备 `--gpu`，省略参数时默认选择 GPU 0。
根目录训练/评估脚本统一用顶部的 `GPU_ID` 变量，在每条命令中明确传入
`--gpu "${GPU_ID}"`，不在 shell 中导出固定的 CUDA 可见设备列表。

| 评估参数 | 策略运行设备 |
| --- | --- |
| 省略设备参数 | 默认物理 GPU 0，模型内部为 `cuda:0` |
| `--gpu 1` | 物理 GPU 1；入口设置 `CUDA_VISIBLE_DEVICES=1` 后内部为 `cuda:0` |
| `--gpu -1` | CPU；不修改仿真渲染的 GPU 环境 |
| `--device cpu` / `--device mps` / `--device cuda:N` | 使用明确的 PyTorch 设备；不改已有可见设备环境，CUDA 的 N 是可见列表中的逻辑编号 |

GPU 可见性在 Python 内、随机种子初始化和模型加载之前设置。`--gpu` 与 `--device`
互斥；评估只有一份模型，不支持 `--gpu 0,1` 或 `--gpu all`，也不是 DDP 推理。
指定 CUDA `--gpu N` 时，入口也为仿真子进程设置 `MUJOCO_EGL_DEVICE_ID=N`，
让 EGL 渲染选择对应的物理 GPU；显式 `--device` 或 CPU 策略模式保留已有渲染设置。
`--workers 4` 创建的是四个仿真进程，它们共享一个策略服务，不会复制四份模型。
CPU 策略模式也不自动切换为 CPU 渲染，原有 EGL/渲染 GPU 设置继续生效。

训练端可按其 running args 使用多 GPU；这一能力不表示评估端也接受多 GPU 列表。
需要把脚本中的实验放到 GPU 1 时，将其顶部变量改为 `GPU_ID=1`；只改变一条命令
也可把该命令的变量引用替换为明确的 `--gpu 1`。
显式的 `--device` 方式适合已有 CUDA 可见列表或 CPU/MPS 调试，不能同时再传 `--gpu`。

## 顺序写训练和评估命令

`train_eval_smolvla.sh` 直接列出原生训练命令，后面紧接评估命令。打开脚本修改
`--dataset_dir`、`--config_file`、`--method_name` 等参数即可；多组实验继续往下追加
各自的训练与评估命令，按书写顺序执行。

```bash
cd /media/wlty_ssd/RoboVLA
bash ./train_eval_smolvla.sh
```

根目录脚本使用 Bash 的 `pipefail`、`source` 等语法，请使用 `bash` 启动，并在
项目根目录执行。

脚本会激活 `wrobo` 环境，训练使用普通的文件入口：

```bash
PYTHONPATH=. python wrobo/methods/SmolVLA/train.py \
    --dataset_dir /path/to/converted/libero_spatial \
    --log_dir ./Logs \
    --config_file ./Config/SmolVLA_LIBERO_AP.yaml \
    --method_name SmolVLA_LIBERO_spatial_seed319 \
    --fold all \
    --num_epoch 120 \
    --warmup_epoch 4 \
    --batch_size 1 \
    --gradient_accumulation_steps 32 \
    --seed 319 \
    --merge_norm_stats

PYTHONPATH=. python wrobo/sim_envs/LIBERO/run_eval.py \
    --log-dir "./Logs/SmolVLA_LIBERO_spatial_seed319/<run_name>/default/fold_all" \
    --method SMOLVLA \
    --libero-env libero \
    --device cuda:0 \
    --suites libero_spatial \
    --workers 4 \
    --num-trials-per-task 50 \
    --replan-steps 5 \
    --seed 319 \
    --save-video \
    --output-dir ./Logs/SmolVLA_LIBERO_spatial_seed319_eval
```

这里的训练和评估是一组实验。要继续训练其他模型或参数组合，复制这两条命令并修改
对应参数；每组使用独立的 `--method_name`，评估的 `--log-dir` 指向实际训练输出目录，
`--output-dir` 指向该组评估结果目录。切换模型时同时修改训练脚本、配置及评估的
`--method`，评估输入输出仍须满足 LIBERO 约定。

`--log-dir` 必须填写 Trainer 打印的完整输出目录，示例中的 `<run_name>` 需要替换。
目录结构为 `<训练log_dir>/<method_name>/<run_name>/<参数设置>/fold_<fold>/`；
`run_name` 按顺序包含 Trainer 名、`BS`、`GA`、`EPOCH`、`SEED`、`PRETRAINED`、`SAMPLE`；
梯度累计次数 `GA` 紧接 batch size `BS`，例如 `__BS_1__GA_32__EPOCH_120`，
不包含数据选择哈希。不同数据或配置的实验使用不同的 `--method_name`；
数据选择仍记录在 `norm_stats_selection.json` 和训练数据快照中。
SmolVLA 不使用 LoRA 时，参数设置目录为 `default`。修改训练参数后，应同步修改评估路径。

`run_eval.py` 在当前模型环境加载权重，并自动调用 `libero` 环境执行仿真，结束时
释放模型、服务和子进程，无需在 shell 中手动启动模型服务。它只读取指定目录中的
`checkpoint_final.pth`、`norm_stats.json` 和 `Config_and_Trainer/` 中保存的 YAML，
不递归搜索实验父目录或选择其他运行的权重。
`--checkpoint-name` 可选择其他明确的权重文件，默认选择 `checkpoint_final.pth`。
`--save-video` 启用评估视频保存；根目录 `train_eval_smolvla.sh` 已开启此选项，
从评估命令中删除它即可关闭。直接调用评估入口时，默认不保存视频。

脚本中是四套件联合训练，训练预算直接由命令中的 epoch、batch 和梯度累计参数决定，
每任务评估局数由 `--num-trials-per-task` 设置。直接修改命令即可改变预算。训练保留现有 Trainer 的数据
划分逻辑；配置的冻结设置、学习率和模型结构也按 YAML 执行。

项目根目录的 `./train_eval_smolvla.sh` 包含八组已启用的 LoRA 训练与评估命令，
采用第一轮 Attention Q/K/V/O 扩展，分别比较 VLAP、AP 在 rank/alpha 16、32、64、128 下的表现：

| 执行顺序 | rank / alpha | 微调范围 | 配置 | batch / 梯度累计 |
| --- | --- | --- | --- | --- |
| 1、3、5、7 | 16、32、64、128 | LoRA Vision、Language、Action、Projection | `Config/SmolVLA_LIBERO_VLAP_LoRA.yaml` | 16 / 2 |
| 2、4、6、8 | 16、32、64、128 | LoRA Action、Projection | `Config/SmolVLA_LIBERO_AP_LoRA.yaml` | 32 / 1 |

八组均为有效 batch 32、120 epoch、4 epoch 预热、种子 319。
根目录脚本只显式写必填项和非默认参数。rank 16 使用默认的 `rank=16`、`alpha=16`，
其余三档显式设置相同的 rank 和 alpha。各组使用默认的 `dropout=0.0` 和 `merged` 保存方式；
VLAP 用 `--lora_parts all`，AP 使用默认的
Action、Projection 适配范围。基础权重保持冻结，YAML 中的冻结标志不会禁止所选
LoRA 适配器训练。脚本通过 `GPU_ID` 和各条命令的 `--gpu` 选择训练及评估显卡。
评估使用默认种子 319、`libero` 环境、每任务 50 局、
每次执行 5 步动作，并读取本组默认的 `checkpoint_final.pth`。
`--save-video` 显式启用拼接视频保存，`--workers 4` 使用四个仿真进程共享一份模型。
各组使用独立的训练和评估目录，名称包含 `LoRA_QKVO`，参数设置目录新增
`_attn_qkvo` 后缀，避免与此前 Q/V 实验混用。
在项目根目录运行 `bash ./train_eval_smolvla.sh`，按 rank 从小到大、每档先 VLAP 再 AP
依次执行八组实验，每组训练后立即评估。所有命令直接写在脚本中；需要跳过某组时，
同时注释该组训练与评估命令，新增实验直接添加对应命令。
默认 LoRA rank/alpha 仍是 16/16，Attention 扩展本身不改变优化器、学习率或训练预算。

模型参数按所属模块归入以下四部分；基础权重与附着其上的 LoRA 适配器归属一致：

| 部分 | 参数范围 |
| --- | --- |
| Vision | 视觉编码器和视觉到语言的 connector |
| Language | 词嵌入、语言层、语言最终 norm 和语言输出头 |
| Action | Action Expert，包括其最终 norm |
| Projection | 状态投影、动作输入/输出投影、动作与时间融合的 MLP |

普通微调中，最后一个向 Action Expert 提供 K/V 的语言层，其 input norm、K/V
仍影响动作预测，随 `freeze_language_encoder` 控制：`False` 时训练，`True` 时冻结。
该层仅服务于最终语言输出的 Q/O/MLP、语言输出头和语言最终 norm 保持冻结；
其后不再影响 Action Expert 的语言层也保持冻结。这个边界由实际语言层与 Expert
层的映射确定，不固定假设为最后一层或最后两层。

本轮扩大所有选中的 Vision、Language、Action Expert Attention：Vision 使用
`q_proj/k_proj/v_proj/out_proj`，Action Expert 使用 `q_proj/k_proj/v_proj/o_proj`。
选择 Language LoRA 时，前面的有效语言层使用 Q/K/V/O，最后一个与 Action Expert
交互的语言层仍只适配 K/V，不给该层 Q/O 或后续无动作依赖的语言尾层添加适配器。
Transformer FFN 不加入 LoRA；Vision connector 和五个状态/动作/时间投影保持原有适配范围。
常规 LoRA 只给所选线性矩阵添加低秩适配器，所有基础权重（包括 norm）保持冻结。
末层 input norm 仍归入 Language 的参数统计，但在 LoRA 模式下不训练。
未选择 Language 的 AP LoRA 不添加任何 Language 适配器。

启动日志保留模型总参数统计，并分别显示 Vision、Language、Action、Projection 的
总参数、可训练参数和冻结参数，便于检查每组实际训练范围。未用于动作损失的参数仍
计入总量，因此可训练比例不必达到 100%。冻结规则细化不改变已保存权重的形状；
旧普通 VLAP 完整断点中的优化器参数列表与新规则不同，应使用其权重初始化新运行，
重新创建优化器。
此前 Q/V 范围 adapter 缺少这轮新增的 K/O 适配器，不能直接按新目标续训；
目标或形状不匹配会报错，已有 merged 权重仍可正常评估。

只评估已有模型时，直接调用上面的 `run_eval.py` 命令。
检查 shell 语法可用 `bash -n ./train_eval_smolvla.sh`。
下面仍保留模型服务和仿真评估的分步命令，便于单独调试。

## 环境分工

在项目根目录 `/media/wlty_ssd/RoboVLA` 启动两个进程：

| 进程 | conda 环境 | 职责 |
| --- | --- | --- |
| `policy_server` | `wrobo`，Python 3.12 | 加载训练产物，处理图像与语言，归一化状态、推理和逆归一化动作 |
| `eval` | `libero`，Python 3.8 | 创建仿真环境，构造观测，执行动作，保存成功率和可选视频 |

两端通过 HTTP 传输 JSON 与数值 NPZ。仿真端不导入训练框架或现代 LeRobot 包。模型端复用 `wrobo.real_envs.utils.policy` 的模型 I/O，不连接真实机器人。

LIBERO、robosuite、MuJoCo 应已安装在 `libero` 环境。评估使用 NumPy 和 tqdm；启用视频时还需要 imageio 及可用的 FFmpeg 后端（例如 imageio-ffmpeg）。无显示器渲染使用 `MUJOCO_GL=egl`。

## 训练产物与输入约定

服务端需要同一次训练保存的以下文件：

- 权重文件，例如 `checkpoint_final.pth` 或实际保存的其他 checkpoint。
- `Config_and_Trainer/` 中复制的训练 YAML。
- 训练目录中的 `norm_stats.json`。
- `Config_and_Trainer/` 中保存的模型和 tokenizer 附件；LoRA 模式还需要相应的 LoRA 配置，adapter 权重需要可访问的原始基础模型。

`--config-file` 应指向训练保存的 YAML，使其所在目录同时包含模型附件。不要把当前工作区中修改过的配置与旧 checkpoint 混用。

当前 `Config/SmolVLA_LIBERO_AP.yaml` 采用以下输入：

| 数据字段 | 在线观测来源 | 维度或处理 |
| --- | --- | --- |
| `observations/image_top` | `agentview_image` | RGB，256×256，原始仿真图像旋转 180° |
| `observations/image_wrist` | `robot0_eye_in_hand_image` | RGB，256×256，原始仿真图像旋转 180° |
| `observations/proprio_ee` | 末端位置、旋转向量、夹爪位置 | 3 + 3 + 2 = 8 |
| `action_rel` | OSC_POSE 末端增量与夹爪命令 | 6 + 1 = 7 |
| `instruction` | 官方任务语言描述 | 字符串 |

评估器还支持转换格式中的 `proprio_joint`（关节位置 7 维 + 夹爪位置 2 维）和 `proprio_states`（夹爪位置 2 维 + 末端位置 3 维 + xyzw 四元数 4 维）。具体状态表示由训练 YAML 的 `data_keys` 决定，必须与训练一致。当前评估支持 `history_width=1` 和 `action_rel`，不将绝对动作转换为相对动作。

图像仅在仿真观测适配时旋转一次。模型端复用 Trainer 的验证图像变换和模型内部 resize，不增加额外 ImageNet 标准化。状态和动作使用训练选择的归一化统计，动作逆归一化后按环境动作范围裁剪，再交给 OSC_POSE；不额外反转夹爪命令。

当前配置从 `lerobot/smolvla_base` 完整加载预训练权重，但 `freeze_language_encoder: True`，因此实际微调会冻结语言编码器。完整加载权重与全参数微调是两个不同设置；实验记录应注明冻结设置或 LoRA 配置。

## 启动模型服务

第一个终端：

```bash
conda activate wrobo
cd /media/wlty_ssd/RoboVLA

python -m wrobo.sim_envs.LIBERO.policy_server \
  --host 127.0.0.1 \
  --port 8000 \
  --method SMOLVLA \
  --checkpoint /path/to/run/checkpoint.pth \
  --config-file /path/to/run/Config_and_Trainer/SmolVLA_LIBERO_AP.yaml \
  --norm-stats-key /path/to/run/norm_stats.json \
  --device cuda:0 \
  --seed 319
```

把 `/path/to/run/...` 替换为真实训练产物路径。可选 `--amp` 开启 CUDA 混合精度，`--compile` 开启推理编译。评估参数应随实验结果记录。

单数据集或 `--merge_norm_stats` 训练得到的平铺统计无需 `--norm-stats-group`。若联合训练保留了每个数据集各自的统计，使用 `--norm-stats-group` 指定 JSON 中的精确组名，并为对应套件使用该组。四套件联合训练建议使用 `--merge_norm_stats`，以便一个模型服务使用同一份统计评估四套件。

## 先评估一个任务

第二个终端：

```bash
conda activate libero
cd /media/wlty_ssd/RoboVLA
export MUJOCO_GL=egl

python -m wrobo.sim_envs.LIBERO.eval \
  --server-url http://127.0.0.1:8000 \
  --suites libero_spatial \
  --task-ids 0 \
  --num-trials-per-task 2 \
  --replan-steps 5 \
  --seed 319 \
  --output-dir ./Logs/libero_eval/spatial_task0_probe \
  --save-video
```

`--task-ids` 只允许在选择单个 suite 时使用，表示官方 benchmark 固定任务顺序中的局部索引。它不是转换后 `instruction_library.json` 中的 task_id。检查任务名称、视频中的视角、机器人动作以及日志中的状态和动作维度。

输出目录须不存在或为空。每次评估使用独立输出目录，避免混合不同 checkpoint 或协议的结果。视频默认不保存，`--save-video` 才启用。

## 正式四套件评估

```bash
python -m wrobo.sim_envs.LIBERO.eval \
  --server-url http://127.0.0.1:8000 \
  --suites libero_spatial libero_object libero_goal libero_10 \
  --num-trials-per-task 50 \
  --replan-steps 5 \
  --seed 319 \
  --output-dir ./Logs/libero_eval/smolvla_four_suites
```

四套件各 10 个任务，每任务 50 局，共 2000 局。`libero_90` 也可单独选择，但不包含在这四套件平均值的默认实验范围内。

## 多进程评估和一致性对照

`run_eval.py` 和独立仿真入口均支持 `--workers N`，默认 `1`。例如在现有评估命令中
添加 `--workers 4`，由 4 个仿真进程共享同一份 GPU 模型。子进程通过 `spawn` 启动，
各自创建 MuJoCo/EGL 环境、动作队列和视频 writer。每个任务的全部 trials 由同一
worker 执行，以复用环境；因此只选择一个任务时，实际只有一个 worker 有工作。
主进程统一更新动态进度条、写入逐局记录并汇总成功率。

模型请求仍串行执行。每个客户端有独立推理会话，保存并恢复 Python、NumPy、
Torch CPU/CUDA 的完整随机状态。每局只按原有规则设种子一次，随后继续该局自己的
随机序列；worker 数量和请求交错不会互相重置随机状态。官方初始状态、环境种子、
动作执行长度和成功判定不随 worker 数量改变。模型端与仿真端需使用同一版代码，
会话协议为 `wrobo_libero_http_v2`。

每个任务在构造环境前固定 Python/NumPy 种子为 319。仿真解释器启动时还会固定
`PYTHONHASHSEED` 为 `--seed`，保证父进程与 spawn 子进程的哈希顺序一致，减少
环境构造顺序的不确定性。`run_eval.py` 自动设置；独立 `python -m ...eval`
入口必要时自动重启一次解释器。直接从 Python 调用 `evaluate()` 时，调用方需在
启动解释器前设置该变量。协议中记录这两个设置，便于对照。

`summary.json` 记录 `protocol.workers` 和端到端仿真耗时 `elapsed_seconds`。
`episodes.jsonl` 最终按选择的 suite、task、初始状态顺序排列；失败或中断时保留
已执行记录。每局额外记录 `action_sha256` 和 `observation_sha256`，分别覆盖实际
执行的裁剪动作，以及每个控制步骤和终帧的观测（包括未重新请求模型的步骤）。浮点状态和动作按 float32
统一编码，图像按旋转后的 RGB 数据编码，便于逐局比较。

对同一个 checkpoint、任务列表、初始状态和种子分别运行 `--workers 1/2/4`，
使用不同的输出目录，然后执行：

```bash
python -m wrobo.sim_envs.LIBERO.compare_workers \
  --evaluation-dirs ./Eval/check_w1 ./Eval/check_w2 ./Eval/check_w4 \
  --compare-videos \
  --output ./Eval/worker_comparison.json
```

工具比较功能协议、模型元数据、种子、动作与观测摘要、步数、成功结果和汇总；
`--compare-videos` 还会比较解码后的每一帧。运行耗时与 worker 数量不参与一致性判定。
全部匹配返回 0，存在差异返回 1，不完整或缺少摘要的输入返回 2。评估加速取决于
仿真、渲染与模型推理的耗时占比，不保证随进程数线性增长。

2026-10-31 的真实 SmolVLA 对照使用同一份 2 epoch checkpoint、Spatial 任务 0–3、
每任务第一个官方初始状态、种子 319、每局 220 步、每 5 步重规划，并保存视频。
模型只加载一次，测试未控制 GPU 独占条件。耗时包括环境构造、渲染、
推理等待和视频保存，不包括模型加载，不能视为独占 GPU 的吞吐基准。

| workers | 4 局耗时 | 相对加速 | 执行动作、结果、解码视频 | 全部原始观测摘要 |
| --- | --- | --- | --- | --- |
| 1 | 81.02 s | 1.00× | 参照 | 参照 |
| 2 | 51.28 s | 1.58× | 一致 | task 0 存在差异 |
| 4 | 49.08 s | 1.65× | 一致 | 一致 |

workers=2 的差异位于 task 0 第 31 个控制步骤的腕部帧，该步没有请求模型。
逐步诊断确认每组实际送入模型的 176 次观测、176 个完整预测块及 884 个状态
输入数组逐字节一致；这里的状态是配置选用的 8 维本体状态，不代表完整仿真状态。
因此不能将本次对照报告为“全部逐像素一致”；严格比较的 `passed` 保留为 `false`。
每组成功局数均为 0/4，此实验只检验执行一致性，不代表四套件基线成功率。
完整协议、逐局摘要、视频和诊断见
[`Eval/LIBERO_workers_comparison_h9boy14u`](../../../Eval/LIBERO_workers_comparison_h9boy14u)。
其他硬件、驱动、checkpoint、AMP 或编译设置应分别验证，不能由本次小样本推出
所有环境的逐位一致性。

| 协议项 | 设置 |
| --- | --- |
| 任务顺序 | 官方 benchmark 的固定任务顺序 |
| 初始状态 | `get_task_init_states()` 提供的官方初始状态，按索引选择 |
| 环境种子 | 固定为 319 |
| 稳定步骤 | 每局初始化后执行 10 步 `[0,0,0,0,0,0,-1]` |
| 图像 | 两路 256×256，各旋转 180° 一次 |
| Spatial 动作步数上限 | 220 |
| Object 动作步数上限 | 280 |
| Goal 动作步数上限 | 300 |
| Long（`libero_10`）动作步数上限 | 520 |
| `libero_90` 动作步数上限 | 400 |
| 成功条件 | 每步检查任务成功，首次成功结束该局；超过上限记失败 |

稳定步骤不计入策略动作步数上限。每局清空动作计划，并通过服务端重置模型随机种子。`--seed` 默认 319，模型种子由 `numpy.SeedSequence([seed, suite_index, task_id, init_state_index])` 派生，因此选择任务子集不会改变同一任务和初始状态的种子；评估种子、每局模型种子及初始状态索引记录在结果中。环境保持同步执行，推理耗时不会自动推进仿真。

三个参数分别控制不同事情：

- `action_chunk_size=50`：一次预测输出 50 个动作，由训练配置决定。
- `--replan-steps 5`：只执行预测的前 5 个动作，再使用新观测预测；其余动作丢弃。
- `num_inference_steps=10`：模型内部 Flow Matching 数值积分的步数，由模型配置决定。

`--replan-steps` 不能超过模型预测长度。正式比较时固定执行长度、图像处理、初始化方式和训练数据版本。

## 结果

评估进度条随终端当前宽度动态调整，缩放窗口后会在下一次刷新时采用新宽度。
每局完成时只刷新一次，后缀用 `success=7/10 (70.0%)` 同时显示成功局数、完成局数
和成功率。详细结果仍保存在下列 JSON 文件中。

输出包含 `summary.json`、逐局 `episodes.jsonl`，以及启用时的 `videos/<suite>/task_XX_trial_XXX.mp4`。汇总记录任务与套件成功率、整体平均值、模型元数据和评估协议。逐局记录任务、官方初始状态索引、种子、步数、推理耗时、动作裁剪数量和成功情况。

启用 `--save-video` 后，每局评估保存一个 MP4，包括成功和失败的回合。
两个相机视角在每个时间步横向拼接：左侧为主视角 `agentview`（`image_top`），
右侧为腕部视角 `robot0_eye_in_hand`（`image_wrist`）。每路图像先按训练数据约定旋转
180°，再拼接为宽 512、高 256 的 RGB 视频，以 20 FPS 保存。
录像从初始化稳定后的首个策略观测开始，包含最后一步动作执行后的画面。
路径相对于 `--output-dir`，并记录在对应的 `episodes.jsonl` 条目中。

通信、仿真或视频写入发生错误时评估会中止，并保存已有记录和错误信息。出错局不会从已执行局数中删除；未完成评估的 `status` 为 `failed` 或 `interrupted`，`overall.success_rate` 为 `null`，已有局的成功率单独放在 `overall.observed_success_rate`，不能当作完整基线结果。修复问题后使用新的空目录重新评估。

训练验证损失与这里的闭环成功率分别报告。保存 checkpoint 标识、训练数据转换设置、冻结或 LoRA 设置、训练预算和种子；多个训练种子各自评估后再汇总均值与标准差。

环境初始化、稳定步骤、动作步数上限及图像方向参考 [OpenVLA LIBERO 评估](https://github.com/openvla/openvla/blob/main/experiments/robot/libero/run_libero_eval.py)；动作分块后执行前 5 步再重新规划参考 [OpenPI LIBERO 评估](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/main.py)。这定义了 wrobo 的评估协议，不等同于复现其他项目的训练配置。
