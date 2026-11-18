# Dataset Format Conversion Tools

This directory contains tools to convert various robot datasets into the wrobo HDF5 format.

## Dataset Structure

After conversion, your dataset should follow this structure:

```
dataset_name/
├── episodes/                    # All trajectory files
│   ├── episode_0.hdf5
│   ├── episode_1.hdf5
│   ├── episode_2.hdf5
│   └── ...
├── dataset_split.json           # Train/val/test split
├── instruction_library.json     # (Optional) Instruction templates and library
└── stats.json                   # (Optional) Normalization statistics
```

### HDF5 Episode File Format

Each episode file contains:
```
episode_X.hdf5
├── attrs: {"dataset": "...", "seq_len": T, "success": True, "instruction": '{"episode":1}', ...}
├── observations/
│   ├── image_wrist: [T, C, H, W] uint8
│   ├── image_top: [T, C, H, W] uint8
│   ├── proprio_state: [T, D] float32
│   └── ...
├── action_abs: [T, D] float32
├── action_rel: [T, D] float32
└── ...
```

### Instruction Format

The HDF5 `instruction` attribute must be a **JSON string** with `episode`
and/or `frames`. These are independent annotations, not an override hierarchy.
Only this format is supported; the old scope-string format is rejected.

```json
{
  "episode": {"task_id": 0, "params": {"object": "blue cube"}},
  "frames": [
    {"start": 0, "end": 40, "task_id": 1},
    {"start": 41, "end": 100, "task_id": 2}
  ]
}
```

- `episode`: overall task; an integer for a library task or a task object.
- `frames`: list of subtask annotations with inclusive integer `start`/`end`,
  `task_id`, and optional template `params`. Ranges must not overlap.
- Either level can be omitted; at least one annotation is required.
- IDs and parameters must match `instruction_library.json`.
- Frame coverage requirements depend on the training mode (see below).
- Store with `root.attrs["instruction"] = json.dumps(metadata)`; do not store a
  Python dict directly in an HDF5 attribute.

## Conversion Scripts

### 1. Convert from Aloha Format

```bash
python convert_Aloha2wrobo.py \
    /path/to/aloha/dataset \
    /path/to/output/dataset \
    --camera-names cam_high cam_left_wrist cam_right_wrist
```

Creates:
- `output/dataset/episodes/episode_*.hdf5`
- `output/dataset/dataset_split.json`

### 2. Convert from LeRobot Format

```bash
python convert_Lerobot2wrobo.py \
    --src /path/to/lerobot/dataset \
    --dst /path/to/output/dataset
```

Creates:
- `output/dataset/episodes/episode_*.hdf5`

To add another LeRobot dataset to the same output directory, pass `--append`:

```bash
python convert_Lerobot2wrobo.py \
    --src /path/to/another/lerobot/dataset \
    --dst /path/to/output/dataset \
    --append
```

New files start after the highest existing `episode_N.hdf5` number in
`output/dataset/episodes/` and use consecutive numbers, even when `--episodes`
selects sparse source indices. For example, if the highest existing file is
`episode_99.hdf5`, the next files are `episode_100.hdf5`, `episode_101.hdf5`, etc.
An empty output directory starts at `episode_0.hdf5`. Existing files are preserved;
if a new output filename already exists when a worker writes it, that conversion
fails instead of overwriting the file. Without `--append`, output filenames use
the source episode indices and existing files with the same names are overwritten.
This converter writes episode files only; it does not update `dataset_split.json`.

### 3. Add Instruction Metadata

Run these examples from the repository root. All attributes use `--attr KEY=VALUE`;
quote the complete assignment with single quotes in Bash. The same supplied annotation **replaces the entire
instruction attribute** in each selected episode; it is not merged or appended.

```bash
# Overall task only, using template parameters
python3 wrobo/tools/add_hdf5_metadata.py /path/to/dataset --attr 'instruction={"episode":{"task_id":0,"params":{"object":"blue cube"}}}'

# Overall library task only
python3 wrobo/tools/add_hdf5_metadata.py /path/to/dataset --attr 'instruction={"episode":1}'

# Frame-level subtasks only
python3 wrobo/tools/add_hdf5_metadata.py /path/to/dataset --attr 'instruction={"frames":[{"start":0,"end":40,"task_id":1},{"start":41,"end":100,"task_id":2}]}'

# Overall task AND subtasks: preview without modifying files
python3 wrobo/tools/add_hdf5_metadata.py /path/to/dataset --attr 'instruction={"episode":{"task_id":0,"params":{"object":"blue cube"}},"frames":[{"start":0,"end":40,"task_id":1},{"start":41,"end":100,"task_id":2}]}' --dry-run

# Update selected episodes only
python3 wrobo/tools/add_hdf5_metadata.py /path/to/dataset --attr 'instruction={"episode":{"task_id":0,"params":{"object":"blue cube"}}}' --episodes 0,1,2
```

Remove `--dry-run` to apply the previewed change. Adjust frame bounds to the
actual episodes; the examples describe frames 0 through 100 inclusive.
Existing datasets are not automatically migrated.
For editing other attributes with the same tool, see
[the HDF5 attribute guide](../documents/edit_hdf5_attrs.md).

### 4. Trim Existing wrobo Datasets

For datasets already containing `episodes/*.hdf5`, use
`python -m wrobo.tools.trim_wrobo_dataset --src /path/to/dataset --dry-run`
from the repository root. Replace `--dry-run` with `--dst /path/to/new_dataset`
to write trimmed episodes without modifying the source.
See [the trimming guide](../documents/trim_wrobo_dataset.md) for options and metadata handling.

## Instruction Library Format

Create `instruction_library.json` in your dataset root:

```json
{
  "0": {
    "name": "pick_and_place",
    "type": "template",
    "slot_meaning": "object",
    "templates": [
      "Pick up the {object} and place it in the plate.",
      "Put the {object} into the plate."
    ]
  },
  "1": {
    "name": "grasp_object",
    "type": "library",
    "instructions": [
      "Grasp the block.",
      "Pick up the block."
    ]
  },
  "2": {
    "name": "place_object",
    "type": "library",
    "instructions": ["Place the block in the plate."]
  }
}
```

## Dataset Split Format

Training datasets need a `dataset_split.json` file, prepared before training or
created during training. K-fold uses folds `"0"` through `"4"`, each with train/val,
and has no test list. Fixed train/val/test splits repeat the same train/val lists
in all five folds and store test once, so mixed datasets share the same `--fold`
argument. The following fixed-split example omits folds `"1"` through `"4"`:

```json
{
  "0": {
    "train": ["episode_0", "episode_1", ...],
    "val": ["episode_10", "episode_11", ...]
  },
  "test": ["episode_20", "episode_21", ...]
}
```

Task selection does not change this format. Training can intersect each fold
with whole-episode task mappings using `--task_ids`, one selection per source:

```bash
--dataset_dir /data/dataset_a /data/dataset_b --task_ids 0,2 all
```

This is independent of episode/frame language modes and also works with language
inputs disabled. See [training_data_selection.md](../documents/training_data_selection.md)
for the mapping contract, unified statistics and strict cache format.

## Usage in Training

The `EpisodicDataset` class loads episodes from the `episodes/` subdirectory:

```python
from wrobo.training.dataset.episodic_dataset import EpisodicDataset

dataset = EpisodicDataset(
    dataset_dir="/path/to/dataset",  # Points to directory containing episodes/
    split="train",
    fold="0",
    data_keys=["observations/image_wrist", "action_abs"],
    history_width=1,
    action_chunk_size=10,
    norm_stats=stats,
)
```

The example above disables language: it loads the split and episodes but does
not load an instruction library, parse annotations, or use instruction modes.

To enable language, include `instruction` in Config's `Training_settings.data_keys`.
Supply `--instruction_mode` in the training running args, one entry per dataset:

```bash
--dataset_dir /path/to/dataset_a /path/to/dataset_b \
--instruction_mode episode_only episode_plus_frame
```

If omitted, every dataset defaults to `episode_only`. An explicitly supplied
list must match the dataset count. If `data_keys` omits `instruction`, all
instruction settings are ignored. This option is not an annotation-tool argument.

| Mode | Selected language |
| --- | --- |
| `episode_only` | Overall task |
| `frame_only` | Subtask covering the sampled frame |
| `episode_plus_frame` | Both overall task and current subtask |
| `frame_with_episode_fallback` | Current subtask, otherwise overall task |

Required missing annotations raise errors. In particular, `episode_plus_frame`
requires both annotation levels at the sampled frame.

For direct dataset construction, pass `instruction_mode=["episode_only"]`,
include `instruction` in `data_keys`, and provide the method's `text_transform`.
Token-based methods use fixed-length tokenization; PI05 preserves raw text here
and tokenizes later with the normalized state. Samples include processed `instruction_episode` and
`instruction_frame` fields plus their `_valid` flags. Unused channels contain
empty-text encodings. The existing model input `instruction` contains the
selected text; when both levels are selected they are joined before tokenization.

See [instruction_modes.md](../documents/instruction_modes.md) for details.

### Training log snapshots

Each fold's log directory contains one merged
`Config_and_Trainer/dataset_split.json`, grouped by dataset and recording only
the actual filtered train/val episode lists, per-source `task_ids`, and the
normalization mode for that fold. Other folds and test lists
are not copied. Each entry records its source path, instruction mode, and
library filename.

Language-enabled datasets keep separate complete libraries named
`dataset0_instruction_library.json`, `dataset1_instruction_library.json`, etc.
Single-dataset runs use the same naming. With language disabled, no libraries
are saved and the corresponding manifest fields are null.

This log snapshot differs from the source dataset split format above; source
split files are unchanged. See
[training_dataset_snapshot.md](../documents/training_dataset_snapshot.md).

## LIBERO 数据集转换

`convert_LIBERO2wrobo.py` 在 LIBERO 仿真环境中回放原始 HDF5 动作，重新生成
图像、本体状态和完整仿真状态，按 suite 保存为独立的 wrobo 数据集。
**默认始终重新渲染为 256×256，不是把原始 128×128 图像放大。**
默认保留成功和失败回放，也不删除 no-op 动作；两个过滤开关可独立启用。
原始 demo 的所有数组和属性另外完整归档，便于检查新旧数据的对应关系。
以下命令均从 RoboVLA 项目根目录执行，不覆盖既有转换结果。

### 原始数据包含哪些信息

五个 suite 为 `libero_spatial`、`libero_object`、`libero_goal`、`libero_90`
和 `libero_10`，分别包含 10、10、10、90、10 个任务 HDF5 文件，共 130 个文件。
每个文件有 `demo_0` 至 `demo_49`，合计 6500 条轨迹。文件 root 只有 `data`
group，root attrs 为空；原始任务、轨迹及字段结构如下：

```text
<task>_demo.hdf5
└── data/
    ├── attrs: bddl_file_name, env_args, env_name, macros_image_convention,
    │          num_demos, problem_info, tag, total
    ├── demo_0/
    │   ├── attrs: num_samples, model_file, init_state
    │   ├── obs/
    │   │   ├── agentview_rgb       uint8 [T,128,128,3]
    │   │   ├── eye_in_hand_rgb     uint8 [T,128,128,3]
    │   │   ├── joint_states       float64 [T,7]
    │   │   ├── gripper_states     float64 [T,2]
    │   │   ├── ee_states          float64 [T,6]
    │   │   ├── ee_pos             float64 [T,3]
    │   │   └── ee_ori             float64 [T,3]
    │   ├── actions                float64 [T,7]
    │   ├── robot_states           float64 [T,9]
    │   ├── states                 float64 [T,S]
    │   ├── rewards                uint8 [T]
    │   └── dones                  uint8 [T]
    ├── demo_1/
    └── ... demo_49/
```

`ee_states = [ee_pos, ee_ori]`，`ee_ori` 是三维 axis-angle。`robot_states`
的 9 维实际为 `[gripper_qpos(2), eef_pos(3), eef_quat_xyzw(4)]`，与
“7 个机械臂关节加 2 个夹爪关节”不同。`states` 是完整 MuJoCo 仿真状态，
其维度 `S` 随任务变化，不能直接解释成关节状态。

`env_args` 和 `problem_info` 是原始 JSON 字符串；前者包含机器人、相机、
控制频率和 OSC_POSE 控制器参数，后者包含任务语言
`language_instruction`。`model_file` 是环境模型 XML 字符串，`init_state`
是 float64 `[S]` 数组，`num_samples` 是轨迹长度。实际下载文件存有 BDDL
路径 `bddl_file_name`，没有 `bddl_file_content` 文本。原数据也没有独立的
`success`、`task_id` 或 timestamp 字段；新输出的成功标签来自实际仿真回放。

上述 schema 来自全部 130 文件的文件级检查，以及每文件首尾两条 demo 的头信息
抽查；转换时还会逐条轨迹校验长度、维度和冗余字段等式。

### 回放、成功判定与可选过滤

每条 demo 在当前 worker 内独立创建和关闭一个 LIBERO 环境，不跨 demo 或
worker 共享环境。环境随机种子统一为 `319`。转换时恢复源 `states[0]`，执行 10 次 dummy action 稳定仿真，
然后依次处理源动作。每个保留动作对应以下过程：

1. 在执行动作前读取当前 observation，重新渲染两路图像，同时记录三种 proprio
   表示及当前完整仿真状态。
2. 执行原始 OSC_POSE 相对动作，不重新缩放或改成绝对动作。
3. 保存这次 `env.step` 实际返回的 reward 和 done。原始 `dones` 不作为回放输出。

所有保留动作执行完后，以最终 `check_success()` 判断这条回放是否成功；
不会因为中途出现成功或 done 就提前截断轨迹。
此流程参考 [OpenVLA 的 LIBERO 数据重建脚本](https://github.com/openvla/openvla/blob/main/experiments/robot/libero/regenerate_libero_dataset.py)。
区别是 OpenVLA 脚本的首帧混用了原始 state；这里每一帧均保存当前回放环境的真实
动作前 observation/state，使图像、proprio、仿真状态与对应动作保持同一时间基准。
原始数据仍保存在归档中，不改写源文件。

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--filter-failed` | 不启用 | 丢弃最终回放失败的整条 demo；不启用时照常保存并记录实际成功状态。 |
| `--filter-noops` | 不启用 | 跳过符合下述规则的动作，不执行这些动作，也不保存对应新帧。 |
| `--image-size` | `256` | 环境实际渲染的正方形图像边长。 |
| `--workers` | `1` | `1` 为串行；大于 `1` 时使用指定数量的独立进程，以 `spawn` 启动。 |

no-op 判定使用源动作的前 6 维运动命令和夹爪命令：

```python
stationary = np.linalg.norm(action[:6]) < 1e-4
is_noop = stationary and (
    previous_kept_action is None
    or action[-1] == previous_kept_action[-1]
)
```

`previous_kept_action` 指上一条保留的动作。还没有保留动作时，只判断前 6 维范数；
之后还要求夹爪命令与上一条保留动作相同。因此机械臂静止但夹爪指令正在切换开合的
动作通常不会被视为 no-op。若启用过滤后整条 demo 的动作都被删除，空轨迹会被
丢弃并记录原因，不写出长度为 0 的 episode。两个过滤开关可以只开一个，也可以
同时开启；不开启时所有非空源 demo 都会尝试保存实际回放结果。

### 按 suite 独立输出与元数据

`--dst` 是输出总根目录，suite 名由转换器追加。无论 `--src` 指向 LIBERO
总根、某个 suite 目录还是其中一个任务文件，实际输出均为 `--dst/<suite>/`。
本次选择的 suite 目标目录必须不存在或为空；其他目录和文件保持原样。
转换器不覆盖、追加或自动迁移已有的非空 suite，应使用新的 `--dst`。

```text
LIBERO_rerender_wrobo/
├── libero_spatial/
│   ├── episodes/episode_*.hdf5
│   ├── data_description.json
│   └── instruction_library.json
├── libero_object/
├── libero_goal/
├── libero_90/
└── libero_10/
```

每个 suite 独立编号。原始输入共 6500 条 demo；实际输出数量取决于选中的任务和
启用的过滤条件，不能把计划数量当成最终写入数量。
`data_description.json` 使用 `schema_version: 5`，记录回放设置、实际成功数、
过滤数量与原因，以及每条 episode 的源长度、输出长度和保留的源动作索引。
所有源任务元数据保存在 `source_tasks`，`tasks` 仅列出有实际输出的任务。
`tasks[task_id].num_episodes/num_steps` 是该任务实际保存的轨迹数和总帧数，
`tasks[task_id].environment` 描述本次回放环境，图像分辨率使用实际输出设置。
顶层和任务的 `image_shapes` 都是解码后的 CHW 形状，默认两路均为 `[3,256,256]`。
原始 `data.attrs` 仅位于 `source_tasks[task_id].source_data_attributes`，
其中 `env_args` 和 `problem_info` 保持原始 JSON 字符串。因此原始 128 分辨率、
`num_demos=50` 仍会保留在源归档信息中，不代表新输出的规格或数量。
`source_tasks[task_id].num_episodes/num_steps` 则来自实际扫描到的源 demo，而不是照抄源属性。
任务到实际输出 episode 的映射保存在 `instruction_library.json`，不在描述文件中
重复登记任务的 episode 列表。

转换开始时实际输出计数为 0，源计划数量使用 `num_source_*`。每完成一个源任务或
间隔约 30 秒，描述文件刷新该 suite 的已完成结果；中间快照的 `output_counts_final`
为 `false`。只有完整转换结束才标记 `status: complete` 和 `output_counts_final: true`，
`num_pending_episodes` 表示尚未完成回放的源轨迹数。

已启动的旧转换进程继续使用启动时加载的代码，更新脚本不会改变其运行行为。
等本次转换全部完成后，可以只刷新描述文件，无需重新渲染或重写 HDF5：

```bash
python -m wrobo.dataset_format_convert.refresh_LIBERO_metadata \
  --dataset-dir /media/data/wlty/RoBoDataset/wrobo/LIBERO \
  --dry-run

python -m wrobo.dataset_format_convert.refresh_LIBERO_metadata \
  --dataset-dir /media/data/wlty/RoBoDataset/wrobo/LIBERO
```

工具也接受单个 suite 目录。它检查实际 episode 文件、图像属性、序列长度和指令库归属，
再更新实际计数与图像规格；原 JSON 备份为 `data_description.json.before_metadata_refresh.bak`。
`--dry-run` 不写文件。任意选中套件仍处于 `in_progress` 或文件归属不一致时会拒绝更新，
避免与正在运行的转换进程互相覆盖。

### 保存字段与源数据归档

模型输入字段保持原有命名，但 observation、仿真状态和反馈来自本次真实回放：

| 输出字段 | shape / 来源 |
| --- | --- |
| `observations/proprio_joint` | `[T,9]`，当前关节位置 7 + 当前夹爪位置 2 |
| `observations/proprio_ee` | `[T,8]`，当前末端位置 3 + axis-angle 3 + 夹爪位置 2 |
| `observations/proprio_states` | `[T,9]`，当前夹爪位置 2 + 末端位置 3 + xyzw 四元数 4 |
| `action_rel` | `[T,7]`，保留的源动作 `[delta_xyz(3), delta_axis_angle(3), gripper_command(1)]` |
| `observations/image_top` | 解码后 uint8 `[T,3,H,W]`，当前 agentview 渲染后旋转 180°；默认 `H=W=256` |
| `observations/image_wrist` | 解码后 uint8 `[T,3,H,W]`，当前 eye-in-hand 渲染后旋转 180°；默认 `H=W=256` |
| `libero/states` | `[T,S]`，每个保留动作执行前的当前完整仿真状态 |
| `rewards`、`dones` | `[T]`，各保留动作执行后 `env.step` 的真实返回值 |
| `libero/source_indices` | `[T]`，输出帧到源动作数组的索引映射 |
| `libero/source/demo` | 完整原始 demo：所有源数组、子 group 及各级 attrs，保持源值和源长度 |

数值状态保留 float64，`action_rel` 保留原动作精度和控制语义；Dataset 读取被选中
的数值输入时才转为 float32。`model_file`、`init_state` 存在 episode root attrs，
对应本次回放实际使用的模型和动作前初始状态；源属性完整保存在归档中。
root `seq_len` 是实际输出长度，原始 `num_samples` 保留在 `libero/source/demo`
的属性中。描述文件各 episode 的 `success` 和 HDF5 root 属性 `replay_success`
记录最终真实回放结果，不从原始 done 猜测。

`libero/source/demo` 是完整归档，包括原始 128×128 图像、原始 `states`、
`rewards`、`dones` 和所有本体状态数组；即使新输出删除部分帧，归档也不裁剪。
因此归档会占用额外磁盘空间，JPEG 只改变新渲染图像的编码，不压缩替换源归档。
需要比较新旧数据时，通过 `libero/source_indices` 查找对应源动作；新回放状态可能
因稳定步骤和 no-op 过滤而不同于源状态，不应要求它们逐数值相同。

### 图像编码与时间索引

两路主图像统一采用 OpenVLA 的 180° 方向协议：转换器将重新渲染图像的
height、width 两个轴同时反转，再转为 CHW 并按所选格式编码。单张 HWC 和
整条 THWC 图像数组的对应写法是：

```python
# HWC: 同时反转高度和宽度，旋转 180°。
img_rotated = img[::-1, ::-1].copy()

# THWC: 保持时间和通道轴不变，同时反转高度和宽度。
images_rotated = images[:, ::-1, ::-1, :].copy()
images_chw = images_rotated.transpose(0, 3, 1, 2)
```

只反转高度是上下翻转，并不等于旋转 180°。OpenVLA 原来在 HDF5 → RLDS 阶段
执行这项方向变换；wrobo 直接读取 HDF5，因此将它放在转换存储阶段完成，无需
引入 RLDS，也不增加 CLI 开关。`libero_replay` 仍返回原始 renderer 图像，
只有转换器写入的 `observations/image_top` 和 `observations/image_wrist` 应用旋转；
`libero/source/demo` 中归档的原始图像保持原值。

`data_description.json` 的 `image_orientation` 为 `rotate_180_from_renderer`，
`image_generation.rotation_degrees` 为 `180`；两路主图像 Dataset 的
`image_orientation` 属性也记录同一方向。数值 proprio、仿真状态、动作、reward、
done 和 `source_indices` 均不随图像旋转改变，不旋转动作或状态坐标。

训练读取的是已经旋转后的主图像，不再旋转。将来 LIBERO 在线评估若直接读取
原始环境 observation，两路图像必须执行同样的 180° 变换，并且只执行一次，
再进入与训练一致的预处理。避免环境和策略预处理重复执行方向变换。
已有输出和正在运行的旧版本转换不会因脚本更新而自动改变；重新生成时使用新的
目标目录，并确保同一数据集目录中的全部 episode 使用同一方向，不能混用新旧主图像。

`--img-encoding` 默认 `png`，可选 `png`、`jpeg`、`gzip`、`raw`：

| 编码 | 磁盘布局 | 解码结果 |
| --- | --- | --- |
| `png`（默认） | 长度为 `T` 的 vlen uint8 PNG 字节数组 | RGB uint8 `[T,C,H,W]`，保留新渲染像素 |
| `jpeg` | 长度为 `T` 的 vlen uint8 JPEG 字节数组 | RGB uint8 `[T,C,H,W]`，像素有损 |
| `gzip` | gzip 压缩的密集 uint8 `[T,C,H,W]` | 保留新渲染像素 |
| `raw` | 未压缩的密集 uint8 `[T,C,H,W]` | 保留新渲染像素 |

PNG/JPEG 的 HDF5 Dataset 形状为 `[T]`，wrobo 根据 `img_encoding` 逐帧解码。
`--jpeg-quality` 默认 `95`，范围为 `0–100`，仅 JPEG 使用；图像 attrs 和
`data_description.json` 记录实际编码、质量和是否无损。
这里的无损是相对于本次 renderer 输出旋转 180° 后的像素，不是声称新图像等于
原始 128×128 图像。
每帧 observation/state 对应动作执行前，reward/done 对应该动作执行后的反馈；
所有新数组与 `source_indices` 共享输出帧索引。

### 环境与转换命令

使用已安装 LIBERO 的环境：Python 3.8、robosuite 1.4、MuJoCo 3.2.3。
先激活环境；脚本默认使用 `MUJOCO_GL=egl`，需要其他渲染后端时可在启动前设置
`MUJOCO_GL`。从 RoboVLA 项目根目录执行：

```bash
conda activate libero
python -m wrobo.dataset_format_convert.convert_LIBERO2wrobo \
  --src /media/data/wlty/RoBoDataset/LIBERO \
  --dst /media/data/wlty/RoBoDataset/LIBERO_rerender_wrobo \
  --image-size 256 --workers 4
```

上例默认不启用失败过滤和 no-op 过滤。需要同时启用时使用独立输出根：

```bash
python -m wrobo.dataset_format_convert.convert_LIBERO2wrobo \
  --src /media/data/wlty/RoBoDataset/LIBERO \
  --dst /media/data/wlty/RoBoDataset/LIBERO_rerender_filtered_wrobo \
  --filter-failed --filter-noops \
  --image-size 256 --workers 4
```

先完整回放一个任务的所有 demo，并使用两个进程和 JPEG：

```bash
python -m wrobo.dataset_format_convert.convert_LIBERO2wrobo \
  --src /media/data/wlty/RoBoDataset/LIBERO/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5 \
  --dst /media/data/wlty/RoBoDataset/LIBERO_rerender_multiprocess_jpeg_wrobo \
  --workers 2 \
  --img-encoding jpeg --jpeg-quality 95
```

`--src` 也支持单个 suite 目录。使用 `--suites libero_spatial libero_goal` 可选择
多个 suite，它们仍独立输出。`--workers 1`（默认）串行处理，大于 `1` 时通过
`spawn` 启动独立进程。每条 demo 都在当前 worker 内创建环境，并在结束时关闭，
不共享环境。进程越多，对 CPU、内存和渲染资源的需求越高。

只检查源数据和转换计划，不创建输出、不启动仿真：

```bash
python -m wrobo.dataset_format_convert.convert_LIBERO2wrobo \
  --src /media/data/wlty/RoBoDataset/LIBERO \
  --dst /media/data/wlty/RoBoDataset/LIBERO_rerender_wrobo \
  --filter-failed --filter-noops --dry-run
```

`--dry-run` 不能预测回放成功数、失败过滤数或 no-op 过滤数。
tqdm 默认显示任务预检和 demo 处理进度；demo 总数为全部选中源 demo，成功写出、
回放失败后丢弃、过滤后变空等已处理结果都推进进度。主进程统一显示进度，worker
不各自创建进度条。

### 训练划分准备、语言库与读取选择

转换阶段不负责训练/验证/测试划分，也不生成 `dataset_split.json`。
各 suite 独立编号并维护各自的语言库，`libero_90` 与 `libero_10` 不混合。
训练时由用户根据实验方案准备划分；`EpisodicDataset` 读取仍需在具体 suite
目录中提供 `dataset_split.json`，格式见上方通用说明。训练阶段的划分策略应与
所选 LIBERO 实验或评估协议对应，本次转换不修改训练代码。

任务语言直接来自 `problem_info.language_instruction` 原文。每个 suite 内
按任务生成独立 task ID，同任务保留下来的 demo 共享该 ID；原数据没有这个整数 ID。
`instruction_library.json` 保持顶层 task ID 到任务条目的格式，每个条目同时保存
原文和该任务实际转换的 episode ID（不含 `.hdf5` 后缀），例如：

```json
{
  "0": {
    "name": "task_name",
    "type": "library",
    "instructions": ["原始任务指令"],
    "episodes": ["episode_0", "episode_1"]
  }
}
```

库中只包含实际保留了至少一个 episode 的任务，不登记被全部过滤的空任务；每个 episode 恰好
属于一个条目，因此完整输出的 `len(library)` 就是该 suite 的实际任务数。
为避免发布未生成轨迹的映射，指令库在全部 HDF5 写入成功后才保存；转换失败时
`data_description.json` 标记为 `failed`，不生成指令库。已有转换结果不会自动更新。
这个结构保留了现有语言读取接口，新增 `episodes` 不影响指令采样。

root `instruction` 属性仍保存结构化 JSON 字符串，例如
`{"episode":{"task_id":0}}`。完整原始语言及任务 metadata 同时
保留在 `data_description.json` 的 source 属性信息中；使用 `episode_only` 即可
读取这类 episode 级语言，不要求生成没有来源的帧级子任务标注。

Config 的 `data_keys` 选择已保存的字段。例如选择关节状态：

```yaml
Training_settings:
  data_keys:
    - observations/image_top
    - observations/image_wrist
    - observations/proprio_joint
    - action_rel
    - instruction
```

改用 `observations/proprio_ee` 或 `observations/proprio_states` 即可选择其余
状态表示，无需重新转换。Dataset 删除 `observations/` 前缀作为输出字典键，
因此上例得到 `proprio_joint` 和 `action_rel`，不会自动改名或拼接多个状态字段。
`history_width` 对所有选中的 observation 字段生效，包括图像和数值状态；
动作字段使用 `action_chunk_size` 的未来窗口。多选 proprio 字段会返回多个独立
Tensor。训练数据目录指向具体 suite，例如
`/media/data/wlty/RoBoDataset/LIBERO_rerender_wrobo/libero_spatial`。

ACT Trainer 会根据 `data_keys` 选择一个 `observations/proprio_*` 和一个根目录
`action_*` 字段；ACT 的状态与动作维度分别由 `Policy.proprio_dim` 和
`Policy.action_dim` 控制。例如上面的 `proprio_joint` 和 `action_rel` 对应：

```yaml
Policy:
  proprio_dim: 9
  action_dim: 7
```

若改用 `proprio_ee`，设置 `proprio_dim: 8`，动作仍为 7 维。旧配置未提供
`action_dim` 时，沿用 `proprio_dim`，已有同维度模型保持兼容。以上是需要合并进
完整 ACT 配置的字段；ACT 训练不需要语言输入，可以从 `data_keys` 中去掉
`instruction`。其他 Trainer 的字段适配需要分别检查；语言开启时需提供对应
方法的 `text_transform`。归一化统计应从训练 split 计算，改变输入字段后避免复用旧运行
的统计。转换检查应覆盖数值和 metadata 可恢复、图像按所选编码解码及描述文件
计数；PNG/gzip/raw 检查与新渲染图像旋转后的像素一致，JPEG 检查解码形状和质量 metadata。用户准备
划分后，再通过 `EpisodicDataset` 验证所选 episode 和字段的读取。格式检查不替代
仿真重放。
