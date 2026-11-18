# 可运行工具

在仓库根目录通过 `python -m wrobo.tools.<工具名>` 运行。数据读写、图像编码和属性校验等共用模块位于 `wrobo/utils/`。

| 工具 | 命令示例 | 用途 |
| --- | --- | --- |
| `add_hdf5_metadata` | `python -m wrobo.tools.add_hdf5_metadata Dataset/my_dataset --attr "success=true" --dry-run` | 添加、修改或删除 HDF5 属性 |
| `repair_hdf5_metadata` | `python -m wrobo.tools.repair_hdf5_metadata Dataset/SO101_pre --dry-run` | 修复采集时因命令行引号错误产生的属性 |
| `trim_wrobo_dataset` | `python -m wrobo.tools.trim_wrobo_dataset --src Dataset/my_dataset --dry-run` | 预览首尾静态帧裁剪，确认后使用 `--dst` 输出新数据集 |
| `summarize_trajectories` | `python -m wrobo.tools.summarize_trajectories Dataset/my_dataset` | 统计 `episodes/` 下每条轨迹及合计的帧数、时长，写入 `trajectory_info.json` |
| `vis_trajectory` | `python -m wrobo.tools.vis_trajectory` | 打印轨迹结构并播放指定相机图像 |

`vis_trajectory` 的轨迹文件路径、相机数据集和播放延时目前在脚本末尾的 `__main__` 中设置。运行前按实际文件修改。

`summarize_trajectories` 接受数据集根目录或其 `episodes/` 目录，只统计其中直接存放的 `.hdf5` / `.h5` 文件。输出的 `episodes` 保存各文件的 `frame_count`、`duration_seconds` 和 `duration_source`；`summary` 保存 `episode_count`、`total_frames` 和 `total_duration_seconds`。帧数根据数组第一维计算，并核对 `seq_len` 及各帧序列长度；不加载或解码图像。

时长以秒为单位，表示首帧到末帧的采样跨度：优先使用 `timestamp[-1] - timestamp[0]`（`timestamp_span`），没有时间戳时使用 `(帧数 - 1) / FPS`（`fps_estimate`）。空轨迹和单帧轨迹的跨度为 0。缺少时间戳及 `FPS` / `fps` 属性时，可传入 `--fps 30` 作为估算频率；`--dry-run` 只打印结果。文件不存在时自动创建，重跑时更新统计并保留其他顶级字段；遇到轨迹格式错误时停止，不写入部分统计。

属性操作详见 [HDF5 属性指南](../documents/edit_hdf5_attrs.md)；裁剪参数详见 [裁剪指南](../documents/trim_wrobo_dataset.md)。裁剪算法与命令行入口合并在 `trim_wrobo_dataset.py` 中，转换器也直接导入该模块复用算法。
