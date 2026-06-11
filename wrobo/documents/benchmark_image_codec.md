# Image Codec Decode Benchmark

This document describes a benchmark measuring the **decode performance** of the three
image storage layouts (`GZIP`, `PNG` and `JPEG`) for RGB images used by our episode HDF5 files, evaluated over the
`sim_transfer_stack_cube` dataset (Aloha simulation, 50 episodes, ~1921 frames each,
4 cameras per frame: 2× 480×640 (`image_top` and `image_angle`) and 2× 240×320(`image_left_wrist` and `image_right_wrist`), RGB `uint8`).

## Experiment

### Goal

For a robot-learning dataloader, the cost that matters is the time to turn what is
stored on disk into a usable `(C, H, W)` `uint8` RGB frame. We compare three layouts
on exactly that metric, under both **sequential** and **random** frame access:

| Layout | Directory | On-disk form | "Decode" work measured |
| --- | --- | --- | --- |
| `GZIP` | `Dataset/ACT_wrobo/sim_transfer_stack_cube_raw` | dense `(T, 3, H, W)` `uint8`, gzip + shuffle + compression_opts=4, 1 frame/chunk | `ds[i]` (h5py read + gzip/shuffle decompress) |
| `PNG` | `Dataset/ACT_wrobo/sim_transfer_stack_cube` | per-frame PNG byte strings (vlen), 1 frame/chunk | `decode_frame(ds[i])` (read byte string + `cv2.imdecode`) |
| `JPEG (Q=95)` | `Dataset/ACT_wrobo/sim_transfer_stack_cube_JPG` | per-frame JPEG byte strings (vlen), 1 frame/chunk | `decode_frame(ds[i])` (read byte string + `cv2.imdecode`) |

The codec under test lives in [`image_codec.py`](/wrobo/utils/image_codec.py); the benchmark driver is
[`benchmark_image_codec.py`](/wrobo/utils/benchmark_image_codec.py).

### Method

- **Unit of measurement.** Wall-clock time to obtain one fully decoded `(C, H, W)`
  `uint8` frame. For `GZip`, read and decompress are fused inside h5py and cannot be
  separated, so for all three layouts we report the combined "time to a usable frame"
  rather than splitting read vs. decode — this keeps the comparison apples-to-apples.
- **Access patterns.** For each camera in each episode we sample the same set of frame
  indices, then visit them (a) in sorted order (`sequential`) and (b) in a fixed
  shuffled order (`random`). Identical work, different order — this isolates the access
  pattern's effect from the amount of work.
- **Cache handling.** By default each file is read once into the OS page cache before
  timing (`warm`), isolating decode/access CPU cost from disk-IO jitter and matching
  steady-state multi-epoch training where files are re-read. `--cold` drops the page
  cache before each file to capture cold-read behavior.
- **Disk size for reference** (whole dataset, 50 episodes): `GZIP` 14 GB, `PNG` 7.8 GB,
  `JPEG` 7.4 GB.

### Environment

| Item | Value |
| --- | --- |
| System | Ubuntu 24.04 |
| CPU | Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz |
| RAM | 64GiB RAM + 8GiB SWAP |
| Storage | 512GiB SSD (System) + 1T SSD (Code and Data) |
| GPU | 1x NVIDIA GeForce RTX 3090 |

## How to run

```bash
# Full dataset, warm cache, with per-camera (per-resolution) breakdown
PYTHONPATH=. python3 wrobo/utils/benchmark_image_codec.py --per-camera

# Cold-read timing (drops page cache before each file)
PYTHONPATH=. python3 wrobo/utils/benchmark_image_codec.py --cold --per-camera

# Quick subset (10 episodes, 200 frames per camera per episode)
PYTHONPATH=. python3 wrobo/utils/benchmark_image_codec.py --episodes 10 --frames-per-episode 200 --per-camera
```
## Results

> Throughput = decoded frames per second (higher is better). Latency = microseconds per
> frame (lower is better). "MiB/s decoded" is the throughput of decoded pixel bytes.

### Warm cache

Overall (all cameras combined, 384544 frames):

| Layout | Access | Throughput (frames/s) | Latency (µs/frame) | Decoded (MiB/s) | All Time (s) |
| --- | --- | --- | --- | --- | --- |
| `GZIP` | sequential | 1218.9 | 820.4 | 702.1 | 315.5 |
| `GZIP` | random | 1209.3 | 826.9 | 696.6 | 318.0 |
| `PNG` | sequential | 897.4 | 1114.3 | 516.9 | 428.5 |
| `PNG` | random | 890.9 | 1122.5 | 513.1 | 431.7 |
| `JPEG (Q=95)` | sequential | 1810.0 | 552.5 | 1042.5 | 212.5 |
| `JPEG (Q=95)` | random | 1781.7 | 561.3 | 1026.3 | 215.86 |

### Cold cache (`--cold`)

Overall (all cameras combined, 384544 frames):

| Layout | Access | Throughput (frames/s) | Latency (µs/frame) | Decoded (MiB/s) | All Time (s) |
| --- | --- | --- | --- | --- | --- |
| `GZIP` | sequential | 1199.7 | 833.5 | 691.0 | 320.5 |
| `GZIP` | random | 838.6 | 1192.5 | 483.0 | 458.6 |
| `PNG` | sequential | 891.4 | 1121.8 | 513.5 | 431.4 |
| `PNG` | random | 768.9 | 1300.6 | 442.9 | 500.1 |
| `JPEG (Q=95)` | sequential | 1791.9 | 558.1 | 1032.1 | 214.6 |
| `JPEG (Q=95)` | random | 1371.9 | 728.9 | 790.2 | 280.3 |


### Per camera Latency (µs/frame):

| Layout | Camera | Resolution | warm seq | warm rnd | cold seq | cold rnd |
| --- | --- | --- | --- | --- | --- | --- |
| `GZIP` | image_top | 480×640 | 1317.1 | 1329.8 | 1342.6 | 1837.0 |
| `GZIP` | image_angle | 480×640 | 1353.2 | 1362.1 | 1373.2 | 1872.9 |
| `GZIP` | image_left_wrist | 240×320 | 299.4 | 301.3 | 302.5 | 525.2 |
| `GZIP` | image_right_wrist | 240×320 | 312.1 | 314.5 | 315.8 | 534.9 |
| `PNG` | image_top | 480×640 | 1680.2 | 1686.2 | 1692.0 | 1945.7 |
| `PNG` | image_angle | 480×640 | 1656.7 | 1659.4 | 1667.3 | 1981.2 |
| `PNG` | image_left_wrist | 240×320 | 569.4 | 580.8 | 572.5 | 650.0 |
| `PNG` | image_right_wrist | 240×320 | 551.0 | 563.6 | 555.5 | 625.4 |
| `JPEG (Q=95)` | image_top | 480×640 | 792.4 | 799.4 | 800.4 | 1007.2 |
| `JPEG (Q=95)` | image_angle | 480×640 | 822.8 | 826.5 | 830.0 | 1137.8 |
| `JPEG (Q=95)` | image_left_wrist | 240×320 | 300.4 | 312.0 | 304.3 | 392.3 |
| `JPEG (Q=95)` | image_right_wrist | 240×320 | 294.5 | 307.1 | 297.6 | 378.3 |

### Storage vs. decode trade-off

| Layout | Dataset size (GiB) | vs. GZIP | Throughput (frames/s, warm seq) | Lossless Compression |
| --- | --- | --- | --- | --- |
| `GZIP` | 14 | 1.0× | 1218.9 | True |
| `PNG` | 7.8 | 0.56× | 897.4 | True |
| `JPEG (Q=95)` | 7.4 | 0.53× | 1810.0 | False |

### Performance during actual training of [ACT/ALOHA](https://arxiv.org/abs/2304.13705)

| Layout | Views | SR (%) | Train Time (h) | Views | SR (%) | Train Time (hours, h) |
| --- | --- | --- | --- | --- | --- | --- |
| `GZIP` | `top` + `angle` | - | - | `top` + `angle` + `left wrist` + `right wrist` | - | - |
| `PNG` |  | 88.00 | 3.44 |  | - | 4.09 |
| `JPEG (Q=95)` |  | - | 3.33 |  | - | 4.08 |

> **Note**
>
> - Use the implementation from wrobo, not the official one. There's some improvements, such as adding history window with learnable Position Embedding (PE) and adding paired learnable PE for different views. 
> - Success Rate (SR) is obtained by training with script [`/wrobo/methods/ACT/train.py`](/wrobo/methods/ACT/train.py) and evaluating with script [`/wrobo/envs/Aloha/eval.py`](/wrobo/envs/Aloha/eval.py) using `checkpoint_best.pth` without temporal aggregation.
> - Training Time statistics refer to the time from the start to the end of running the training script.

## Conclusion

- **Actual Situation.** In actual training, which involves **multi-process** data loading, there's no significant performance difference among the three (maybe the training scale is not large enough).
- **Sequential vs. random.** Because all three layouts use one-frame-per-chunk storage, warm-cache random access is expected to cost about the same as sequential (no cross-chunk locality to exploit). Confirm whether the cold-cache run shows a random penalty from disk seeks.
- **Storage vs. decode trade-off.** `PNG`/`JPEG` roughly halve on-disk size vs. `GZIP` with comparative success rates.
