# SmolVLA in wrobo

This directory is a standalone port of LeRobot SmolVLA. It does not import
`lerobot`; the only external model dependency is `transformers` (plus
`safetensors`/`huggingface_hub` when a checkpoint is loaded).

## Model and training

The computation is the official SmolVLA flow-matching model:

`SigLIP image embeddings + token embeddings + padded state -> SmolVLM + action expert -> velocity`

Training samples `t ~ Beta(1.5, 1.0)`, interpolates `x_t = t * noise + (1-t) * action`,
and minimizes MSE between the predicted and target velocity. Inference uses the
official forward Euler integration from `t=1` to `t=0` with 10 steps by default.

The official architecture defaults are used: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`,
16 VLM/expert layers, `expert_width_multiplier=0.75`, `chunk_size=50`,
`max_state_dim=max_action_dim=32`, and `512x512` left/top padded images. All of
the architecture values stay fixed in code; the wrobo config only exposes the
dataset-dependent state/action dimensions and inference controls.

## Loading modes

Use `pretrained_load_mode` explicitly when possible:

```yaml
Policy:
  pretrained_load_mode: none       # random VLM + random action expert
  # pretrained_load_mode: vlm
  # pretrained_policy_path: HuggingFaceTB/SmolVLM2-500M-Video-Instruct
  # pretrained_load_mode: full
  # pretrained_policy_path: lerobot/smolvla_base
```

For backward compatibility, omitting `pretrained_load_mode` infers it from
`pretrained_policy_path`: no path means `none`, a path containing `smolvlm` means
`vlm`, and every other path means `full`.

`full` accepts a local directory/file or a Hugging Face repository containing
`model.safetensors` (also `pytorch_model.bin`/`model.bin`). Shape mismatches are
accepted only for the explicit `state_proj`, `action_in_proj`, and
`action_out_proj` allowlist. A mismatch in any other layer is an error.

## wrobo contract

Training call:

```python
policy(
    image={"camera_name": tensor[B, T, C, H, W]},
    camera_names=["camera_name"],
    proprio_state=tensor[B, T, state_dim],
    language=tensor[B, tokenizer_max_length] or list[str],
    actions=tensor[B, chunk_size, action_dim],
)
# -> {"model_pred": velocity, "target": target_velocity}
```

The Policy only performs model computation. The Trainer computes the elementwise
Flow Matching MSE and applies `is_pad` before reducing it to the training loss.

Inference omits `actions` and returns `{"action_chunk": ..., "pred": ...}` with
shape `[B, chunk_size, action_dim]`. The wrobo dataset pipeline only converts uint8
frames to float tensors in `[0,1]`. Matching official SmolVLA, every policy call
then applies aspect-ratio-preserving resize with left/top zero padding followed by
the SigLIP scaling `image * 2 - 1`. Training and deployment therefore share the
same deterministic model-side preprocessing.

When `instruction` is present in `Training_settings.data_keys`, the trainer uses
the SmolVLM tokenizer with a fixed `tokenizer_max_length` (48 by default), adds the
official trailing newline, and stores tokenizer/config artifacts under the run's
`Config_and_Trainer/smolvlm_config` directory for checkpoint loading and deployment.

The trainer uses wrobo's existing `CosineAnnealingLRScheduler`, configured with
`num_epoch`, `warmup_epoch`, and optional `Training_settings.min_lr`. Scheduler
advancement remains owned by the common trainer epoch loop.

The shared Trainer rejects non-finite training/validation losses and checks the
gradient norm before clipping and updating weights. BF16/FP32 failures stop with
the stage and, for invalid gradients, up to five parameter names; invalid metrics
cannot become a best checkpoint. FP16 GradScaler retains its normal overflow
skip/recovery behavior. Compiled backward is traced with autocast disabled to
match the shared backward call outside the method's AMP context. See the
[shared Trainer documentation](../../documents/abc_trainer.md).

For every method, `--compile` uses the shared Trainer: it first sets
`torch.set_float32_matmul_precision("high")`, then calls `torch.compile` with
`inplace_buffers=False`. The process-wide `high` setting permits TF32 or other
supported reduced internal FP32 matmul precision without changing parameter
storage dtypes or disabling AMP. The buffer option is applied per compiled network
without changing global compiler configuration. Model constructors do not compile
themselves; deployment uses the independent, explicit `ModelOptions.compile` setting.
On PyTorch 2.11.0+cu130 and an RTX 3090, the AP LoRA rank-32, batch-32 LIBERO
configuration reproduced a finite first-batch loss followed by extremely large
compiled gradients and an overflowing norm. Disabling Inductor's in-place buffer reuse
removed the failure in the controlled first-batch comparison while preserving
the compiled forward loss. Compilation and BF16 AMP remain enabled; learning
rate, batch size, LoRA rank and gradient clipping are unchanged.
Avoiding these buffer aliases can require more activation memory, and its
memory and throughput costs need measurement for the actual configuration.
The finite-value checks above remain active.

A checkpoint whose trainable weights or optimizer moments already contain NaN
cannot be repaired by these checks. Start from the original finite pretrained
weights rather than resuming that checkpoint. Do not replace NaNs with zero or
assume that a smaller learning rate resolves an unlocalized numerical failure.

## Fine-tuning

For full fine-tuning, use `freeze_vision_encoder` and
`freeze_language_encoder` independently; setting both to `true` freezes the
complete VLM while leaving the action expert trainable.

Enable PEFT LoRA with `--use_lora` and choose the adapted parts with
`--lora_parts`. The four groups separate the language backbone, visual input
path, action expert, and the state/action/time projections:

| Part | LoRA targets |
| --- | --- |
| `language` | Earlier action-related language layers use `q_proj`, `k_proj`, `v_proj`, and `o_proj`; the final action-related layer uses only `k_proj` and `v_proj`. Language submodules without an action-loss dependency are excluded. |
| `vision` | Vision encoder attention `q_proj`, `k_proj`, `v_proj`, and `out_proj`, plus linear layers in the vision-language connector. |
| `action` | Action expert attention `q_proj`, `k_proj`, `v_proj`, and `o_proj`. |
| `projection` | `state_proj`, `action_in_proj`, `action_out_proj`, `action_time_mlp_in`, and `action_time_mlp_out`. |

The connector maps image features into the language backbone's embedding space;
grouping it with `vision` makes that flag adapt the complete visual input path.
This first expansion adds attention K/O to every selected backbone. Vision names
its attention output projection `out_proj`; Language and Action use `o_proj`.
Transformer FFNs are not LoRA targets, and all base norms remain frozen. The
connector and five state/action/time projections retain their existing targets.
The default is `--lora_parts action projection`.
Start with these groups; add
`language` when instructions need adaptation or `vision` when the image domain
changes. Adapting either backbone also requires more activation memory during
backpropagation. All base-model weights remain frozen in LoRA mode, and selected
adapters are trainable regardless of the YAML encoder-freezing flags.
`--lora_rank`, `--lora_alpha`, and `--lora_dropout` apply to every selected group;
their defaults are 16, 16, and 0 respectively. Use `--lora_parts all` alone to
select all four groups.

The current [official LeRobot SmolVLA defaults](https://github.com/huggingface/lerobot/blob/5aa74557f84c54d4b458f8b9643c5aa2982acfed/src/lerobot/policies/smolvla/modeling_smolvla.py#L420-L429)
target only the action expert's Q/V projections and the five state/action/time
projections, with `modules_to_save=[]`. They do not add Vision or Language LoRA,
or train base norm/bias parameters. wrobo's current `action projection` selection
also includes action attention K/O; `all` additionally selects this port's Vision
and Language groups using the table above. It does not mean all linear layers.

The [official LIBERO Spatial PEFT example](https://huggingface.co/docs/lerobot/peft_training#training-smolvla)
explicitly uses rank 64, alpha 64, LR 1e-3, final LR 1e-4, 100000 optimizer
updates and batch size 32. These are an example recipe for the Spatial subset,
not the generic PEFT constructor defaults or a four-suite benchmark guarantee.
The eight Q/K/V/O experiments in `train_eval_smolvla.sh` compare VLAP and AP at
rank/alpha 16, 32, 64, and 128, running VLAP then AP at each ascending rank.
Each training command is immediately followed by its evaluation command. VLAP uses
LR 1e-4 and 30000 updates, so these experiments do not reproduce that recipe. Differences in target coverage,
rank, optimization and training data must be separated with controlled comparisons.

Run from the repository root, replacing the dataset path with your dataset.
For example, adapt the language backbone together with the default groups:

```bash
python -m wrobo.methods.SmolVLA.train \
  --dataset_dir /path/to/dataset \
  --log_dir ./Logs \
  --config_file Config/SmolVLA_SO101.yaml \
  --method_name SmolVLA --fold all --num_epoch 100 --batch_size 16 \
  --use_lora --lora_parts language action projection
```

To adapt all groups:

```bash
python -m wrobo.methods.SmolVLA.train \
  --dataset_dir /path/to/dataset \
  --log_dir ./Logs \
  --config_file Config/SmolVLA_SO101.yaml \
  --method_name SmolVLA --fold all --num_epoch 100 --batch_size 16 \
  --use_lora --lora_parts all
```

`--lora_save_mode merged` (the default) writes standalone deployment weights with
the adapter merged into a plain `SmolVLAPolicy`. `--lora_save_mode adapter` writes
only the LoRA tensors and therefore requires `pretrained_load_mode: full` so the
recorded base checkpoint can reconstruct the frozen weights. Retrainable checkpoints
always store the compact adapter plus optimizer/scheduler state. The resolved LoRA
and base-model settings are stored with the run so deployment and resume rebuild the
same structure automatically.
Run directories include the selected parts and an `_attn_qkvo` suffix, separating
this expansion from earlier Q/V runs. `--continue_train` loads from that directory
using the same selection. Loading adapter checkpoints with mismatched targets or
ranks raises an error. Earlier Q/V adapters cannot directly resume into the Q/K/V/O
structure; existing merged deployment checkpoints can still be evaluated.

Merged export builds a CPU deployment state dictionary from temporary per-layer
weight copies. It follows PEFT `safe_merge`'s dtype/rounding rule and preserves
the base checkpoint dtype. It never calls merge/unmerge on the live training
model: frozen base values, adapters, gradients and train/eval mode remain unchanged,
including when writing fails. This avoids BF16 base drift when exporting final
followed by best. It does not eliminate quantization inherent in a BF16 merged
checkpoint.

Files exported before this fix remain unchanged. A previously affected best
deployment file must be regenerated from the original pretrained base and the
saved `checkpoint_best_retrainable.pth` adapters. The repaired LIBERO LoRA runs
use the separate filename `checkpoint_best_fixed.pth`, selectable by
`run_eval.py --checkpoint-name checkpoint_best_fixed.pth`; final remains the
default checkpoint.

Run the dependency-free contract test with:

```bash
PYTHONPATH=. conda run -n wltyrobo python -m unittest -v \
  wrobo.methods.SmolVLA.test_model
```

## LIBERO evaluation

Use `Config/SmolVLA_LIBERO.yaml` for the converted LIBERO inputs: two RGB cameras,
8-dimensional `proprio_ee`, 7-dimensional `action_rel`, and task instructions.
The [LIBERO evaluation guide](../../sim_envs/LIBERO/README.md) explains the model
server in the `wrobo` environment and the evaluation process in the Python 3.8
`libero` environment, including official initial states, action chunk execution,
success reports and optional videos. These scripts evaluate policies and do not
collect demonstrations.

## Diagnosing color instructions

`python -m wrobo.methods.SmolVLA.diagnose_language --help` provides offline,
paired instruction comparisons with fixed noise, token checks, action traces,
and synchronized model timings. See
[`smolvla_language_diagnosis.md`](../../documents/smolvla_language_diagnosis.md)
for commands matching the saved SO101 runs and deployment sentence.

