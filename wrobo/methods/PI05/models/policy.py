"""PI0.5 wrobo I/O: quantile-normalized state -> discrete text prompt."""
import numpy as np
import torch
from wrobo.methods.PI0.models.policy import PI0Policy
from wrobo.methods.PI05.models.configuration import PI05Config
from wrobo.methods.PI05.models.modeling_pi05 import PI05Pytorch


def build_prompts(language, state):
    """Match LeRobot Pi05PrepareStateTokenizerProcessorStep exactly.

    Use real robot dimensions, not a 32-dimension zero-padded state. Do not clip
    out-of-quantile-range values: np.digitize semantics match upstream.
    """
    if state.ndim == 3:
        if state.shape[1] != 1:
            raise ValueError("PI05 requires history_width=1")
        state = state[:, 0]
    if state.ndim != 2 or not torch.isfinite(state).all():
        raise ValueError("PI05 expects finite normalized state [B,D]")
    batch = state.shape[0]
    if language is None:
        language = [""] * batch
    elif isinstance(language, str):
        language = [language] * batch
    if not isinstance(language, (list, tuple)) or len(language) != batch:
        raise TypeError("PI05 needs raw task strings, not pre-tokenized PI0 instructions")
    bins = np.digitize(state.detach().float().cpu().numpy(), np.linspace(-1, 1, 257)[:-1]) - 1
    prompts = []
    for task, values in zip(language, bins, strict=True):
        if not isinstance(task, str):
            raise TypeError("Each PI05 task must be a string")
        task = task.strip().replace("_", " ").replace("\n", " ")
        prompts.append(f"Task: {task}, State: {' '.join(map(str, values))};\nAction: ")
    return prompts


class PI05Policy(PI0Policy):
    config_class = PI05Config
    model_class = PI05Pytorch
    policy_type = "pi05"

    @torch.compiler.disable
    def _prepare_language(self, language, state):
        """Keep state-to-text and tokenizer work outside the compiled model graph."""
        prompts = build_prompts(language, state)
        return dict(self.tokenizer(
            prompts, padding="max_length", truncation=True,
            max_length=self.config.tokenizer_max_length, return_tensors="pt",
        ))

    def forward(self, image, proprio_state, language=None, actions=None,
                camera_names=None, noise=None, time=None):
        encoded = self._prepare_language(language, proprio_state)
        return super().forward(
            image=image, proprio_state=proprio_state, language=encoded,
            actions=actions, camera_names=camera_names, noise=noise, time=time,
        )

    def _get_default_peft_targets(self):
        # Exact standard (non-MEM) LeRobot PI0.5 defaults. state_proj is absent.
        common = "state_proj|action_in_proj|action_out_proj|time_mlp_in|time_mlp_out"
        return {"target_modules": rf"(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj|model\.({common}))",
                "modules_to_save": []}
