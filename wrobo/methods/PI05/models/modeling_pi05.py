# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""PI0.5 flow model: shared PI-Gemma core, time-conditioned action AdaRMS.

The inherited PI0 flow/inference plumbing passes a state argument, but PI0.5
intentionally does not project it: the policy has already put state in the prompt.
"""
import torch
from torch import nn
from torch.nn import functional as F
from wrobo.methods.PI0.models.modeling_pi0 import (
    PI0Pytorch, PaliGemmaWithExpertModel, get_gemma_config,
)
from wrobo.methods.PI0.models.utils import create_sinusoidal_pos_embedding


class PI05Pytorch(PI0Pytorch):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.rtc_processor = None
        vlm = get_gemma_config(config.paligemma_variant)
        expert = get_gemma_config(config.action_expert_variant)
        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            vlm, expert, use_adarms=[False, True], precision=config.dtype,
            image_size=config.image_resolution[0],
            freeze_vision_encoder=config.freeze_vision_encoder,
            freeze_language_encoder=config.freeze_language_encoder,
        )
        self.action_in_proj = nn.Linear(config.max_action_dim, expert.width)
        self.action_out_proj = nn.Linear(expert.width, config.max_action_dim)
        self.time_mlp_in = nn.Linear(expert.width, expert.width)
        self.time_mlp_out = nn.Linear(expert.width, expert.width)
        self.gradient_checkpointing_enabled = False

    def embed_suffix(self, state, noisy_actions, timestep):
        # No continuous state token in PI0.5; state is encoded in the text prefix.
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features,
            min_period=self.config.min_period, max_period=self.config.max_period,
            device=timestep.device,
        ).to(timestep.dtype)
        action_emb = self._apply_checkpoint(self.action_in_proj, noisy_actions)

        def time_mlp(value):
            return F.silu(self.time_mlp_out(F.silu(self.time_mlp_in(value))))

        cond = self._apply_checkpoint(time_mlp, time_emb)
        batch, length = action_emb.shape[:2]
        masks = torch.ones(batch, length, dtype=torch.bool, device=action_emb.device)
        blocks = torch.zeros(batch, length, dtype=action_emb.dtype, device=action_emb.device)
        blocks[:, 0] = 1
        return action_emb, masks, blocks, cond
