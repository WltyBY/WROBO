import math

import torch
import torch.nn as nn


_REQUIRED_COMMON_KEYS = (
    "action_dim",
    "cond_dim",
    "action_chunk_size",
    "hidden_dim",
    "n_heads",
    "n_layers",
)


def _require(config, key):
    if key not in config or config[key] is None:
        raise KeyError(f"action expert config requires {key!r}.")
    return config[key]


def _get(config, key, default):
    value = config.get(key, default)
    return default if value is None else value


def _validate_common_config(config):
    for key in _REQUIRED_COMMON_KEYS:
        _require(config, key)
    hidden_dim = _require(config, "hidden_dim")
    n_heads = _require(config, "n_heads")
    if n_heads <= 0:
        raise ValueError(f"n_heads must be positive, got {n_heads}.")
    if hidden_dim % n_heads != 0:
        raise ValueError(
            f"hidden_dim must be divisible by n_heads, got hidden_dim={hidden_dim}, "
            f"n_heads={n_heads}."
        )


def _build_transformer_decoder(
    hidden_dim,
    n_heads,
    n_layers,
    ffn_mult=4,
    dropout=0.0,
    activation="gelu",
    batch_first=True,
    norm_first=True,
):
    decoder_layer = nn.TransformerDecoderLayer(
        d_model=hidden_dim,
        nhead=n_heads,
        dim_feedforward=ffn_mult * hidden_dim,
        dropout=dropout,
        activation=activation,
        batch_first=batch_first,
        norm_first=norm_first,
    )
    return nn.TransformerDecoder(decoder_layer, num_layers=n_layers)


def _prepare_sample_batch(memory, mem_mask, n_samples):
    batch_size = memory.size(0)
    return_samples_dim = n_samples is not None
    n_samples = 1 if n_samples is None else n_samples
    if not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError(f"n_samples must be a positive integer or None, got {n_samples!r}.")
    memory = memory.repeat_interleave(n_samples, dim=0)
    if mem_mask is not None:
        mem_mask = mem_mask.repeat_interleave(n_samples, dim=0)
    return memory, mem_mask, batch_size, n_samples, return_samples_dim


def _restore_sample_batch(samples, batch_size, n_samples, return_samples_dim):
    samples = samples.reshape(batch_size, n_samples, *samples.shape[1:])
    if return_samples_dim:
        return samples
    return samples[:, 0]


def _shift_sigmas(sigmas, shift):
    return shift * sigmas / (1 + (shift - 1) * sigmas)


def _make_alpha_bar_schedule(config):
    """Cumulative-product alpha schedule (alpha_bar) for the diffusion experts.

    Only alpha_bar is ever consumed by the samplers / q_sample / target math, so the
    intermediate betas and alphas are computed locally and not returned.
    """
    num_steps = _get(config, "num_train_timesteps", 1000)
    schedule = _get(config, "beta_schedule", "cosine").lower()

    if schedule == "linear":
        betas = torch.linspace(
            _get(config, "beta_start", 1e-4),
            _get(config, "beta_end", 2e-2),
            num_steps,
        )
        return torch.cumprod(1.0 - betas, dim=0)

    if schedule != "cosine":
        raise ValueError(f"beta_schedule must be 'linear' or 'cosine', got {schedule!r}.")

    s = _get(config, "cosine_s", 0.008)
    t = torch.linspace(0, num_steps, num_steps + 1)
    f = torch.cos(((t / num_steps + s) / (1 + s)) * math.pi / 2) ** 2
    alpha_bar_full = f / f[0]
    betas = 1.0 - (alpha_bar_full[1:] / alpha_bar_full[:-1])
    betas = betas.clamp(1e-8, 0.999)
    return torch.cumprod(1.0 - betas, dim=0)



class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t):
        half_dim = self.dim // 2
        device = t.device
        dtype = torch.float32 if not torch.is_floating_point(t) else t.dtype
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half_dim, device=device, dtype=dtype)
            / max(half_dim, 1)
        )
        args = t.to(dtype).unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
        return emb


class ConditionalActionTransformer(nn.Module):
    """Transformer decoder for conditional stochastic action experts.

    x_t:      (B, chunk, action_dim)
    t:        (B,) integer or continuous timestep-like values
    memory:   (B, N_mem, cond_dim)
    mem_mask: (B, N_mem) bool key-padding mask, or None
    """

    def __init__(self, config):
        super().__init__()
        _validate_common_config(config)
        self.config = dict(config)

        self.action_dim = _require(config, "action_dim")
        self.cond_dim = _require(config, "cond_dim")
        self.action_chunk_size = _require(config, "action_chunk_size")
        self.hidden_dim = _require(config, "hidden_dim")
        self.time_emb_dim = _get(config, "time_emb_dim", self.hidden_dim)
        self.action_pos_emb_dim = _get(config, "action_pos_emb_dim", self.hidden_dim)

        self.time_emb = SinusoidalTimeEmbedding(
            self.time_emb_dim,
            max_period=_get(config, "time_max_period", 10000),
        )
        self.action_pos_emb = nn.Embedding(self.action_chunk_size, self.action_pos_emb_dim)
        self.in_proj = nn.Linear(
            self.action_dim + self.time_emb_dim + self.action_pos_emb_dim,
            self.hidden_dim,
        )
        self.mem_in = (
            nn.Identity()
            if self.cond_dim == self.hidden_dim
            else nn.Linear(self.cond_dim, self.hidden_dim)
        )
        self.decoder = _build_transformer_decoder(
            hidden_dim=self.hidden_dim,
            n_heads=_require(config, "n_heads"),
            n_layers=_require(config, "n_layers"),
            ffn_mult=_get(config, "ffn_mult", 4),
            dropout=_get(config, "dropout", 0.0),
            activation=_get(config, "activation", "gelu"),
            batch_first=_get(config, "batch_first", True),
            norm_first=_get(config, "norm_first", True),
        )
        self.out_norm = nn.LayerNorm(self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.action_dim)

    def forward(self, x_t, t, memory, mem_mask=None):
        """
        Inputs:
            x_t:      (B, chunk, action_dim)
            t:        (B,) integer or continuous timestep-like values
            memory:   (B, N_mem, cond_dim)
            mem_mask: (B, N_mem) bool key-padding mask, or None
        Returns:
            (B, chunk, action_dim)
        """
        B, chunk_size, _ = x_t.shape
        if chunk_size > self.action_chunk_size:
            raise ValueError(
                f"Got chunk_size={chunk_size}, but configured action_chunk_size={self.action_chunk_size}."
            )

        t_emb = self.time_emb(t).unsqueeze(1).expand(-1, chunk_size, -1)
        step_ids = torch.arange(chunk_size, device=x_t.device)
        pos_emb = self.action_pos_emb(step_ids).unsqueeze(0).expand(B, -1, -1)
        q = self.in_proj(torch.cat([x_t, t_emb, pos_emb], dim=-1))
        mem = self.mem_in(memory)
        out = self.decoder(tgt=q, memory=mem, memory_key_padding_mask=mem_mask)
        return self.out_proj(self.out_norm(out))


class RegressionActionExpert(nn.Module):
    """ACT-style deterministic action-chunk regression expert."""

    def __init__(self, config):
        super().__init__()
        _validate_common_config(config)
        self.config = dict(config)

        self.action_dim = _require(config, "action_dim")
        self.cond_dim = _require(config, "cond_dim")
        self.action_chunk_size = _require(config, "action_chunk_size")
        self.hidden_dim = _require(config, "hidden_dim")

        self.query_emb = nn.Embedding(self.action_chunk_size, self.hidden_dim)
        self.mem_in = (
            nn.Identity()
            if self.cond_dim == self.hidden_dim
            else nn.Linear(self.cond_dim, self.hidden_dim)
        )
        self.decoder = _build_transformer_decoder(
            hidden_dim=self.hidden_dim,
            n_heads=_require(config, "n_heads"),
            n_layers=_require(config, "n_layers"),
            ffn_mult=_get(config, "ffn_mult", 4),
            dropout=_get(config, "dropout", 0.0),
            activation=_get(config, "activation", "gelu"),
            batch_first=_get(config, "batch_first", True),
            norm_first=_get(config, "norm_first", True),
        )
        self.out_norm = nn.LayerNorm(self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.action_dim)

    def predict(self, memory, mem_mask=None):
        B = memory.size(0)
        q = self.query_emb.weight.unsqueeze(0).expand(B, -1, -1)
        mem = self.mem_in(memory)
        out = self.decoder(tgt=q, memory=mem, memory_key_padding_mask=mem_mask)
        return self.out_proj(self.out_norm(out))

    def sample(self, memory, mem_mask=None, n_samples=None):
        memory, mem_mask, batch_size, n_samples, return_samples_dim = _prepare_sample_batch(
            memory, mem_mask, n_samples
        )
        pred = self.predict(memory, mem_mask)
        return _restore_sample_batch(pred, batch_size, n_samples, return_samples_dim)

    def forward(self, memory, mem_mask=None, actions=None, return_sample=False):
        pred = self.predict(memory, mem_mask)
        return {"model_pred": pred, "pred": pred, "action": pred}


class DDPMActionExpert(nn.Module):
    """Conditional diffusion action expert with stochastic DDPM ancestral sampling."""

    def __init__(self, config):
        super().__init__()
        _validate_common_config(config)
        self.config = dict(config)

        self.action_dim = _require(config, "action_dim")
        self.action_chunk_size = _require(config, "action_chunk_size")
        self.num_train_timesteps = _get(config, "num_train_timesteps", 1000)
        # Default to a small, fast sampler (10x cheaper than the legacy 1000-step
        # rollout). Capped at num_train_timesteps so a small training schedule stays
        # valid. Override via config for a quality/latency trade-off.
        self.num_inference_steps = _get(
            config, "num_inference_steps", min(100, self.num_train_timesteps)
        )
        if self.num_train_timesteps <= 0:
            raise ValueError(
                f"num_train_timesteps must be positive, got {self.num_train_timesteps}."
            )
        if self.num_inference_steps <= 0:
            raise ValueError(
                f"num_inference_steps must be positive, got {self.num_inference_steps}."
            )
        if self.num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                "num_inference_steps must be <= num_train_timesteps, got "
                f"num_inference_steps={self.num_inference_steps}, "
                f"num_train_timesteps={self.num_train_timesteps}."
            )
        self.prediction_type = _get(config, "prediction_type", "v").lower()
        if self.prediction_type not in {"v", "eps", "x0"}:
            raise ValueError(
                "prediction_type must be one of {'v', 'eps', 'x0'}, "
                f"got {self.prediction_type!r}."
            )
        self.x0_clip = _get(config, "x0_clip", None)

        model_config = dict(config)
        model_config.setdefault("time_emb_dim", _require(config, "hidden_dim"))
        model_config.setdefault("action_pos_emb_dim", _require(config, "hidden_dim"))
        self.model = ConditionalActionTransformer(model_config)

        alpha_bar = _make_alpha_bar_schedule(self.config)
        self.register_buffer("alpha_bar", alpha_bar)

    def q_sample(self, x0, t, noise):
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1)
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise

    def training_target(self, x0, t, noise):
        if self.prediction_type == "eps":
            return noise
        if self.prediction_type == "x0":
            return x0

        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1)
        return torch.sqrt(alpha_bar_t) * noise - torch.sqrt(1 - alpha_bar_t) * x0

    def _prediction_to_x0_eps(self, x_t, t_index, model_pred):
        alpha_bar_t = self.alpha_bar[t_index]
        sqrt_ab = torch.sqrt(alpha_bar_t)
        sqrt_1mab = torch.sqrt(1 - alpha_bar_t)

        if self.prediction_type == "eps":
            x0_pred = (x_t - sqrt_1mab * model_pred) / sqrt_ab.clamp_min(1e-8)
            eps_pred = model_pred
        elif self.prediction_type == "x0":
            x0_pred = model_pred
            eps_pred = (x_t - sqrt_ab * x0_pred) / sqrt_1mab.clamp_min(1e-8)
        else:
            x0_pred = sqrt_ab * x_t - sqrt_1mab * model_pred
            eps_pred = sqrt_1mab * x_t + sqrt_ab * model_pred

        if self.x0_clip is not None:
            x0_pred = x0_pred.clamp(-self.x0_clip, self.x0_clip)
        return x0_pred, eps_pred

    def _inference_timesteps(self, device):
        steps = torch.linspace(
            self.num_train_timesteps - 1,
            0,
            self.num_inference_steps,
            device=device,
        )
        return steps.round().long().unique_consecutive()

    def _ddpm_step(self, x_t, t_index, prev_t_index, model_pred):
        x0_pred, eps_pred = self._prediction_to_x0_eps(x_t, t_index, model_pred)
        if prev_t_index < 0:
            return x0_pred

        alpha_bar_t = self.alpha_bar[t_index]
        alpha_bar_prev = self.alpha_bar[prev_t_index]
        alpha_ratio = alpha_bar_t / alpha_bar_prev
        beta_ratio = 1 - alpha_ratio
        mean = (x_t - beta_ratio / torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(
            alpha_ratio
        )
        posterior_var = beta_ratio * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
        return mean + torch.sqrt(posterior_var.clamp_min(1e-20)) * torch.randn_like(x_t)

    @torch.no_grad()
    def sample(self, memory, mem_mask=None, n_samples=None):
        memory, mem_mask, batch_size, n_samples, return_samples_dim = _prepare_sample_batch(
            memory, mem_mask, n_samples
        )
        B = memory.size(0)

        x_t = torch.randn(
            B,
            self.action_chunk_size,
            self.action_dim,
            device=memory.device,
            dtype=memory.dtype,
        )
        timesteps = self._inference_timesteps(memory.device)
        for idx, t_index in enumerate(timesteps):
            t = torch.full((B,), t_index, device=memory.device, dtype=torch.long)
            model_pred = self.model(x_t, t, memory, mem_mask)
            prev_t_index = timesteps[idx + 1] if idx + 1 < len(timesteps) else -1
            x_t = self._ddpm_step(x_t, t_index, prev_t_index, model_pred)
        return _restore_sample_batch(x_t, batch_size, n_samples, return_samples_dim)

    def forward(self, memory, mem_mask=None, actions=None, return_sample=False):
        if actions is None:
            pred = self.sample(memory, mem_mask)
            return {"model_pred": pred, "pred": pred, "action": pred}

        B = actions.size(0)
        t = torch.randint(0, self.num_train_timesteps, (B,), device=actions.device)
        noise = torch.randn_like(actions)
        x_t = self.q_sample(actions, t, noise)
        model_pred = self.model(x_t, t, memory, mem_mask)
        target = self.training_target(actions, t, noise)
        out = {
            "model_pred": model_pred,
            "target": target,
            "x_t": x_t,
            "t": t,
        }
        if return_sample:
            pred = self.sample(memory, mem_mask)
            out.update({"pred": pred, "action": pred})
        return out


class DDIMActionExpert(DDPMActionExpert):
    """Conditional diffusion action expert with deterministic DDIM sampling."""

    @torch.no_grad()
    def sample(self, memory, mem_mask=None, n_samples=None):
        memory, mem_mask, batch_size, n_samples, return_samples_dim = _prepare_sample_batch(
            memory, mem_mask, n_samples
        )
        B = memory.size(0)

        x_t = torch.randn(
            B,
            self.action_chunk_size,
            self.action_dim,
            device=memory.device,
            dtype=memory.dtype,
        )
        timesteps = self._inference_timesteps(memory.device)
        for idx, t_index in enumerate(timesteps):
            t = torch.full((B,), t_index, device=memory.device, dtype=torch.long)
            model_pred = self.model(x_t, t, memory, mem_mask)
            x0_pred, eps_pred = self._prediction_to_x0_eps(x_t, t_index, model_pred)

            if idx + 1 < len(timesteps):
                prev_t = timesteps[idx + 1]
                alpha_bar_prev = self.alpha_bar[prev_t]
                x_t = torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(
                    1 - alpha_bar_prev
                ) * eps_pred
            else:
                x_t = x0_pred
        return _restore_sample_batch(x_t, batch_size, n_samples, return_samples_dim)


class FlowMatchingActionExpert(nn.Module):
    """Conditional flow-matching action expert."""

    def __init__(self, config):
        super().__init__()
        _validate_common_config(config)
        self.config = dict(config)

        self.action_dim = _require(config, "action_dim")
        self.action_chunk_size = _require(config, "action_chunk_size")
        self.num_train_timesteps = _get(config, "num_train_timesteps", 1000)
        self.num_inference_steps = _get(config, "num_inference_steps", 20)
        if self.num_train_timesteps <= 0:
            raise ValueError(
                f"num_train_timesteps must be positive, got {self.num_train_timesteps}."
            )
        if self.num_inference_steps <= 0:
            raise ValueError(
                f"num_inference_steps must be positive, got {self.num_inference_steps}."
            )
        self.sigma_max = _get(config, "sigma_max", 1.0)
        self.sigma_min = _get(config, "sigma_min", 0.003 / 1.002)
        self.shift = _get(config, "shift", 3.0)
        if self.sigma_min < 0:
            raise ValueError(f"sigma_min must be non-negative, got {self.sigma_min}.")
        if self.sigma_max <= self.sigma_min:
            raise ValueError(
                f"sigma_max must be greater than sigma_min, got sigma_max={self.sigma_max}, "
                f"sigma_min={self.sigma_min}."
            )
        if self.shift <= 0:
            raise ValueError(f"shift must be positive, got {self.shift}.")

        model_config = dict(config)
        model_config.setdefault("time_emb_dim", _require(config, "hidden_dim"))
        model_config.setdefault("action_pos_emb_dim", _require(config, "hidden_dim"))
        self.model = ConditionalActionTransformer(model_config)

        sigmas = self._build_inference_sigmas()
        self.register_buffer("inference_sigmas", sigmas)

    def _build_inference_sigmas(self):
        sigmas = torch.linspace(self.sigma_max, self.sigma_min, self.num_inference_steps)
        return _shift_sigmas(sigmas, self.shift)

    def _sample_training_sigmas(self, batch_size, device, dtype):
        sigmas = torch.rand(batch_size, device=device, dtype=dtype)
        sigmas = self.sigma_min + (self.sigma_max - self.sigma_min) * sigmas
        return _shift_sigmas(sigmas, self.shift)

    @torch.no_grad()
    def sample(self, memory, mem_mask=None, n_samples=None):
        memory, mem_mask, batch_size, n_samples, return_samples_dim = _prepare_sample_batch(
            memory, mem_mask, n_samples
        )
        B = memory.size(0)

        x = torch.randn(
            B,
            self.action_chunk_size,
            self.action_dim,
            device=memory.device,
            dtype=memory.dtype,
        )
        # Integrate the probability-flow ODE from sigma_max (pure noise) down to 0
        # (clean action) with explicit Euler steps. A terminal sigma=0 is appended so
        # the final step lands exactly on the data manifold.
        sigmas = self.inference_sigmas.to(device=memory.device, dtype=memory.dtype)
        sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])
        for idx in range(len(sigmas) - 1):
            sigma, sigma_next = sigmas[idx], sigmas[idx + 1]
            # (sigma * T) is a 0-d tensor; expand to (B,) for the time embedding (avoids
            # torch.full with a tensor fill value, which is brittle across torch versions).
            t = (sigma * self.num_train_timesteps).expand(B)
            velocity = self.model(x, t, memory, mem_mask)
            x = x + velocity * (sigma_next - sigma)
        return _restore_sample_batch(x, batch_size, n_samples, return_samples_dim)

    def forward(self, memory, mem_mask=None, actions=None, return_sample=False):
        if actions is None:
            pred = self.sample(memory, mem_mask)
            return {"model_pred": pred, "pred": pred, "action": pred}

        B = actions.size(0)
        sigma = self._sample_training_sigmas(B, actions.device, actions.dtype)
        sigma_b = sigma.view(B, 1, 1)
        noise = torch.randn_like(actions)
        x_t = (1 - sigma_b) * actions + sigma_b * noise
        target = noise - actions
        t = sigma * self.num_train_timesteps
        model_pred = self.model(x_t, t, memory, mem_mask)
        out = {
            "model_pred": model_pred,
            "target": target,
            "x_t": x_t,
            "sigma": sigma,
            "t": t,
        }
        if return_sample:
            pred = self.sample(memory, mem_mask)
            out.update({"pred": pred, "action": pred})
        return out


def build_action_expert(config):
    config = dict(config)
    expert_type = _require(config, "type").lower()
    if expert_type == "regression":
        return RegressionActionExpert(config)
    if expert_type == "ddpm":
        return DDPMActionExpert(config)
    if expert_type == "ddim":
        return DDIMActionExpert(config)
    if expert_type == "flow_matching":
        return FlowMatchingActionExpert(config)
    raise ValueError(
        "Unsupported action expert type "
        f"{expert_type!r}. Supported types: 'regression', 'ddpm', 'ddim', 'flow_matching'."
    )
