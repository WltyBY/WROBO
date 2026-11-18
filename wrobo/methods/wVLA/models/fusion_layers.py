import math

import torch
import torch.nn as nn


class SinusoidalPositionEmbedding(nn.Module):
    """Deterministic 1D sinusoidal positions for variable-length token streams."""

    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, positions):
        half_dim = self.dim // 2
        dtype = torch.float32 if not torch.is_floating_point(positions) else positions.dtype
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half_dim, device=positions.device, dtype=dtype)
            / max(half_dim, 1)
        )
        args = positions.to(dtype).unsqueeze(-1) * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
        return emb


class TransformerFusionLayer(nn.Module):
    """Fuse sequence-form vision, language, and proprioception tokens.

    This layer assumes every modality has already been flattened into a token
    sequence before fusion. It does not flatten image feature maps, add singleton
    sequence dimensions, or otherwise infer modality-specific shapes.

    Expected tensor layouts:
      - batch_first=True:  ``(B, T, D)``
      - batch_first=False: ``(T, B, D)``

    Masks follow PyTorch Transformer convention regardless of ``batch_first``:
    ``(B, T)`` bool, True for tokens that should be ignored.
    """

    def __init__(
        self,
        embed_dim=256,
        n_heads=4,
        n_layers=2,
        ffn_mult=4,
        dropout=0.0,
        camera_names=None,
        use_cls_token=False,
        activation="gelu",
        batch_first=True,
        norm_first=True,
        vision_dim=None,
        language_dim=None,
        proprio_dim=None,
    ):
        super().__init__()
        if n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {n_heads}.")
        if embed_dim % n_heads != 0:
            raise ValueError(
                f"embed_dim must be divisible by n_heads, got embed_dim={embed_dim}, "
                f"n_heads={n_heads}."
            )

        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token
        self.batch_first = batch_first

        # Per-modality input projections with explicit in_features (no nn.LazyLinear):
        # the caller passes each modality's feature width — vision_dim (backbone feature
        # channels), language_dim (text-encoder hidden size), proprio_dim (proprioception
        # state dim). A projection is built only when its dim is provided; feeding a
        # modality whose dim was None is a configuration error (raised in _encode_sequence).
        self.vision_proj = (
            nn.Linear(vision_dim, embed_dim) if vision_dim is not None else None
        )
        self.language_proj = (
            nn.Linear(language_dim, embed_dim) if language_dim is not None else None
        )
        self.proprio_proj = (
            nn.Linear(proprio_dim, embed_dim) if proprio_dim is not None else None
        )

        # type ids: 0=vision, 1=language, 2=proprio, 3=optional summary token
        self.type_emb = nn.Embedding(4, embed_dim)
        self.pos_emb = SinusoidalPositionEmbedding(embed_dim)

        self.camera_names = list(camera_names) if camera_names is not None else []
        if len(self.camera_names) > 1:
            # one (1, embed_dim) embedding per named view
            self.cam_embed = nn.ModuleDict(
                {name: nn.Embedding(1, embed_dim) for name in self.camera_names}
            )
        else:
            # single (or unknown) camera: no view to disambiguate, so no embedding.
            self.cam_embed = None

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.register_parameter("cls_token", None)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ffn_mult * embed_dim,
            dropout=dropout,
            activation=activation,
            batch_first=batch_first,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.in_norm = nn.LayerNorm(embed_dim)
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        vision=None,
        language=None,
        proprio_state=None,
        language_mask=None,
        vision_mask=None,
        proprio_mask=None,
    ):
        """Return ``(memory, key_padding_mask)`` for the action expert."""
        if isinstance(language, (tuple, list)):
            if len(language) != 2:
                raise ValueError("language tuple/list must be (tokens, mask).")
            language, tuple_mask = language
            if language_mask is None:
                language_mask = tuple_mask

        token_groups = []
        mask_groups = []
        batch_size = None
        device = None

        if vision is not None:
            vision_tokens, vision_mask = self._encode_vision(vision, vision_mask)
            token_groups.append(vision_tokens)
            mask_groups.append(vision_mask)
            batch_size = self._batch_size(vision_tokens)
            device = vision_tokens.device

        if language is not None:
            language_tokens, language_mask = self._encode_sequence(
                tokens=language,
                mask=language_mask,
                proj=self.language_proj,
                type_id=1,
                name="language",
            )
            token_groups.append(language_tokens)
            mask_groups.append(language_mask)
            batch_size = self._check_batch_size(batch_size, self._batch_size(language_tokens))
            device = language_tokens.device if device is None else device

        if proprio_state is not None:
            proprio_tokens, proprio_mask = self._encode_sequence(
                tokens=proprio_state,
                mask=proprio_mask,
                proj=self.proprio_proj,
                type_id=2,
                name="proprio_state",
            )
            token_groups.append(proprio_tokens)
            mask_groups.append(proprio_mask)
            batch_size = self._check_batch_size(batch_size, self._batch_size(proprio_tokens))
            device = proprio_tokens.device if device is None else device

        if not token_groups:
            raise ValueError("At least one of vision, language, or proprio_state is required.")

        seq_dim = 1 if self.batch_first else 0
        memory = torch.cat(token_groups, dim=seq_dim)
        key_padding_mask = torch.cat(mask_groups, dim=1)

        if self.use_cls_token:
            cls = self._make_cls_token(batch_size, device, memory.dtype)
            cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
            memory = torch.cat([cls, memory], dim=seq_dim)
            key_padding_mask = torch.cat([cls_mask, key_padding_mask], dim=1)

        memory = self.in_norm(memory)
        memory = self.encoder(memory, src_key_padding_mask=key_padding_mask)
        return self.out_norm(memory), key_padding_mask

    def _encode_vision(self, vision, vision_mask=None):
        if torch.is_tensor(vision):
            return self._encode_sequence(
                tokens=vision,
                mask=vision_mask,
                proj=self.vision_proj,
                type_id=0,
                name="vision",
            )

        if not isinstance(vision, dict):
            raise TypeError(
                "vision must be a sequence tensor or a dict of named camera sequence tensors, "
                f"got {type(vision).__name__}."
            )

        if vision_mask is not None and not isinstance(vision_mask, dict):
            raise TypeError("vision_mask must be a dict when vision is a dict.")

        if self.camera_names and any(name in vision for name in self.camera_names):
            items = [(name, vision[name]) for name in self.camera_names if name in vision]
        else:
            items = list(vision.items())
        if not items:
            raise ValueError("vision dict must contain at least one camera sequence tensor.")

        encoded = []
        masks = []
        batch_size = None
        for name, tokens in items:
            mask = None if vision_mask is None else vision_mask.get(name)
            tokens, mask = self._encode_sequence(
                tokens=tokens,
                mask=mask,
                proj=self.vision_proj,
                type_id=0,
                name=f"vision[{name}]",
            )
            if self.cam_embed is not None and name in self.cam_embed:
                tokens = tokens + self._camera_embedding(name, tokens.device, tokens.dtype)
            encoded.append(tokens)
            masks.append(mask)
            batch_size = self._check_batch_size(batch_size, self._batch_size(tokens))

        seq_dim = 1 if self.batch_first else 0
        return torch.cat(encoded, dim=seq_dim), torch.cat(masks, dim=1)

    def _encode_sequence(self, tokens, mask, proj, type_id, name):
        if proj is None:
            raise ValueError(
                f"received {name} tokens but no {name} projection was built; pass the "
                f"matching *_dim to TransformerFusionLayer so its nn.Linear is created."
            )
        if not torch.is_tensor(tokens):
            raise TypeError(f"{name} must be a tensor, got {type(tokens).__name__}.")
        if tokens.dim() != 3:
            expected = "(B, T, D)" if self.batch_first else "(T, B, D)"
            raise ValueError(f"{name} must have shape {expected}, got {tuple(tokens.shape)}.")

        batch_size = self._batch_size(tokens)
        seq_len = self._seq_len(tokens)
        tokens = proj(tokens)
        tokens = tokens + self._type_embedding(type_id, tokens.device, tokens.dtype)
        tokens = tokens + self._position_embedding(seq_len, tokens.device, tokens.dtype)
        mask = self._normalize_mask(mask, batch_size, seq_len, tokens.device)
        return tokens, mask

    def _type_embedding(self, type_id, device, dtype):
        return self.type_emb.weight[type_id].to(device=device, dtype=dtype).view(1, 1, -1)

    def _position_embedding(self, seq_len, device, dtype):
        pos = torch.arange(seq_len, device=device)
        pos = self.pos_emb(pos).to(dtype=dtype)
        if self.batch_first:
            return pos.unsqueeze(0)
        return pos.unsqueeze(1)

    def _camera_embedding(self, camera_name, device, dtype):
        cam_id = torch.zeros(1, dtype=torch.long, device=device)
        return self.cam_embed[camera_name](cam_id).to(dtype=dtype).view(1, 1, -1)

    def _make_cls_token(self, batch_size, device, dtype):
        cls = self.cls_token.to(device=device, dtype=dtype)
        cls = cls + self._type_embedding(3, device, dtype)
        if self.batch_first:
            return cls.expand(batch_size, -1, -1)
        return cls.transpose(0, 1).expand(-1, batch_size, -1)

    def _batch_size(self, tokens):
        return tokens.size(0) if self.batch_first else tokens.size(1)

    def _seq_len(self, tokens):
        return tokens.size(1) if self.batch_first else tokens.size(0)

    @staticmethod
    def _check_batch_size(current, new):
        if current is not None and current != new:
            raise ValueError(f"All modalities must share batch size, got {current} and {new}.")
        return new

    @staticmethod
    def _normalize_mask(mask, batch_size, length, device):
        if mask is None:
            return torch.zeros(batch_size, length, dtype=torch.bool, device=device)
        if mask.shape != (batch_size, length):
            raise ValueError(f"mask must have shape {(batch_size, length)}, got {tuple(mask.shape)}.")
        return mask.to(device=device, dtype=torch.bool)


_FUSION_LAYER_REGISTRY = {
    "TransformerFusionLayer": TransformerFusionLayer,
}


def build_fusion_layer(config):
    """Build a fusion layer from a yaml-style config dictionary.

    The yaml must name the class explicitly, for example::

        fusion_layer:
          type: TransformerFusionLayer
          embed_dim: 256
          n_heads: 4
          n_layers: 2

    If the class name is misspelled, the error reports both the provided value
    and the available fusion-layer classes.
    """
    config = dict(config or {})
    class_name = config.pop("type", None)
    if class_name is None:
        class_name = config.pop("class_name", None)
    if class_name is None:
        class_name = config.pop("name", None)
    if class_name is None:
        raise KeyError(
            "fusion layer config requires a class name under 'type' "
            "(or 'class_name'/'name'). Available fusion-layer classes: "
            f"{sorted(_FUSION_LAYER_REGISTRY)}."
        )

    if not isinstance(class_name, str):
        raise TypeError(
            "fusion layer class name must be a string, got "
            f"{class_name!r} ({type(class_name).__name__}). Available fusion-layer "
            f"classes: {sorted(_FUSION_LAYER_REGISTRY)}."
        )

    if class_name not in _FUSION_LAYER_REGISTRY:
        raise ValueError(
            "Unsupported fusion layer class "
            f"{class_name!r}. Available fusion-layer classes: "
            f"{sorted(_FUSION_LAYER_REGISTRY)}."
        )

    if "cond_dim" in config and "embed_dim" not in config:
        config["embed_dim"] = config.pop("cond_dim")
    if "d_model" in config and "embed_dim" not in config:
        config["embed_dim"] = config.pop("d_model")
    return _FUSION_LAYER_REGISTRY[class_name](**config)
