"""wVLA policy: assemble a vision-language-action model from the modular building blocks.

This module is the wVLA counterpart of ACT's ``ACTPolicy`` (and miniVLA's
``VLADiffusionPolicy``): it wires four interchangeable, separately-tested components
into one ``nn.Module`` and exposes the same train/eval forward contract the rest of the
pipeline expects.

Component responsibilities (one file each, mirrored from the ACT ``models`` layout):
  - vision_encoders.PretrainedTorchvisionBackbone   -- pretrained CNN/ViT feature maps.
  - language_encoders.PretrainedTransformersLanguageEncoder -- per-token text features.
  - fusion_layers.TransformerFusionLayer            -- fuse modality token streams into
    a single conditioning memory.
  - action_experts.*                                -- decode the fused memory into an
    action chunk (regression / ddpm / ddim / flow_matching), chosen by the yaml
    ``action_expert.type``.

The policy owns only the *glue*. It turns the raw observation bundle (a list of camera
tensors with an optional history axis, a proprioception history, and optional tokenized
language) into the sequence-form tokens each component expects, then threads the fused
memory into the action expert. Every modality-specific architecture choice lives in the
component's own config block, so the yaml ``Policy`` block stays the single source of
truth (matching ``ACTPolicy``).

Forward contract:
  train (actions given): returns the action expert's training dict, including
        ``model_pred`` (plus ``target``/``loss_type`` for the stochastic experts); the
        loss is reduced in the Trainer (cf. ACTTrainer / miniVLATrainer).
  eval  (actions=None):  returns a dict containing ``pred``/``action``; the evaluator
        reads ``self.policy(...)["pred"]`` (cf. AlohaEvaluator).
"""

import torch
import torch.nn as nn

from wrobo.methods.wVLA.models.vision_encoders import PretrainedTorchvisionBackbone
from wrobo.methods.wVLA.models.language_encoders import build_language_encoder
from wrobo.methods.wVLA.models.fusion_layers import build_fusion_layer
from wrobo.methods.wVLA.models.action_experts import build_action_expert


class wVLAPolicy(nn.Module):
    """Vision-language-action policy built from the modular wVLA components.

    Args:
        config: the yaml ``Policy`` dictionary. Recognized sub-blocks:
            - ``vision_encoder``:   kwargs for ``PretrainedTorchvisionBackbone``.
            - ``language_encoder``: kwargs for ``build_language_encoder`` (optional;
              omit the block to build a vision-only policy).
            - ``fusion_layer``:     kwargs for ``build_fusion_layer``.
            - ``action_expert``:    kwargs for ``build_action_expert``.
            - ``camera_names``:     view names injected by the Trainer from ``data_keys``
              (same rule as ACT); bound name-wise inside the fusion layer so each camera
              keeps its own learnable embedding across save/load.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        vision_cfg = dict(self._require_block(config, "vision_encoder"))
        fusion_cfg = dict(self._require_block(config, "fusion_layer"))
        action_cfg = dict(self._require_block(config, "action_expert"))
        language_cfg = config.get("language_encoder")

        # Camera view names, derived by the Trainer from the image data_keys. They key
        # the per-camera vision dict so the fusion layer can attach a name-unique
        # embedding to each view. The order is the same the dataset/collater uses to
        # build the "image" list, so list index i <-> camera_names[i].
        self.camera_names = list(config.get("camera_names") or [])

        # Vision backbone: produces feature maps; the names of the maps to keep come from
        # the backbone's return_layers output names (e.g. {"layer4": "output"}).
        self.vision_encoder = PretrainedTorchvisionBackbone(**vision_cfg)
        self._vision_output_keys = list(self.vision_encoder.return_layers.values())

        # Language encoder is optional: a vision-only wVLA simply drops the block.
        self.language_encoder = (
            build_language_encoder(language_cfg) if language_cfg is not None else None
        )

        # Resolve each modality's feature width so the fusion layer can build explicit
        # nn.Linear projections (no nn.LazyLinear). Sources:
        #   - vision:   probed backbone output channels (vision_encoder.feature_dims).
        #   - language: the text encoder's hidden size (encoder.output_dim), or None when
        #               vision-only so no language projection is built.
        #   - proprio:  proprio_dim, given explicitly in the Policy config (there is no
        #               proprio encoder; the fusion Linear is its projection).
        vision_dim = self._resolve_vision_dim()
        language_dim = (
            self.language_encoder.output_dim
            if self.language_encoder is not None
            else None
        )
        proprio_dim = config.get("proprio_dim")
        if proprio_dim is None:
            raise KeyError(
                "wVLA Policy config requires 'proprio_dim' (the proprioception state "
                "width); there is no proprio encoder, so the fusion projection is sized "
                f"from it directly. Got keys: {sorted(config)}."
            )

        # Fusion: bind the camera names so the per-view embeddings are created/loaded by
        # name (the yaml may also set camera_names explicitly; respect that if present),
        # and pass the resolved modality dims so the input projections are sized now.
        fusion_cfg.setdefault("camera_names", self.camera_names)
        fusion_cfg.setdefault("vision_dim", vision_dim)
        fusion_cfg.setdefault("language_dim", language_dim)
        fusion_cfg.setdefault("proprio_dim", proprio_dim)
        self.fusion_layer = build_fusion_layer(fusion_cfg)

        # Action expert: condition on the fused memory. cond_dim must match the fusion
        # embed_dim; we default it from the fusion config so the yaml can omit it.
        action_cfg.setdefault("cond_dim", fusion_cfg.get("embed_dim"))
        self.action_expert = build_action_expert(action_cfg)

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        image,
        proprio_state,
        lang_tokens=None,
        attention_mask=None,
        actions=None,
        is_pad=None,
        return_sample=False,
    ):
        """
        Inputs:
            image:         dict of {camera_name: (B, his_width, C, H, W)}
            proprio_state: (B, his_width, state_dim).
            lang_tokens:   tokenizer output for the language encoder — a tensor of
                           ``input_ids`` (B, T_text) or the dict a HF tokenizer returns.
                           Pass None for a vision-only policy / step.
            attention_mask:(B, T_text) language attention mask, if not already carried
                           inside ``lang_tokens``.
            actions:       (B, chunk, action_dim) GT chunk at train time, None at eval.
            is_pad:        (B, chunk) padded-step mask. Accepted for API parity with ACT;
                           the action experts do not consume it (the Trainer masks the
                           loss), so it is ignored here.
            return_sample: for stochastic experts, also decode a sample during training.

        Returns: the action expert's output dict. ``model_pred`` is present at train
            time; ``pred``/``action`` are present at eval time (and at train time for the
            deterministic regression expert).
        """
        memory, mem_mask = self.encode_obs(
            image=image,
            proprio_state=proprio_state,
            lang_tokens=lang_tokens,
            attention_mask=attention_mask,
        )
        return self.action_expert(
            memory,
            mem_mask=mem_mask,
            actions=actions,
            return_sample=return_sample,
        )

    def encode_obs(self, image, proprio_state, lang_tokens=None, attention_mask=None):
        """Encode every modality to sequence-form tokens and fuse them.

        Returns ``(memory, mem_mask)`` for the action expert:
            memory:   (B, N_mem, embed_dim) fused conditioning tokens.
            mem_mask: (B, N_mem) bool key-padding mask, True at ignored positions.
        """
        vision = self._encode_vision(image)
        language = (
            self._encode_language(lang_tokens, attention_mask)
            if (self.language_encoder is not None and lang_tokens is not None)
            else None
        )
        return self.fusion_layer(
            vision=vision,
            language=language,
            proprio_state=proprio_state,
        )

    # ------------------------------------------------------------------ #
    # Per-modality encoding helpers                                      #
    # ------------------------------------------------------------------ #
    def _encode_vision(self, image):
        """Turn the per-camera image stack into a dict of (B, T_view, C_feat) tokens.

        image: dict of {camera_name: (B, his_width, C, H, W)}
        """
        if not isinstance(image, dict):
            raise TypeError(f"image must be dict, got {type(image)}")

        vision = {}
        for name, cam in image.items():
            if cam.dim() != 5:
                raise ValueError(
                    "each camera tensor must be (B, his_width, C, H, W), got "
                    f"{tuple(cam.shape)} for view {name!r}."
                )
            bs, his_width, c, h, w = cam.shape
            flat = cam.reshape(bs * his_width, c, h, w)
            feats = self.vision_encoder(flat)

            view_tokens = []
            for key in self._vision_output_keys:
                fmap = feats[key]  # (B*his, C_feat, H', W')
                c_feat = fmap.shape[1]
                # (B*his, C_feat, H', W') -> (B, his*H'*W', C_feat)
                tokens = (
                    fmap.flatten(2)
                    .transpose(1, 2)
                    .reshape(bs, his_width, -1, c_feat)
                    .reshape(bs, -1, c_feat)
                )
                view_tokens.append(tokens)
            channels = {t.shape[-1] for t in view_tokens}
            if len(channels) > 1:
                raise ValueError(
                    "vision return layers produce different channel counts "
                    f"{sorted(channels)} for view {name!r}; keep a single return layer "
                    "or layers with equal channels."
                )
            vision[name] = torch.cat(view_tokens, dim=1)
        return vision

    def _encode_language(self, lang_tokens, attention_mask):
        """Run the language encoder; returns ``(seq, mask)`` the fusion layer accepts."""
        return self.language_encoder(lang_tokens, attention_mask=attention_mask)

    def _resolve_vision_dim(self):
        """Vision feature width fed to the fusion projection.

        Tokens from every kept return layer are concatenated along the sequence axis (see
        _encode_vision), so they must share a channel count; that shared count is the
        projection's in_features. The backbone probes these channels once at construction
        (vision_encoder.feature_dims: output_name -> channels).
        """
        dims = {
            self.vision_encoder.feature_dims[key]
            for key in self._vision_output_keys
        }
        if len(dims) > 1:
            raise ValueError(
                "vision return layers produce different channel counts "
                f"{sorted(dims)}; keep a single return layer or layers with equal "
                "channels so the fusion vision projection has one input width."
            )
        return next(iter(dims))

    @staticmethod
    def _require_block(config, key):
        block = config.get(key)
        if block is None:
            raise KeyError(
                f"wVLA Policy config requires a {key!r} block. "
                f"Got keys: {sorted(config)}."
            )
        return block


def _smoke_test():
    """Offline shape smoke test on ACT-like inputs (multi-camera, history, chunk).

    Uses a tiny local DistilBERT config so no weights are downloaded, and runs both a
    training forward (actions given) and an eval forward (actions=None).
    """
    import tempfile

    from transformers import AutoConfig

    bs, his_width, chunk = 2, 1, 100
    camera_names = ["image_top", "image_wrist"]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Tiny, weightless language config to keep the test offline and fast.
        AutoConfig.for_model(
            "distilbert",
            vocab_size=100,
            dim=64,
            hidden_dim=128,
            n_heads=4,
            n_layers=2,
        ).save_pretrained(tmpdir)

        config = {
            "camera_names": camera_names,
            "proprio_dim": 14,
            "vision_encoder": {
                "backbone_name": "ResNet18",
                "backbone_weights_name": None,
                "return_layers": {"layer4": "output"},
                "freeze_norm": True,
            },
            "language_encoder": {
                "backbone_name": tmpdir,
                "backbone_weights_name": None,
                "freeze_backbone": False,
                "local_files_only": True,
            },
            "fusion_layer": {
                "type": "TransformerFusionLayer",
                "embed_dim": 256,
                "n_heads": 4,
                "n_layers": 2,
            },
            "action_expert": {
                "type": "regression",
                "action_dim": 14,
                "action_chunk_size": chunk,
                "hidden_dim": 256,
                "n_heads": 4,
                "n_layers": 4,
            },
        }

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy = wVLAPolicy(config).to(device)

        image = [
            torch.randn(bs, his_width, 3, 240, 320, device=device)
            for _ in camera_names
        ]
        proprio_state = torch.randn(bs, his_width, 14, device=device)
        lang_tokens = torch.randint(0, 100, (bs, 16), device=device)
        attention_mask = torch.ones(bs, 16, dtype=torch.long, device=device)
        actions = torch.randn(bs, chunk, 14, device=device)

        out = policy(
            image=image,
            proprio_state=proprio_state,
            lang_tokens=lang_tokens,
            attention_mask=attention_mask,
            actions=actions,
        )
        print(f"[train] model_pred shape: {tuple(out['model_pred'].shape)}")

        policy.eval()
        with torch.no_grad():
            pred = policy(
                image=image,
                proprio_state=proprio_state,
                lang_tokens=lang_tokens,
                attention_mask=attention_mask,
            )["pred"]
        print(f"[eval]  pred shape:        {tuple(pred.shape)}")

        # Vision-only path (no language) should still fuse and decode.
        with torch.no_grad():
            vision_only = policy(image=image, proprio_state=proprio_state)["pred"]
        print(f"[eval]  vision-only pred:  {tuple(vision_only.shape)}")

        n_params = sum(p.numel() for p in policy.parameters())
        print(f"Total parameters: {n_params:,}")


if __name__ == "__main__":
    _smoke_test()

