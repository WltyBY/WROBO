import torch
from torch import nn

from wrobo.training.utils.mics import is_main_process


class PretrainedTransformersLanguageEncoder(nn.Module):
    """Transformers-based pretrained language encoder.

    This module is the language-side counterpart of
    ``PretrainedTorchvisionBackbone``: it owns the pretrained transformer
    structure/weights, while the Trainer/dataset owns tokenization. Callers pass
    tokenizer outputs such as ``input_ids`` and ``attention_mask`` instead of raw
    strings.

    The forward contract mirrors miniVLA's ``TextEncoderTinyGRU`` so language
    tokens can be fused with vision/proprioception memory tokens directly:

      - seq:  ``(B, T_text, hidden_size)`` per-token language features
      - mask: ``(B, T_text)`` bool key-padding mask, True at padded positions

    Args:
        backbone_name: Hugging Face model/config id or local config/checkpoint path.
        backbone_weights_name: Controls pretrained weight loading. ``"DEFAULT"`` or
            ``True`` loads weights from ``backbone_name``; ``None``/``False`` builds
            only the architecture from config; any other string is treated as a
            Hugging Face id or local checkpoint path to load weights from.
        freeze_backbone: Freeze the transformer backbone parameters.
            Modality-dimension alignment is handled by the fusion layer rather than
            by this encoder.
        config_kwargs: Extra kwargs for ``AutoConfig.from_pretrained``.
        model_kwargs: Extra kwargs for ``AutoModel.from_pretrained``.
        trust_remote_code: Passed to transformers loading APIs.
        local_files_only: Passed to transformers loading APIs.
        output_hidden_states: Ask the transformer to return all hidden states and
            use ``hidden_state_index`` as the sequence output. Otherwise use
            ``last_hidden_state``.
        hidden_state_index: Hidden-state tuple index used when
            ``output_hidden_states`` is enabled.
    """

    def __init__(
        self,
        backbone_name,
        backbone_weights_name="DEFAULT",
        freeze_backbone=False,
        config_kwargs=None,
        model_kwargs=None,
        trust_remote_code=False,
        local_files_only=False,
        output_hidden_states=False,
        hidden_state_index=-1,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone_weights_name = backbone_weights_name
        self.freeze_backbone = freeze_backbone
        self.config_kwargs = dict(config_kwargs or {})
        self.model_kwargs = dict(model_kwargs or {})
        self.trust_remote_code = trust_remote_code
        self.local_files_only = local_files_only
        self.output_hidden_states = output_hidden_states
        self.hidden_state_index = hidden_state_index

        self.model = self.build_backbone()
        self.output_dim = self._infer_hidden_size(self.model.config)

        if self.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        """Encode tokenized text.

        ``input_ids`` may be either a tensor or a dictionary returned by a
        transformers tokenizer. Additional tokenizer fields, such as
        ``token_type_ids``, can be passed through ``kwargs``.
        """
        if isinstance(input_ids, dict):
            batch = dict(input_ids)
            input_ids = batch.pop("input_ids", None)
            attention_mask = batch.pop("attention_mask", attention_mask)
            kwargs = {**batch, **kwargs}
        if input_ids is None:
            raise ValueError(
                "input_ids is required for PretrainedTransformersLanguageEncoder."
            )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=self.output_hidden_states,
            return_dict=True,
            **self._filter_none_kwargs(kwargs),
        )
        seq = self._select_sequence_output(outputs)
        mask = self._make_padding_mask(input_ids, attention_mask)
        return seq, mask

    def build_backbone(self):
        AutoConfig, AutoModel = _require_transformers()

        config = AutoConfig.from_pretrained(
            self.backbone_name,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local_files_only,
            **self.config_kwargs,
        )
        kwargs = dict(self.model_kwargs)
        kwargs.setdefault("trust_remote_code", self.trust_remote_code)
        kwargs.setdefault("local_files_only", self.local_files_only)

        backbone_weights_source = self._resolve_backbone_weights_source()
        if backbone_weights_source is None:
            return AutoModel.from_config(
                config, trust_remote_code=self.trust_remote_code
            )

        if is_main_process():
            return AutoModel.from_pretrained(
                backbone_weights_source,
                config=config,
                **kwargs,
            )

        # Avoid duplicated downloads on non-main ranks. DDP will synchronize rank-0
        # parameters during wrapping, so all ranks still receive pretrained weights.
        return AutoModel.from_config(config, trust_remote_code=self.trust_remote_code)

    @staticmethod
    def _is_no_weights_name(weight_name):
        if weight_name is None or weight_name is False:
            return True
        if isinstance(weight_name, str):
            return weight_name.strip().lower() in {"", "none", "null", "false", "no", "off"}
        return False

    def _resolve_backbone_weights_source(self):
        weight_name = self.backbone_weights_name
        if self._is_no_weights_name(weight_name):
            return None
        if weight_name is True:
            return self.backbone_name
        if isinstance(weight_name, str) and weight_name.upper() == "DEFAULT":
            return self.backbone_name
        return weight_name

    def _select_sequence_output(self, outputs):
        if self.output_hidden_states:
            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states is None:
                raise RuntimeError(
                    "output_hidden_states=True but the transformer did not return hidden_states."
                )
            return hidden_states[self.hidden_state_index]
        if (
            hasattr(outputs, "last_hidden_state")
            and outputs.last_hidden_state is not None
        ):
            return outputs.last_hidden_state
        if isinstance(outputs, (tuple, list)) and outputs:
            return outputs[0]
        raise RuntimeError("Cannot find a sequence output in transformer outputs.")

    @staticmethod
    def _make_padding_mask(input_ids, attention_mask):
        if attention_mask is None:
            return torch.zeros(
                input_ids.shape, dtype=torch.bool, device=input_ids.device
            )
        return attention_mask == 0

    @staticmethod
    def _filter_none_kwargs(kwargs):
        return {key: value for key, value in kwargs.items() if value is not None}

    @staticmethod
    def _infer_hidden_size(config):
        for attr_name in (
            "hidden_size",
            "d_model",
            "dim",
            "n_embd",
            "emb_dim",
            "word_embed_proj_dim",
        ):
            value = getattr(config, attr_name, None)
            if value is not None:
                return value
        raise ValueError(
            f"Cannot infer transformer hidden size from config type "
            f"{type(config).__name__}. Extend _infer_hidden_size for this architecture."
        )


def build_language_encoder(config):
    """Build a transformers language encoder from a yaml-style config dictionary."""
    return PretrainedTransformersLanguageEncoder(**dict(config))


def _require_transformers():
    try:
        from transformers import AutoConfig, AutoModel
    except ImportError as exc:
        raise ImportError(
            "PretrainedTransformersLanguageEncoder requires the 'transformers' package. "
            "Install it in the active training environment first."
        ) from exc
    return AutoConfig, AutoModel


RUN_PRETRAINED_LOAD_TEST = True
PRETRAINED_SMOKE_TEST_MODELS = (
    "distilbert-base-uncased",
    "bert-base-uncased",
)


def _validate_transformers_language_encoders():
    try:
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as exc:
        print(f"[SKIP] transformers is not installed: {exc}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print(f"run_pretrained_load_test={RUN_PRETRAINED_LOAD_TEST}")

    if not RUN_PRETRAINED_LOAD_TEST:
        _validate_random_initialized_architectures(AutoConfig, device)
        return

    # Download/cache validation is intentionally narrow: only these smoke-test
    # models touch Hugging Face Hub or local pretrained checkpoints. The encoder
    # still receives tokenizer outputs, but tokenization remains a validation/data
    # pipeline concern rather than a module responsibility.
    for backbone_name in PRETRAINED_SMOKE_TEST_MODELS:
        try:
            tokenizer = AutoTokenizer.from_pretrained(backbone_name)
            batch = tokenizer(
                ["pick up the red cube", "move the blue cube to the right"],
                max_length=16,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            # {'input_ids': ..., 'token_type_ids': ..., 'attention_mask': ...}
            batch = {key: value.to(device) for key, value in batch.items()}
            encoder = PretrainedTransformersLanguageEncoder(
                backbone_name=backbone_name,
                backbone_weights_name="DEFAULT",
                freeze_backbone=False,
            ).to(device)
            encoder.eval()
            with torch.no_grad():
                seq, mask = encoder(batch)
            print(
                f"[OK]   {backbone_name:<28} seq={tuple(seq.shape)} mask={tuple(mask.shape)}"
            )
        except Exception as exc:
            print(
                f"[FAIL] {backbone_name:<28} "
                f"{_format_pretrained_load_error(backbone_name, exc)}"
            )


def _format_pretrained_load_error(backbone_name, exc):
    message = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, OSError):
        return (
            f"{message}\n"
            f"       Pretrained validation needs a reachable Hugging Face Hub entry or a "
            f"local/cache directory containing config.json and weights for {backbone_name!r}. "
            "If this machine cannot access huggingface.co, pre-download the model or put "
            "a local checkpoint path in PRETRAINED_SMOKE_TEST_MODELS."
        )
    return message


def _validate_random_initialized_architectures(AutoConfig, device):
    """Offline architecture smoke test for five representative text encoders.

    RUN_PRETRAINED_LOAD_TEST=False means: do not download tokenizers or weights.
    We still exercise the real encoder construction path by writing tiny local
    configs and loading them with backbone_weights_name=None.
    """
    import tempfile

    ok_count = 0
    fail_count = 0
    for backbone_name, config in _iter_tiny_text_model_configs(AutoConfig):
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                config.save_pretrained(tmpdir)
                batch = {
                    "input_ids": torch.randint(
                        0, config.vocab_size, (2, 16), device=device
                    ),
                    "attention_mask": torch.ones(
                        2, 16, dtype=torch.long, device=device
                    ),
                }
                batch["attention_mask"][:, -3:] = 0
                encoder = PretrainedTransformersLanguageEncoder(
                    backbone_name=tmpdir,
                    backbone_weights_name=None,
                    freeze_backbone=False,
                    local_files_only=True,
                ).to(device)
                encoder.eval()
                with torch.no_grad():
                    seq, mask = encoder(batch)
            ok_count += 1
            print(
                f"[OK]   {backbone_name:<28} seq={tuple(seq.shape)} mask={tuple(mask.shape)}"
            )
        except Exception as exc:
            fail_count += 1
            print(f"[FAIL] {backbone_name:<28} {type(exc).__name__}: {exc}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    total = ok_count + fail_count
    print(
        f"\nOffline architecture summary: {ok_count}/{total} passed, {fail_count} failed."
    )


def _iter_tiny_text_model_configs(AutoConfig):
    """Yield tiny configs for five representative text encoder families."""
    specs = (
        (
            "bert",
            {
                "vocab_size": 100,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
            },
        ),
        (
            "distilbert",
            {
                "vocab_size": 100,
                "dim": 64,
                "hidden_dim": 128,
                "n_heads": 4,
                "n_layers": 2,
            },
        ),
        (
            "roberta",
            {
                "vocab_size": 100,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
            },
        ),
        (
            "deberta-v2",
            {
                "vocab_size": 100,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
            },
        ),
        (
            "mpnet",
            {
                "vocab_size": 100,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
            },
        ),
    )

    for model_type, kwargs in specs:
        yield model_type, AutoConfig.for_model(model_type, **kwargs)


if __name__ == "__main__":
    import os

    os.environ.update(
        {
            "HTTP_PROXY": "http://127.0.0.1:7897",
            "HTTPS_PROXY": "http://127.0.0.1:7897",
            "http_proxy": "http://127.0.0.1:7897",
            "https_proxy": "http://127.0.0.1:7897",
        }
    )
    _validate_transformers_language_encoders()
