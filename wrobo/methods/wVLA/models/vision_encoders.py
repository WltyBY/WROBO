import inspect

import torch
import torchvision

from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

from wrobo.training.utils.mics import is_main_process


class PretrainedTorchvisionBackbone(nn.Module):
    def __init__(
        self,
        backbone_name,
        return_layers,
        backbone_weights_name=None,
        freeze_norm=True,
        freeze_backbone=False,
        backbone_kwargs=None,
        probe_input_size=None,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone_weights_name = backbone_weights_name
        self.return_layers = self._normalize_return_layers(return_layers)
        self.freeze_norm = freeze_norm
        self.freeze_backbone = freeze_backbone
        self.backbone_kwargs = dict(backbone_kwargs or {})
        self.probe_input_size = probe_input_size

        backbone = self.build_backbone()
        self._validate_return_layers(backbone)
        # freeze_norm freezes only norm layers; freeze_backbone (the vision-side analog of
        # the language encoder's freeze_backbone) freezes the whole backbone. When the
        # whole backbone is frozen, norm freezing is redundant.
        if self.freeze_norm and not self.freeze_backbone:
            self._freeze_norm_layers(backbone)

        self.model = IntermediateLayerGetter(backbone, return_layers=self.return_layers)

        if self.freeze_backbone:
            # Stop gradients on every backbone parameter. The module is additionally kept
            # in eval() via train() below so BatchNorm running stats and dropout are also
            # frozen — requires_grad=False alone would still let BN stats drift.
            for param in self.model.parameters():
                param.requires_grad = False

        # Output channel count per return layer (output name -> channels), probed once at
        # construction so downstream modules (e.g. the fusion layer's vision projection)
        # can size an explicit nn.Linear instead of relying on nn.LazyLinear. The channel
        # dimension is independent of the probe's spatial size, so the probe only needs an
        # input the backbone accepts (see _probe_feature_dims).
        self.feature_dims = self._probe_feature_dims()

    def forward(self, x):
        return self.model(x)

    def train(self, mode: bool = True):
        super().train(mode)
        # A fully-frozen backbone must always run in eval mode (frozen BN stats, no
        # dropout), regardless of the parent's train/eval state. Otherwise fall back to
        # keeping just the norm layers in eval when freeze_norm is set.
        if self.freeze_backbone:
            self.model.eval()
        elif self.freeze_norm:
            self._set_norm_layers_eval(self)
        return self

    def build_backbone(self):
        backbone_fn = getattr(torchvision.models, self.backbone_name.lower(), None)
        if backbone_fn is None:
            raise ValueError(
                f"Unsupported torchvision backbone name: {self.backbone_name}"
            )

        kwargs = dict(self.backbone_kwargs)
        self._set_feature_extractor_defaults(kwargs)
        self._validate_backbone_kwargs(backbone_fn, kwargs)

        weights = self._resolve_weights()
        try:
            return backbone_fn(weights=weights if is_main_process() else None, **kwargs)
        except TypeError as exc:
            if "weights" not in str(exc):
                raise
            return backbone_fn(
                pretrained=weights is not None and is_main_process(),
                **kwargs,
            )

    def _probe_feature_dims(self):
        """Return ``{output_name: channels}`` for each return layer.

        Runs a single dummy forward to read each feature map's channel dimension. That
        dimension does not depend on the spatial input size, so the probe only needs an
        input the backbone accepts. Candidate sizes are tried in order (see
        ``_candidate_probe_sizes``) and the first that forwards successfully is used.
        """
        was_training = self.training
        # eval() so frozen/regular norm layers tolerate a batch size of 1 during probing.
        self.eval()
        errors = []
        try:
            with torch.no_grad():
                for size in self._candidate_probe_sizes():
                    x = torch.zeros(1, 3, size, size)
                    try:
                        outputs = self.model(x)
                    except Exception as exc:  # try the next candidate size
                        errors.append(f"{size}: {type(exc).__name__}: {exc}")
                        continue
                    return {
                        name: tensor.shape[1] for name, tensor in outputs.items()
                    }
        finally:
            self.train(was_training)

        raise RuntimeError(
            f"Could not probe feature dims for backbone {self.backbone_name!r} with "
            f"return_layers {self.return_layers}. Tried input sizes and failed:\n  "
            + "\n  ".join(errors)
            + "\nPass probe_input_size=<int> with a resolution this backbone accepts."
        )

    def _candidate_probe_sizes(self):
        """Ordered probe resolutions: explicit override, then the weights' canonical
        crop size, then a fallback ladder covering common backbone requirements."""
        sizes = []
        if self.probe_input_size is not None:
            sizes.append(int(self.probe_input_size))

        # torchvision weights enums expose the official input transform; its crop_size is
        # the canonical resolution for that backbone (e.g. inception_v3 -> 299, ViT -> its
        # configured size). This is metadata only, so it needs no weight download.
        weights = self._resolve_weights()
        crop = self._weights_crop_size(weights)
        if crop is not None:
            sizes.append(crop)

        sizes.extend([224, 256, 299, 384, 518])
        # de-duplicate while preserving order
        seen = set()
        ordered = []
        for size in sizes:
            if size not in seen:
                seen.add(size)
                ordered.append(size)
        return ordered

    @staticmethod
    def _weights_crop_size(weights):
        if weights is None or isinstance(weights, str):
            return None
        try:
            transform = weights.transforms()
        except Exception:
            return None
        crop_size = getattr(transform, "crop_size", None)
        if crop_size is None:
            return None
        if isinstance(crop_size, (list, tuple)):
            return int(crop_size[0])
        return int(crop_size)

    def _set_feature_extractor_defaults(self, kwargs):
        if self.backbone_name.lower() in ("inception_v3", "googlenet"):
            kwargs.setdefault("aux_logits", False)
            kwargs.setdefault("init_weights", False)

    def _validate_return_layers(self, backbone):
        children = dict(backbone.named_children())
        missing_layers = sorted(set(self.return_layers) - set(children))
        if missing_layers:
            raise ValueError(
                f"return_layers contains layers that do not exist in {self.backbone_name}: "
                f"{missing_layers}. Available top-level layers: {list(children)}"
            )

        module_list_layers = [
            layer_name
            for layer_name in self.return_layers
            if isinstance(children[layer_name], nn.ModuleList)
        ]
        if module_list_layers:
            raise TypeError(
                f"return_layers for {self.backbone_name} contains nn.ModuleList layers "
                f"that cannot be called directly by IntermediateLayerGetter: "
                f"{module_list_layers}. Choose a forwardable child module, or implement "
                "a model-specific adapter for this backbone."
            )

    @staticmethod
    def _validate_backbone_kwargs(backbone_fn, kwargs):
        signature = inspect.signature(backbone_fn)
        if any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        ):
            return

        valid_kwargs = set(signature.parameters)
        invalid_kwargs = sorted(set(kwargs) - valid_kwargs)
        if invalid_kwargs:
            raise TypeError(
                f"Unsupported kwargs for {backbone_fn.__name__}: {invalid_kwargs}. "
                f"Valid kwargs: {sorted(valid_kwargs)}"
            )

    @staticmethod
    def _is_norm_layer(module):
        norm_types = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.GroupNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
            nn.LocalResponseNorm,
        )
        return isinstance(module, norm_types)

    @classmethod
    def _freeze_norm_layers(cls, module):
        for submodule in module.modules():
            if not cls._is_norm_layer(submodule):
                continue
            submodule.eval()
            for param in submodule.parameters(recurse=False):
                param.requires_grad = False

    @classmethod
    def _set_norm_layers_eval(cls, module):
        for submodule in module.modules():
            if cls._is_norm_layer(submodule):
                submodule.eval()

    def _normalize_return_layers(self, return_layers):
        if isinstance(return_layers, list):
            return {layer_name: layer_name for layer_name in return_layers}
        if isinstance(return_layers, dict):
            return return_layers
        raise TypeError(
            "return_layers must be a list of backbone layer names "
            "or a dict mapping backbone layer names to output names"
        )

    @staticmethod
    def _is_no_weights_name(weight_name):
        if weight_name is None or weight_name is False:
            return True
        if isinstance(weight_name, str):
            return weight_name.strip().lower() in {"", "none", "null", "false", "no", "off"}
        return False

    def _resolve_weights(self):
        weight_name = self.backbone_weights_name
        if self._is_no_weights_name(weight_name):
            return None
        if weight_name is True:
            weight_name = "DEFAULT"

        get_model_weights = getattr(torchvision.models, "get_model_weights", None)
        weights_enum = None
        if get_model_weights is not None:
            try:
                weights_enum = get_model_weights(self.backbone_name.lower())
            except (KeyError, ValueError):
                weights_enum = None

        if weights_enum is None:
            normalized_name = self.backbone_name.lower().replace("_", "")
            for attr_name in dir(torchvision.models):
                if not attr_name.endswith("_Weights"):
                    continue
                normalized_attr_name = (
                    attr_name.removesuffix("_Weights").lower().replace("_", "")
                )
                if normalized_attr_name == normalized_name:
                    weights_enum = getattr(torchvision.models, attr_name)
                    break

        if weights_enum is None:
            return weight_name

        if not isinstance(weight_name, str):
            return weight_name

        weight_name = weight_name.upper()
        if hasattr(weights_enum, weight_name):
            return getattr(weights_enum, weight_name)
        try:
            return weights_enum[weight_name]
        except KeyError as exc:
            valid_weights = [name for name in dir(weights_enum) if name.isupper()]
            raise ValueError(
                f"Unsupported weights '{self.backbone_weights_name}' for {self.backbone_name}. "
                f"Available weights: {valid_weights}"
            ) from exc


def _infer_validation_return_layer(backbone_name, backbone):
    children = dict(backbone.named_children())
    if not children:
        return None

    # ViT's encoder is not directly reachable through IntermediateLayerGetter because
    # torchvision reshapes/permutes tokens in the parent forward before calling it.
    # Returning conv_proj still validates that this wrapper can expose a useful feature map.
    if backbone_name.lower().startswith("vit_") and "conv_proj" in children:
        return "conv_proj"
    if backbone_name.lower().startswith("maxvit_") and "stem" in children:
        return "stem"

    preferred_layers = (
        "layer4",  # ResNet / ResNeXt / Wide-ResNet
        "features",  # AlexNet / VGG / DenseNet / MobileNet / EfficientNet / ConvNeXt / Swin
        "trunk_output",  # RegNet
        "blocks",  # MaxViT
        "layers",  # MNASNet
        "Mixed_7c",  # Inception v3
        "inception5b",  # GoogLeNet
        "conv5",  # ShuffleNet v2
        "conv_proj",  # fallback for some patchify-style models
    )
    for layer_name in preferred_layers:
        if layer_name in children:
            return layer_name

    ignored_suffixes = ("classifier", "classif", "head", "heads", "fc")
    ignored_exact = {"avgpool", "dropout", "flatten"}
    for layer_name in reversed(children):
        lowered = layer_name.lower()
        if lowered in ignored_exact or lowered.endswith(ignored_suffixes):
            continue
        return layer_name
    return next(reversed(children))


def _iter_torchvision_model_names():
    list_models = getattr(torchvision.models, "list_models", None)
    if list_models is not None:
        return sorted(list_models())

    model_names = []
    for name in dir(torchvision.models):
        if name.startswith("_") or name.endswith("_Weights"):
            continue
        obj = getattr(torchvision.models, name)
        if not callable(obj):
            continue
        try:
            signature = inspect.signature(obj)
        except (TypeError, ValueError):
            continue
        if "weights" in signature.parameters or "pretrained" in signature.parameters:
            model_names.append(name)
    return sorted(model_names)


def _validation_backbone_kwargs(backbone_name):
    if backbone_name == "inception_v3":
        return {"aux_logits": False, "init_weights": False}
    if backbone_name == "googlenet":
        return {"aux_logits": False, "init_weights": False}
    return {}


def _validate_default_weight_resolution(backbone_name):
    probe = PretrainedTorchvisionBackbone.__new__(PretrainedTorchvisionBackbone)
    probe.backbone_name = backbone_name
    probe.backbone_weights_name = "DEFAULT"
    weights = probe._resolve_weights()
    if isinstance(weights, str):
        raise RuntimeError(
            f"DEFAULT weights for {backbone_name} were not resolved to a torchvision "
            f"weights enum: {weights!r}"
        )
    return weights


def _validate_pretrained_load(backbone_name, return_layer, backbone_kwargs, device, x):
    model = PretrainedTorchvisionBackbone(
        backbone_name=backbone_name,
        backbone_weights_name="DEFAULT",
        return_layers={return_layer: "output"},
        freeze_norm=True,
        backbone_kwargs=backbone_kwargs,
    ).to(device)
    model.eval()
    with torch.no_grad():
        output = model(x)
    if "output" not in output:
        raise RuntimeError(
            f"missing output key after pretrained load, got {list(output)}"
        )
    return tuple(output["output"].shape)


def _validate_torchvision_backbones():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 3, 320, 320, device=device)
    results = []
    weight_results = []
    pretrained_results = []

    print(f"torchvision={torchvision.__version__}")
    print(f"device={device}")
    print(f"run_pretrained_load_test={RUN_PRETRAINED_LOAD_TEST}")

    for backbone_name in _iter_torchvision_model_names():
        backbone_fn = getattr(torchvision.models, backbone_name, None)
        if backbone_fn is None:
            continue

        try:
            backbone_kwargs = _validation_backbone_kwargs(backbone_name)
            raw_backbone = backbone_fn(weights=None, **backbone_kwargs)
            return_layer = _infer_validation_return_layer(backbone_name, raw_backbone)
            del raw_backbone
            if return_layer is None:
                raise RuntimeError("cannot infer a return layer")

            model = PretrainedTorchvisionBackbone(
                backbone_name=backbone_name,
                backbone_weights_name=None,
                return_layers={return_layer: "output"},
                freeze_norm=True,
                backbone_kwargs=backbone_kwargs,
            ).to(device)
            model.eval()

            with torch.no_grad():
                output = model(x)

            if "output" not in output:
                raise RuntimeError(f"missing output key, got {list(output)}")
            output_shape = tuple(output["output"].shape)

            norm_layers = [
                module for module in model.modules() if model._is_norm_layer(module)
            ]
            norm_params_frozen = all(
                not param.requires_grad
                for module in norm_layers
                for param in module.parameters(recurse=False)
            )
            model.train()
            norm_layers_eval_after_train = all(
                not module.training for module in norm_layers
            )
            if norm_layers and (
                not norm_params_frozen or not norm_layers_eval_after_train
            ):
                raise RuntimeError(
                    "freeze_norm failed: "
                    f"params_frozen={norm_params_frozen}, "
                    f"eval_after_train={norm_layers_eval_after_train}"
                )

            weights = _validate_default_weight_resolution(backbone_name)
            weight_results.append((backbone_name, True, repr(weights), None))

            pretrained_shape = None
            if (
                RUN_PRETRAINED_LOAD_TEST
                and backbone_name in PRETRAINED_SMOKE_TEST_MODELS
            ):
                try:
                    pretrained_shape = _validate_pretrained_load(
                        backbone_name,
                        return_layer,
                        backbone_kwargs,
                        device,
                        x,
                    )
                    pretrained_results.append(
                        (backbone_name, True, pretrained_shape, None)
                    )
                except Exception as exc:
                    pretrained_results.append((backbone_name, False, None, repr(exc)))

            results.append((backbone_name, True, return_layer, output_shape, None))
            message = (
                f"[OK]   {backbone_name:<28} return={return_layer:<12} "
                f"shape={str(output_shape):<20} weights={weights:<45} "
            )
            if pretrained_shape is not None:
                message += f" pretrained_shape={pretrained_shape}"
            print(message)
        except Exception as exc:
            results.append((backbone_name, False, None, None, repr(exc)))
            weight_results.append((backbone_name, False, None, repr(exc)))
            print(f"[FAIL] {backbone_name:<28} {type(exc).__name__}: {exc}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    ok_count = sum(ok for _, ok, _, _, _ in results)
    fail_count = len(results) - ok_count
    weight_ok_count = sum(ok for _, ok, _, _ in weight_results)
    weight_fail_count = len(weight_results) - weight_ok_count
    pretrained_ok_count = sum(ok for _, ok, _, _ in pretrained_results)
    pretrained_fail_count = len(pretrained_results) - pretrained_ok_count

    print(f"\nForward summary: {ok_count}/{len(results)} passed, {fail_count} failed.")
    print(
        f"DEFAULT weights resolution summary: "
        f"{weight_ok_count}/{len(weight_results)} passed, {weight_fail_count} failed."
    )
    if RUN_PRETRAINED_LOAD_TEST:
        print(
            f"Pretrained load smoke summary: "
            f"{pretrained_ok_count}/{len(pretrained_results)} passed, "
            f"{pretrained_fail_count} failed."
        )
    else:
        print(
            "Pretrained load smoke summary: skipped "
            "(set RUN_PRETRAINED_LOAD_TEST = True to enable downloads/cache loads)."
        )

    if fail_count:
        print("Failed models:")
        for backbone_name, ok, _, _, error in results:
            if not ok:
                print(f"  - {backbone_name}: {error}")
    if weight_fail_count:
        print("Failed DEFAULT weights resolution:")
        for backbone_name, ok, _, error in weight_results:
            if not ok:
                print(f"  - {backbone_name}: {error}")
    if pretrained_fail_count:
        print("Failed pretrained load smoke tests:")
        for backbone_name, ok, _, error in pretrained_results:
            if not ok:
                print(f"  - {backbone_name}: {error}")


RUN_PRETRAINED_LOAD_TEST = True
PRETRAINED_SMOKE_TEST_MODELS = (
    "resnet18",
    "mobilenet_v3_small",
    "convnext_tiny",
    "vit_b_16",
)


if __name__ == "__main__":
    _validate_torchvision_backbones()
