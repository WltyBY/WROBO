import torch
import torchvision

from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import FrozenBatchNorm2d

from wrobo.training.utils.mics import is_main_process


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed

    Args:
        num_features (int): Number of features ``C`` from an expected input of size ``(N, C, H, W)``
        eps (float): a value added to the denominator for numerical stability. Default: 1e-5
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
    ):
        super(FrozenBatchNorm2d, self).__init__()
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: Tensor) -> Tensor:
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class ResNetBackBone(nn.Module):
    def __init__(
        self, backbone_name, backbone_weights_name=None, return_interm_layers=False, dilation=False
    ):
        super().__init__()
        assert backbone_name.lower() in [
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
        ], f"Unsupported backbone name: {backbone_name}"
        self.backbone_name = backbone_name
        self.backbone_weights_name = backbone_weights_name
        self.dilation = dilation

        if return_interm_layers:
            return_layers = {"layer1": "1", "layer2": "2", "layer3": "3", "layer4": "4"}
        else:
            return_layers = {"layer4": "4"}

        self.model = IntermediateLayerGetter(
            self.build_backbone(), return_layers=return_layers
        )
        self.num_channels = 512 if self.backbone_name.lower() in ('resnet18', 'resnet34') else 2048

    def forward(self, x):
        return self.model(x)

    def build_backbone(self):
        backbone_fn = getattr(torchvision.models, self.backbone_name.lower())
        # If 'backbone_pretrained' is not specified, it will be None
        weight_name = self.backbone_weights_name
        weight_name = weight_name.upper() if weight_name is not None else None

        # load pretrained weights if specified
        # note: new version of torchvision uses 'weights' instead of 'pretrained' argument,
        # but we want to maintain compatibility with older versions
        try:
            backbone = backbone_fn(
                weights=weight_name,
                replace_stride_with_dilation=[False, False, self.dilation],
                pretrained=is_main_process(),
                norm_layer=FrozenBatchNorm2d,
            )
        except TypeError:
            # For old version of torchvision
            backbone = backbone_fn(
                replace_stride_with_dilation=[False, False, self.dilation],
                pretrained=weight_name is not None and is_main_process(),
                norm_layer=FrozenBatchNorm2d,
            )

        return backbone
