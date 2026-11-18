from collections.abc import Mapping

import torch
from torch._dynamo import OptimizedModule


# Only for the .pth checkpoint saved in our repository.
def load_pretrained_weights(
    network,
    fname: str,
    skip_strings_in_pretrained=None,
    verbose: bool = False,
):
    """Load model weights from either supported wrobo checkpoint format.

    Full/retrainable checkpoints store weights under ``network_weights``; inference
    checkpoints are a bare state dict. Optimizer and other training state are ignored.
    """
    if skip_strings_in_pretrained is None:
        skip_strings_in_pretrained = []

    saved_model = torch.load(fname, map_location="cpu", weights_only=False)
    if not isinstance(saved_model, Mapping):
        raise TypeError(
            f"Unsupported checkpoint format in {fname}: expected a mapping, "
            f"got {type(saved_model).__name__}"
        )

    if "network_weights" in saved_model:
        pretrained_dict = saved_model["network_weights"]
        checkpoint_format = "full"
    else:
        pretrained_dict = saved_model
        checkpoint_format = "weights-only"

    if not isinstance(pretrained_dict, Mapping):
        raise TypeError(
            f"Unsupported {checkpoint_format} checkpoint in {fname}: model weights "
            "must be a state-dict mapping"
        )

    mod = network
    while isinstance(mod, (OptimizedModule, torch.nn.parallel.DistributedDataParallel)):
        mod = mod._orig_mod if isinstance(mod, OptimizedModule) else mod.module

    # Remove 'module.' prefix from pretrained weights (DDP compatibility)
    cleaned_pretrained = {}
    for k, v in pretrained_dict.items():
        if not isinstance(k, str):
            raise TypeError(
                f"Invalid state-dict key in {fname}: expected str, got {type(k).__name__}"
            )
        if k.startswith("module."):
            k = k[7:]  # strip 'module.' prefix
        cleaned_pretrained[k] = v
    pretrained_dict = cleaned_pretrained

    model_dict = mod.state_dict()

    # Verify all non-skipped layers exist and match shapes
    for key, _ in model_dict.items():
        if all([i not in key for i in skip_strings_in_pretrained]):
            assert key in pretrained_dict, (
                f"Key {key} is missing in the pretrained model weights. "
                f"The pretrained weights do not seem to be compatible with your network."
            )
            assert model_dict[key].shape == pretrained_dict[key].shape, (
                f"The shape of the parameters of key {key} is not the same. "
                f"Pretrained model: {pretrained_dict[key].shape}; "
                f"your network: {model_dict[key].shape}."
            )

    # Filter pretrained dict to only include compatible layers
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if k in model_dict.keys()
        and all([i not in k for i in skip_strings_in_pretrained])
    }

    model_dict.update(pretrained_dict)

    print(
        f"################### Loading {checkpoint_format} pretrained weights from file ",
        fname,
        "###################",
    )
    if verbose:
        print(
            "Below is the list of overlapping blocks in pretrained model and loaded model architecture:"
        )
        for key, value in pretrained_dict.items():
            print(key, "shape", value.shape)
        print("################### Done ###################")
    mod.load_state_dict(model_dict)
