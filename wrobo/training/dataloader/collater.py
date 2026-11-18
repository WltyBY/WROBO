import torch
from typing import Dict, List, Any

class BaseCollater:
    """
    Collate function for EpisodicDataset.
    Stacks all tensor keys along a new batch dimension.
    Non-tensor values (if any) are kept as lists.

    Now handles flat dictionary format where each camera is a separate key.
    Example input batch:
        [{"image_angle": tensor, "image_wrist": tensor, "proprio_state": tensor}, ...]
    Example output:
        {"image_angle": stacked_tensor, "image_wrist": stacked_tensor, "proprio_state": stacked_tensor}
    """
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        collated = {}
        for key in batch[0].keys():
            values = [item[key] for item in batch]
            if all(isinstance(v, torch.Tensor) for v in values):
                collated[key] = torch.stack(values)
            else:
                collated[key] = values
        return collated