import torch
from typing import Dict, List, Any

class BaseCollater:
    """
    Collate function for EpisodicDataset.
    Stacks all tensor keys along a new batch dimension.
    Non-tensor values (if any) are kept as lists.
    """
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        collated = {}
        for key in batch[0].keys():
            values = [item[key] for item in batch]
            if all(isinstance(v, torch.Tensor) for v in values):
                collated[key] = torch.stack(values)
            else:
                collated[key] = values
        
        img_lst = []
        num_cam = len(batch[0]["image"])
        for i in range(num_cam):
             img_lst.append(torch.stack([b["image"][i] for b in batch], dim=0))
        collated["image"] = img_lst
        return collated