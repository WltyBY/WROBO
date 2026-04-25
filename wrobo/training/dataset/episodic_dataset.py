import os
import h5py
import torch

import numpy as np

from typing import List, Dict
from collections import OrderedDict
from torch.utils.data import Dataset

from wrobo.utils.file_operations import open_json


class EpisodicDataset(Dataset):
    """
    each episode is stored in a separate HDF5 file, with the following structure:
    ├── attrs: {"dataset": "ACT", "success": True, "comment": "<comment>", ...}
    ├── observations (Group)        # images
    │   ├── image_wrist (Dataset)   # [T, C, H, W] - uint8
    │   ├── image_third (Dataset)   # [T, C, H, W] - uint8
    │   ├── proprio_state (Dataset) # proprioceptive state, [T, D1] - float32
    │   └── ... (other datasets, e.g. "proprio_vel", other cameras, etc.)
    ├── action_absolute (Dataset)  # absolute action sequence, [T, D2] - float32
    ├── action_relative (Dataset)  # relative action sequence, [T, D1] - float32
    └── ... (other datasets, e.g. "rewards", etc.)

    data_keys: e.g. ["observations/image_wrist", "observations/image_front", "observations/proprio_state", "action_absolute", ...]
    """

    def __init__(
        self,
        dataset_dir: str,
        split: str,
        fold: str,
        data_keys: List,
        history_width: int,
        action_chunk_size: int,
        norm_stats: Dict,
        img_transforms=None,
        biased_sample: bool = False,
    ):
        super(EpisodicDataset).__init__()
        self.dataset_dir = dataset_dir
        self.split = split
        self.fold = fold
        self.data_keys = data_keys
        self.history_width = history_width
        self.action_chunk_size = action_chunk_size
        self.norm_stats = norm_stats
        self.img_transforms = img_transforms
        self.biased_sample = biased_sample

        self.cam_keys = [
            key.replace("observations/", "")
            for key in data_keys
            if "image" in key and "observations" in key
        ]

        self.episode_ids = self._get_episode_ids()
        self._h5_cache = OrderedDict()
        self._h5_cache_size = min(len(self), 256)
        self._h5_rdcc_nbytes = 1024**2 * 32  # 32MB cache for each file

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, idx):
        """
        return a dict of data for the episode at idx, with keys specified in self.data_keys.
        Tensors are with shape [num_cam, history_width, C, H, W] for images and [history_width, D] for proprioceptive states (Observations),
        and [action_chunk_size, D] for actions. If the episode is shorter than history_width or action_chunk_size, it will be padded with zeros.
        """
        episode_id = self.episode_ids[idx]
        episode_path = os.path.join(self.dataset_dir, f"{episode_id}.hdf5")

        data_dict = dict()
        root = self._get_h5_file(episode_path)
        original_action_shape = root["/action_absolute"].shape
        episode_len = original_action_shape[0]

        # sample a timestep
        if self.biased_sample:
            # bias towards sampling later timesteps, which have more future and complex actions
            u = np.random.beta(a=2.0, b=1.0)
            sample_ts = int(u * (episode_len - 1))
        else:
            sample_ts = np.random.randint(episode_len)

        for key in self.data_keys:
            # assert (
            #     "/" + key in root
            # ), f"Key {key} not found in episode file {episode_path}, available keys: {[k for k in root.keys()]}"

            if key.startswith("observations"):
                # Observations are no need for chunking, just load the data at start_ts
                obs_seq = root["/" + key]

                start_idx = sample_ts - self.history_width + 1
                end_idx = sample_ts + 1
                if start_idx < 0:
                    pad_len = -start_idx
                    start_idx = 0

                    obs_chunk = obs_seq[:end_idx]
                    if obs_chunk.shape[0] == 0:
                        raise ValueError(f"Empty observation in {episode_path}")

                    pad = np.repeat(obs_chunk[:1], pad_len, axis=0)

                    obs_chunk = np.concatenate(
                        [pad, obs_chunk], axis=0
                    )  # history_width, D or history_width, C, H, W for images
                else:
                    obs_chunk = obs_seq[start_idx:end_idx]

                if "image" in key:
                    # For image
                    obs = torch.from_numpy(
                        obs_chunk
                    )  # / 255.0 is no need, while using transforms.ToTensor().
                    if self.img_transforms:
                        obs = torch.stack([self.img_transforms(frame) for frame in obs])
                    else:
                        obs = obs.float() / 255.0
                else:
                    obs = torch.from_numpy(obs_chunk).float()

                data_dict[key.replace("observations/", "")] = obs
            else:
                # Actions need to be chunked here
                actions_seq = root["/" + key][sample_ts:]

                action_len = episode_len - sample_ts
                padded_action = np.zeros(
                    (self.action_chunk_size, *actions_seq.shape[1:]),
                    dtype=np.float32,
                )
                if action_len >= self.action_chunk_size:
                    # if enough actions, then get the first chunk of actions
                    padded_action[:] = actions_seq[: self.action_chunk_size]
                    if "is_pad" not in data_dict:
                        is_pad = np.zeros(
                            self.action_chunk_size
                        )  # all real data, no padding
                else:
                    # if not enough actions, then pad with zeros to the chunk size
                    padded_action[:action_len] = actions_seq
                    if "is_pad" not in data_dict:
                        is_pad = np.zeros(self.action_chunk_size)
                        is_pad[action_len:] = 1  # 1 for padding steps, 0 for real data

                # normalization for actions
                if key in self.norm_stats:
                    mean = np.asarray(self.norm_stats[key]["mean"])
                    std = np.asarray(self.norm_stats[key]["std"])
                    padded_action = (padded_action - mean) / (std + 1e-6)

                data_dict[key] = torch.from_numpy(padded_action).float()
                if "is_pad" not in data_dict:
                    data_dict["is_pad"] = torch.from_numpy(is_pad).bool()

        # stack a num_cameras dimension for image observations, to be compatible with the model input
        if self.cam_keys:
            image_lst = [data_dict[key] for key in self.cam_keys]
            # [history_width, C, H, W] -> [num_cam, history_width, C, H, W]
            data_dict["image"] = torch.stack(image_lst, dim=0)
            for key in self.cam_keys:
                del data_dict[key]

        return data_dict

    def _get_episode_ids(self):
        split_file_path = os.path.join(self.dataset_dir, "dataset_split.json")
        dataset_split = open_json(split_file_path)
        if self.fold == "all":
            if self.split in ["train", "val"]:
                ids = dataset_split["0"]["train"] + dataset_split["0"]["val"]
            elif self.split == "test":
                ids = dataset_split[self.split]
            else:
                raise Exception('Para:split should be "train", "val" or "test".')
        else:
            if self.split in ["train", "val"]:
                ids = dataset_split[self.fold][self.split]
            elif self.split == "test":
                ids = dataset_split[self.split]
            else:
                raise Exception('Para:split should be "train", "val" or "test".')
        return ids

    def _get_h5_file(self, path):
        if path in self._h5_cache:
            # move to end to show that it was recently used
            self._h5_cache.move_to_end(path)
            return self._h5_cache[path]

        # For newly opened file
        f = h5py.File(
            path, "r", libver="latest", swmr=True, rdcc_nbytes=self._h5_rdcc_nbytes
        )

        self._h5_cache[path] = f

        # If cache is full, remove the least recently used file
        if len(self._h5_cache) > self._h5_cache_size:
            old_path, old_file = self._h5_cache.popitem(last=False)
            try:
                old_file.close()
            except:
                pass

        return f

    def close(self):
        for f in self._h5_cache.values():
            try:
                f.close()
            except:
                pass
        self._h5_cache = {}

    def __del__(self):
        self.close()
