import os
import h5py
import torch
import warnings

import numpy as np

from typing import List, Dict, Union
from collections import OrderedDict
from torch.utils.data import Dataset

from wrobo.utils.image_codec import get_encoding, decode_frames, is_encoded
from wrobo.utils.normalization import normalization_parameters
from wrobo.training.dataset.instruction import INSTRUCTION_ATTR, build_instruction_provider, normalize_instruction_modes
from wrobo.training.dataset.selection import _validated_task_ids, resolve_episode_ids
from wrobo.training.dataset.sampling import (
    SamplingMode,
    normalize_biased_samples,
    normalize_uniform_sample_probs,
    sample_timestep,
)


class EpisodicDataset(Dataset):
    """
    each episode is stored in a separate HDF5 file, with the following structure:
    ├── attrs: {"dataset": "ACT", "seq_len": T, "success": True, "comment": "<comment>", ...}
    ├── observations (Group)        # images
    │   ├── image_wrist (Dataset)   # [T, C, H, W] - uint8
    │   ├── image_top (Dataset)     # [T, C, H, W] - uint8
    │   ├── proprio_state (Dataset) # proprioceptive state, [T, D1] - float32
    │   └── ... (other datasets, e.g. "proprio_vel", other cameras, etc.)
    ├── action_abs (Dataset)        # absolute action sequence, [T, D1] - float32
    ├── action_rel (Dataset)        # relative action sequence, [T, D1] - float32
    └── ... (other datasets, e.g. "ee_pos_abs", "ee_pos_rel", "rewards", etc.)

    data_keys: e.g. ["observations/image_wrist", "observations/image_top", "observations/proprio_state", "action_abs", ...]

    Multi-dataset support:
    - dataset_dir can be a single string (single dataset) or a list of strings (multiple datasets)
    - When using multiple datasets, each dataset must have its own dataset_split.json
    - Episode IDs are automatically prefixed with dataset index to avoid collisions
    - biased_sample can be a shared mode or a list of modes in dataset_dir order
    - uniform_sample_prob mixes uniform draws into early/late sampling per item
    - task_ids selects complete trajectories independently for each dataset
    - normalization_mode chooses standard mean/std or quantile center/half-width
    - norm_stats can be:
      * Unified dict (same stats for all datasets)
      * Per-dataset dict: {dataset_dir: {key: {mean, std}}}
    """

    def __init__(
        self,
        dataset_dir: Union[str, List[str]],
        split: str,
        fold: str,
        data_keys: List,
        history_width: int,
        action_chunk_size: int,
        norm_stats: Union[Dict, Dict[str, Dict]],
        img_transforms=None,
        text_transform=None,
        biased_sample: Union[SamplingMode, List[SamplingMode]] = "uniform",
        instruction_mode=None,
        uniform_sample_prob: Union[float, List[float]] = 0.0,
        task_ids=None,
        normalization_mode="standard",
    ):
        super(EpisodicDataset).__init__()
        self._h5_cache = OrderedDict()

        # Normalize dataset_dir to always be a list
        if isinstance(dataset_dir, str):
            self.dataset_dirs = [dataset_dir]
        elif isinstance(dataset_dir, list):
            self.dataset_dirs = dataset_dir
        else:
            raise TypeError(
                f"dataset_dir must be str or list of str, got {type(dataset_dir)}"
            )

        self.is_multi_dataset = len(self.dataset_dirs) > 1

        self.dataset_dir = self.dataset_dirs[0] if len(self.dataset_dirs) == 1 else self.dataset_dirs

        if task_ids is None:
            self.task_ids = [None] * len(self.dataset_dirs)
        elif not isinstance(task_ids, list) or len(task_ids) != len(self.dataset_dirs):
            raise ValueError("task_ids must contain one selection per dataset_dir")
        else:
            self.task_ids = [_validated_task_ids(selection) for selection in task_ids]
        if normalization_mode not in ("standard", "quantile"):
            raise ValueError("normalization_mode must be 'standard' or 'quantile'")
        self.normalization_mode = normalization_mode

        self.split = split
        self.fold = fold
        self.data_keys = data_keys
        self.history_width = history_width
        self.action_chunk_size = action_chunk_size

        self.per_dataset_norm_stats = isinstance(norm_stats, dict) and any(
            directory in norm_stats for directory in self.dataset_dirs
        )
        if self.per_dataset_norm_stats:
            missing = [directory for directory in self.dataset_dirs if directory not in norm_stats]
            if missing:
                raise ValueError(f"Normalization statistics are missing dataset groups: {missing}")
            self.norm_stats_dict = {directory: norm_stats[directory] for directory in self.dataset_dirs}
        else:
            self.norm_stats_dict = {d: norm_stats for d in self.dataset_dirs}

        self.norm_stats = norm_stats
        self.img_transforms = img_transforms
        self.text_transform = text_transform
        self.biased_sample = biased_sample
        self.biased_samples = normalize_biased_samples(biased_sample, len(self.dataset_dirs))
        self.uniform_sample_prob = uniform_sample_prob
        self.uniform_sample_probs = normalize_uniform_sample_probs(
            uniform_sample_prob, len(self.dataset_dirs)
        )

        self.cam_keys = [
            key.replace("observations/", "")
            for key in data_keys
            if "image" in key and "observations" in key
        ]

        # Optional language-instruction channel. Each enabled dataset must provide a
        # unified instruction_library.json and string-valued episode metadata.
        self.use_instruction = "instruction" in data_keys
        self.instruction_modes = (
            normalize_instruction_modes(instruction_mode, len(self.dataset_dirs))
            if self.use_instruction else None
        )
        # Build instruction providers for each dataset directory
        if self.use_instruction:
            self.instruction_providers = {
                dataset_dir: build_instruction_provider(dataset_dir)
                for dataset_dir in self.dataset_dirs
            }
        else:
            self.instruction_providers = None

        # When language is consumed, a tokenizer is mandatory: the dataset's job is to
        # yield model-ready tensors (the text-side analog of img_transforms turning raw
        # frames into normalized tensors), so it must not leak a raw string. Methods
        # that don't request "instruction" are unaffected (text_transform stays None).
        if self.use_instruction and self.text_transform is None:
            raise ValueError(
                "'instruction' is in data_keys but no text_transform (tokenizer) was "
                "provided. Pass a tokenizer so the dataset can return token ids."
            )

        # Build episode ID mapping: {episode_id: dataset_dir}
        self.episode_ids, self.episode_to_dataset = self._get_episode_ids()
        # Optimize cache size for better memory usage
        # For 64GB machine with batch_size=8: use smaller cache to reduce memory footprint
        self._h5_cache_size = min(len(self), 64)  # Reduced from 256 to 64
        self._h5_rdcc_nbytes = 1024**2 * 16  # Reduced from 32MB to 16MB per file

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, idx):
        """
        return a dict of data for the episode at idx, with keys specified in self.data_keys.
        Tensors are with shape [num_cam, history_width, C, H, W] for images and [history_width, D] for proprioceptive states (Observations),
        and [action_chunk_size, D] for actions. If the episode is shorter than history_width or action_chunk_size, it will be padded with zeros.
        """
        episode_id = self.episode_ids[idx]
        dataset_dir = self.episode_to_dataset[episode_id]
        dataset_idx = (
            int(episode_id.split("_", 1)[0][len("dataset"):])
            if self.is_multi_dataset else 0
        )

        # Get the appropriate norm_stats for this episode's dataset
        current_norm_stats = self.norm_stats_dict[dataset_dir]

        # For multi-dataset, strip the prefix to get the actual episode filename
        if self.is_multi_dataset:
            # episode_id format: "dataset{i}_episode_xxx"
            # Extract the actual episode name after the prefix
            actual_episode_id = "_".join(episode_id.split("_")[1:])
        else:
            actual_episode_id = episode_id

        # Episodes are in episodes/ subdirectory
        episode_path = os.path.join(dataset_dir, "episodes", f"{actual_episode_id}.hdf5")

        data_dict = dict()
        root = self._get_h5_file(episode_path)
        episode_len = root.attrs["seq_len"]

        # Select the distribution independently each time this item is sampled.
        sample_ts = sample_timestep(
            episode_len,
            self.biased_samples[dataset_idx],
            self.uniform_sample_probs[dataset_idx],
        )

        for key in self.data_keys:
            # The language instruction is not an hdf5 dataset; it is produced by the
            # instruction provider after this tensor loop. Skip it here so the existing
            # observation/action handling stays untouched.
            if key == "instruction":
                continue

            # assert (
            #     "/" + key in root
            # ), f"Key {key} not found in episode file {episode_path}, available keys: {[k for k in root.keys()]}"

            if key.startswith("observations"):
                obs_seq = root["/" + key]
                # image datasets may be stored as per-frame encoded byte strings;
                # capture the encoding now (slicing returns a plain array w/o attrs).
                img_is_encoded = (
                    is_encoded(get_encoding(obs_seq)) if "image" in key else False
                )

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
                    # Decode per-frame byte strings back to a dense (his_width, C, H, W)
                    # uint8 array if the dataset is encoded (png/jpg); "raw" is already
                    # a dense array. Padding above operated on the (object) byte strings,
                    # which is fine since np.repeat/concatenate just duplicate elements.
                    if img_is_encoded:
                        obs_chunk = decode_frames(obs_chunk)

                    # For image
                    obs = torch.from_numpy(
                        obs_chunk
                    )  # / 255.0 is no need, while using transforms.ToTensor().

                    if self.img_transforms:
                        obs = torch.stack([self.img_transforms(frame) for frame in obs])
                    else:
                        # No transform: scale to [0, 1] then normalize per-channel with
                        # the dataset's image stats (computed on /255 pixels in
                        # get_norm_stats), mirroring ToTensor + Normalize.
                        obs = obs.float() / 255.0  # (his_width, C, H, W)
                        norm_key = key.replace("observations/", "")
                        if norm_key in current_norm_stats:
                            # stats are per-channel (C,); broadcast over H, W
                            mean, std = normalization_parameters(current_norm_stats[norm_key], "standard")
                            mean = torch.as_tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
                            std = torch.as_tensor(std, dtype=torch.float32).view(1, -1, 1, 1)
                            obs = (obs - mean) / std
                        else:
                            warnings.warn(
                                f"No normalization stats for image key '{norm_key}', "
                                "using only /255 scaling. Please check if this is intended.",
                                UserWarning,
                                stacklevel=2,
                            )
                else:
                    # Numeric observations use the model's explicit normalization mode.
                    norm_key = key.replace("observations/", "")
                    obs_chunk = np.asarray(obs_chunk, dtype=np.float32)
                    if norm_key in current_norm_stats:
                        mean, std = normalization_parameters(current_norm_stats[norm_key], self.normalization_mode)
                        mean, std = np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)
                        obs_chunk = (obs_chunk - mean) / std
                    else:
                        warnings.warn(
                            f"No normalization stats for observation key '{norm_key}', using unnormalized data. "
                            "Please check if this is intended.",
                            UserWarning,
                            stacklevel=2,
                        )
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
                if key in current_norm_stats:
                    mean, std = normalization_parameters(current_norm_stats[key], self.normalization_mode)
                    mean, std = np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)
                    padded_action = (padded_action - mean) / std
                else:
                    warnings.warn(
                        f"No normalization stats for action key '{key}', using unnormalized data. "
                        "Please check if this is intended.",
                        UserWarning,
                        stacklevel=2,
                    )

                data_dict[key] = torch.from_numpy(padded_action).float()
                if "is_pad" not in data_dict:
                    data_dict["is_pad"] = torch.from_numpy(is_pad).bool()

        # Language instruction. The provider yields a raw string; the text_transform
        # (tokenizer) turns it into model-ready token ids, mirroring how img_transforms
        # turns raw frames into normalized tensors. Random selection augments language
        # at train time; a stable per-episode choice keeps val/test reproducible while
        # still varying wording across episodes.
        if self.use_instruction:
            provider = self.instruction_providers[dataset_dir]
            if INSTRUCTION_ATTR not in root.attrs:
                raise ValueError(
                    f"Episode {actual_episode_id} has no '{INSTRUCTION_ATTR}' in metadata"
                )
            instruction_str = root.attrs[INSTRUCTION_ATTR]

            channels = provider.get_channels(
                instruction_str=instruction_str,
                frame_idx=sample_ts,
                mode=self.instruction_modes[dataset_idx],
                episode_id=actual_episode_id,
                pick_random=(self.split == "train"),
            )
            for channel, text in channels.items():
                data_dict[f"instruction_{channel}"] = self.text_transform(text or "")
                data_dict[f"instruction_{channel}_valid"] = torch.tensor(text is not None)
            instruction = " ".join(text for text in channels.values() if text is not None)
            data_dict["instruction"] = self.text_transform(instruction)

        return data_dict

    def _get_episode_ids(self):
        """
        Load episode IDs from all dataset directories.

        Returns:
            tuple: (episode_ids, episode_to_dataset)
                - episode_ids: List of episode IDs (prefixed with dataset index for multi-dataset)
                - episode_to_dataset: Dict mapping episode_id to its dataset directory
        """
        all_ids = []
        episode_to_dataset = {}

        for dataset_idx, dataset_dir in enumerate(self.dataset_dirs):
            ids = resolve_episode_ids(dataset_dir, self.fold, self.split, self.task_ids[dataset_idx])

            # Prefix episode IDs to avoid collisions across datasets
            if self.is_multi_dataset:
                prefix = f"dataset{dataset_idx}_"
                prefixed_ids = [f"{prefix}{ep_id}" for ep_id in ids]
            else:
                prefixed_ids = ids

            # Build mapping
            for prefixed_id in prefixed_ids:
                episode_to_dataset[prefixed_id] = dataset_dir

            all_ids.extend(prefixed_ids)

        return all_ids, episode_to_dataset

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
            except Exception:
                pass

        return f

    def close(self):
        for f in self._h5_cache.values():
            try:
                f.close()
            except Exception:
                pass
        self._h5_cache = {}

    def __del__(self):
        self.close()
