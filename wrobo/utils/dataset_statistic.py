import os
import h5py
import numpy as np

from typing import Dict, List


def get_norm_stats(dataset_dir, case_lst=None, num_sample=1000, seed=319):
    if case_lst is None:
        file_paths = [
            os.path.join(dataset_dir, f)
            for f in os.listdir(dataset_dir)
            if f.endswith(".hdf5")
        ]
    else:
        file_paths = [
            os.path.join(dataset_dir, f if f.endswith(".hdf5") else f + ".hdf5")
            for f in case_lst
        ]
        
    if not file_paths:
        raise ValueError(f"No episode files found in {dataset_dir}")

    rng = np.random.default_rng(seed)

    data_buffers: Dict[str, List[np.ndarray]] = {}
    for file_path in file_paths:
        with h5py.File(file_path, "r") as root:
            # find all "propri" related keys under "observations"
            obs_group = root.get("observations")
            if obs_group is None:
                continue

            proprio_keys = [key for key in obs_group.keys() if "proprio" in key.lower()]
            if not proprio_keys:
                continue

            first_key = proprio_keys[0]
            T = obs_group[first_key].shape[0]

            # sample time indices for this episode
            if T > num_sample:
                sample_indices = rng.choice(T, size=num_sample, replace=False)
                sample_indices.sort()
            else:
                sample_indices = np.arange(T)

            # sample and buffer data for each proprio key
            for key in proprio_keys:
                sampled = obs_group[key][sample_indices]  # (min(num_sample, T), dim)
                if key not in data_buffers:
                    data_buffers[key] = []
                data_buffers[key].append(sampled)

            action_keys = [key for key in root.keys() if "action" in key.lower()]
            if not action_keys:
                continue

            for key in action_keys:
                sampled_actions = root[key][sample_indices]
                if key not in data_buffers:
                    data_buffers[key] = []
                data_buffers[key].append(sampled_actions)

    norm_stats = {}
    for key, samples in data_buffers.items():
        if not samples:
            continue
        # stack data to (N*num_sample, dim)
        stacked = np.concatenate(data_buffers[key], axis=0)
        mean = stacked.mean(axis=0)  # (dim,)
        std = stacked.std(axis=0).clip(min=1e-8)
        norm_stats[key] = {"mean": mean, "std": std}

    return norm_stats
