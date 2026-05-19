import os
import h5py

import numpy as np

from wrobo.utils.vis import print_h5_structure


def convert_hdf5_format(src_path, dst_path, comment, dataset_name="ACT", success=True):
    with h5py.File(src_path, "r", rdcc_nbytes=1024**2 * 2) as src, h5py.File(
        dst_path, "w", rdcc_nbytes=1024**2 * 2
    ) as dst:
        dst.attrs["dataset"] = dataset_name
        dst.attrs["success"] = success
        dst.attrs["comment"] = comment

        obs_grp = dst.create_group("observations")

        # images: (T, H, W, C) -> (T, C, H, W)
        img_src = src["observations/images/top"][:]  # shape: (T, H, W, C)
        T, H, W, C = img_src.shape
        dst.attrs["seq_len"] = T

        img_target = np.transpose(img_src, (0, 3, 1, 2))  # (T, C, H, W)
        obs_grp.create_dataset(
            "image_top",
            data=img_target,
            dtype="uint8",
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            chunks=(1, C, H, W),
        )

        # qpos -> proprio_state
        qpos = src["observations/qpos"][:]
        obs_grp.create_dataset(
            "proprio_state",
            data=qpos,
            dtype="float32",
            compression="gzip",
            compression_opts=1,
            shuffle=True,
            chunks=(50, 14),
        )

        # qvel -> proprio_vel
        if "observations/qvel" in src:
            qvel = src["observations/qvel"][:]
            obs_grp.create_dataset(
                "proprio_vel",
                data=qvel,
                dtype="float32",
                compression="gzip",
                compression_opts=1,
                shuffle=True,
                chunks=(50, 14),
            )

        action = src["action"][:]
        dst.create_dataset(
            "action_abs",
            data=action,
            dtype="float32",
            compression="gzip",
            compression_opts=1,
            shuffle=True,
            chunks=(50, 14),
        )


if __name__ == "__main__":
    comment = "sim_transfer_cube_compressed"
    folder_path = "./Dataset/ACT/sim_transfer_cube_scripted/"
    saved_folder = f"./Dataset/ACT_wrobo/{comment}/"
    file_path_lst = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".hdf5") and f.startswith("episode_")
    ]
    os.makedirs(saved_folder, exist_ok=True)

    print_h5_structure(file_path_lst[0])
    for file in file_path_lst:
        convert_hdf5_format(
            file, os.path.join(saved_folder, os.path.basename(file)), comment
        )

    saved_file_path_lst = [
        os.path.join(saved_folder, f)
        for f in os.listdir(saved_folder)
        if f.endswith(".hdf5") and f.startswith("episode_")
    ]
    print_h5_structure(saved_file_path_lst[0])
