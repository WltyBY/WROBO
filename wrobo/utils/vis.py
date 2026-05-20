import h5py
import matplotlib.pyplot as plt
import numpy as np


def print_h5_structure(filename):
    def _print(name, obj, indent=""):
        if isinstance(obj, h5py.Group):
            print(f"{indent}├── [Group] {name}")
            for key, val in obj.items():
                _print(key, val, indent + "│   ")
        elif isinstance(obj, h5py.Dataset):
            print(
                f"{indent}├── [Dataset] {name} (shape: {obj.shape}, dtype: {obj.dtype})"
            )

    with h5py.File(filename, "r") as f:
        print(f"📁 HDF5 file: {filename}")
        if f.attrs:
            str_attrs = str(dict(f.attrs))
            if len(str_attrs) > 150:
                str_attrs = str_attrs[:150] + " ... (truncated)"
        else:
            str_attrs = "No attributes"
        print(f"├── [Attr]: {str_attrs}")
        for key, val in f.items():
            _print(key, val, "")


def visualize_hdf5_video(hdf5_path, image_dataset_path="observations/image_top", delay=0.02):
    # img in [T, C, H, W] no matter the img is RGB or grayscale, then transpose to [T, H, W, C] for visualization
    with h5py.File(hdf5_path, "r") as f:
        images = f[image_dataset_path]

        # check the shape and transpose if needed
        assert images.ndim in [
            4
        ], f"Expected images to have 4 dimensions [T, C, H, W] or [T, H, W, C], got {images.ndim}"

        # Gray (C=1), RGB (C=3), RGBA (C=4)
        if images.shape[3] in [1, 3, 4]:
            # data already in (T, H, W, C)
            pass
        elif images.shape[1] in [1, 3, 4]:
            # datas saved in (T, C, H, W), transpose to (T, H, W, C)
            images = np.transpose(images, (0, 2, 3, 1))

        num_frames = images.shape[0]
        print(f"Total Number of Frames: {num_frames}")

        plt.ion()
        fig, ax = plt.subplots()
        first_frame = images[0]
        # If the image values are in the range 0-255, normalize to 0-1; if already in 0-1, no need to change
        if first_frame.dtype == np.uint8:
            first_frame = first_frame / 255.0
        im = ax.imshow(first_frame)
        plt.axis("off")

        for i in range(num_frames):
            frame = images[i]
            if frame.dtype == np.uint8:
                frame = frame / 255.0
            im.set_data(frame)
            plt.pause(delay)
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    hdf5_file_path = "./Dataset/ACT_wrobo/sim_insertion/episode_0.hdf5"
    vis_img_dataset = "observations/image_right_wrist"
    print_h5_structure(hdf5_file_path)
    visualize_hdf5_video(hdf5_file_path, vis_img_dataset, delay=0.02)
