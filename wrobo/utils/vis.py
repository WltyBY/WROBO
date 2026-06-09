import h5py
import matplotlib.pyplot as plt
import numpy as np

from wrobo.utils.image_codec import get_encoding, decode_frame, RAW

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
    # Each frame is rendered as (H, W, C) for matplotlib. Image datasets may be stored
    # either as a dense 4-D array (T, C, H, W)/(T, H, W, C) ["raw"], or as a 1-D
    # variable-length array of PNG/JPEG byte strings ["png"/"jpg"], which we decode.
    with h5py.File(hdf5_path, "r") as f:
        images = f[image_dataset_path]
        encoding = get_encoding(images)

        def get_frame_hwc(i):
            """Return frame i as an (H, W, C) array, decoding if needed."""
            if encoding != RAW:
                # decode byte string -> (C, H, W) RGB -> (H, W, C)
                return np.transpose(decode_frame(images[i]), (1, 2, 0))
            frame = images[i]
            # dense layout: (C, H, W) -> (H, W, C); (H, W, C) stays as-is
            if frame.ndim == 3 and frame.shape[0] in [1, 3, 4]:
                frame = np.transpose(frame, (1, 2, 0))
            return frame

        if encoding != RAW:
            num_frames = images.shape[0]  # (T,)
        else:
            assert images.ndim == 4, (
                f"Expected raw images to have 4 dimensions [T, C, H, W] or "
                f"[T, H, W, C], got {images.ndim}"
            )
            num_frames = images.shape[0]
        print(f"Total Number of Frames: {num_frames}")

        plt.ion()
        fig, ax = plt.subplots()
        first_frame = get_frame_hwc(0)
        # If the image values are in the range 0-255, normalize to 0-1; if already in 0-1, no need to change
        if first_frame.dtype == np.uint8:
            first_frame = first_frame / 255.0
        im = ax.imshow(first_frame)
        plt.axis("off")

        for i in range(num_frames):
            frame = get_frame_hwc(i)
            if frame.dtype == np.uint8:
                frame = frame / 255.0
            im.set_data(frame)
            plt.pause(delay)
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    hdf5_file_path = "./Dataset/ACT_wrobo/sim_transfer_stack_cube/episode_0.hdf5"
    vis_img_dataset = "observations/image_angle"
    print_h5_structure(hdf5_file_path)
    visualize_hdf5_video(hdf5_file_path, vis_img_dataset, delay=0.02)
