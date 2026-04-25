import os
import h5py
import csv
import shutil
import yaml
import json
import pickle

from typing import List
from multiprocessing import Pool


def open_yaml(file_path, mode="r"):
    with open(file_path, mode) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def save_yaml(data, save_path, mode="w"):
    with open(save_path, mode) as f:
        yaml.dump(data=data, stream=f, allow_unicode=True)


def open_json(file_path, mode="r"):
    with open(file_path, mode) as f:
        data = json.load(f)
    return data


def save_json(data, save_path, mode="w", sort_keys=True, indent=4):
    with open(save_path, mode) as f:
        json.dump(data, f, sort_keys=sort_keys, indent=indent)


def open_pickle(file_path, mode="rb"):
    with open(file_path, mode) as f:
        data = pickle.load(f)
    return data


def save_pickle(data, save_path, mode="wb"):
    with open(save_path, mode) as f:
        pickle.dump(data, f)


def save_csv(list_of_lists, save_path, mode="a"):
    with open(save_path, mode, encoding="utf-8", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(list_of_lists)


def copy_file_to_dstFolder(srcfile, dstfolder):
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        # get the file's name
        _, fname = os.path.split(srcfile)
        if not os.path.exists(dstfolder):
            os.makedirs(dstfolder)
        dst_path = os.path.join(dstfolder, fname)
        # copy file from src path to dst path
        shutil.copy(srcfile, dst_path)
        print("Copy %s -> %s" % (srcfile, dst_path))


def check_workers_alive_and_busy(
    export_pool: Pool,
    worker_list: List,
    results_list: List,
    allowed_num_queued: int = 0,
):
    """

    returns True if the number of results that are not ready is greater than the number of available workers + allowed_num_queued
    """
    alive = [i.is_alive() for i in worker_list]
    if not all(alive):
        raise RuntimeError("Some background workers are no longer alive")

    not_ready = [not i.ready() for i in results_list]
    if sum(not_ready) >= (len(export_pool._pool) + allowed_num_queued):
        return True
    return False


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
            if len(str_attrs) > 100:
                str_attrs = str_attrs[:100] + " ... (truncated)"
        else:
            str_attrs = "No attributes"
        print(f"├── [Attr]: {str_attrs}")
        for key, val in f.items():
            _print(key, val, "")
