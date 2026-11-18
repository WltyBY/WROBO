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

