from pathlib import Path
import h5py
from pathlib import Path
from torch.utils.data import (
    DataLoader,
    Dataset)
import numpy as np
from torchvision.transforms import Compose
import transformer.isolate_handlanguage_transformer.features as features
import transformer.isolate_handlanguage_transformer.config as config
import copy
import torch
from typing import Tuple, List
import json


def read_dataset(input_dir: Path = config.read_dataset_dir) -> Tuple[List, List, List, int]:
    dataset_dir = Path(input_dir)
    files = list(dataset_dir.iterdir())
    print(files)
    hdf5_files = [fin for fin in files if ".hdf5" in fin.name]

    train_hdf5files = [fin for fin in hdf5_files if config.test_number not in fin.name]
    val_hdf5files = [fin for fin in hdf5_files if config.val_number in fin.name]
    test_hdf5files = [fin for fin in hdf5_files if config.test_number in fin.name]
    
    dictionary = [fin for fin in files if ".json" in fin.name][0]
    with open(dictionary, "r") as f:
        key2token = json.load(f)

    VOCAB = len(key2token)


    return train_hdf5files, val_hdf5files, test_hdf5files, VOCAB

class HDF5Dataset(Dataset):
    def __init__(self,
                 hdf5files,
                 load_into_ram=False,
                 convert_to_channel_first=False,
                 pre_transforms=None,
                 transforms=None):
        self.convert_to_channel_first = convert_to_channel_first
        self.pre_transforms = pre_transforms
        self.load_into_ram = load_into_ram
        data_info = []
        # Load file pointers.
        for fin in hdf5files:
            swap = 1 if "_swap" in fin.name else 0
            # filename should be [pid].hdf5 or [pid]_swap.hdf5
            pid = int(fin.stem.split("_")[0])
            with h5py.File(fin.resolve(), "r") as f:
                keys = list(f.keys())
                for key in keys:
                    if load_into_ram:
                        data = {"feature": f[key]["feature"][:],
                                "token": f[key]["token"][:]}
                        if self.convert_to_channel_first:
                            feature = data["feature"]
                            # `[T, J, C] -> [C, T, J]`
                            feature = np.transpose(feature, [2, 0, 1])
                            data["feature"] = feature
                        if self.pre_transforms:
                            data = self.pre_transforms(data)
                    else:
                        data = None
                    data_info.append({
                        "file": fin,
                        "data_key": key,
                        "swap": swap,
                        "pid": pid,
                        "data": data})
        self.data_info = data_info

        # Check and assign transforms.
        self.transforms = self._check_transforms(transforms)

    def _check_transforms(self, transforms):
        # Check transforms.
        if transforms:
            if isinstance(transforms, Compose):
                _transforms = transforms.transforms
            else:
                _transforms = transforms
            check_totensor = False
            for trans in _transforms:
                if isinstance(trans, features.ToTensor):
                    check_totensor = True
                    break
            message = "Dataset should return torch.Tensor but transforms does " \
                + "not include ToTensor class."
            assert check_totensor, message

        if transforms is None:
            transforms = Compose([features.ToTensor()])
        elif not isinstance(transforms, Compose):
            transforms = Compose(transforms)
        return transforms

    def __getitem__(self, index):
        info = self.data_info[index]
        if info["data"]:
            data = info["data"]
        else:
            with h5py.File(info["file"], "r") as f:
                data = {"feature": f[info["data_key"]]["feature"][:],
                        "token": f[info["data_key"]]["token"][:]}
        _data = copy.deepcopy(data)
        if self.load_into_ram is False:
            if self.convert_to_channel_first:
                feature = _data["feature"]
                # `[T, J, C] -> [C, T, J]`
                feature = np.transpose(feature, [2, 0, 1])
                _data["feature"] = feature
            if self.pre_transforms:
                _data = self.pre_transforms(_data)
        _data = self.transforms(_data)
        return _data

    def __len__(self):
        return len(self.data_info)
    
def merge_padded_batch(batch,
                       feature_shape,
                       token_shape,
                       feature_padding_val=0,
                       token_padding_val=0):
    feature_batch = [sample["feature"] for sample in batch]
    token_batch = [sample["token"] for sample in batch]
    # ==========================================================
    # Merge feature.
    # ==========================================================
    # `[B, C, T, J]`
    merged_shape = [len(batch), *feature_shape]
    # Use maximum frame length in a batch as padded length.

    if merged_shape[2] == -1:
        tlen = max([feature.shape[1] for feature in feature_batch])
        merged_shape[2] = tlen
    merged_feature = merge(feature_batch, merged_shape, padding_val=feature_padding_val)

    # ==========================================================
    # Merge token.
    # ==========================================================
    # `[B, L]`
    merged_shape = [len(batch), *token_shape]
    # Use maximum token length in a batch as padded length.
    if merged_shape[1] == -1:
        tlen = max([token.shape[0] for token in token_batch])
        merged_shape[1] = tlen
    merged_token = merge(token_batch, merged_shape, padding_val=token_padding_val)

    # Generate padding mask.
    # Pad: 0, Signal: 1
    # The frames which all channels and landmarks are equals to padding value
    # should be padded.
    feature_pad_mask = merged_feature == feature_padding_val
    feature_pad_mask = torch.all(feature_pad_mask, dim=1)
    feature_pad_mask = torch.all(feature_pad_mask, dim=-1)
    feature_pad_mask = torch.logical_not(feature_pad_mask)
    token_pad_mask = torch.logical_not(merged_token == token_padding_val)

    retval = {
        "feature": merged_feature,
        "token": merged_token,
        "feature_pad_mask": feature_pad_mask,
        "token_pad_mask": token_pad_mask}
    return retval

def merge(sequences, merged_shape, padding_val=0):
    merged = torch.full(tuple(merged_shape),
                        padding_val,
                        dtype=sequences[0].dtype)
    if len(merged_shape) == 2:
        for i, seq in enumerate(sequences):
            merged[i,
                   :seq.shape[0]] = seq
    if len(merged_shape) == 3:
        for i, seq in enumerate(sequences):
            merged[i,
                   :seq.shape[0],
                   :seq.shape[1]] = seq
    if len(merged_shape) == 4:
        for i, seq in enumerate(sequences):
            merged[i,
                   :seq.shape[0],
                   :seq.shape[1],
                   :seq.shape[2]] = seq
    if len(merged_shape) == 5:
        for i, seq in enumerate(sequences):
            merged[i,
                   :seq.shape[0],
                   :seq.shape[1],
                   :seq.shape[2],
                   :seq.shape[3]] = seq
    return merged
