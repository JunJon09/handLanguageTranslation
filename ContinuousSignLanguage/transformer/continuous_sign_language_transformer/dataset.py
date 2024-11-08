import transformer.continuous_sign_language_transformer.config as config
from typing import Tuple, List
from pathlib import Path
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
