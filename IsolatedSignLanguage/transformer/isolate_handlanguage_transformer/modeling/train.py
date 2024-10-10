import isolate_handlanguage_transformer.dataset as dataset
import isolate_handlanguage_transformer.features as features
import isolate_handlanguage_transformer.config as config
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from functools import partial
import os
from typing import Tuple, List

def set_dataloader(train_hdf5files: List, val_hdf5files: List, test_hdf5files: List) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_hdf5files, val_hdf5files, test_hdf5files = dataset.read_dataset()
    _, use_landmarks = features.get_fullbody_landmarks()
    trans_select_feature = features.SelectLandmarksAndFeature(landmarks=use_landmarks, features=config.use_features)
    trans_repnan = features.ReplaceNan()
    trans_norm = features.PartsBasedNormalization(align_mode="framewise", scale_mode="unique")

    pre_transforms = Compose([trans_select_feature,
                            trans_repnan,
                            trans_norm])

    transforms = Compose([features.ToTensor()])
    train_dataset = dataset.HDF5Dataset(train_hdf5files, pre_transforms=pre_transforms, transforms=transforms,load_into_ram=config.load_into_ram)
    val_dataset = dataset.HDF5Dataset(val_hdf5files, pre_transforms=pre_transforms,transforms=transforms, load_into_ram=config.load_into_ram)
    test_dataset = dataset.HDF5Dataset(test_hdf5files,pre_transforms=pre_transforms,
    transforms=transforms, load_into_ram=config.load_into_ram)

    feature_shape = (len(config.use_features), -1, len(use_landmarks))
    token_shape = (1,)
    num_workers = os.cpu_count()
    merge_fn = partial(dataset.merge_padded_batch,
                   feature_shape=feature_shape,
                   token_shape=token_shape,
                   feature_padding_val=0.0,
                   token_padding_val=0)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=merge_fn, num_workers=num_workers, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, collate_fn=merge_fn, num_workers=num_workers, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=merge_fn, num_workers=num_workers, shuffle=False)
    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":
    train_hdf5files, val_hdf5files, test_hdf5files = dataset.read_dataset()
    train_dataloader, val_dataloader, test_dataloader = set_dataloader(train_hdf5files, val_hdf5files, test_hdf5files)
    data = next(iter(train_dataloader))
    feature_origin = data["feature"]

    print(feature_origin.shape)