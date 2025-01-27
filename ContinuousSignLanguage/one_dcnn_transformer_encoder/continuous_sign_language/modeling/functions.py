import one_dcnn_transformer_encoder.continuous_sign_language.features as features
import one_dcnn_transformer_encoder.continuous_sign_language.config as config
import one_dcnn_transformer_encoder.continuous_sign_language.dataset as dataset
from torchvision.transforms import Compose
import os
from functools import partial
from torch.utils.data import DataLoader

def set_dataloader(key2token, train_hdf5files, val_hdf5files, test_hdf5files):
    _, use_landmarks = features.get_fullbody_landmarks()
    trans_select_feature = features.SelectLandmarksAndFeature(landmarks=use_landmarks, features=config.use_features)
    trans_repnan = features.ReplaceNan()
    trans_norm = features.PartsBasedNormalization(align_mode="framewise", scale_mode="unique")

    pre_transforms = Compose([
        trans_select_feature,
        trans_repnan,
        trans_norm
    ])
    train_transforms = Compose([
        features.ToTensor()
    ])

    val_transforms = Compose([
        features.ToTensor()
    ])

    test_transforms = Compose([
       features.ToTensor()
    ])

    train_dataset = dataset.HDF5Dataset(train_hdf5files,
    pre_transforms=pre_transforms, transforms=train_transforms, load_into_ram=config.load_into_ram)
    val_dataset = dataset.HDF5Dataset(val_hdf5files,
        pre_transforms=pre_transforms, transforms=val_transforms, load_into_ram=config.load_into_ram)
    test_dataset = dataset.HDF5Dataset(test_hdf5files,
        pre_transforms=pre_transforms, transforms=test_transforms, load_into_ram=config.load_into_ram)

    feature_shape = (len(config.use_features), -1, len(use_landmarks))
    token_shape = (-1,)
    num_workers = os.cpu_count()
    merge_fn = partial(dataset.merge_padded_batch,
                   feature_shape=feature_shape,
                   token_shape=token_shape,
                   feature_padding_val=0.0,
                   token_padding_val=key2token["<pad>"])

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=merge_fn, num_workers=num_workers, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, collate_fn=merge_fn, num_workers=num_workers, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=merge_fn, num_workers=num_workers, shuffle=False)
    in_channels = len(use_landmarks) * len(config.use_features)

    return train_dataloader, val_dataloader, test_dataloader, in_channels