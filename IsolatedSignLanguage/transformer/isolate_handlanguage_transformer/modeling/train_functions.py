import transformer.isolate_handlanguage_transformer.features as features
import transformer.isolate_handlanguage_transformer.config as config
import transformer.isolate_handlanguage_transformer.dataset as dataset
import transformer.isolate_handlanguage_transformer.modeling.config as model_config
import transformer.models.model as model
import os
import time
from inspect import signature
import numpy as np
from typing import Tuple, List
import torch
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from functools import partial


def set_dataloader(train_hdf5files: List, val_hdf5files: List, test_hdf5files: List) -> Tuple[DataLoader, DataLoader, DataLoader]:
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

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=merge_fn, num_workers=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, collate_fn=merge_fn, num_workers=num_workers, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=merge_fn, num_workers=num_workers, shuffle=False)
    in_channels = len(use_landmarks) * len(config.use_features)
    return train_dataloader, val_dataloader, test_dataloader, in_channels


def train_loop(dataloader, model, loss_fn, optimizer, device, use_mask=True,
               return_pred_times=False):
    num_batches = len(dataloader)
    train_loss = 0
    size = len(dataloader.dataset)

    # Inspect model signature.
    sig = signature(model.forward)
    use_mask = True if "feature_pad_mask" in sig.parameters and use_mask is True else False

    # Collect prediction time.
    pred_times = []

    # Switch to training mode.
    model.train()
    # Main loop.
    print("Start training.")
    start = time.perf_counter()
    for batch_idx, batch_sample in enumerate(dataloader):
        feature = batch_sample["feature"]
        token = batch_sample["token"]
        feature = feature.to(device)
        token = token.to(device)
        frames = feature.shape[-2]

        # Predict.
        pred_start = time.perf_counter()
        if use_mask:
            feature_pad_mask = batch_sample["feature_pad_mask"]
            feature_pad_mask = feature_pad_mask.to(device)
            pred = model(feature, feature_pad_mask=feature_pad_mask)
        else:
            pred = model(feature)
        pred_end = time.perf_counter()
        pred_times.append([frames, pred_end - pred_start])

        # Compute loss.
        loss = loss_fn(pred, token.squeeze(-1))

        # Back propagation.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # Print current loss per 100 steps.
        if batch_idx % 100 == 0:
            loss = loss.item()
            steps = batch_idx * len(feature)
            print(f"loss:{loss:>7f} [{steps:>5d}/{size:>5d}]")
    print(f"Done. Time:{time.perf_counter()-start}")
    # Average loss.
    train_loss /= num_batches
    print("Training performance: \n",
          f"Avg loss:{train_loss:>8f}\n")
    pred_times = np.array(pred_times)
    retval = (train_loss, pred_times) if return_pred_times else train_loss
    return retval

def val_loop(dataloader, model, loss_fn, device, use_mask=True,
             return_pred_times=False):
    num_batches = len(dataloader)
    val_loss = 0

    # Inspect model signature.
    sig = signature(model.forward)
    use_mask = True if "feature_pad_mask" in sig.parameters and use_mask is True else False

    # Collect prediction time.
    pred_times = []

    # Switch to evaluation mode.
    model.eval()
    # Main loop.
    print("Start validation.")
    start = time.perf_counter()
    with torch.no_grad():
        for batch_sample in dataloader:
            feature = batch_sample["feature"]
            token = batch_sample["token"]
            feature = feature.to(device)
            token = token.to(device)
            frames = feature.shape[-2]

            # Predict.
            pred_start = time.perf_counter()
            if use_mask:
                feature_pad_mask = batch_sample["feature_pad_mask"]
                feature_pad_mask = feature_pad_mask.to(device)
                pred = model(feature, feature_pad_mask=feature_pad_mask)
            else:
                pred = model(feature)
            pred_end = time.perf_counter()
            pred_times.append([frames, pred_end - pred_start])

            val_loss += loss_fn(pred, token.squeeze(-1)).item()
    print(f"Done. Time:{time.perf_counter()-start}")

    # Average loss.
    val_loss /= num_batches
    print("Validation performance: \n",
          f"Avg loss:{val_loss:>8f}\n")
    pred_times = np.array(pred_times)
    retval = (val_loss, pred_times) if return_pred_times else val_loss
    return retval


def test_loop(dataloader, model, device, use_mask=False,
              return_pred_times=False):
    size = len(dataloader.dataset)
    correct = 0

    # Inspect model signature.
    sig = signature(model.forward)
    use_mask = True if "feature_pad_mask" in sig.parameters and use_mask is True else False

    # Collect prediction time.
    pred_times = []

    # Switch to evaluation mode.
    model.eval()
    # Main loop.
    print("Start evaluation.")
    start = time.perf_counter()
    with torch.no_grad():
        for batch_sample in dataloader:
            feature = batch_sample["feature"]
            token = batch_sample["token"]
            feature = feature.to(device)
            token = token.to(device)
            frames = feature.shape[-2]

            # Predict.
            pred_start = time.perf_counter()
            if use_mask:
                feature_pad_mask = batch_sample["feature_pad_mask"]
                feature_pad_mask = feature_pad_mask.to(device)
                pred = model(feature, feature_pad_mask=feature_pad_mask)
            else:
                pred = model(feature)
            pred_end = time.perf_counter()
            pred_times.append([frames, pred_end - pred_start])

            pred_ids = pred.argmax(dim=1).unsqueeze(-1)
            count = (pred_ids == token).sum().detach().cpu().numpy()
            correct += int(count)
    print(f"Done. Time:{time.perf_counter()-start}")

    acc = correct / size * 100
    print("Test performance: \n",
          f"Accuracy:{acc:>0.1f}%")
    pred_times = np.array(pred_times)
    retval = (acc, pred_times) if return_pred_times else acc
    return retval

def save_model(save_path, model_default_dict, optimizer_dict, epoch, val_loss):
    torch.save({
        'model_state_dict': model_default_dict,
        'optimizer_state_dict': optimizer_dict,
        'epoch': epoch,
        'val_losses': val_loss,
    }, save_path)

    print(f"モデルとオプティマイザの状態を {save_path} に保存しました。")


def load_model(save_path: str, device: str = "cpu") -> Tuple[model.TransformerModel, torch.optim.Adam, int, np.ndarray, list]:

    checkpoint = torch.load(save_path, map_location=device)

    train_hdf5files, val_hdf5files, test_hdf5files, VOCAB = dataset.read_dataset()
    train_dataloader, val_dataloader, test_dataloader, in_channels = set_dataloader(train_hdf5files, val_hdf5files, test_hdf5files)

    out_channels = VOCAB

    model_loaded = model.TransformerModel(
        in_channels=in_channels,
        inter_channels=model_config.inter_channels,
        out_channels=out_channels,
        activation=model_config.activation,
        tren_num_layers=model_config.tren_num_layers,
        tren_num_heads=model_config.tren_num_heads,
        tren_dim_ffw=model_config.tren_dim_ffw,
        tren_dropout_pe=model_config.tren_dropout_pe,
        tren_dropout=model_config.tren_dropout,
        tren_layer_norm_eps=model_config.tren_layer_norm_eps,
        tren_norm_first=model_config.tren_norm_first,
        tren_add_bias=model_config.tren_add_bias,
        tren_add_tailnorm=model_config.tren_add_tailnorm)
    
    model_loaded.load_state_dict(checkpoint['model_state_dict'])
    model_loaded.to(device)
    
    # オプティマイザの再構築
    optimizer_loaded = torch.optim.Adam(model_loaded.parameters(), lr=model_config.lr)
    optimizer_loaded.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch_loaded = checkpoint.get('epoch', None)
    val_losses_loaded = checkpoint.get('val_losses', None)
    
    print(f"エポック {epoch_loaded} までのモデルをロードしました。")
    
    return model_loaded, optimizer_loaded, epoch_loaded, val_losses_loaded, test_dataloader
