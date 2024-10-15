import transformer.isolate_handlanguage_transformer.dataset as dataset
import transformer.isolate_handlanguage_transformer.features as features
import transformer.isolate_handlanguage_transformer.config as config
import transformer.isolate_handlanguage_transformer.plots as plot
import transformer.models.model as model
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from functools import partial
import os
from typing import Tuple, List
from torch import nn
import torch
import time
from inspect import signature
import numpy as np



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


if __name__ == "__main__":
    train_hdf5files, val_hdf5files, test_hdf5files, VOCAB = dataset.read_dataset()
    train_dataloader, val_dataloader, test_dataloader, in_channels = set_dataloader(train_hdf5files, val_hdf5files, test_hdf5files)

    inter_channels = 64
    out_channels = VOCAB
    activation = "relu"
    tren_num_layers = 6
    tren_num_heads = 2
    tren_dim_ffw = 256
    tren_dropout_pe = 0.1
    tren_dropout = 0.1
    tren_layer_norm_eps = 1e-5
    tren_norm_first = False
    tren_add_bias = True
    tren_add_tailnorm = False

    tren_norm_first = False
    tren_add_tailnorm = False

    model_default = model.TransformerModel(
        in_channels=in_channels,
        inter_channels=inter_channels,
        out_channels=out_channels,
        activation=activation,
        tren_num_layers=tren_num_layers,
        tren_num_heads=tren_num_heads,
        tren_dim_ffw=tren_dim_ffw,
        tren_dropout_pe=tren_dropout_pe,
        tren_dropout=tren_dropout,
        tren_layer_norm_eps=tren_layer_norm_eps,
        tren_norm_first=tren_norm_first,
        tren_add_bias=tren_add_bias,
        tren_add_tailnorm=tren_add_tailnorm)
    print(model_default)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 3e-4
    epochs = 50
    eval_every_n_epochs = 1



    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.Adam(model_default.parameters(), lr=lr)

    #Train, validation, and evaluation.
    model_default.to(device)
    train_losses = []
    val_losses = []
    test_accs = []
    print("Start training.")
    for epoch in range(epochs):
        print("-" * 80)
        print(f"Epoch {epoch+1}")

        train_losses = train_loop(train_dataloader, model_default, loss_fn, optimizer, device)
        val_loss = val_loop(val_dataloader, model_default, loss_fn, device)
        val_losses.append(val_loss)

        if (epoch+1) % eval_every_n_epochs == 0:
            acc = test_loop(test_dataloader, model_default, device)
            test_accs.append(acc)
    train_losses_default = np.array(train_losses)
    val_losses_default = np.array(val_losses)
    test_accs_default = np.array(test_accs)

    print(f"Minimum validation loss:{val_losses_default.min()} at {np.argmin(val_losses_default)+1} epoch.")
    print(f"Maximum accuracy:{test_accs_default.max()} at {np.argmax(test_accs_default)*eval_every_n_epochs+1} epoch.")

    save_dir = "transformer/models"
    os.makedirs(save_dir, exist_ok=True)

    # 保存ファイルのパスを設定
    save_path = os.path.join(save_dir, "transformer_model.pth")

    # モデルとオプティマイザの状態を保存
    torch.save({
        'model_state_dict': model_default.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epochs,
        'val_losses': val_losses_default,
        # 他に保存したい情報があれば追加
    }, save_path)

    print(f"モデルとオプティマイザの状態を {save_path} に保存しました。")
    plot.loss_plot(val_losses_default=val_losses_default)
    plot.test_data_plot(test_accs_default=test_accs_default)

