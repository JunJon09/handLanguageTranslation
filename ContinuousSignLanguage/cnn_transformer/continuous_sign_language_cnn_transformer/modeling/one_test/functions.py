import cnn_transformer.continuous_sign_language_cnn_transformer.features as features
import cnn_transformer.continuous_sign_language_cnn_transformer.config as config
import cnn_transformer.continuous_sign_language_cnn_transformer.dataset as dataset
import cnn_transformer.continuous_sign_language_cnn_transformer.modeling.config as model_config
import cnn_transformer.models.cnn_transformer as models
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
import os
import time
from inspect import signature
import numpy as np
from typing import Tuple, List
import torch
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from functools import partial
from torch import nn
import torch.nn.functional as F
from nltk.metrics.distance import edit_distance
import itertools

def set_dataloader(key2token, train_hdf5files, val_hdf5files, test_hdf5files):
    _, use_landmarks = features.get_fullbody_landmarks()
    trans_select_feature = features.SelectLandmarksAndFeature(landmarks=use_landmarks, features=config.use_features)
    trans_repnan = features.ReplaceNan()
    trans_norm = features.PartsBasedNormalization(align_mode="framewise", scale_mode="unique")
    trans_insert_token = features.InsertTokensForS2S(sos_token=key2token["<sos>"], eos_token=key2token["<eos>"])

    pre_transforms = Compose([
        trans_select_feature,
        trans_repnan,
        trans_insert_token,
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

def inference(model, feature, start_id, end_id, max_seqlen):
   
    if isinstance(model, models.CNNTransformerModel):
        pred_ids, _ = model.inference(feature,
                                      start_id,
                                      end_id,
                                      max_seqlen=max_seqlen)
    else:
        raise NotImplementedError(f"Unknown model type:{type(model)}.")
    return pred_ids


def test_loop_csir_s2s(dataloader,
                       model,
                       device,
                       start_id,
                       end_id,
                       max_seqlen=62,
                       return_pred_times=False,
                       verbose_num=1):
    size = len(dataloader.dataset)
    total_wer = 0

    # Collect prediction time.
    pred_times = []

    # Switch to evaluation mode.
    model.eval()
    # Main loop.
    print("Start test.")
    start = time.perf_counter()
    with torch.no_grad():
        for batch_idx, batch_sample in enumerate(dataloader):
            if batch_idx != 25:
                continue
            feature = batch_sample["feature"]
            tokens = batch_sample["token"]
            tokens_pad_mask = batch_sample["token_pad_mask"]
     
            feature = feature.to(device)
            tokens = tokens.to(device)
            tokens_pad_mask = tokens_pad_mask.to(device)
     
            frames = feature.shape[-2]
     
            # Predict.
            pred_start = time.perf_counter()
            pred_ids = inference(model, feature, start_id, end_id, max_seqlen)
            pred_end = time.perf_counter()
            pred_times.append([frames, pred_end - pred_start])
     
            # Compute WER.
            # <sos> and <eos> should be removed because they may boost performance.
            # print(tokens)
            # print(pred_ids)
            if batch_idx < verbose_num:
                print("="*40)
                print("Verbose output")
            if batch_idx < verbose_num:
                print(f"Tokens_w_keywords: {tokens}")
                print(f"Preds_w_keywords: {pred_ids}")

            tokens = tokens[tokens_pad_mask]
            if len(tokens.shape) == 2:
                tokens = tokens[0, 1:-1]
            else:
                tokens = tokens[1:-1]
            # pred_ids = pred_ids[0, 1:-1]
            pred_ids = [pid for pid in pred_ids[0] if pid not in [start_id, end_id]]
            if batch_idx < verbose_num:
                print(f"Tokens_wo_keywords: {tokens}")
                print(f"Preds_wo_keywords: {pred_ids}")

            ref_length = len(tokens)
            tokens = tokens.tolist()
            pred_ids = [int(x) for x in pred_ids]

            return tokens, pred_ids
    return "OK"

def load_model(save_path: str, device: str = "cpu"):

    checkpoint = torch.load(save_path, map_location=device)

    train_hdf5files, val_hdf5files, test_hdf5files, key2token = dataset.read_dataset()
    train_dataloader, val_dataloader, test_dataloader, in_channels = set_dataloader(key2token, train_hdf5files, val_hdf5files, test_hdf5files)

    VOCAB = len(key2token)
    out_channels = VOCAB
    pad_token = key2token["<pad>"]


    model_loaded = models.CNNTransformerModel(
        in_channels=in_channels,
        inter_channels=model_config.inter_channels,
        kernel_size=model_config.kernel_size,
        stride=model_config.stride,
        out_channels=out_channels,
        padding_val=pad_token,
        activation=model_config.activation,
        tren_num_layers=model_config.tren_num_layers,
        tren_num_heads=model_config.tren_num_heads,
        tren_dim_ffw=model_config.tren_dim_ffw,
        tren_dropout_pe=model_config.tren_dropout_pe,
        tren_dropout=model_config.tren_dropout,
        tren_norm_type_sattn=model_config.norm_type,
        tren_norm_type_ffw=model_config.norm_type,
        tren_norm_type_tail=model_config.norm_type,
        tren_norm_eps=model_config.tren_norm_eps,
        tren_norm_first=model_config.tren_norm_first,
        tren_add_bias=model_config.tren_add_bias,
        tren_add_tailnorm=model_config.tren_add_tailnorm,
        trde_num_layers=model_config.trde_num_layers,
        trde_num_heads=model_config.trde_num_heads,
        trde_dim_ffw=model_config.trde_dim_ffw,
        trde_dropout_pe=model_config.trde_dropout_pe,
        trde_dropout=model_config.trde_dropout,
        trde_norm_type_sattn=model_config.norm_type,
        trde_norm_type_cattn=model_config.norm_type,
        trde_norm_type_ffw=model_config.norm_type,
        trde_norm_type_tail=model_config.norm_type,
        trde_norm_eps=model_config.trde_norm_eps,
        trde_norm_first=model_config.trde_norm_first,
        trde_add_bias=model_config.trde_add_bias,
        trde_add_tailnorm=model_config.trde_add_tailnorm)
    
    model_loaded.load_state_dict(checkpoint['model_state_dict'])
    model_loaded.to(device)
    
    # オプティマイザの再構築
    optimizer_loaded = torch.optim.Adam(model_loaded.parameters(), lr=model_config.lr)
    optimizer_loaded.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch_loaded = checkpoint.get('epoch', None)
    
    print(f"エポック {epoch_loaded} までのモデルをロードしました。")
    
    return model_loaded, optimizer_loaded, epoch_loaded, test_dataloader, key2token