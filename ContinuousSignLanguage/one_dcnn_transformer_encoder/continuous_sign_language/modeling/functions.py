import one_dcnn_transformer_encoder.continuous_sign_language.features as features
import one_dcnn_transformer_encoder.continuous_sign_language.config as config
import one_dcnn_transformer_encoder.continuous_sign_language.dataset as dataset
from torchvision.transforms import Compose
import os
from functools import partial
from torch.utils.data import DataLoader
import time
from torch import nn
import numpy as np
import torch
from jiwer import wer, cer, mer


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

    feature_shape = (len(config.use_features), -1, len(use_landmarks) + config.spatial_spatial_feature)
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
    in_channels = (len(use_landmarks) + config.spatial_spatial_feature) * len(config.use_features)

    return train_dataloader, val_dataloader, test_dataloader, in_channels


def train_loop(dataloader, model, optimizer, device, return_pred_times=False):
    num_batches = len(dataloader)
    train_loss = 0
    size = len(dataloader.dataset)

    # Collect prediction time.
    pred_times = []

    # Switch to training mode.
    model.train()
    # Main loop.
    print("Start training.")
    start = time.perf_counter()
    tokens_causal_mask = None
    for batch_idx, batch_sample in enumerate(dataloader):
        feature = batch_sample["feature"]
        feature_pad_mask = batch_sample["feature_pad_mask"]
        tokens = batch_sample["token"]
        tokens_pad_mask = batch_sample["token_pad_mask"]
        #check_tokens_format(tokens, tokens_pad_mask, start_id, end_id)

        feature = feature.to(device)
        feature_pad_mask = feature_pad_mask.to(device)
        tokens = tokens.to(device)
        tokens_pad_mask = tokens_pad_mask.to(device)

        frames = feature.shape[-2]

        # Predict.

        input_lengths = [len(feature[i][0]) / 2 if len(feature[i][0])%2==0 else len(feature[i][0])//2 -1 for i in range(len(feature))]
        input_lengths = torch.tensor(input_lengths, dtype=torch.long)
        target_lengths = [len(tokens[i]) for i in range(len(tokens))]
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)
        pred_start = time.perf_counter()
        loss, log_probs = model.forward(
            src_feature=feature,
            tgt_feature=tokens,
            src_causal_mask=None,
            src_padding_mask=feature_pad_mask,
            input_lengths=input_lengths, #後修正
            target_lengths=target_lengths, #修正
            mode="train",
        )
        print("loss", loss)
        pred_end = time.perf_counter()
        pred_times.append([frames, pred_end - pred_start])

        # Back propagation.
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 勾配クリッピングを追加
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

def val_loop(dataloader, model, device, return_pred_times=False):
    num_batches = len(dataloader)
    val_loss = 0

    # Collect prediction time.
    pred_times = []

    # Switch to evaluation mode.
    model.eval()
    # Main loop.
    print("Start validation.")
    start = time.perf_counter()
    tokens_causal_mask = None
    with torch.no_grad():
        for batch_idx, batch_sample in enumerate(dataloader):
            feature = batch_sample["feature"]
            feature_pad_mask = batch_sample["feature_pad_mask"]
            tokens = batch_sample["token"]
            tokens_pad_mask = batch_sample["token_pad_mask"]
            feature = feature.to(device)
            feature_pad_mask = feature_pad_mask.to(device)
            tokens = tokens.to(device)
            tokens_pad_mask = tokens_pad_mask.to(device)
     
            frames = feature.shape[-2]
     
            # Predict.
            input_lengths = torch.tensor(
                [feature[i].shape[-1] - 20 for i in range(len(feature))], 
                dtype=torch.long
            )
            target_lengths = [len(tokens[i]) for i in range(len(tokens))]
            target_lengths = torch.tensor(target_lengths, dtype=torch.long)
            pred_start = time.perf_counter()
            val_loss, log_probs = model.forward(
                src_feature=feature,
                tgt_feature=tokens,
                src_causal_mask=None,
                src_padding_mask=feature_pad_mask,
                input_lengths=input_lengths, #後修正
                target_lengths=target_lengths, #修正
                mode="eval",
            )
            pred_end = time.perf_counter()
            pred_times.append([frames, pred_end - pred_start])
            print("val_loss", val_loss)
            # Compute loss.
            # Preds do not include <start>, so skip that of tokens.
     
            tokens = tokens.tolist()
            reference_text = [' '.join(map(str, seq)) for seq in tokens]
            hypothesis_text = [' '.join(map(str, seq)) for seq in log_probs]
            wer_score = wer(reference_text, hypothesis_text)
            print(f"Batch {batch_idx}: WER: {wer_score:.10f}")
    print(f"Done. Time:{time.perf_counter()-start}")
    # Average loss.
    val_loss /= num_batches
    print("Validation performance: \n",
          f"Avg loss:{val_loss:>8f}\n")
    pred_times = np.array(pred_times)
    retval = (val_loss, pred_times) if return_pred_times else val_loss
    return retval

def test_loop(dataloader,
                       model,
                       device,
                       return_pred_times=False,
                       blank_id=100):
    size = len(dataloader.dataset)
    hypothesis_text_list = []
    reference_text_list = []

    # Collect prediction time.
    pred_times = []
    # Switch to evaluation mode.
    model.eval()
    # Main loop.
    print("Start test.")
    start = time.perf_counter()
    with torch.no_grad():
        for batch_idx, batch_sample in enumerate(dataloader):
            feature = batch_sample["feature"]
            feature_pad_mask = batch_sample["feature_pad_mask"]
            tokens = batch_sample["token"]
            tokens_pad_mask = batch_sample["token_pad_mask"]
     
            feature = feature.to(device)
            tokens = tokens.to(device)
            tokens_pad_mask = tokens_pad_mask.to(device)
     
            frames = feature.shape[-2]
     
            # Predict.
            pred_start = time.perf_counter()
            log_probs = model.forward(
                src_feature=feature,
                tgt_feature=tokens,
                src_causal_mask=None,      # オプション、今回は使用しない
                src_padding_mask=None,     # オプション、今回は使用しない
                input_lengths=100, #後修正
                target_lengths=20, #修正
                mode="test",
                blank_id=blank_id
            )
            pred_end = time.perf_counter()
            pred_times.append([frames, pred_end - pred_start])

            tokens = tokens.tolist()
            reference_text = [' '.join(map(str, seq)) for seq in tokens]
            hypothesis_text = [' '.join(map(str, seq)) for seq in log_probs]
            
            reference_text_list.append(reference_text[0])
            hypothesis_text_list.append(hypothesis_text[0])
           
            
    print(reference_text_list, hypothesis_text_list)

    print(f"Done. Time:{time.perf_counter()-start}")

    # Average WER.
    # Create a dictionary to store WER per label
    label_wer = {}
    for ref, hyp in zip(reference_text_list, hypothesis_text_list):
        ref_label = ref  # Get the first token as label

        if ref_label not in label_wer:
            label_wer[ref_label] = {"refs": [], "hyps": []}
        label_wer[ref_label]["refs"].append(ref)
        label_wer[ref_label]["hyps"].append(hyp)
    # Calculate and print WER for each label
    print("\nWER per label:")
    for label in label_wer:
        label_refs = label_wer[label]["refs"]
        label_hyps = label_wer[label]["hyps"]
        label_wer_score = wer(label_refs, label_hyps)
        print(f"Label {label}: {label_wer_score:.10f} ({len(label_refs)} samples)")
    awer = wer(reference_text_list, hypothesis_text_list)
    print("Test performance: \n",
          f"Avg WER:{awer:>0.10f}\n")
    pred_times = np.array(pred_times)
    error_rate_cer = cer(reference_text_list, hypothesis_text_list)
    error_rate_mer = mer(reference_text_list, hypothesis_text_list)
    print(f"Overall WER: {awer}")
    print(f"Overall CER: {error_rate_cer}")
    print(f"Overall MER: {error_rate_mer}")
    retval = (awer, pred_times) if return_pred_times else awer
    return retval

def save_model(save_path, model_default_dict, optimizer_dict, epoch):
    torch.save({
        'model_state_dict': model_default_dict,
        'optimizer_state_dict': optimizer_dict,
        'epoch': epoch,
    }, save_path)

    print(f"モデルとオプティマイザの状態を {save_path} に保存しました。")

def load_model(model, save_path: str, device: str = "cpu"):

    checkpoint = torch.load(save_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # オプティマイザの再構築
    optimizer_loaded = torch.optim.Adam(model.parameters(), lr=3e-4)
    optimizer_loaded.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch_loaded = checkpoint.get('epoch', None)
    
    print(f"エポック {epoch_loaded} までのモデルをロードしました。")
    
    return model, optimizer_loaded, epoch_loaded
