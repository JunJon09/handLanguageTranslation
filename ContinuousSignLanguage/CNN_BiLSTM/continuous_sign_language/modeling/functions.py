import CNN_BiLSTM.continuous_sign_language.features as features
import CNN_BiLSTM.continuous_sign_language.config as config
import CNN_BiLSTM.continuous_sign_language.dataset as dataset
import CNN_BiLSTM.continuous_sign_language.modeling.middle_dataset_relation as middle_dataset_relation
import CNN_BiLSTM.continuous_sign_language
from torchvision.transforms import Compose
import os
from functools import partial
from torch.utils.data import DataLoader
import time
from torch import nn
import numpy as np
import torch
from jiwer import wer, cer, mer
from torch.cuda.amp import GradScaler



def set_dataloader(key2token, train_hdf5files, val_hdf5files, test_hdf5files):
    _, use_landmarks = features.get_fullbody_landmarks()
    trans_select_feature = features.SelectLandmarksAndFeature(
        landmarks=use_landmarks, features=config.use_features
    )
    trans_repnan = features.ReplaceNan()
    trans_norm = features.PartsBasedNormalization(
        align_mode="framewise", scale_mode="unique"
    )

    pre_transforms = Compose([trans_select_feature, trans_repnan, trans_norm])
    train_transforms = Compose([features.ToTensor()])

    val_transforms = Compose([features.ToTensor()])

    test_transforms = Compose([features.ToTensor()])

    train_dataset = dataset.HDF5Dataset(
        train_hdf5files,
        pre_transforms=pre_transforms,
        transforms=train_transforms,
        load_into_ram=config.load_into_ram,
    )
    val_dataset = dataset.HDF5Dataset(
        val_hdf5files,
        pre_transforms=pre_transforms,
        transforms=val_transforms,
        load_into_ram=config.load_into_ram,
    )
    test_dataset = dataset.HDF5Dataset(
        test_hdf5files,
        pre_transforms=pre_transforms,
        transforms=test_transforms,
        load_into_ram=config.load_into_ram,
    )

    feature_shape = (
        len(config.use_features),
        -1,
        len(use_landmarks)
    )
    token_shape = (-1,)
    num_workers = os.cpu_count()
    merge_fn = partial(
        dataset.merge_padded_batch,
        feature_shape=feature_shape,
        token_shape=token_shape,
        feature_padding_val=0.0,
        token_padding_val=key2token["<pad>"],
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=merge_fn,
        num_workers=num_workers,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        collate_fn=merge_fn,
        num_workers=num_workers,
        shuffle=False,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=merge_fn,
        num_workers=num_workers,
        shuffle=False,
    )
    in_channels = len(use_landmarks) * len(config.use_features)

    return train_dataloader, val_dataloader, test_dataloader, in_channels


def train_loop(
    dataloader, model, optimizer, scheduler, device, return_pred_times=False
):
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
    scaler = GradScaler()
    for batch_idx, batch_sample in enumerate(dataloader):
        feature = batch_sample["feature"]
        spatial_feature = batch_sample["spatial_feature"]
        tokens = batch_sample["token"]
        feature_pad_mask = batch_sample["feature_pad_mask"]
        spatial_feature_pad_mask = batch_sample["spatial_feature_pad_mask"]
        tokens_pad_mask = batch_sample["token_pad_mask"]
        feature_lengths = batch_sample["feature_lengths"]
        # check_tokens_format(tokens, tokens_pad_mask, start_id, end_id)

        feature = feature.to(device)
        spatial_feature = spatial_feature.to(device)
        tokens = tokens.to(device)
        feature_pad_mask = feature_pad_mask.to(device)
        spatial_feature_pad_mask = spatial_feature_pad_mask.to(device)
        tokens_pad_mask = tokens_pad_mask.to(device)
        frames = feature.shape[-2]

        input_lengths = feature_lengths
        target_lengths = target_lengths = torch.sum(tokens_pad_mask, dim=1)
        pred_start = time.perf_counter()
        optimizer.zero_grad()

        ret_dict = model.forward(
            src_feature=feature,
            spatial_feature=spatial_feature,
            tgt_feature=tokens,
            src_causal_mask=None,
            src_padding_mask=feature_pad_mask,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            mode="train",
        )
        loss = model.criterion_calculation(ret_dict, tokens, target_lengths)

        pred_end = time.perf_counter()
        pred_times.append([frames, pred_end - pred_start])

        # NaNチェック - エラーが発生した場合の対策
        if torch.isnan(loss).any():
            print("警告: NaNが検出されました。このバッチをスキップします")
            continue

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 勾配クリッピングの値を小さくして安定性を向上
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        # optimizer.step()

        train_loss += loss.item()
        del loss
        del ret_dict

        # Print current loss per 100 steps.
        # if batch_idx % 100 == 0:
        #     loss = loss.item()
        #     steps = batch_idx * len(feature)
        #     # ロジットの分布を確認するための診断情報を追加
        #     with torch.no_grad():
        #         classifier_output = model.classifier.weight.clone()
        #         blank_weight_norm = torch.norm(classifier_output[:, model.blank_id])
        #         other_weight_norm = torch.norm(classifier_output) - blank_weight_norm
        #         print(
        #             f"ブランク重みのノルム: {blank_weight_norm.item():.4f}, 他の重みのノルム平均: {other_weight_norm.item()/(classifier_output.size(1)-1):.4f}"
        #         )

        #     print(f"loss:{loss:>7f} [{steps:>5d}/{size:>5d}]")

    # 学習率スケジューラを更新
    if scheduler is not None:
        scheduler.step()
        print(f"Learning rate updated to: {scheduler.get_last_lr()[0]:.6f}")

    print(f"Done. Time:{time.perf_counter()-start}")
    # Average loss.
    train_loss /= num_batches
    print("Training performance: \n", f"Avg loss:{train_loss:>8f}\n")
    pred_times = np.array(pred_times)
    return train_loss


def val_loop(dataloader, model, device, return_pred_times=False, current_epoch=None):
    num_batches = len(dataloader)
    val_loss = 0

    pred_times = []
    hypothesis_text_list = []
    hypothesis_text_conv_list = []
    reference_text_list = []

    # Switch to evaluation mode.
    model.eval()
    # Main loop.
    print("Start validation.")
    start = time.perf_counter()
    tokens_causal_mask = None
    with torch.no_grad():
        for batch_idx, batch_sample in enumerate(dataloader):
            feature = batch_sample["feature"]
            spatial_feature = batch_sample["spatial_feature"]
            tokens = batch_sample["token"]
            feature_pad_mask = batch_sample["feature_pad_mask"]
            spatial_feature_pad_mask = batch_sample["spatial_feature_pad_mask"]
            tokens_pad_mask = batch_sample["token_pad_mask"]
            feature_lengths = batch_sample["feature_lengths"]

            feature = feature.to(device)
            spatial_feature = spatial_feature.to(device)
            tokens = tokens.to(device)
            feature_pad_mask = feature_pad_mask.to(device)
            spatial_feature_pad_mask = spatial_feature_pad_mask.to(device)
            tokens_pad_mask = tokens_pad_mask.to(device)

            frames = feature.shape[-2]

            # Predict.
            input_lengths = feature_lengths
            target_lengths = target_lengths = torch.sum(tokens_pad_mask, dim=1)
            pred_start = time.perf_counter()
            ret_dict = model.forward(
                src_feature=feature,
                spatial_feature=spatial_feature,
                tgt_feature=tokens,
                src_causal_mask=None,
                src_padding_mask=feature_pad_mask,
                input_lengths=input_lengths,
                target_lengths=target_lengths,
                mode="eval",
                current_epoch=current_epoch,  # エポック情報の追加
            )
            loss = model.criterion_calculation(ret_dict, tokens, target_lengths)
            pred_end = time.perf_counter()
            pred_times.append([frames, pred_end - pred_start])
            pred = ret_dict["recognized_sents"]
            conv_pred = ret_dict["conv_sents"]
            

            tokens = tokens.tolist()
            hypothesis_text_conv_list.append(" ".join(map(str, tokens)))
            reference_text = [" ".join(map(str, seq)) for seq in tokens]

            pred_words = [[middle_dataset_relation.middle_dataset_relation_dict[word] for word, idx in sample] for sample in pred]
            hypothesis_text = [" ".join(map(str, seq)) for seq in pred_words]

            conv_pred_words = [[middle_dataset_relation.middle_dataset_relation_dict[word] for word, idx in sample] for sample in conv_pred]
            hypothesis_text_conv =[" ".join(map(str,seq)) for seq in conv_pred_words]
            
            
            reference_text_list.append(reference_text[0])
            hypothesis_text_list.append(hypothesis_text[0])
            hypothesis_text_conv_list.append(hypothesis_text_conv[0])

            val_loss += loss.item()
            del loss
            del ret_dict

    wer_score = wer(reference_text, hypothesis_text)
    val_loss /= num_batches
    print(f"Wer: {wer_score:.10f}, Avg loss: {val_loss:.10f}")
    pred_times = np.array(pred_times)
    label_wer = {}
    print(len(reference_text_list), "reference_text_list")
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
    print("Test performance: \n", f"Avg WER:{awer:>0.10f}\n")
    pred_times = np.array(pred_times)
    error_rate_cer = cer(reference_text_list, hypothesis_text_list)
    error_rate_mer = mer(reference_text_list, hypothesis_text_list)
    print(f"Overall WER: {awer}")
    print(f"Overall CER: {error_rate_cer}")
    print(f"Overall MER: {error_rate_mer}")
    return val_loss, wer_score

def test_loop(dataloader, model, device, return_pred_times=False, blank_id=100):
    size = len(dataloader.dataset)
    hypothesis_text_list = []
    hypothesis_text_conv_list = []
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
            spatial_feature = batch_sample["spatial_feature"]
            tokens = batch_sample["token"]
            feature_pad_mask = batch_sample["feature_pad_mask"]
            spatial_feature_pad_mask = batch_sample["spatial_feature_pad_mask"]
            tokens_pad_mask = batch_sample["token_pad_mask"]
            feature_lengths = batch_sample["feature_lengths"]

            feature = feature.to(device)
            spatial_feature = spatial_feature.to(device)
            tokens = tokens.to(device)
            tokens_pad_mask = tokens_pad_mask.to(device)
            feature_pad_mask = (
                feature_pad_mask.to(device) if feature_pad_mask is not None else None
            )
            spatial_feature_pad_mask = spatial_feature_pad_mask.to(device)

            frames = feature.shape[-2]

            # 実際のシーケンス長を計算
            # 入力シーケンス長（CNNで縮小された後の長さ）
            # 後述のモデルのForwardパスで処理される形に変形
            N, C, T, J = feature.shape
            src_feature_reshaped = (
                feature.permute(0, 3, 1, 2).contiguous().view(N, C * J, T)
            )

            # CNN通過後の長さを推定（ストライドとカーネルサイズに基づく）
            # 簡易的な方法：CNNの出力長さ = (入力長さ + 2*パディング - カーネルサイズ) / ストライド + 1
            from CNN_BiLSTM.continuous_sign_language.modeling import (
                config as model_config,
            )

            kernel_size = model_config.kernel_size
            stride = model_config.stride
            padding = model_config.padding

            input_lengths = feature_lengths

            # ターゲット長の計算（実際のトークン長を使用）
            target_lengths = torch.sum(tokens_pad_mask, dim=1)

            # Predict.
            pred_start = time.perf_counter()
            ret_dict = model.forward(
                src_feature=feature,
                spatial_feature=spatial_feature,
                tgt_feature=tokens,
                src_causal_mask=None,
                src_padding_mask=feature_pad_mask,
                input_lengths=input_lengths,
                target_lengths=target_lengths,
                mode="test",
                blank_id=0,
            )

            pred_end = time.perf_counter()
            pred_times.append([frames, pred_end - pred_start])
            pred = ret_dict["recognized_sents"]
            conv_pred = ret_dict["conv_sents"]

            tokens = tokens.tolist()
            reference_text = [" ".join(map(str, seq)) for seq in tokens]

            pred_words = [[middle_dataset_relation.middle_dataset_relation_dict[word] for word, idx in sample] for sample in pred]
            hypothesis_text = [" ".join(map(str, seq)) for seq in pred_words]

            conv_pred_words = [[middle_dataset_relation.middle_dataset_relation_dict[word] for word, idx in sample] for sample in conv_pred]
            hypothesis_text_conv =[" ".join(map(str,seq)) for seq in conv_pred_words]
            
            
            reference_text_list.append(reference_text[0])
            hypothesis_text_list.append(hypothesis_text[0])
            hypothesis_text_conv_list.append(hypothesis_text_conv[0])   
    print(reference_text_list)
    print(hypothesis_text_list)
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
    print("Test performance: \n", f"Avg WER:{awer:>0.10f}\n")
    pred_times = np.array(pred_times)
    error_rate_cer = cer(reference_text_list, hypothesis_text_list)
    error_rate_mer = mer(reference_text_list, hypothesis_text_list)
    print(f"Overall WER: {awer}")
    print(f"Overall CER: {error_rate_cer}")
    print(f"Overall MER: {error_rate_mer}")
    retval = (awer, pred_times) if return_pred_times else awer
    return retval


def save_model(save_path, model_default_dict, optimizer_dict, epoch):
    torch.save(
        {
            "model_state_dict": model_default_dict,
            "optimizer_state_dict": optimizer_dict,
            "epoch": epoch,
        },
        save_path,
    )

    print(f"モデルとオプティマイザの状態を {save_path} に保存しました。")


def load_model(model, save_path: str, device: str = "cpu"):

    checkpoint = torch.load(save_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # オプティマイザの再構築
    optimizer_loaded = torch.optim.Adam(model.parameters(), lr=3e-4)
    optimizer_loaded.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch_loaded = checkpoint.get("epoch", None)

    print(f"エポック {epoch_loaded} までのモデルをロードしました。")

    return model, optimizer_loaded, epoch_loaded
