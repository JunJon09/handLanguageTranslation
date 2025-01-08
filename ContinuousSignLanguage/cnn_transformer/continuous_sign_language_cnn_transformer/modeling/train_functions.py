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

def forward(model, feature, tokens, feature_pad_mask, tokens_pad_mask, spatial_feature,
            tokens_causal_mask=None):
    
    if isinstance(model, models.CNNTransformerModel):
        if tokens_causal_mask is None:
            tokens_causal_mask = make_causal_mask(tokens_pad_mask)
        if tokens_causal_mask.shape[-1] != tokens_pad_mask.shape[-1]:
            tokens_causal_mask = make_causal_mask(tokens_pad_mask)
        preds = model(src_feature=feature,
                      tgt_feature=tokens,
                      src_causal_mask=None,
                      src_padding_mask=feature_pad_mask,
                      tgt_causal_mask=tokens_causal_mask,
                      tgt_padding_mask=tokens_pad_mask)
    else:
        raise NotImplementedError(f"Unknown model type:{type(model)}.")
    return preds, tokens_causal_mask


def check_tokens_format(tokens, tokens_pad_mask, start_id, end_id):
    # Check token's format.
    end_indices0 = np.arange(len(tokens))
    end_indices1 = tokens_pad_mask.sum(dim=-1).detach().cpu().numpy() - 1
    message = "The start and/or end ids are not included in tokens. " \
        f"Please check data format. start_id:{start_id}, " \
        f"end_id:{end_id}, enc_indices:{end_indices1}, tokens:{tokens}"
    ref_tokens = tokens.detach().cpu().numpy()
    assert (ref_tokens[:, 0] == start_id).all(), message
    assert (ref_tokens[end_indices0, end_indices1] == end_id).all(), message


def train_loop_csir_s2s(dataloader,
                        model,
                        loss_fn,
                        optimizer,
                        device,
                        start_id,
                        end_id,
                        return_pred_times=False):
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
        spatial_feature = "Kk"
        """
        追加でデータが
        """
        check_tokens_format(tokens, tokens_pad_mask, start_id, end_id)

        feature = feature.to(device)
        feature_pad_mask = feature_pad_mask.to(device)
        tokens = tokens.to(device)
        tokens_pad_mask = tokens_pad_mask.to(device)

        frames = feature.shape[-2]

        # Predict.
        pred_start = time.perf_counter()
        preds, tokens_causal_mask = forward(model, feature, tokens,
                                            feature_pad_mask, tokens_pad_mask,
                                            spatial_feature, tokens_causal_mask
                                            )
        pred_end = time.perf_counter()
        pred_times.append([frames, pred_end - pred_start])

        # Compute loss.
        # Preds do not include <start>, so skip that of tokens.
        loss = 0
        if isinstance(loss_fn, nn.CrossEntropyLoss):
            for t_index in range(1, tokens.shape[-1]):
                pred = preds[:, t_index-1, :]
                token = tokens[:, t_index]
                loss += loss_fn(pred, token)
            loss /= tokens.shape[-1]
        # LabelSmoothingCrossEntropyLoss
        else:
            # `[N, T, C] -> [N, C, T]`
            preds = preds.permute([0, 2, 1])
            # Remove prediction after the last token.
            if preds.shape[-1] == tokens.shape[-1]:
                preds = preds[:, :, :-1]
            loss = loss_fn(preds, tokens[:, 1:])

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


def val_loop_csir_s2s(dataloader,
                      model,
                      loss_fn,
                      device,
                      start_id,
                      end_id,
                      return_pred_times=False):
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
     
            check_tokens_format(tokens, tokens_pad_mask, start_id, end_id)
     
            feature = feature.to(device)
            feature_pad_mask = feature_pad_mask.to(device)
            tokens = tokens.to(device)
            tokens_pad_mask = tokens_pad_mask.to(device)
     
            frames = feature.shape[-2]
     
            # Predict.
            pred_start = time.perf_counter()
            preds, tokens_causal_mask = forward(model, feature, tokens,
                                                feature_pad_mask, tokens_pad_mask,
                                                tokens_causal_mask)
            pred_end = time.perf_counter()
            pred_times.append([frames, pred_end - pred_start])
     
            # Compute loss.
            # Preds do not include <start>, so skip that of tokens.
            loss = 0
            if isinstance(loss_fn, nn.CrossEntropyLoss):
                for t_index in range(1, tokens.shape[-1]):
                    pred = preds[:, t_index-1, :]
                    token = tokens[:, t_index]
                    loss += loss_fn(pred, token)
                loss /= tokens.shape[-1]
            # LabelSmoothingCrossEntropyLoss
            else:
                # `[N, T, C] -> [N, C, T]`
                preds = preds.permute([0, 2, 1])
                # Remove prediction after the last token.
                if preds.shape[-1] == tokens.shape[-1]:
                    preds = preds[:, :, :-1]
                loss = loss_fn(preds, tokens[:, 1:])
     
            val_loss += loss.item()
    print(f"Done. Time:{time.perf_counter()-start}")

    # Average loss.
    val_loss /= num_batches
    print("Validation performance: \n",
          f"Avg loss:{val_loss:>8f}\n")
    pred_times = np.array(pred_times)
    retval = (val_loss, pred_times) if return_pred_times else val_loss
    return retval


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
            feature = batch_sample["feature"]
            tokens = batch_sample["token"]
            tokens_pad_mask = batch_sample["token_pad_mask"]
     
            check_tokens_format(tokens, tokens_pad_mask, start_id, end_id)
     
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
            print(pred_ids)
            pred_ids = remove_consecutive_duplicates([int(x) for x in pred_ids]) # [7, 6]


            wer = edit_distance(tokens, pred_ids)
            print("*"*100)
            print(ref_length, tokens, pred_ids, wer)
            print("*"*100)
            wer /= ref_length
            print(wer)
            total_wer += wer
            if batch_idx < verbose_num:
                print(f"WER: {wer}")
                print("="*40)
    print(f"Done. Time:{time.perf_counter()-start}")

    # Average WER.
    awer = total_wer / size * 100
    print("Test performance: \n",
          f"Avg WER:{awer:>0.1f}%\n")
    pred_times = np.array(pred_times)
    retval = (awer, pred_times) if return_pred_times else awer
    return retval

def remove_consecutive_duplicates(lst):
    return [key for key, _ in itertools.groupby(lst)]


class LabelSmoothingCrossEntropyLoss(nn.Module):
    """Cross-entropy loss with label smoothing.

    For the detail, please refer
    "Rethinking the Inception Architecture for Computer Vision"
    https://arxiv.org/abs/1512.00567
    """
    def __init__(self, weight=None, ignore_indices=None, reduction="none",
                 label_smoothing=0.0):
        super().__init__()
        self.weight = weight
        if isinstance(ignore_indices, int):
            self.ignore_indices = [ignore_indices]
        else:
            self.ignore_indices = ignore_indices
        assert reduction in ["none",
                             "mean_batch_prior", "mean_temporal_prior",
                             "sum"]
        self.reduction = reduction
        assert label_smoothing >= 0.0
        assert label_smoothing <= 1.0
        self.label_smoothing = label_smoothing

    def _isnotin_ignore(self, target):
        # Please refer
        # https://github.com/pytorch/pytorch/issues/3025
        # pylint error of torch.tensor() should be solved in the future release.
        # https://github.com/pytorch/pytorch/issues/24807
        ignore = torch.tensor(self.ignore_indices, dtype=target.dtype,
                              device=target.device)
        isin = (target[..., None] == ignore).any(-1)
        return isin.bitwise_not()

    def _calc_loss(self, logit_t, target_t):
        logit_mask = torch.ones(logit_t.shape[-1],
                                dtype=logit_t.dtype,
                                device=logit_t.device)
        target_mask = torch.ones(target_t.shape,
                                 dtype=logit_t.dtype,
                                 device=logit_t.device)
        if self.ignore_indices is not None:
            logit_mask[self.ignore_indices] = 0
            target_mask = self._isnotin_ignore(target_t).float()
        if self.weight is None:
            weight = torch.ones(logit_t.shape[-1],
                                dtype=logit_t.dtype,
                                device=logit_t.device)
        else:
            weight = self.weight.to(dtype=logit_t.dtype, device=logit_t.device)
        # Calculate CE.
        logprobs = F.log_softmax(logit_t, dim=-1)
        logprobs_m = logprobs * weight * logit_mask
        nll_loss = -logprobs_m.gather(dim=-1, index=target_t.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs_m.sum(dim=-1) / logit_mask.sum()
        smooth_loss *= target_mask
        loss = (1 - self.label_smoothing) * nll_loss + self.label_smoothing * smooth_loss
        return loss

    def forward(self, logit, target):
        """Perform forward computation.

        # Args:
          - logit: `[N, C]` or `[N, C, T]`
          - target: `[N]` or [N, T]
        """
        # Check format.
        if len(logit.shape) == 2:
            logit = logit.unsqueeze(-1)
        if len(target.shape) == 1:
            target = target.unsqueeze(-1)
        assert len(logit.shape) == 3, f"{logit.shape}"
        assert len(target.shape) == 2, f"{target.shape}"
        assert logit.shape[0] == target.shape[0], f"{logit.shape, target.shape}"
        assert logit.shape[-1] == target.shape[-1], f"{logit.shape, target.shape}"

        loss = 0
        for t in range(target.shape[-1]):
            _loss = self._calc_loss(logit[:, :, t], target[:, t])
            # Reduction should be conducted in a loop when reduction is
            # mean_batch_prior.
            if self.reduction == "mean_batch_prior":
                if self.ignore_indices is not None:
                    denom = len([t for t in target[:, t]
                                 if t not in self.ignore_indices])
                else:
                    denom = logit.shape[0]
                _loss /= max(denom, 1)
            loss += _loss

        # Reduction.
        if self.reduction == "sum":
            loss = loss.sum()
        # Temporal Normalization.
        if self.reduction == "mean_batch_prior":
            loss = loss.sum() / target.shape[-1]
        if self.reduction == "mean_temporal_prior":
            target_lengths = self._isnotin_ignore(target).sum(dim=-1)
            loss /= torch.clamp(target_lengths, min=1)
            loss = loss.mean()
        return loss


def make_causal_mask(ref_mask,
                     lookahead=0):
    """Make causal mask.
    # Args:
      - ref_mask: `[N, T (query)]`
        The reference mask to make causal mask.
      - lookahead: lookahead frame
    # Returns:
      - ref_mask: `[T (query), T (key)]`
    """
    causal_mask = ref_mask.new_ones([ref_mask.size(1), ref_mask.size(1)],
                                    dtype=ref_mask.dtype)
    causal_mask = torch.tril(causal_mask,
                             diagonal=lookahead,
                             out=causal_mask).unsqueeze(0)
    return causal_mask

def save_model(save_path, model_default_dict, optimizer_dict, epoch, val_loss):
    torch.save({
        'model_state_dict': model_default_dict,
        'optimizer_state_dict': optimizer_dict,
        'epoch': epoch,
        'val_losses': val_loss,
    }, save_path)

    print(f"モデルとオプティマイザの状態を {save_path} に保存しました。")

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
    val_losses_loaded = checkpoint.get('val_losses', None)
    
    print(f"エポック {epoch_loaded} までのモデルをロードしました。")
    
    return model_loaded, optimizer_loaded, epoch_loaded, val_losses_loaded, test_dataloader, key2token
