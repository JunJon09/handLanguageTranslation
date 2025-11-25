import CNN_BiLSTM.continuous_sign_language.features as features
import CNN_BiLSTM.continuous_sign_language.config as config
import CNN_BiLSTM.continuous_sign_language.dataset as dataset
import CNN_BiLSTM.continuous_sign_language.modeling.config as model_config
import CNN_BiLSTM.continuous_sign_language.plots as plots
from torchvision.transforms import Compose
import os
from functools import partial
from torch.utils.data import DataLoader
import time
from torch import nn
import numpy as np
import torch
from jiwer import wer, cer, mer
import logging
import CNN_BiLSTM.continuous_sign_language.modeling.phoenx as phoenx

def set_dataloader(key2token, mode_hdf5files, val_hdf5files, mode):
    _, use_landmarks = features.get_fullbody_landmarks()
    trans_select_feature = features.SelectLandmarksAndFeature(
        landmarks=use_landmarks, features=config.use_features
    )
    trans_repnan = features.ReplaceNan()
    trans_norm = features.PartsBasedNormalization(
        align_mode="framewise", scale_mode="unique"
    )

    pre_transforms = Compose([trans_select_feature, trans_repnan, trans_norm])

    feature_shape = (len(config.use_features), -1, len(use_landmarks))
    token_shape = (-1,)
    num_workers = os.cpu_count()
    merge_fn = partial(
        dataset.merge_padded_batch,
        feature_shape=feature_shape,
        token_shape=token_shape,
        feature_padding_val=0.0,
        token_padding_val=key2token["<pad>"],
    )
    in_channels = len(use_landmarks) * len(config.use_features)

    val_transforms = Compose([features.ToTensor()])

    val_dataset = dataset.HDF5Dataset(
            val_hdf5files,
            pre_transforms=pre_transforms,
            transforms=val_transforms,
            load_into_ram=config.load_into_ram,
    )

    val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            collate_fn=merge_fn,
            num_workers=num_workers,
            shuffle=False,
    )
    if mode == "train":
        train_transforms = Compose([features.ToTensor()])

        train_dataset = dataset.HDF5Dataset(
            mode_hdf5files,
            pre_transforms=pre_transforms,
            transforms=train_transforms,
            load_into_ram=config.load_into_ram,
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            collate_fn=merge_fn,
            num_workers=num_workers,
            shuffle=True,
        )
        return train_dataloader, val_dataloader, in_channels

    elif mode == "test":
        test_transforms = Compose([features.ToTensor()])

        test_dataset = dataset.HDF5Dataset(
            mode_hdf5files,
            pre_transforms=pre_transforms,
            transforms=test_transforms,
            load_into_ram=config.load_into_ram,
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            collate_fn=merge_fn,
            num_workers=num_workers,
            shuffle=False,
        )

        return test_dataloader, val_dataloader, in_channels



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
        loss, log_probs = model.forward(
            src_feature=feature,
            spatial_feature=spatial_feature,
            tgt_feature=tokens,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            mode="train",
        )
        pred_end = time.perf_counter()
        pred_times.append([frames, pred_end - pred_start])

        # Back propagation.
        optimizer.zero_grad()
        loss.backward()
        # 勾配クリッピング（過学習対策）
        clip_norm = model_config.grad_clip_norm if hasattr(model_config, 'grad_clip_norm') else 0.5
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()

        train_loss += loss.item()

        # Print current loss per 100 steps.
        if batch_idx % 100 == 0:
            loss = loss.item()
            steps = batch_idx * len(feature)
            # ロジットの分布を確認するための診断情報を追加
            with torch.no_grad():
                classifier_output = model.classifier.weight.clone()
                blank_weight_norm = torch.norm(classifier_output[:, model.blank_id])
                other_weight_norm = torch.norm(classifier_output) - blank_weight_norm
                print(
                    f"ブランク重みのノルム: {blank_weight_norm.item():.4f}, 他の重みのノルム平均: {other_weight_norm.item()/(classifier_output.size(1)-1):.4f}"
                )

            print(f"loss:{loss:>7f} [{steps:>5d}/{size:>5d}]")

    # 学習率スケジューラを更新
    if scheduler is not None:
        scheduler.step()
        print(f"Learning rate updated to: {scheduler.get_last_lr()[0]:.6f}")

    print(f"Done. Time:{time.perf_counter()-start}")
    # Average loss.
    train_loss /= num_batches
    print("Training performance: \n", f"Avg loss:{train_loss:>8f}\n")
    pred_times = np.array(pred_times)
    retval = (train_loss, pred_times) if return_pred_times else train_loss
    return retval


def val_loop(dataloader, model, device, return_pred_times=False, current_epoch=None):
    num_batches = len(dataloader)
    val_loss = 0

    # Collect prediction time.
    pred_times = []

    # Switch to evaluation mode.
    model.eval()
    # Main loop.
    print("Start validation.")
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
            feature_pad_mask = feature_pad_mask.to(device)
            spatial_feature_pad_mask = spatial_feature_pad_mask.to(device)
            tokens_pad_mask = tokens_pad_mask.to(device)

            frames = feature.shape[-2]

            # Predict.
            input_lengths = feature_lengths
            target_lengths = target_lengths = torch.sum(tokens_pad_mask, dim=1)
            pred_start = time.perf_counter()
            loss, log_probs = model.forward(
                src_feature=feature,
                spatial_feature=spatial_feature,
                tgt_feature=tokens,
                input_lengths=input_lengths,
                target_lengths=target_lengths,
                mode="eval",
            )
            pred_end = time.perf_counter()
            pred_times.append([frames, pred_end - pred_start])
            val_loss += loss.item()
            tokens = tokens.tolist()
    print(f"Done. Time:{time.perf_counter()-start}")
    # Average loss.
    val_loss /= num_batches
    print("Validation performance: \n", f"Avg loss:{val_loss:>8f}\n")
    pred_times = np.array(pred_times)
    retval = (val_loss, pred_times) if return_pred_times else val_loss
    return retval


def test_loop(
    dataloader,
    model,
    device,
    return_pred_times=False,
):

    size = len(dataloader.dataset)
    hypothesis_text_list = []
    hypothesis_text_conv_list = []
    reference_text_list = []
    topk_stats = {
        'total_segments': 0,
        'rank_distribution': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 'out_of_topk': 0},
        'correct_in_topk': 0,
        'details': []  # 詳細なログ用
    }

    # Collect prediction time.
    pred_times = []
    # Switch to evaluation mode.
    model.eval()
    # Main loop.
    logging.info("Start test.")
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

            input_lengths = feature_lengths

            # ターゲット長の計算（実際のトークン長を使用）
            target_lengths = torch.sum(tokens_pad_mask, dim=1)

            # Predict.
            pred_start = time.perf_counter()
            forward_result = model.forward(
                src_feature=feature,
                spatial_feature=spatial_feature,
                tgt_feature=tokens,
                input_lengths=input_lengths,
                target_lengths=target_lengths,
                mode="test",
            )
            
            # 戻り値の数に応じて処理を分岐
            if len(forward_result) == 3:
                pred, conv_pred, sequence_logits = forward_result
                topk_result = None
            if len(forward_result) == 4:
                pred, conv_pred, sequence_logits, topk_result = forward_result
            else:
                pred, conv_pred = forward_result
                sequence_logits = None
                topk_result = None

            pred_end = time.perf_counter()
            pred_times.append([frames, pred_end - pred_start])

            tokens = tokens.tolist()
            reference_text = [" ".join(map(str, seq)) for seq in tokens]
            
            pred_words = [
                [
                    phoenx.phon[word]
                    for word, idx in sample
                ]
                for sample in pred
            ]
            hypothesis_text = [" ".join(map(str, seq)) for seq in pred_words]

            conv_pred_words = [
                [
                    phoenx.phon.get(word, '<UNK>')
                    for word, idx in sample
                ]
                for sample in conv_pred
            ]
            hypothesis_text_conv = [" ".join(map(str, seq)) for seq in conv_pred_words]

            if topk_result is not None:
                analyze_topk_predictions(topk_result, tokens[0], batch_idx, topk_stats)

            reference_text_list.append(reference_text[0])
            hypothesis_text_list.append(hypothesis_text[0])
            hypothesis_text_conv_list.append(hypothesis_text_conv[0])

    logging.info(f"参照文リスト: {reference_text_list}")
    logging.info(f"予測文リスト: {hypothesis_text_list}")

    # WER評価指標の計算
    wer_metrics = calculate_wer_metrics(reference_text_list, hypothesis_text_list)
    awer = wer_metrics["awer"] if wer_metrics else 0.0

    print(f"テスト完了. 時間:{time.perf_counter()-start:.2f}秒")
    print("テストサイズ:", size)
    print(f"テスト平均時間: {(time.perf_counter()-start)/size:.4f}秒/サンプル")
    pred_times = np.array(pred_times)


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

    logging.info(f"モデルとオプティマイザの状態を {save_path} に保存しました。")


def load_model(model, save_path: str, device: str = "cpu"):

    checkpoint = torch.load(save_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # オプティマイザの再構築
    optimizer_loaded = torch.optim.Adam(model.parameters(), lr=3e-4)
    optimizer_loaded.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch_loaded = checkpoint.get("epoch", None)

    logging.info(f"エポック {epoch_loaded} までのモデルをロードしました。")

    return model, optimizer_loaded, epoch_loaded


def calculate_wer_metrics(reference_text_list, hypothesis_text_list):
    """
    WER関連の評価指標を計算する関数

    Args:
        reference_text_list: 正解テキストのリスト
        hypothesis_text_list: 予測テキストのリスト

    Returns:
        dict: 各種評価指標の辞書
    """
    try:
        # ラベル別WERの計算
        label_wer = {}
        # 単語ごとの誤り回数を集計
        word_error_counts = {}

        for ref, hyp in zip(reference_text_list, hypothesis_text_list):
            ref_label = ref  # Get the first token as label

            if ref_label not in label_wer:
                label_wer[ref_label] = {"refs": [], "hyps": []}
            label_wer[ref_label]["refs"].append(ref)
            label_wer[ref_label]["hyps"].append(hyp)

            # 単語ごとに誤りをカウント
            ref_words = ref.split()
            hyp_words = hyp.split()

            # 各単語について正誤を判定
            for ref_word in ref_words:
                if ref_word not in word_error_counts:
                    word_error_counts[ref_word] = {
                        "correct": 0,
                        "incorrect": 0,
                        "total": 0,
                    }

                word_error_counts[ref_word]["total"] += 1

                # 単語が予測に含まれているかチェック
                if ref_word in hyp_words:
                    word_error_counts[ref_word]["correct"] += 1
                else:
                    word_error_counts[ref_word]["incorrect"] += 1

        # Calculate and log WER for each label
        logging.info("WER per label:")
        for label in label_wer:
            label_refs = label_wer[label]["refs"]
            label_hyps = label_wer[label]["hyps"]
            label_wer_score = wer(label_refs, label_hyps)
            logging.info(
                f"Label {label}: {label_wer_score:.10f} ({len(label_refs)} samples)"
            )

        # 単語ごとの誤り回数をログ出力
        logging.info("\n=== 単語ごとの誤り回数 ===")
        for word in sorted(
            word_error_counts.keys(),
            key=lambda x: word_error_counts[x]["incorrect"],
            reverse=True,
        ):
            stats = word_error_counts[word]
            error_rate = (
                stats["incorrect"] / stats["total"] if stats["total"] > 0 else 0
            )
            logging.info(
                f"単語 '{word}': 誤り {stats['incorrect']}回 / 合計 {stats['total']}回 (誤り率: {error_rate:.2%})"
            )

        # 全体的な評価指標を計算
        awer = wer(reference_text_list, hypothesis_text_list)
        error_rate_cer = cer(reference_text_list, hypothesis_text_list)
        error_rate_mer = mer(reference_text_list, hypothesis_text_list)

        # ログ出力
        logging.info(f"Test performance - Avg WER: {awer:>0.10f}")
        logging.info(f"Overall WER: {awer}")
        logging.info(f"Overall CER: {error_rate_cer}")
        logging.info(f"Overall MER: {error_rate_mer}")

        # エラー数が多い順
        sorted_words_count_desc = sorted(
            word_error_counts.items(), 
            key=lambda x: x[1]["incorrect"], 
            reverse=True
        )

        # エラー数が少ない順
        sorted_words_count_asc = sorted(
            word_error_counts.items(), 
            key=lambda x: x[1]["incorrect"], 
            reverse=False
        )

        # エラー率が高い順
        sorted_words_rate_desc = sorted(
            word_error_counts.items(), 
            key=lambda x: x[1]["incorrect"] / x[1]["total"] if x[1]["total"] > 0 else 0, 
            reverse=True
        )

        # エラー率が低い順
        sorted_words_rate_asc = sorted(
            word_error_counts.items(), 
            key=lambda x: x[1]["incorrect"] / x[1]["total"] if x[1]["total"] > 0 else 0, 
            reverse=False
)
        # 4つのグラフを生成
        plots.plot_word_error_distribution(word_error_counts, sorted_words_count_desc, "count_desc.png")
        plots.plot_word_error_distribution(word_error_counts, sorted_words_count_asc, "count_asc.png")
        plots.plot_word_error_distribution(word_error_counts, sorted_words_rate_desc, "rate_desc.png")
        plots.plot_word_error_distribution(word_error_counts, sorted_words_rate_asc, "rate_asc.png")

        return {
            "awer": awer,
            "cer": error_rate_cer,
            "mer": error_rate_mer,
            "label_wer": label_wer,
        }

    except Exception as e:
        logging.error(f"WER計算でエラー: {e}")
        return None

def analyze_topk_predictions(topk_result, reference_tokens, batch_idx, topk_stats):
    """
    TOP-K予測結果を分析し、統計情報を更新する
    
    Args:
        topk_result: AnalysisDecodeWithTopKの戻り値
        reference_tokens: 正解トークン列（リスト）
        batch_idx: バッチインデックス
        topk_stats: 統計情報を格納する辞書（更新される）
        model_decoder: デコーダオブジェクト（blank_id取得用）
    """
    decoded_words = topk_result['decoded']
    segment_analysis = topk_result['segment_analysis']
    
    logging.info(f"\n=== サンプル {batch_idx + 1} の TOP-K 分析 ===")
    logging.info(f"正解トークン列: {reference_tokens}")
    logging.info(f"デコード結果: {[w for w, _ in decoded_words]}")
    # 各セグメントの分析
    for seg_idx, seg in enumerate(segment_analysis):
        pass
        # topk_stats['total_segments'] += 1
        
        # selected_token = seg['token']
        # selected_word = seg['word']
        # rank = seg['rank']
        # confidence = seg['confidence']
        # top_candidates = seg['top_k_candidates']
        
        # # 順位分布を更新
        # if rank == -1:
        #     topk_stats['rank_distribution']['out_of_topk'] += 1
        # else:
        #     topk_stats['rank_distribution'][rank] += 1
        
        # # このセグメントが正解かどうかを判定
        # is_correct = selected_token in reference_tokens
        # if is_correct and rank <= 5:
        #     topk_stats['correct_in_topk'] += 1
        
        # # 詳細ログ
        # logging.info(f"\n  セグメント {seg_idx + 1}:")
        # logging.info(f"    選択: {selected_word} (トークンID: {selected_token})")
        # logging.info(f"    確信度: {confidence:.4f}")
        # logging.info(f"    TOP-5 順位: {rank if rank != -1 else '圏外'}")
        # logging.info(f"    正解含む: {'✓' if is_correct else '✗'}")
        # logging.info(f"    TOP-5 候補:")
        
        # for i, cand in enumerate(top_candidates, 1):
        #     is_answer = '← 正解' if cand['token'] in reference_tokens else ''
        #     logging.info(
        #         f"      {i}位: {cand['word']} (ID:{cand['token']}) "
        #         f"確率:{cand['prob']:.4f} {is_answer}"
        #     )
        
        # # 詳細データを保存
        # topk_stats['details'].append({
        #     'sample_idx': batch_idx,
        #     'segment_idx': seg_idx,
        #     'selected_token': selected_token,
        #     'selected_word': selected_word,
        #     'rank': rank,
        #     'confidence': confidence,
        #     'is_correct': is_correct,
        #     'reference_tokens': reference_tokens,
        #     'top_candidates': top_candidates
        # })
