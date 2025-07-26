import CNN_BiLSTM.continuous_sign_language.features as features
import CNN_BiLSTM.continuous_sign_language.config as config
import CNN_BiLSTM.continuous_sign_language.dataset as dataset
import CNN_BiLSTM.continuous_sign_language.modeling.middle_dataset_relation as middle_dataset_relation
from CNN_BiLSTM.continuous_sign_language.plots import (
    plot_attention_matrix,
    plot_attention_statistics,
    plot_attention_focus_over_time,
    visualize_ctc_alignment_path,
    extract_multilayer_features,
    plot_multilayer_feature_visualization,
    analyze_feature_separation,
)
import CNN_BiLSTM.continuous_sign_language.modeling.config as model_config
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

        # Predict.
        input_lengths = feature_lengths
        target_lengths = target_lengths = torch.sum(tokens_pad_mask, dim=1)
        pred_start = time.perf_counter()
        loss, log_probs = model.forward(
            src_feature=feature,
            spatial_feature=spatial_feature,
            tgt_feature=tokens,
            src_causal_mask=None,
            src_padding_mask=feature_pad_mask,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            mode="train",
        )
        print("loss", loss)
        pred_end = time.perf_counter()
        pred_times.append([frames, pred_end - pred_start])

        # NaNチェック - エラーが発生した場合の対策
        if torch.isnan(loss).any():
            print("警告: NaNが検出されました。このバッチをスキップします")
            continue

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
            val_loss, log_probs = model.forward(
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
            pred_end = time.perf_counter()
            pred_times.append([frames, pred_end - pred_start])
            print("val_loss", val_loss)
            # Compute loss.
            # Preds do not include <start>, so skip that of tokens.

            tokens = tokens.tolist()
            # reference_text = [" ".join(map(str, seq)) for seq in tokens]
            # hypothesis_text = [" ".join(map(str, seq)) for seq in log_probs]
            # wer_score = wer(reference_text, hypothesis_text)
            # print(f"Batch {batch_idx}: WER: {wer_score:.10f}")
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
    blank_id=100,
    visualize_attention=False,
    generate_confusion_matrix=False,
    visualize_confidence=False,
    visualize_multilayer_features=False,
    multilayer_method="both",
):
    """
    テストループ関数 - 予測精度評価と可視化・分析機能を統合

    Args:
        dataloader: テストデータローダー
        model: 学習済みモデル
        device: 計算デバイス ('cuda' or 'cpu')
        return_pred_times: 予測時間も返すかどうか
        blank_id: ブランクトークンのID
        visualize_attention: Attention可視化を有効にするか
        generate_confusion_matrix: 混同行列を生成するか
        visualize_confidence: 予測信頼度可視化を有効にするか
        visualize_multilayer_features: 多層特徴量可視化を有効にするか
        multilayer_method: 多層特徴量可視化手法 ('tsne', 'umap', 'both')

    Returns:
        float or tuple: WER値、または (WER, 予測時間) のタプル

    Note:
        以下の機能が統合されています:
        - 基本的な予測精度評価 (WER, CER, MER)
        - Attention重み可視化
        - CTC Alignment Path可視化  
        - 混同行列生成・分析
        - 予測信頼度可視化
        - 多層特徴量可視化 (CNN空間パターン、BiLSTM時系列、Attention重要度、最終統合特徴量)
        - CTC Alignment Path可視化
        - 混同行列分析
        - 予測信頼度可視化（新機能）
    """

    size = len(dataloader.dataset)
    hypothesis_text_list = []
    hypothesis_text_conv_list = []
    reference_text_list = []

    # 混同行列用のデータ収集
    if generate_confusion_matrix:
        prediction_labels = []
        ground_truth_labels = []
        logging.info("混同行列生成モードを有効化")

    # Attention可視化の設定
    max_visualize_samples = 20  # 最大可視化サンプル数
    output_dir, visualize_count = setup_visualization_environment(
        visualize_attention, max_visualize_samples
    )
    if visualize_attention:
        model.enable_attention_visualization()

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
                src_causal_mask=None,
                src_padding_mask=feature_pad_mask,
                input_lengths=input_lengths,
                target_lengths=target_lengths,
                mode="test",
                blank_id=0,
            )
            
            # 戻り値の数に応じて処理を分岐
            if len(forward_result) == 3:
                pred, conv_pred, sequence_logits = forward_result
            else:
                pred, conv_pred = forward_result
                sequence_logits = None

            pred_end = time.perf_counter()
            pred_times.append([frames, pred_end - pred_start])

            tokens = tokens.tolist()
            reference_text = [" ".join(map(str, seq)) for seq in tokens]
            
            pred_words = [
                [
                    middle_dataset_relation.middle_dataset_relation_dict[word]
                    for word, idx in sample
                ]
                for sample in pred
            ]
            hypothesis_text = [" ".join(map(str, seq)) for seq in pred_words]

            conv_pred_words = [
                [
                    middle_dataset_relation.middle_dataset_relation_dict.get(word, '<UNK>')
                    for word, idx in sample
                ]
                for sample in conv_pred
            ]
            hypothesis_text_conv = [" ".join(map(str, seq)) for seq in conv_pred_words]

            # # Attention & CTC & 信頼度可視化処理
            # if (visualize_attention or visualize_confidence) and visualize_count < max_visualize_samples:
            #     # Attention & CTC可視化
            #     success_attention, success_ctc = False, False
            #     if visualize_attention:
            #         success_attention, success_ctc = process_attention_visualization(
            #             model=model,
            #             batch_idx=batch_idx,
            #             feature=feature,
            #             spatial_feature=spatial_feature,
            #             tokens=tokens,
            #             feature_pad_mask=feature_pad_mask,
            #             input_lengths=input_lengths,
            #             target_lengths=target_lengths,
            #             reference_text=reference_text,
            #             hypothesis_text=hypothesis_text,
            #             output_dir=output_dir,
            #             max_samples=max_visualize_samples,
            #         )                # 信頼度可視化
            #     success_confidence, success_word_confidence = False, False
            #     if visualize_confidence:
            #         # log_probsを準備
            #         log_probs = None
            #         try:
            #             # 既に取得したsequence_logitsからlog_probsを計算
            #             if sequence_logits is not None:
            #                 log_probs = sequence_logits.log_softmax(-1)  # (T, B, C)
            #                 logging.info(f"信頼度可視化用log_probsを取得しました。形状: {log_probs.shape}")
            #                 # NaNや無限大値をチェック
            #                 if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
            #                     logging.warning("log_probsに無効な値が含まれています")
            #                     log_probs = None
            #             else:
            #                 logging.warning("sequence_logitsが利用できません")
            #         except Exception as e:
            #             logging.error(f"log_probs取得中にエラー: {e}")
            #             log_probs = None

            #         # 予測結果を準備
            #         pred_for_confidence = []
            #         if len(pred) > 0 and len(pred[0]) > 0:
            #             # predの構造: [[(word_id, confidence), ...], ...]
            #             pred_for_confidence = [item[0] if isinstance(item, tuple) else item for item in pred[0]]
            #             logging.info(f"信頼度可視化用予測データ: {len(pred_for_confidence)}個の単語")
            #         else:
            #             logging.warning("予測結果が空のため、信頼度可視化をスキップします")

            #         # 信頼度可視化を実行（データが揃っている場合のみ）
            #         if log_probs is not None:
            #             success_confidence, success_word_confidence = (
            #                 process_confidence_visualization(
            #                     log_probs=log_probs,
            #                     predictions=pred_for_confidence,
            #                     batch_idx=batch_idx,
            #                     output_dir=output_dir,
            #                     vocab_dict=middle_dataset_relation.middle_dataset_relation_dict,
            #                 )
            #             )
            #         else:
            #             logging.warning("log_probsが利用できないため、信頼度可視化をスキップします")

            #     # 多層特徴量可視化
            #     success_multilayer = False
            #     if visualize_multilayer_features:
            #         try:
            #             success_multilayer = process_multilayer_feature_visualization(
            #                 model=model,
            #                 feature=feature,
            #                 spatial_feature=spatial_feature,
            #                 tokens=tokens,
            #                 feature_pad_mask=feature_pad_mask,
            #                 input_lengths=input_lengths,
            #                 target_lengths=target_lengths,
            #                 pred=pred,
            #                 batch_idx=batch_idx,
            #                 output_dir=output_dir,
            #                 vocab_dict=middle_dataset_relation.middle_dataset_relation_dict,
            #                 method=multilayer_method
            #             )
                        
            #             if success_multilayer:
            #                 logging.info("多層特徴量可視化が成功しました")
            #             else:
            #                 logging.warning("多層特徴量可視化に失敗しました")
                            
            #         except Exception as e:
            #             logging.error(f"多層特徴量可視化でエラー: {e}")
            #             success_multilayer = False

            #     # 可視化結果をチェック
            #     any_success = False
            #     if visualize_attention:
            #         any_success = success_attention or success_ctc
            #     if visualize_confidence:
            #         any_success = (
            #             any_success or success_confidence or success_word_confidence
            #         )
            #     if visualize_multilayer_features:
            #         any_success = any_success or success_multilayer

            #     if any_success:
            #         visualize_count += 1
            #         results = []
            #         if visualize_attention:
            #             results.append(
            #                 f"Attention: {'成功' if success_attention else '失敗'}"
            #             )
            #             results.append(f"CTC: {'成功' if success_ctc else '失敗'}")
            #         if visualize_confidence:
            #             results.append(
            #                 f"信頼度: {'成功' if success_confidence else '失敗'}"
            #             )
            #             results.append(
            #                 f"単語信頼度: {'成功' if success_word_confidence else '失敗'}"
            #             )
            #         if visualize_multilayer_features:
            #             results.append(
            #                 f"多層特徴量: {'成功' if success_multilayer else '失敗'}"
            #             )

            #         logging.info(f"可視化完了 ({', '.join(results)})")

            reference_text_list.append(reference_text[0])
            hypothesis_text_list.append(hypothesis_text[0])
            hypothesis_text_conv_list.append(hypothesis_text_conv[0])

            # 混同行列用のラベル収集
            if generate_confusion_matrix:
                # 単語レベルでの予測と正解を収集
                ref_words, pred_words = collect_prediction_labels(
                    reference_text[0], hypothesis_text[0]
                )
                ground_truth_labels.extend(ref_words)
                prediction_labels.extend(pred_words)

    logging.info(f"参照文リスト: {reference_text_list}")
    logging.info(f"予測文リスト: {hypothesis_text_list}")
    logging.info(f"テスト完了. 時間:{time.perf_counter()-start:.2f}秒")

    # WER評価指標の計算
    wer_metrics = calculate_wer_metrics(reference_text_list, hypothesis_text_list)
    awer = wer_metrics["awer"] if wer_metrics else 0.0
    pred_times = np.array(pred_times)

    # Attention & CTC可視化の後処理
    finalize_visualization(
        model=model,
        visualize_attention=visualize_attention,
        visualize_count=visualize_count,
        max_visualize_samples=max_visualize_samples,
        output_dir=output_dir,
    )

    # 混同行列の生成
    if generate_confusion_matrix:
        success = generate_confusion_matrix_analysis(
            prediction_labels=prediction_labels,
            ground_truth_labels=ground_truth_labels,
            save_dir=config.plot_save_dir,
        )

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


def visualize_attention_weights(
    model, batch_idx, reference_text, hypothesis_text, output_dir
):
    """
    Attention重みの可視化処理を実行

    Args:
        model: 学習済みモデル
        batch_idx: バッチインデックス
        reference_text: 参照文のリスト
        hypothesis_text: 予測文のリスト
        output_dir: 出力ディレクトリ

    Returns:
        bool: 可視化が成功したかどうか
    """
    try:
        attention_weights = model.get_attention_weights()

        if attention_weights is None:
            logging.warning(f"Batch {batch_idx}: Attention重みが取得できませんでした")
            return False

        # 予測結果の評価
        ref_text = reference_text[0]
        hyp_text = hypothesis_text[0]
        is_correct = ref_text == hyp_text

        # ファイル名の設定
        sample_name = f"batch_{batch_idx}_sample_0"
        status = "correct" if is_correct else "incorrect"

        logging.info(f"Attention可視化開始: {sample_name} ({status})")

        # 1. Attentionマトリックス
        plot_attention_matrix(
            attention_weights,
            sample_idx=0,
            save_path=os.path.join(
                output_dir, f"attention_matrix_{sample_name}_{status}.png"
            ),
            title=f"Attention Matrix ({status})",
        )

        # 2. 統計情報
        plot_attention_statistics(
            attention_weights[0:1],
            save_path=os.path.join(
                output_dir, f"attention_stats_{sample_name}_{status}.png"
            ),
        )

        # 3. 時間的変化
        plot_attention_focus_over_time(
            attention_weights[0:1],
            save_path=os.path.join(
                output_dir, f"attention_focus_{sample_name}_{status}.png"
            ),
        )

        logging.info(f"可視化完了 - 参照文: {ref_text}")
        logging.info(f"可視化完了 - 予測文: {hyp_text}")

        return True

    except Exception as e:
        logging.error(f"Attention可視化でエラー: {e}")
        return False


def visualize_ctc_alignment(
    model,
    batch_idx,
    feature,
    spatial_feature,
    tokens,
    feature_pad_mask,
    input_lengths,
    target_lengths,
    reference_text,
    hypothesis_text,
    output_dir,
    sequence_logits=None,
):
    """
    CTC Alignment Pathの可視化処理を実行

    Args:
        model: 学習済みモデル
        batch_idx: バッチインデックス
        feature: 特徴量
        spatial_feature: 空間特徴量
        tokens: トークン
        feature_pad_mask: パディングマスク
        input_lengths: 入力長
        target_lengths: ターゲット長
        reference_text: 参照文のリスト
        hypothesis_text: 予測文のリスト
        output_dir: 出力ディレクトリ

    Returns:
        bool: 可視化が成功したかどうか
    """
    try:
        # tokensを元のtensor形式に準備
        original_tokens = tokens
        if isinstance(tokens, list):
            original_tokens = torch.tensor(tokens, device=feature.device)

        # CTCログ確率を取得
        ctc_log_probs = None
        try:
            if sequence_logits is not None:
                # 既に取得済みのsequence_logitsを使用
                ctc_log_probs = sequence_logits.log_softmax(-1)  # (T, B, C)
                logging.info(f"CTC log_probsを取得 (既存): {ctc_log_probs.shape}")
                # NaNや無限大値をチェック
                if torch.isnan(ctc_log_probs).any() or torch.isinf(ctc_log_probs).any():
                    logging.warning("CTC log_probsに無効な値が含まれています")
                    ctc_log_probs = None
            else:
                # sequence_logitsが渡されていない場合は再計算
                logging.info("sequence_logitsが利用できないため、CTC可視化のため再計算を実行します")
                with torch.no_grad():
                    forward_result = model.forward(
                        src_feature=feature,
                        spatial_feature=spatial_feature,
                        tgt_feature=original_tokens,
                        src_causal_mask=None,
                        src_padding_mask=feature_pad_mask,
                        input_lengths=input_lengths,
                        target_lengths=target_lengths,
                        mode="test",  # testモードでsequence_logitsを取得
                        blank_id=0,
                    )
                    # 戻り値の数に応じて処理を分岐
                    if len(forward_result) == 3:
                        _, _, sequence_logits_tmp = forward_result
                        if sequence_logits_tmp is not None:
                            ctc_log_probs = sequence_logits_tmp.log_softmax(-1)  # (T, B, C)
                            logging.info(f"CTC log_probsを取得 (再計算): {ctc_log_probs.shape}")
                    else:
                        logging.warning("再計算でもsequence_logitsが取得できませんでした")
        except Exception as e:
            logging.error(f"CTC log_probs取得中にエラー: {e}")
            ctc_log_probs = None

        if ctc_log_probs is None:
            logging.warning("CTC log_probsが取得できませんでした")
            return False

        # 予測結果の評価
        ref_text = reference_text[0]
        hyp_text = hypothesis_text[0]
        is_correct = ref_text == hyp_text

        # ファイル名の設定
        sample_name = f"batch_{batch_idx}_sample_0"
        status = "correct" if is_correct else "incorrect"

        logging.info(f"CTC可視化開始: {sample_name} ({status})")

        # CTC可視化用のディレクトリを作成
        ctc_output_dir = os.path.join(output_dir, f"ctc_{sample_name}_{status}")
        os.makedirs(ctc_output_dir, exist_ok=True)

        # デコード結果を整理（可視化用）
        # hypothesis_textから予測されたトークンを抽出
        decoded_tokens = []
        if hyp_text:
            # 予測文を単語に分割してトークンIDに変換
            pred_words = hyp_text.split()
            # middle_dataset_relation_dictの逆引き
            reverse_dict = {
                v: k
                for k, v in middle_dataset_relation.middle_dataset_relation_dict.items()
            }
            decoded_tokens = [
                reverse_dict.get(word, 0) for word in pred_words if word in reverse_dict
            ]

        target_tokens = []
        if isinstance(tokens, list) and len(tokens) > 0:
            target_tokens = tokens[0] if isinstance(tokens[0], list) else tokens
        elif hasattr(tokens, "tolist"):
            # tokensがtensorの場合の処理
            tokens_list = tokens.tolist()
            target_tokens = tokens_list[0] if len(tokens_list) > 0 else []

        # CTC可視化を実行
        success = visualize_ctc_alignment_path(
            log_probs=ctc_log_probs,
            decoded_sequence=decoded_tokens,
            target_sequence=target_tokens,
            vocab_dict=middle_dataset_relation.middle_dataset_relation_dict,
            blank_id=0,
            sample_idx=0,
            save_dir=ctc_output_dir,
        )

        if success:
            logging.info(f"CTC可視化完了 - 参照文: {ref_text}")
            logging.info(f"CTC可視化完了 - 予測文: {hyp_text}")
            logging.info(f"CTC出力ディレクトリ: {ctc_output_dir}")

        return success

    except Exception as e:
        logging.error(f"CTC可視化でエラー: {e}")
        return False


def generate_confusion_matrix_analysis(
    prediction_labels, ground_truth_labels, save_dir=None
):
    """
    混同行列分析を実行する独立関数

    Args:
        prediction_labels: 予測ラベルのリスト
        ground_truth_labels: 正解ラベルのリスト
        save_dir: 保存ディレクトリ (None の場合は config.plot_save_dir を使用)

    Returns:
        bool: 成功したかどうか
    """
    try:
        if len(prediction_labels) == 0 or len(ground_truth_labels) == 0:
            logging.warning("混同行列生成用のデータが不足しています")
            return False

        if len(prediction_labels) != len(ground_truth_labels):
            logging.warning(
                f"予測ラベル数({len(prediction_labels)})と正解ラベル数({len(ground_truth_labels)})が一致しません"
            )
            return False

        logging.info(f"混同行列生成開始 - 総サンプル数: {len(prediction_labels)}")

        # 語彙辞書を作成（単語名をそのまま使用）
        unique_words = sorted(set(ground_truth_labels + prediction_labels))
        vocab_dict = {word: word for word in unique_words}

        # 保存パスを決定
        if save_dir is None:
            save_dir = config.plot_save_dir
        save_path = os.path.join(save_dir, "word_level_confusion_matrix.png")

        # 混同行列を生成
        from CNN_BiLSTM.continuous_sign_language.plots import (
            analyze_word_level_confusion,
        )

        success = analyze_word_level_confusion(
            predictions=prediction_labels,
            ground_truth=ground_truth_labels,
            vocab_dict=vocab_dict,
            save_path=save_path,
        )

        if success:
            logging.info("混同行列の生成が完了しました")
            logging.info(f"保存先: {save_path}")
        else:
            logging.warning("混同行列の生成に失敗しました")

        return success

    except Exception as e:
        logging.error(f"混同行列分析でエラー: {e}")
        return False


def collect_prediction_labels(reference_text, hypothesis_text):
    """
    予測結果から単語レベルのラベルを収集する関数

    Args:
        reference_text: 正解テキスト
        hypothesis_text: 予測テキスト

    Returns:
        tuple: (ground_truth_words, prediction_words)
    """
    try:
        # 単語レベルでの予測と正解を収集
        ref_words = reference_text.split()
        pred_words = hypothesis_text.split()

        # 単語ごとにラベルを収集（長さが異なる場合は短い方に合わせる）
        min_len = min(len(ref_words), len(pred_words))

        ground_truth_words = []
        prediction_words = []

        for i in range(min_len):
            ground_truth_words.append(ref_words[i])
            prediction_words.append(pred_words[i])

        return ground_truth_words, prediction_words

    except Exception as e:
        logging.error(f"ラベル収集でエラー: {e}")
        return [], []


def process_attention_visualization(
    model,
    batch_idx,
    feature,
    spatial_feature,
    tokens,
    feature_pad_mask,
    input_lengths,
    target_lengths,
    reference_text,
    hypothesis_text,
    output_dir,
    max_samples=10,
):
    """
    Attention可視化処理を実行する独立関数

    Args:
        model: モデルインスタンス
        batch_idx: バッチインデックス
        feature: 入力特徴量
        spatial_feature: 空間特徴量
        tokens: トークン
        feature_pad_mask: パディングマスク
        input_lengths: 入力長
        target_lengths: ターゲット長
        reference_text: 正解テキスト
        hypothesis_text: 予測テキスト
        output_dir: 出力ディレクトリ
        max_samples: 最大可視化サンプル数

    Returns:
        tuple: (success_attention, success_ctc)
    """
    try:
        success_attention = visualize_attention_weights(
            model=model,
            batch_idx=batch_idx,
            reference_text=reference_text,
            hypothesis_text=hypothesis_text,
            output_dir=output_dir,
        )

        # CTC Alignment Path可視化
        success_ctc = visualize_ctc_alignment(
            model=model,
            batch_idx=batch_idx,
            feature=feature,
            spatial_feature=spatial_feature,
            tokens=tokens,
            feature_pad_mask=feature_pad_mask,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            reference_text=reference_text,
            hypothesis_text=hypothesis_text,
            output_dir=output_dir,
        )

        return success_attention, success_ctc

    except Exception as e:
        logging.error(f"Attention可視化処理でエラー: {e}")
        return False, False


def setup_visualization_environment(visualize_attention, max_visualize_samples=10):
    """
    可視化環境をセットアップする関数

    Args:
        visualize_attention: 可視化を有効にするかどうか
        max_visualize_samples: 最大可視化サンプル数

    Returns:
        tuple: (output_dir, visualize_count) または (None, 0)
    """
    if visualize_attention:
        output_dir = os.path.join(config.plot_save_dir, "attention_test")
        os.makedirs(output_dir, exist_ok=True)
        visualize_count = 0
        logging.info("Attention可視化を有効化")
        logging.info(f"出力ディレクトリ: {output_dir}")
        logging.info(f"最大可視化サンプル数: {max_visualize_samples}")
        return output_dir, visualize_count
    else:
        return None, 0


def finalize_visualization(
    model, visualize_attention, visualize_count, max_visualize_samples, output_dir
):
    """
    可視化処理の後処理を行う関数

    Args:
        model: モデルインスタンス
        visualize_attention: 可視化が有効だったかどうか
        visualize_count: 実際に可視化したサンプル数
        max_visualize_samples: 最大可視化サンプル数
        output_dir: 出力ディレクトリ
    """
    if visualize_attention:
        model.disable_attention_visualization()
        logging.info("可視化処理完了")
        logging.info(f"可視化サンプル数: {visualize_count}/{max_visualize_samples}")
        logging.info(f"出力ディレクトリ: {output_dir}")
        logging.info("  - Attention重み可視化")
        logging.info("  - CTC Alignment Path可視化")


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
        for ref, hyp in zip(reference_text_list, hypothesis_text_list):
            ref_label = ref  # Get the first token as label

            if ref_label not in label_wer:
                label_wer[ref_label] = {"refs": [], "hyps": []}
            label_wer[ref_label]["refs"].append(ref)
            label_wer[ref_label]["hyps"].append(hyp)

        # Calculate and log WER for each label
        logging.info("WER per label:")
        for label in label_wer:
            label_refs = label_wer[label]["refs"]
            label_hyps = label_wer[label]["hyps"]
            label_wer_score = wer(label_refs, label_hyps)
            logging.info(
                f"Label {label}: {label_wer_score:.10f} ({len(label_refs)} samples)"
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

        return {
            "awer": awer,
            "cer": error_rate_cer,
            "mer": error_rate_mer,
            "label_wer": label_wer,
        }

    except Exception as e:
        logging.error(f"WER計算でエラー: {e}")
        return None


def process_confidence_visualization(
    log_probs, predictions, batch_idx, output_dir, vocab_dict=None
):
    """
    予測信頼度可視化処理を実行する独立関数

    Args:
        log_probs: CTC出力の対数確率
        predictions: 予測結果
        batch_idx: バッチインデックス
        output_dir: 出力ディレクトリ
        vocab_dict: 語彙辞書

    Returns:
        tuple: (success_confidence, success_word_confidence)
    """
    try:
        from CNN_BiLSTM.continuous_sign_language.plots import (
            plot_prediction_confidence_over_time,
            plot_word_level_confidence_timeline,
        )

        success_confidence = False
        success_word_confidence = False

        # 時系列信頼度可視化
        if log_probs is not None:
            confidence_path = os.path.join(
                output_dir, f"confidence_timeline_batch_{batch_idx}.png"
            )
            success_confidence = plot_prediction_confidence_over_time(
                log_probs=log_probs,
                vocab_dict=vocab_dict,
                save_path=confidence_path,
                sample_idx=0,
            )
            if success_confidence:
                logging.info(f"時系列信頼度可視化完了: {confidence_path}")
            else:
                logging.warning("時系列信頼度可視化に失敗しました")
        else:
            logging.warning("log_probsがNullのため、時系列信頼度可視化をスキップします")

        # 単語レベル信頼度可視化
        if log_probs is not None and predictions and len(predictions) > 0:
            word_confidence_path = os.path.join(
                output_dir, f"word_confidence_batch_{batch_idx}.png"
            )
            success_word_confidence = plot_word_level_confidence_timeline(
                predictions=predictions,
                log_probs=log_probs,
                vocab_dict=vocab_dict,
                save_path=word_confidence_path,
                sample_idx=0,
            )
            if success_word_confidence:
                logging.info(f"単語レベル信頼度可視化完了: {word_confidence_path}")
            else:
                logging.warning("単語レベル信頼度可視化に失敗しました")
        else:
            if log_probs is None:
                logging.warning("log_probsがNullのため、単語レベル信頼度可視化をスキップします")
            if not predictions or len(predictions) == 0:
                logging.warning("予測結果が空のため、単語レベル信頼度可視化をスキップします")

        return success_confidence, success_word_confidence

    except Exception as e:
        logging.error(f"信頼度可視化処理でエラー: {e}")
        import traceback
        logging.error(f"詳細なエラー情報: {traceback.format_exc()}")
        return False, False


def process_multilayer_feature_visualization(
    model,
    feature,
    spatial_feature,
    tokens,
    feature_pad_mask,
    input_lengths,
    target_lengths,
    pred,
    batch_idx,
    output_dir,
    vocab_dict=None,
    method="both"
):
    """
    手話認識の多層特徴量可視化を統合処理
    
    CNN出力→空間的パターン、BiLSTM隠れ状態→時系列ダイナミクス、
    Attention重み→重要度マップ、最終層直前→統合的判断
    の各層特徴量を抽出・可視化・分析する包括的な処理
    
    Args:
        model: CNNBiLSTMモデル
        feature: 入力特徴量
        spatial_feature: 空間特徴量
        tokens: ターゲットトークン
        feature_pad_mask: パディングマスク
        input_lengths: 入力長
        target_lengths: ターゲット長
        pred: 予測結果
        batch_idx: バッチインデックス
        output_dir: 出力ディレクトリ
        vocab_dict: 語彙辞書
        method: 可視化手法 ('tsne', 'umap', 'both')
    
    Returns:
        bool: 処理成功フラグ
    """
    try:
        logging.info(f"多層特徴量可視化開始 - バッチ {batch_idx}")
        
        # 多層特徴量を抽出
        features_dict = extract_multilayer_features(
            model=model,
            feature=feature,
            spatial_feature=spatial_feature,
            tokens=tokens,
            feature_pad_mask=feature_pad_mask,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            blank_id=0,
        )
        
        if not features_dict or len(features_dict) <= 1:  # メタデータのみの場合
            logging.warning("有効な特徴量が抽出されませんでした")
            return False
        
        # ラベルを準備（予測結果から）
        labels = None
        if pred and len(pred) > 0 and len(pred[0]) > 0:
            # 予測結果から最初のサンプルのラベルを作成
            sample_predictions = pred[0] if isinstance(pred[0], list) else pred
            if sample_predictions:
                # 予測された単語IDをラベルとして使用
                labels = np.array([item[0] if isinstance(item, tuple) else item for item in sample_predictions[:10]])  # 最初の10単語
                # バッチサイズに合わせて拡張
                batch_size = list(features_dict.values())[0]['features'].shape[0] if features_dict else 1
                if len(labels) < batch_size:
                    # 不足分は最後の値で埋める
                    labels = np.pad(labels, (0, batch_size - len(labels)), mode='edge')
                elif len(labels) > batch_size:
                    # 超過分を切り詰める
                    labels = labels[:batch_size]
            else:
                logging.warning("予測結果が空のため、ラベルなしで可視化します")
        else:
            logging.warning("予測結果が利用できないため、ラベルなしで可視化します")
        
        # 出力ディレクトリを設定
        multilayer_output_dir = os.path.join(output_dir, f"multilayer_features_batch_{batch_idx}")
        os.makedirs(multilayer_output_dir, exist_ok=True)
        
        # 多層特徴量可視化を実行
        success = plot_multilayer_feature_visualization(
            features_dict=features_dict,
            labels=labels,
            vocab_dict=vocab_dict,
            save_dir=multilayer_output_dir,
            sample_idx=0,
            method=method
        )
        
        if success:
            logging.info(f"多層特徴量可視化完了: {multilayer_output_dir}")
            
            # 特徴量分離度分析も実行
            if labels is not None:
                separation_results = analyze_feature_separation(
                    features_dict=features_dict,
                    labels=labels,
                    vocab_dict=vocab_dict,
                    save_dir=multilayer_output_dir,
                    sample_idx=0
                )
                
                if separation_results:
                    logging.info("特徴量分離度分析も完了しました")
        else:
            logging.warning("多層特徴量可視化に失敗しました")
        
        return success
        
    except Exception as e:
        logging.error(f"多層特徴量可視化統合処理でエラー: {e}")
        import traceback
        logging.error(f"詳細なエラー情報: {traceback.format_exc()}")
        return False
