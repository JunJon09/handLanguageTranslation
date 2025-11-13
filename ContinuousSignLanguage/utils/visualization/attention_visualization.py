"""
Attention重みとCTC Alignmentの可視化機能

このモジュールはAttention機構とCTCアライメントパスの可視化を提供します。
"""

import os
import logging
import torch
import numpy as np
from typing import List, Optional, Dict, Any


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

        # plots.pyから可視化関数をインポート
        try:
            # 相対パスでplots.pyを見つける
            current_dir = os.path.dirname(os.path.abspath(__file__))
            cnn_bilstm_path = os.path.join(
                current_dir, "..", "..", "CNN_BiLSTM", "continuous_sign_language"
            )

            import sys

            if cnn_bilstm_path not in sys.path:
                sys.path.insert(0, cnn_bilstm_path)

            from plots import (
                plot_attention_matrix,
                plot_attention_statistics,
                plot_attention_focus_over_time,
            )
        except ImportError as e:
            logging.error(f"plots.pyからの可視化関数のインポートに失敗: {e}")
            return False

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
        sequence_logits: 事前計算済みのCTCロジット（オプション）

    Returns:
        bool: 可視化が成功したかどうか
    """
    try:
        # plots.pyから可視化関数をインポート
        try:
            # 相対パスでplots.pyを見つける
            current_dir = os.path.dirname(os.path.abspath(__file__))
            cnn_bilstm_path = os.path.join(
                current_dir, "..", "..", "CNN_BiLSTM", "continuous_sign_language"
            )

            import sys

            if cnn_bilstm_path not in sys.path:
                sys.path.insert(0, cnn_bilstm_path)

            from plots import visualize_ctc_alignment_path
            import CNN_BiLSTM.continuous_sign_language.modeling.middle_dataset_relation as middle_dataset_relation
        except ImportError as e:
            logging.error(f"plots.pyまたは関連モジュールのインポートに失敗: {e}")
            return False

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
                logging.info(
                    "sequence_logitsが利用できないため、CTC可視化のため再計算を実行します"
                )
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
                            ctc_log_probs = sequence_logits_tmp.log_softmax(
                                -1
                            )  # (T, B, C)
                            logging.info(
                                f"CTC log_probsを取得 (再計算): {ctc_log_probs.shape}"
                            )
                    else:
                        logging.warning(
                            "再計算でもsequence_logitsが取得できませんでした"
                        )
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
