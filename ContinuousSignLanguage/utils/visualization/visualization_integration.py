"""
視覚化統合プロセス関数

このモジュールは各種可視化処理の統合管理機能を提供します。
"""

import os
import logging
from typing import List, Optional, Dict, Any, Tuple

from .attention_visualization import (
    visualize_attention_weights,
    visualize_ctc_alignment,
)
from .confidence_visualization import process_confidence_visualization
from .confusion_matrix_visualization import (
    generate_confusion_matrix_analysis,
    collect_prediction_labels,
)
from .feature_visualization import process_multilayer_feature_visualization


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
    Attention可視化処理を実行する統合関数

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
        # Attention重み可視化
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
        # configモジュールをインポート
        try:
            # 相対パスでconfigを見つける
            current_dir = os.path.dirname(os.path.abspath(__file__))
            cnn_bilstm_path = os.path.join(
                current_dir, "..", "..", "CNN_BiLSTM", "continuous_sign_language"
            )

            import sys

            if cnn_bilstm_path not in sys.path:
                sys.path.insert(0, cnn_bilstm_path)

            import config
        except ImportError as e:
            logging.error(f"configモジュールのインポートに失敗: {e}")
            output_dir = os.path.join("outputs", "attention_test")
        else:
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
        # jiwer.pyからインポート
        try:
            from jiwer import wer, cer, mer
        except ImportError as e:
            logging.error(f"jiwerモジュールのインポートに失敗: {e}")
            return None

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
