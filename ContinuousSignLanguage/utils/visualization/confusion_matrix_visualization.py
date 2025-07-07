"""
混同行列（Confusion Matrix）可視化機能

このモジュールは単語レベルでの混同行列分析と可視化を提供します。
"""

import os
import logging
from typing import List, Optional, Dict, Any


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

            from plots import analyze_word_level_confusion
            import CNN_BiLSTM.continuous_sign_language.config as config
        except ImportError as e:
            logging.error(f"plots.pyまたはconfigモジュールのインポートに失敗: {e}")
            return False

        # 保存パスを決定
        if save_dir is None:
            save_dir = config.plot_save_dir
        save_path = os.path.join(save_dir, "word_level_confusion_matrix.png")

        # 混同行列を生成
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
