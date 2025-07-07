"""
信頼度（Confidence）可視化機能

このモジュールは予測信頼度の時系列可視化と単語レベル信頼度分析を提供します。
"""

import os
import logging
from typing import List, Optional, Dict, Any


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
                plot_prediction_confidence_over_time,
                plot_word_level_confidence_timeline,
            )
        except ImportError as e:
            logging.error(f"plots.pyからの信頼度可視化関数のインポートに失敗: {e}")
            return False, False

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
                logging.warning(
                    "log_probsがNullのため、単語レベル信頼度可視化をスキップします"
                )
            if not predictions or len(predictions) == 0:
                logging.warning(
                    "予測結果が空のため、単語レベル信頼度可視化をスキップします"
                )

        return success_confidence, success_word_confidence

    except Exception as e:
        logging.error(f"信頼度可視化処理でエラー: {e}")
        import traceback

        logging.error(f"詳細なエラー情報: {traceback.format_exc()}")
        return False, False
