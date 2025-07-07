"""
連続手話認識システム用 視覚化ユーティリティ

このモジュールは以下の視覚化機能を提供します:
- Attention重みの可視化
- CTC Alignmentの可視化
- 信頼度（Confidence）の可視化
- 混同行列（Confusion Matrix）の可視化
- 多層特徴量の可視化

他のプロジェクトからも利用可能な汎用的な実装を提供します。
"""

import os
import logging
import torch
import numpy as np
from typing import List, Optional, Dict, Any, Tuple


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
            logging.warning(f"バッチ {batch_idx}: Attention重みが取得できませんでした")
            return False

        if len(attention_weights) == 0:
            logging.warning(f"バッチ {batch_idx}: Attention重みが空です")
            return False

        # plots.pyから可視化関数をインポート（動的インポート）
        try:
            # 相対パスでplots.pyを見つける
            current_dir = os.path.dirname(os.path.abspath(__file__))
            cnn_bilstm_path = os.path.join(
                current_dir, "..", "CNN_BiLSTM", "continuous_sign_language"
            )

            import sys

            if cnn_bilstm_path not in sys.path:
                sys.path.insert(0, cnn_bilstm_path)

            from plots import (
                plot_attention_matrix,
                plot_attention_focus,
                plot_attention_stats,
            )
        except ImportError as e:
            logging.error(f"plots.pyからの可視化関数のインポートに失敗: {e}")
            return False

        os.makedirs(output_dir, exist_ok=True)

        success_count = 0
        for i, (ref, hyp) in enumerate(zip(reference_text, hypothesis_text)):
            if i < len(attention_weights):
                try:
                    # 正解/不正解の判定
                    is_correct = ref.strip() == hyp.strip()
                    status = "correct" if is_correct else "incorrect"

                    # Attention Matrix可視化
                    matrix_path = os.path.join(
                        output_dir,
                        f"attention_matrix_batch_{batch_idx}_sample_{i}_{status}.png",
                    )
                    plot_attention_matrix(attention_weights[i], ref, hyp, matrix_path)

                    # Attention Focus可視化
                    focus_path = os.path.join(
                        output_dir,
                        f"attention_focus_batch_{batch_idx}_sample_{i}_{status}.png",
                    )
                    plot_attention_focus(attention_weights[i], ref, hyp, focus_path)

                    # Attention統計可視化
                    stats_path = os.path.join(
                        output_dir,
                        f"attention_stats_batch_{batch_idx}_sample_{i}_{status}.png",
                    )
                    plot_attention_stats(attention_weights[i], ref, hyp, stats_path)

                    success_count += 1
                    logging.info(
                        f"バッチ {batch_idx} サンプル {i}: Attention可視化完了 ({status})"
                    )

                except Exception as e:
                    logging.error(
                        f"バッチ {batch_idx} サンプル {i}: Attention可視化エラー: {e}"
                    )

        logging.info(
            f"バッチ {batch_idx}: {success_count}/{len(reference_text)} のAttention可視化が完了"
        )
        return success_count > 0

    except Exception as e:
        logging.error(f"バッチ {batch_idx}: Attention可視化で予期しないエラー: {e}")
        return False


def visualize_ctc_alignment(
    sequence_logits, reference_text, hypothesis_text, batch_idx, output_dir, vocab
):
    """
    CTC Alignmentの可視化処理を実行

    Args:
        sequence_logits: CTCの予測logits
        reference_text: 参照文のリスト
        hypothesis_text: 予測文のリスト
        batch_idx: バッチインデックス
        output_dir: 出力ディレクトリ
        vocab: 語彙辞書

    Returns:
        bool: 可視化が成功したかどうか
    """
    try:
        if sequence_logits is None:
            logging.warning(f"バッチ {batch_idx}: CTClogitsが取得できませんでした")
            return False

        # plots.pyから可視化関数をインポート
        try:
            # 相対パスでplots.pyを見つける
            current_dir = os.path.dirname(os.path.abspath(__file__))
            cnn_bilstm_path = os.path.join(
                current_dir, "..", "CNN_BiLSTM", "continuous_sign_language"
            )

            import sys

            if cnn_bilstm_path not in sys.path:
                sys.path.insert(0, cnn_bilstm_path)

            from plots import plot_ctc_heatmap, plot_ctc_prob_time, plot_ctc_statistics
        except ImportError as e:
            logging.error(f"plots.pyからのCTC可視化関数のインポートに失敗: {e}")
            return False

        os.makedirs(output_dir, exist_ok=True)

        success_count = 0
        for i, (ref, hyp) in enumerate(zip(reference_text, hypothesis_text)):
            if i < len(sequence_logits):
                try:
                    # 正解/不正解の判定
                    is_correct = ref.strip() == hyp.strip()
                    status = "correct" if is_correct else "incorrect"

                    # サンプル別ディレクトリ作成
                    sample_dir = os.path.join(
                        output_dir, f"ctc_batch_{batch_idx}_sample_{i}_{status}"
                    )
                    os.makedirs(sample_dir, exist_ok=True)

                    # CTC確率のソフトマックス計算
                    log_probs = torch.nn.functional.log_softmax(
                        sequence_logits[i], dim=-1
                    )
                    probs = torch.exp(log_probs)

                    # CTC Heatmap可視化
                    heatmap_path = os.path.join(
                        sample_dir, f"ctc_heatmap_sample_{i}.png"
                    )
                    plot_ctc_heatmap(probs, ref, hyp, heatmap_path, vocab)

                    # CTC確率時系列可視化
                    prob_time_path = os.path.join(
                        sample_dir, f"ctc_prob_time_sample_{i}.png"
                    )
                    plot_ctc_prob_time(probs, ref, hyp, prob_time_path, vocab)

                    # CTC統計可視化
                    stats_path = os.path.join(
                        sample_dir, f"ctc_statistics_sample_{i}.png"
                    )
                    plot_ctc_statistics(probs, ref, hyp, stats_path, vocab)

                    success_count += 1
                    logging.info(
                        f"バッチ {batch_idx} サンプル {i}: CTC可視化完了 ({status})"
                    )

                except Exception as e:
                    logging.error(
                        f"バッチ {batch_idx} サンプル {i}: CTC可視化エラー: {e}"
                    )

        logging.info(
            f"バッチ {batch_idx}: {success_count}/{len(reference_text)} のCTC可視化が完了"
        )
        return success_count > 0

    except Exception as e:
        logging.error(f"バッチ {batch_idx}: CTC可視化で予期しないエラー: {e}")
        return False


def process_attention_visualization(
    model,
    reference_text: List[str],
    hypothesis_text: List[str],
    batch_idx: int,
    config: Dict[str, Any],
    output_base_dir: str = "outputs",
) -> bool:
    """
    Attention可視化の統合処理
    """
    try:
        if not config.get("enabled", False):
            return True

        sample_rate = config.get("sample_rate", 1.0)
        save_path = config.get("save_path", "attention")

        # サンプリング判定
        if np.random.random() > sample_rate:
            logging.debug(
                f"バッチ {batch_idx}: サンプリングによりAttention可視化をスキップ"
            )
            return True

        output_dir = os.path.join(output_base_dir, save_path)

        return visualize_attention_weights(
            model, batch_idx, reference_text, hypothesis_text, output_dir
        )

    except Exception as e:
        logging.error(f"Attention可視化統合処理エラー: {e}")
        return False


def process_ctc_visualization(
    sequence_logits,
    reference_text: List[str],
    hypothesis_text: List[str],
    batch_idx: int,
    config: Dict[str, Any],
    vocab: Dict,
    output_base_dir: str = "outputs",
) -> bool:
    """
    CTC可視化の統合処理
    """
    try:
        if not config.get("enabled", False):
            return True

        sample_rate = config.get("sample_rate", 1.0)
        save_path = config.get("save_path", "ctc")

        # サンプリング判定
        if np.random.random() > sample_rate:
            logging.debug(f"バッチ {batch_idx}: サンプリングによりCTC可視化をスキップ")
            return True

        output_dir = os.path.join(output_base_dir, save_path)

        return visualize_ctc_alignment(
            sequence_logits,
            reference_text,
            hypothesis_text,
            batch_idx,
            output_dir,
            vocab,
        )

    except Exception as e:
        logging.error(f"CTC可視化統合処理エラー: {e}")
        return False


def process_confidence_visualization(
    log_probs,
    reference_text: List[str],
    hypothesis_text: List[str],
    batch_idx: int,
    config: Dict[str, Any],
    vocab: Optional[Dict] = None,
    output_base_dir: str = "outputs",
) -> bool:
    """
    信頼度可視化の統合処理
    """
    try:
        if not config.get("enabled", False):
            return True

        if log_probs is None:
            logging.warning("log_probsがNoneのため、信頼度可視化をスキップします")
            return False

        sample_rate = config.get("sample_rate", 1.0)
        save_path = config.get("save_path", "confidence")
        threshold = config.get("threshold", 0.5)
        show_entropy = config.get("show_entropy", True)

        # サンプリング判定
        if np.random.random() > sample_rate:
            logging.debug(
                f"バッチ {batch_idx}: サンプリングにより信頼度可視化をスキップ"
            )
            return True

        # plots.pyから可視化関数をインポート
        try:
            # 相対パスでplots.pyを見つける
            current_dir = os.path.dirname(os.path.abspath(__file__))
            cnn_bilstm_path = os.path.join(
                current_dir, "..", "CNN_BiLSTM", "continuous_sign_language"
            )

            import sys

            if cnn_bilstm_path not in sys.path:
                sys.path.insert(0, cnn_bilstm_path)

            from plots import plot_confidence_timeline, plot_word_confidence
        except ImportError as e:
            logging.error(f"plots.pyからの信頼度可視化関数のインポートに失敗: {e}")
            return False

        output_dir = os.path.join(output_base_dir, save_path)
        os.makedirs(output_dir, exist_ok=True)

        # 信頼度時系列可視化
        confidence_timeline_path = os.path.join(
            output_dir, f"confidence_timeline_batch_{batch_idx}.png"
        )
        plot_confidence_timeline(
            log_probs,
            reference_text,
            hypothesis_text,
            confidence_timeline_path,
            threshold,
            show_entropy,
        )

        # 単語別信頼度可視化
        word_confidence_path = os.path.join(
            output_dir, f"word_confidence_batch_{batch_idx}.png"
        )
        plot_word_confidence(
            log_probs,
            reference_text,
            hypothesis_text,
            word_confidence_path,
            threshold,
            vocab,
        )

        logging.info(f"バッチ {batch_idx}: 信頼度可視化完了")
        return True

    except Exception as e:
        logging.error(f"信頼度可視化統合処理エラー: {e}")
        return False


def process_multilayer_feature_visualization(
    model, data_loader, config: Dict[str, Any], output_base_dir: str = "outputs"
) -> bool:
    """
    多層特徴量可視化の統合処理
    """
    try:
        if not config.get("enabled", False):
            return True

        save_path = config.get("save_path", "multilayer_features")
        method = config.get("method", "umap")
        layers = config.get("layers", ["cnn", "bilstm", "attention", "final"])
        perplexity = config.get("perplexity", 30)
        n_neighbors = config.get("n_neighbors", 15)

        # plots.pyから可視化関数をインポート
        try:
            # 相対パスでplots.pyを見つける
            current_dir = os.path.dirname(os.path.abspath(__file__))
            cnn_bilstm_path = os.path.join(
                current_dir, "..", "CNN_BiLSTM", "continuous_sign_language"
            )

            import sys

            if cnn_bilstm_path not in sys.path:
                sys.path.insert(0, cnn_bilstm_path)

            from plots import (
                extract_multilayer_features,
                plot_multilayer_feature_visualization,
                analyze_feature_separation,
            )
        except ImportError as e:
            logging.error(f"plots.pyからの特徴量可視化関数のインポートに失敗: {e}")
            return False

        output_dir = os.path.join(output_base_dir, save_path)

        # 最初のバッチのみ処理（計算量削減）
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= 10:  # 最大10バッチまで
                break

            batch_dir = os.path.join(
                output_dir, f"multilayer_features_batch_{batch_idx}"
            )
            os.makedirs(batch_dir, exist_ok=True)

            # 多層特徴量抽出
            features_dict = extract_multilayer_features(
                model, batch, layers, max_samples=1  # サンプル数制限
            )

            if not features_dict:
                logging.warning(f"バッチ {batch_idx}: 特徴量抽出に失敗")
                continue

            # 各サンプルの可視化
            for sample_idx in range(min(1, len(list(features_dict.values())[0]))):
                sample_features = {
                    layer: features[sample_idx : sample_idx + 1]
                    for layer, features in features_dict.items()
                }

                # 特徴量可視化
                plot_multilayer_feature_visualization(
                    sample_features,
                    method=method,
                    save_path=os.path.join(
                        batch_dir,
                        f"multilayer_features_{method}_sample_{sample_idx}.png",
                    ),
                    perplexity=perplexity,
                    n_neighbors=n_neighbors,
                )

                # 分離度分析
                analyze_feature_separation(
                    sample_features,
                    save_path=os.path.join(
                        batch_dir, f"feature_separation_sample_{sample_idx}.png"
                    ),
                )

                logging.info(
                    f"バッチ {batch_idx} サンプル {sample_idx}: 多層特徴量可視化完了"
                )

        return True

    except Exception as e:
        logging.error(f"多層特徴量可視化統合処理エラー: {e}")
        return False


def process_confusion_matrix_visualization(
    all_predictions: List[str],
    all_references: List[str],
    config: Dict[str, Any],
    output_base_dir: str = "outputs",
) -> bool:
    """
    混同行列可視化の統合処理
    """
    try:
        if not config.get("enabled", False):
            return True

        save_path = config.get("save_path", "confusion_matrix")
        normalize = config.get("normalize", "true")
        show_metrics = config.get("show_metrics", True)

        # plots.pyから可視化関数をインポート
        try:
            # 相対パスでplots.pyを見つける
            current_dir = os.path.dirname(os.path.abspath(__file__))
            cnn_bilstm_path = os.path.join(
                current_dir, "..", "CNN_BiLSTM", "continuous_sign_language"
            )

            import sys

            if cnn_bilstm_path not in sys.path:
                sys.path.insert(0, cnn_bilstm_path)

            from plots import plot_confusion_matrix_analysis
        except ImportError as e:
            logging.error(f"plots.pyからの混同行列可視化関数のインポートに失敗: {e}")
            return False

        output_dir = os.path.join(output_base_dir, save_path)
        os.makedirs(output_dir, exist_ok=True)

        # 混同行列可視化
        confusion_path = os.path.join(output_dir, "word_level_confusion_matrix.png")
        plot_confusion_matrix_analysis(
            all_predictions, all_references, confusion_path, normalize, show_metrics
        )

        logging.info("混同行列可視化完了")
        return True

    except Exception as e:
        logging.error(f"混同行列可視化統合処理エラー: {e}")
        return False
