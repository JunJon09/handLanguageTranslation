"""
多層特徴量可視化機能

このモジュールはCNN、BiLSTM、Attentionなど各層の特徴量可視化と分析を提供します。
"""

import os
import logging
import numpy as np
from typing import List, Optional, Dict, Any


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
    method="both",
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
                extract_multilayer_features,
                plot_multilayer_feature_visualization,
                analyze_feature_separation,
            )
        except ImportError as e:
            logging.error(f"plots.pyからの特徴量可視化関数のインポートに失敗: {e}")
            return False

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
                labels = np.array(
                    [
                        item[0] if isinstance(item, tuple) else item
                        for item in sample_predictions[:10]
                    ]
                )  # 最初の10単語
                # バッチサイズに合わせて拡張
                batch_size = (
                    list(features_dict.values())[0]["features"].shape[0]
                    if features_dict
                    else 1
                )
                if len(labels) < batch_size:
                    # 不足分は最後の値で埋める
                    labels = np.pad(labels, (0, batch_size - len(labels)), mode="edge")
                elif len(labels) > batch_size:
                    # 超過分を切り詰める
                    labels = labels[:batch_size]
            else:
                logging.warning("予測結果が空のため、ラベルなしで可視化します")
        else:
            logging.warning("予測結果が利用できないため、ラベルなしで可視化します")

        # 出力ディレクトリを設定
        multilayer_output_dir = os.path.join(
            output_dir, f"multilayer_features_batch_{batch_idx}"
        )
        os.makedirs(multilayer_output_dir, exist_ok=True)

        # 多層特徴量可視化を実行
        success = plot_multilayer_feature_visualization(
            features_dict=features_dict,
            labels=labels,
            vocab_dict=vocab_dict,
            save_dir=multilayer_output_dir,
            sample_idx=0,
            method=method,
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
                    sample_idx=0,
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
