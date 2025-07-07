"""
test_loop統合モジュール - テストループの可視化・分析処理を統合的に管理

このモジュールは、test_loop関数で使用される可視化・分析機能を統合的に管理し、
test_loop関数の肥大化を解消するための統合関数を提供します。

機能:
- 全可視化・分析処理の統合実行
- バッチ単位での可視化処理
- エラーハンドリング・ログ管理
- パラメータ設定の統一管理

使用例:
    # test_loop内での使用
    visualizer = TestLoopVisualizer(config)
    visualizer.process_batch(model, batch_data, batch_idx, predictions)
    visualizer.finalize()
"""

import logging
import os
import time
import torch
from typing import Dict, List, Tuple, Optional, Any, Union


# 修正されたインポート文
import utils.visualization.attention_visualization as attention_viz
import utils.visualization.confidence_visualization as confidence_viz
import utils.visualization.feature_visualization as feature_viz
import utils.visualization.confusion_matrix_visualization as confusion_viz
import utils.visualization.visualization_integration as viz_integration

# 使用する関数のエイリアス
process_attention_visualization = viz_integration.process_attention_visualization
process_confidence_visualization = confidence_viz.process_confidence_visualization
process_multilayer_feature_visualization = feature_viz.process_multilayer_feature_visualization
generate_confusion_matrix_analysis = confusion_viz.generate_confusion_matrix_analysis
collect_prediction_labels = confusion_viz.collect_prediction_labels
setup_visualization_environment = viz_integration.setup_visualization_environment
finalize_visualization = viz_integration.finalize_visualization
calculate_wer_metrics = viz_integration.calculate_wer_metrics

class TestLoopVisualizer:
    """
    テストループ用の統合可視化クラス

    test_loop関数内での可視化・分析処理を統合的に管理し、
    関数の肥大化を解消します。
    """

    def __init__(
        self,
        visualize_attention: bool = False,
        generate_confusion_matrix: bool = False,
        visualize_confidence: bool = False,
        visualize_multilayer_features: bool = False,
        multilayer_method: str = "both",
        max_visualize_samples: int = 10,
        output_base_dir: Optional[str] = None,
    ):
        """
        初期化

        Args:
            visualize_attention: Attention可視化を有効にするか
            generate_confusion_matrix: 混同行列を生成するか
            visualize_confidence: 予測信頼度可視化を有効にするか
            visualize_multilayer_features: 多層特徴量可視化を有効にするか
            multilayer_method: 多層特徴量可視化手法 ('tsne', 'umap', 'both')
            max_visualize_samples: 最大可視化サンプル数
            output_base_dir: 出力ディレクトリのベースパス
        """
        self.visualize_attention = visualize_attention
        self.generate_confusion_matrix = generate_confusion_matrix
        self.visualize_confidence = visualize_confidence
        self.visualize_multilayer_features = visualize_multilayer_features
        self.multilayer_method = multilayer_method
        self.max_visualize_samples = max_visualize_samples

        # 可視化環境の設定
        self.output_dir, self.visualize_count = setup_visualization_environment(
            visualize_attention, max_visualize_samples, output_base_dir
        )

        # 混同行列用のデータ収集
        if self.generate_confusion_matrix:
            self.prediction_labels = []
            self.ground_truth_labels = []
            logging.info("混同行列生成モードを有効化")

        # 統計情報
        self.stats = {
            "attention_success": 0,
            "ctc_success": 0,
            "confidence_success": 0,
            "word_confidence_success": 0,
            "multilayer_success": 0,
            "total_processed": 0,
        }

        logging.info(f"TestLoopVisualizer初期化完了 (出力先: {self.output_dir})")

    def setup_model_for_visualization(self, model: torch.nn.Module) -> None:
        """
        モデルを可視化用に設定

        Args:
            model: 対象モデル
        """
        if self.visualize_attention and hasattr(
            model, "enable_attention_visualization"
        ):
            model.enable_attention_visualization()
            logging.info("モデルのAttention可視化を有効化")

    def process_batch_visualization(
        self,
        model: torch.nn.Module,
        batch_data: Dict[str, Any],
        batch_idx: int,
        predictions: Dict[str, Any],
        vocab_dict: Optional[Dict[int, str]] = None,
    ) -> bool:
        """
        バッチ単位での可視化・分析処理

        Args:
            model: 学習済みモデル
            batch_data: バッチデータ（feature, tokens等を含む辞書）
            batch_idx: バッチインデックス
            predictions: 予測結果（pred, conv_pred, sequence_logits等を含む辞書）
            vocab_dict: 語彙辞書

        Returns:
            bool: 可視化処理が実行されたかどうか
        """
        if self.visualize_count >= self.max_visualize_samples:
            return False

        if not (
            self.visualize_attention
            or self.visualize_confidence
            or self.visualize_multilayer_features
        ):
            return False

        success_flags = {}
        any_success = False

        try:
            # Attention可視化処理
            if self.visualize_attention:
                success_attention, success_ctc = self._process_attention_batch(
                    model, batch_data, batch_idx, predictions
                )
                success_flags.update(
                    {"attention": success_attention, "ctc": success_ctc}
                )
                any_success = any_success or success_attention or success_ctc
                if success_attention:
                    self.stats["attention_success"] += 1
                if success_ctc:
                    self.stats["ctc_success"] += 1

            # 信頼度可視化処理
            if self.visualize_confidence:
                success_confidence, success_word_confidence = (
                    self._process_confidence_batch(
                        batch_data, batch_idx, predictions, vocab_dict
                    )
                )
                success_flags.update(
                    {
                        "confidence": success_confidence,
                        "word_confidence": success_word_confidence,
                    }
                )
                any_success = (
                    any_success or success_confidence or success_word_confidence
                )
                if success_confidence:
                    self.stats["confidence_success"] += 1
                if success_word_confidence:
                    self.stats["word_confidence_success"] += 1

            # 多層特徴量可視化処理
            if self.visualize_multilayer_features:
                success_multilayer = self._process_multilayer_batch(
                    model, batch_data, batch_idx, predictions, vocab_dict
                )
                success_flags["multilayer"] = success_multilayer
                any_success = any_success or success_multilayer
                if success_multilayer:
                    self.stats["multilayer_success"] += 1

            # 成功した場合のカウント更新
            if any_success:
                self.visualize_count += 1
                self.stats["total_processed"] += 1
                self._log_visualization_results(success_flags)

        except Exception as e:
            logging.error(f"バッチ{batch_idx}の可視化処理でエラー: {e}")
            return False

        return any_success

    def collect_confusion_matrix_data(
        self,
        reference_text: str,
        hypothesis_text: str,
    ) -> None:
        """
        混同行列用のデータを収集

        Args:
            reference_text: 正解テキスト
            hypothesis_text: 予測テキスト
        """
        if not self.generate_confusion_matrix:
            return

        try:
            ref_words, pred_words = collect_prediction_labels(
                reference_text, hypothesis_text
            )
            self.ground_truth_labels.extend(ref_words)
            self.prediction_labels.extend(pred_words)
        except Exception as e:
            logging.error(f"混同行列データ収集でエラー: {e}")

    def finalize_visualization(
        self,
        model: torch.nn.Module,
        save_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        可視化処理の後処理と統計情報の出力

        Args:
            model: 対象モデル
            save_dir: 混同行列保存ディレクトリ

        Returns:
            Dict[str, Any]: 処理結果の統計情報
        """
        results = {}

        try:
            # Attention可視化の後処理
            finalize_visualization(
                model=model,
                visualize_attention=self.visualize_attention,
                visualize_count=self.visualize_count,
                max_visualize_samples=self.max_visualize_samples,
                output_dir=self.output_dir,
            )

            # 混同行列の生成
            if self.generate_confusion_matrix and self.prediction_labels:
                confusion_success = generate_confusion_matrix_analysis(
                    prediction_labels=self.prediction_labels,
                    ground_truth_labels=self.ground_truth_labels,
                    save_dir=save_dir or self.output_dir,
                )
                results["confusion_matrix"] = confusion_success
                logging.info(f"混同行列生成: {'成功' if confusion_success else '失敗'}")

            # 統計情報の出力
            results["stats"] = self.stats.copy()
            self._log_final_statistics()

        except Exception as e:
            logging.error(f"可視化後処理でエラー: {e}")
            results["error"] = str(e)

        return results

    def finalize(
        self,
        model: torch.nn.Module,
        save_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        可視化処理の終了処理（finalize_visualizationのエイリアス）

        Args:
            model: 対象モデル
            save_dir: 混同行列保存ディレクトリ

        Returns:
            Dict[str, Any]: 処理結果の統計情報
        """
        return self.finalize_visualization(model, save_dir)

    def _process_attention_batch(
        self,
        model: torch.nn.Module,
        batch_data: Dict[str, Any],
        batch_idx: int,
        predictions: Dict[str, Any],
    ) -> Tuple[bool, bool]:
        """Attention可視化のバッチ処理"""
        try:
            return process_attention_visualization(
                model=model,
                batch_idx=batch_idx,
                feature=batch_data["feature"],
                spatial_feature=batch_data["spatial_feature"],
                tokens=batch_data["tokens"],
                feature_pad_mask=batch_data.get("feature_pad_mask"),
                input_lengths=batch_data["input_lengths"],
                target_lengths=batch_data["target_lengths"],
                reference_text=predictions["reference_text"],
                hypothesis_text=predictions["hypothesis_text"],
                output_dir=self.output_dir,
                max_samples=self.max_visualize_samples,
            )
        except Exception as e:
            logging.error(f"Attention可視化でエラー: {e}")
            return False, False

    def _process_confidence_batch(
        self,
        batch_data: Dict[str, Any],
        batch_idx: int,
        predictions: Dict[str, Any],
        vocab_dict: Optional[Dict[int, str]],
    ) -> Tuple[bool, bool]:
        """信頼度可視化のバッチ処理"""
        try:
            # log_probsを準備
            log_probs = None
            sequence_logits = predictions.get("sequence_logits")

            if sequence_logits is not None:
                log_probs = sequence_logits.log_softmax(-1)  # (T, B, C)
                logging.debug(f"信頼度可視化用log_probs形状: {log_probs.shape}")

                # NaNや無限大値をチェック
                if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
                    logging.warning("log_probsに無効な値が含まれています")
                    log_probs = None
            else:
                logging.warning("sequence_logitsが利用できません")
                return False, False

            # 予測結果を準備
            pred = predictions.get("pred", [])
            pred_for_confidence = []
            if len(pred) > 0 and len(pred[0]) > 0:
                pred_for_confidence = [
                    item[0] if isinstance(item, tuple) else item for item in pred[0]
                ]
                logging.debug(
                    f"信頼度可視化用予測データ: {len(pred_for_confidence)}個の単語"
                )
            else:
                logging.warning("予測結果が空のため、信頼度可視化をスキップします")
                return False, False

            # 信頼度可視化を実行
            if log_probs is not None:
                return process_confidence_visualization(
                    log_probs=log_probs,
                    predictions=pred_for_confidence,
                    batch_idx=batch_idx,
                    output_dir=self.output_dir,
                    vocab_dict=vocab_dict,
                )
            else:
                logging.warning(
                    "log_probsが利用できないため、信頼度可視化をスキップします"
                )
                return False, False

        except Exception as e:
            logging.error(f"信頼度可視化でエラー: {e}")
            return False, False

    def _process_multilayer_batch(
        self,
        model: torch.nn.Module,
        batch_data: Dict[str, Any],
        batch_idx: int,
        predictions: Dict[str, Any],
        vocab_dict: Optional[Dict[int, str]],
    ) -> bool:
        """多層特徴量可視化のバッチ処理"""
        try:
            return process_multilayer_feature_visualization(
                model=model,
                feature=batch_data["feature"],
                spatial_feature=batch_data["spatial_feature"],
                tokens=batch_data["tokens"],
                feature_pad_mask=batch_data.get("feature_pad_mask"),
                input_lengths=batch_data["input_lengths"],
                target_lengths=batch_data["target_lengths"],
                pred=predictions.get("pred", []),
                batch_idx=batch_idx,
                output_dir=self.output_dir,
                vocab_dict=vocab_dict,
                method=self.multilayer_method,
            )
        except Exception as e:
            logging.error(f"多層特徴量可視化でエラー: {e}")
            return False

    def _log_visualization_results(self, success_flags: Dict[str, bool]) -> None:
        """可視化結果のログ出力"""
        results = []
        if "attention" in success_flags:
            results.append(
                f"Attention: {'成功' if success_flags['attention'] else '失敗'}"
            )
        if "ctc" in success_flags:
            results.append(f"CTC: {'成功' if success_flags['ctc'] else '失敗'}")
        if "confidence" in success_flags:
            results.append(
                f"信頼度: {'成功' if success_flags['confidence'] else '失敗'}"
            )
        if "word_confidence" in success_flags:
            results.append(
                f"単語信頼度: {'成功' if success_flags['word_confidence'] else '失敗'}"
            )
        if "multilayer" in success_flags:
            results.append(
                f"多層特徴量: {'成功' if success_flags['multilayer'] else '失敗'}"
            )

        if results:
            logging.info(f"可視化完了 ({', '.join(results)})")

    def _log_final_statistics(self) -> None:
        """最終統計情報のログ出力"""
        logging.info("=== 可視化処理統計 ===")
        logging.info(f"処理バッチ数: {self.stats['total_processed']}")
        if self.visualize_attention:
            logging.info(f"Attention成功: {self.stats['attention_success']}")
            logging.info(f"CTC成功: {self.stats['ctc_success']}")
        if self.visualize_confidence:
            logging.info(f"信頼度成功: {self.stats['confidence_success']}")
            logging.info(f"単語信頼度成功: {self.stats['word_confidence_success']}")
        if self.visualize_multilayer_features:
            logging.info(f"多層特徴量成功: {self.stats['multilayer_success']}")
        logging.info("==================")


def run_all_test_visualizations(
    model: torch.nn.Module,
    batch_data: Dict[str, Any],
    batch_idx: int,
    predictions: Dict[str, Any],
    vocab_dict: Optional[Dict[int, str]] = None,
    visualize_attention: bool = False,
    generate_confusion_matrix: bool = False,
    visualize_confidence: bool = False,
    visualize_multilayer_features: bool = False,
    multilayer_method: str = "both",
    max_visualize_samples: int = 10,
    output_dir: Optional[str] = None,
    visualizer: Optional[TestLoopVisualizer] = None,
) -> bool:
    """
    テストループ用の統合可視化関数（単発実行用）

    TestLoopVisualizerを使わずに単発で可視化処理を実行したい場合に使用

    Args:
        model: 学習済みモデル
        batch_data: バッチデータ
        batch_idx: バッチインデックス
        predictions: 予測結果
        vocab_dict: 語彙辞書
        visualize_attention: Attention可視化を有効にするか
        generate_confusion_matrix: 混同行列を生成するか
        visualize_confidence: 予測信頼度可視化を有効にするか
        visualize_multilayer_features: 多層特徴量可視化を有効にするか
        multilayer_method: 多層特徴量可視化手法
        max_visualize_samples: 最大可視化サンプル数
        output_dir: 出力ディレクトリ
        visualizer: 既存のビジュアライザーインスタンス（あれば）

    Returns:
        bool: 可視化処理が実行されたかどうか
    """
    if visualizer is None:
        visualizer = TestLoopVisualizer(
            visualize_attention=visualize_attention,
            generate_confusion_matrix=generate_confusion_matrix,
            visualize_confidence=visualize_confidence,
            visualize_multilayer_features=visualize_multilayer_features,
            multilayer_method=multilayer_method,
            max_visualize_samples=max_visualize_samples,
            output_base_dir=output_dir,
        )
        visualizer.setup_model_for_visualization(model)

    return visualizer.process_batch_visualization(
        model=model,
        batch_data=batch_data,
        batch_idx=batch_idx,
        predictions=predictions,
        vocab_dict=vocab_dict,
    )


def run_complete_test_loop_visualizations(
    model: torch.nn.Module,
    dataloader: Any,
    device: str,
    middle_dataset_relation_dict: Dict[int, str],
    visualize_attention: bool = False,
    generate_confusion_matrix: bool = False,
    visualize_confidence: bool = False,
    visualize_multilayer_features: bool = False,
    multilayer_method: str = "both",
    max_visualize_samples: int = 10,
    config_plot_save_dir: Optional[str] = None,
) -> Tuple[Dict[str, float], List[List], List[str], List[str], List[str]]:
    """
    test_loop関数の可視化・分析処理を完全に統合した関数

    この関数は、test_loop関数内で行われていた全ての可視化・分析処理を
    統合的に実行し、test_loop関数の肥大化を解消します。

    Args:
        model: 学習済みモデル
        dataloader: テストデータローダー
        device: 実行デバイス
        middle_dataset_relation_dict: 語彙辞書
        visualize_attention: Attention可視化を有効にするか
        generate_confusion_matrix: 混同行列を生成するか
        visualize_confidence: 予測信頼度可視化を有効にするか
        visualize_multilayer_features: 多層特徴量可視化を有効にするか
        multilayer_method: 多層特徴量可視化手法
        max_visualize_samples: 最大可視化サンプル数
        config_plot_save_dir: プロット保存ディレクトリ

    Returns:
        Tuple containing:
        - wer_metrics: WER評価指標の辞書
        - pred_times: 予測時間のリスト
        - reference_text_list: 参照文のリスト
        - hypothesis_text_list: 予測文のリスト
        - hypothesis_text_conv_list: Conv予測文のリスト
    """
    # 結果収集用リスト
    hypothesis_text_list = []
    hypothesis_text_conv_list = []
    reference_text_list = []
    pred_times = []

    # 可視化環境の設定
    output_dir, visualize_count = setup_visualization_environment(
        visualize_attention, max_visualize_samples
    )

    # 混同行列用のデータ収集初期化
    prediction_labels = []
    ground_truth_labels = []
    if generate_confusion_matrix:
        logging.info("混同行列生成モードを有効化")

    # Attention可視化の設定
    if visualize_attention:
        model.enable_attention_visualization()

    # メインループは test_loop 関数で実行されるため、
    # ここでは可視化・分析のサポート関数として機能

    # 統計情報の準備
    visualization_stats = {
        "attention_success": 0,
        "ctc_success": 0,
        "confidence_success": 0,
        "word_confidence_success": 0,
        "multilayer_success": 0,
        "total_processed": 0,
    }

    return {
        "output_dir": output_dir,
        "visualize_count": visualize_count,
        "prediction_labels": prediction_labels,
        "ground_truth_labels": ground_truth_labels,
        "visualization_stats": visualization_stats,
        "max_visualize_samples": max_visualize_samples,
    }


def process_batch_visualizations_integrated(
    model: torch.nn.Module,
    batch_idx: int,
    feature: torch.Tensor,
    spatial_feature: torch.Tensor,
    tokens: List,
    feature_pad_mask: Optional[torch.Tensor],
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    reference_text: List[str],
    hypothesis_text: List[str],
    pred: List,
    sequence_logits: Optional[torch.Tensor],
    middle_dataset_relation_dict: Dict[int, str],
    visualization_state: Dict[str, Any],
    visualize_attention: bool = False,
    visualize_confidence: bool = False,
    visualize_multilayer_features: bool = False,
    multilayer_method: str = "both",
) -> Dict[str, Any]:
    """
    バッチ単位での可視化処理を統合実行

    Args:
        model: 学習済みモデル
        batch_idx: バッチインデックス
        feature: 入力特徴量
        spatial_feature: 空間特徴量
        tokens: トークンリスト
        feature_pad_mask: 特徴量パディングマスク
        input_lengths: 入力長
        target_lengths: ターゲット長
        reference_text: 参照テキスト
        hypothesis_text: 予測テキスト
        pred: 予測結果
        sequence_logits: シーケンスロジット
        middle_dataset_relation_dict: 語彙辞書
        visualization_state: 可視化状態
        visualize_attention: Attention可視化フラグ
        visualize_confidence: 信頼度可視化フラグ
        visualize_multilayer_features: 多層特徴量可視化フラグ
        multilayer_method: 多層特徴量可視化手法

    Returns:
        Dict: 更新された可視化状態
    """
    output_dir = visualization_state["output_dir"]
    visualize_count = visualization_state["visualize_count"]
    max_visualize_samples = visualization_state["max_visualize_samples"]
    visualization_stats = visualization_state["visualization_stats"]

    # 可視化サンプル数制限チェック
    if visualize_count >= max_visualize_samples:
        return visualization_state

    if not (
        visualize_attention or visualize_confidence or visualize_multilayer_features
    ):
        return visualization_state

    any_success = False

    # Attention可視化処理
    success_attention, success_ctc = False, False
    if visualize_attention:
        try:
            success_attention, success_ctc = process_attention_visualization(
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
                max_samples=max_visualize_samples,
            )

            if success_attention:
                visualization_stats["attention_success"] += 1
            if success_ctc:
                visualization_stats["ctc_success"] += 1

            any_success = success_attention or success_ctc

        except Exception as e:
            logging.error(f"Attention可視化でエラー: {e}")

    # 信頼度可視化処理
    success_confidence, success_word_confidence = False, False
    if visualize_confidence:
        try:
            # log_probsを準備
            log_probs = None
            if sequence_logits is not None:
                log_probs = sequence_logits.log_softmax(-1)  # (T, B, C)
                logging.info(
                    f"信頼度可視化用log_probsを取得しました。形状: {log_probs.shape}"
                )

                # NaNや無限大値をチェック
                if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
                    logging.warning("log_probsに無効な値が含まれています")
                    log_probs = None
            else:
                logging.warning("sequence_logitsが利用できません")

            # 予測結果を準備
            pred_for_confidence = []
            if len(pred) > 0 and len(pred[0]) > 0:
                pred_for_confidence = [
                    item[0] if isinstance(item, tuple) else item for item in pred[0]
                ]
                logging.info(
                    f"信頼度可視化用予測データ: {len(pred_for_confidence)}個の単語"
                )
            else:
                logging.warning("予測結果が空のため、信頼度可視化をスキップします")

            # 信頼度可視化を実行
            if log_probs is not None:
                success_confidence, success_word_confidence = (
                    process_confidence_visualization(
                        log_probs=log_probs,
                        predictions=pred_for_confidence,
                        batch_idx=batch_idx,
                        output_dir=output_dir,
                        vocab_dict=middle_dataset_relation_dict,
                    )
                )

                if success_confidence:
                    visualization_stats["confidence_success"] += 1
                if success_word_confidence:
                    visualization_stats["word_confidence_success"] += 1

                any_success = (
                    any_success or success_confidence or success_word_confidence
                )
            else:
                logging.warning(
                    "log_probsが利用できないため、信頼度可視化をスキップします"
                )

        except Exception as e:
            logging.error(f"信頼度可視化でエラー: {e}")

    # 多層特徴量可視化処理
    success_multilayer = False
    if visualize_multilayer_features:
        try:
            success_multilayer = process_multilayer_feature_visualization(
                model=model,
                feature=feature,
                spatial_feature=spatial_feature,
                tokens=tokens,
                feature_pad_mask=feature_pad_mask,
                input_lengths=input_lengths,
                target_lengths=target_lengths,
                pred=pred,
                batch_idx=batch_idx,
                output_dir=output_dir,
                vocab_dict=middle_dataset_relation_dict,
                method=multilayer_method,
            )

            if success_multilayer:
                visualization_stats["multilayer_success"] += 1
                logging.info("多層特徴量可視化が成功しました")
                any_success = True
            else:
                logging.warning("多層特徴量可視化に失敗しました")

        except Exception as e:
            logging.error(f"多層特徴量可視化でエラー: {e}")

    # 可視化結果をログ出力
    if any_success:
        visualize_count += 1
        visualization_stats["total_processed"] += 1

        results = []
        if visualize_attention:
            results.append(f"Attention: {'成功' if success_attention else '失敗'}")
            results.append(f"CTC: {'成功' if success_ctc else '失敗'}")
        if visualize_confidence:
            results.append(f"信頼度: {'成功' if success_confidence else '失敗'}")
            results.append(
                f"単語信頼度: {'成功' if success_word_confidence else '失敗'}"
            )
        if visualize_multilayer_features:
            results.append(f"多層特徴量: {'成功' if success_multilayer else '失敗'}")

        logging.info(f"可視化完了 ({', '.join(results)})")

    # 状態を更新
    visualization_state["visualize_count"] = visualize_count
    visualization_state["visualization_stats"] = visualization_stats

    return visualization_state


def finalize_all_visualizations(
    model: torch.nn.Module,
    visualization_state: Dict[str, Any],
    reference_text_list: List[str],
    hypothesis_text_list: List[str],
    visualize_attention: bool = False,
    generate_confusion_matrix: bool = False,
    config_plot_save_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    全可視化処理の最終化

    Args:
        model: 学習済みモデル
        visualization_state: 可視化状態
        reference_text_list: 参照文リスト
        hypothesis_text_list: 予測文リスト
        visualize_attention: Attention可視化フラグ
        generate_confusion_matrix: 混同行列生成フラグ
        config_plot_save_dir: プロット保存ディレクトリ

    Returns:
        Dict: WER評価指標とその他の結果
    """
    output_dir = visualization_state["output_dir"]
    visualize_count = visualization_state["visualize_count"]
    max_visualize_samples = visualization_state["max_visualize_samples"]
    prediction_labels = visualization_state["prediction_labels"]
    ground_truth_labels = visualization_state["ground_truth_labels"]
    visualization_stats = visualization_state["visualization_stats"]

    # WER評価指標の計算
    wer_metrics = calculate_wer_metrics(reference_text_list, hypothesis_text_list)

    # Attention可視化の後処理
    finalize_visualization(
        model=model,
        visualize_attention=visualize_attention,
        visualize_count=visualize_count,
        max_visualize_samples=max_visualize_samples,
        output_dir=output_dir,
    )

    # 混同行列の生成
    confusion_matrix_success = False
    if generate_confusion_matrix and prediction_labels and ground_truth_labels:
        confusion_matrix_success = generate_confusion_matrix_analysis(
            prediction_labels=prediction_labels,
            ground_truth_labels=ground_truth_labels,
            save_dir=config_plot_save_dir,
        )

    # 統計情報をログ出力
    if visualization_stats["total_processed"] > 0:
        logging.info(f"可視化統計:")
        logging.info(f"  処理済みサンプル数: {visualization_stats['total_processed']}")
        logging.info(f"  Attention成功: {visualization_stats['attention_success']}")
        logging.info(f"  CTC成功: {visualization_stats['ctc_success']}")
        logging.info(f"  信頼度成功: {visualization_stats['confidence_success']}")
        logging.info(
            f"  単語信頼度成功: {visualization_stats['word_confidence_success']}"
        )
        logging.info(f"  多層特徴量成功: {visualization_stats['multilayer_success']}")
        if generate_confusion_matrix:
            logging.info(
                f"  混同行列生成: {'成功' if confusion_matrix_success else '失敗'}"
            )

    return {
        "wer_metrics": wer_metrics,
        "confusion_matrix_success": confusion_matrix_success,
        "visualization_stats": visualization_stats,
    }


def collect_confusion_matrix_labels(
    reference_text: str,
    hypothesis_text: str,
    prediction_labels: List[str],
    ground_truth_labels: List[str],
) -> Tuple[List[str], List[str]]:
    """
    混同行列用のラベル収集（統合版）

    Args:
        reference_text: 参照テキスト
        hypothesis_text: 予測テキスト
        prediction_labels: 予測ラベルリスト（蓄積用）
        ground_truth_labels: 正解ラベルリスト（蓄積用）

    Returns:
        Tuple: (更新された予測ラベルリスト, 更新された正解ラベルリスト)
    """
    # 単語レベルでの予測と正解を収集
    ref_words, pred_words = collect_prediction_labels(reference_text, hypothesis_text)

    # 蓄積リストに追加
    ground_truth_labels.extend(ref_words)
    prediction_labels.extend(pred_words)

    return prediction_labels, ground_truth_labels
