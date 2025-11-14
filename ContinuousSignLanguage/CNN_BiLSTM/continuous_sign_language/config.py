import numpy as np
import os
from datetime import datetime

# dataset
read_dataset_dir = "../hdf5/phoenix_2014_t/"
test_number = "003"
val_number = "002"


USE_LIP_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]

USE_LIP_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]

USE_LIP_CORNERS_CENTER = [324, 318, 402, 317, 14, 87, 178, 88, 95]
# # 耳の点（MediaPipeでは限定的）
EAR_POINTS = [162, 389] # MediaPipeには明確な右耳下部はない
USE_NOSE = [1, 2]


USE_POSE = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]) + 478

USE_LHAND = np.arange(478 + 33, 478 + 33 + 21)
USE_RHAND = np.arange(478 + 33 + 21, 478 + 33 + 21 + 21)

use_features = ["x", "y"]
load_into_ram = True
batch_size = 16
spatial_spatial_feature = 12 * 2

# plots
plot_save_dir = "CNN_BiLSTM/reports/figures"
plot_loss_save_path = "cnn_transformer_loss.png"
plot_loss_train_save_path = "cnn_transformer_train_loss.png"
plot_loss_val_save_path = "cnn_transformer_val_loss.png"
plot_loss_train_val_save_path = "cnn_transformer_train_val_loss.png"
plot_wer_save_path = "cnn_transformer_wer.png"

# =============================================================================
# 総合評価関連パス設定 (Comprehensive Evaluation Paths)
# =============================================================================

# ベースディレクトリの設定
evaluation_base_dir = "CNN_BiLSTM/reports/figures"
evaluation_results_dir = "CNN_BiLSTM/reports/results"

# 5つの総合評価指標の保存ディレクトリ
attention_vis_dir = os.path.join(evaluation_base_dir, "attention_weights")
ctc_vis_dir = os.path.join(evaluation_base_dir, "ctc_alignment")
confusion_matrix_dir = os.path.join(evaluation_base_dir, "confusion_matrix")
confidence_vis_dir = os.path.join(evaluation_base_dir, "confidence_timeline")
feature_vis_dir = os.path.join(evaluation_base_dir, "multilayer_features")

# 統合レポート
comprehensive_report_dir = os.path.join(evaluation_results_dir, "evaluation")
comprehensive_report_path = os.path.join(
    comprehensive_report_dir, "comprehensive_report.html"
)
performance_metrics_path = os.path.join(
    comprehensive_report_dir, "performance_metrics.json"
)

# =============================================================================
# 5つの総合評価指標の設定 (5 Comprehensive Evaluation Metrics)
# =============================================================================

# 1. Attention/Alignment Matrix 設定
attention_config = {
    "enabled": True,
    "save_dir": attention_vis_dir,
    "heatmap_size": (12, 8),
    "colormap": "Blues",
    "save_alignment_matrix": True,
    "save_attention_weights": True,
    "max_samples": 10,
}

# 2. CTC Alignment Path 設定
ctc_config = {
    "enabled": True,
    "save_dir": ctc_vis_dir,
    "show_best_path": True,
    "show_probability_distribution": True,
    "heatmap_size": (15, 10),
    "colormap": "viridis",
    "max_samples": 10,
}

# 3. Confusion Matrix 設定
confusion_matrix_config = {
    "enabled": True,
    "save_dir": confusion_matrix_dir,
    "normalize": "true",  # 'true', 'pred', 'all', None
    "show_values": True,
    "figure_size": (10, 8),
    "colormap": "Blues",
    "save_misclassification_analysis": True,
}

# 4. 時系列予測確率/信頼度分析設定
confidence_config = {
    "enabled": True,
    "save_dir": confidence_vis_dir,
    "confidence_threshold": 0.5,
    "timeline_analysis": True,
    "word_level_analysis": True,
    "figure_size": (15, 6),
    "save_statistics": True,
    "max_samples": 10,
}

# 5. Feature Visualization 設定
feature_vis_config = {
    "enabled": True,
    "save_dir": feature_vis_dir,
    "methods": ["tsne", "umap"],  # 'tsne', 'umap', 'both'
    "layers": ["cnn_output", "bilstm_hidden", "attention_weights", "final_features"],
    "tsne_perplexity": 30,
    "tsne_n_components": 2,
    "umap_n_neighbors": 15,
    "umap_min_dist": 0.1,
    "figure_size": (12, 10),
    "save_separation_analysis": True,
    "max_samples": 10,
}

# =============================================================================
# 統合評価設定 (Integrated Evaluation Configuration)
# =============================================================================

evaluation_config = {
    "attention": attention_config,
    "ctc": ctc_config,
    "confusion_matrix": confusion_matrix_config,
    "confidence": confidence_config,
    "feature_visualization": feature_vis_config,
    # 統合レポート設定
    "generate_comprehensive_report": True,
    "report_template": "comprehensive_evaluation_template.html",
    "include_good_bad_examples": True,
    "auto_analysis": True,
    # パフォーマンス設定
    "max_visualization_samples": 10,
    "parallel_processing": False,
    "memory_efficient_mode": True,
    "save_intermediate_results": True,
    # 出力設定
    "image_format": "png",
    "dpi": 300,
    "save_raw_data": True,
}

# =============================================================================
# ヘルパー関数 (Helper Functions)
# =============================================================================


def create_evaluation_directories():
    """評価に必要なディレクトリを作成"""
    directories = [
        evaluation_base_dir,
        attention_vis_dir,
        ctc_vis_dir,
        confusion_matrix_dir,
        confidence_vis_dir,
        feature_vis_dir,
        comprehensive_report_dir,
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


def get_visualization_save_path(vis_type, batch_idx, sample_idx=0):
    """可視化ファイルの保存パスを生成"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if vis_type == "attention":
        return os.path.join(
            attention_vis_dir,
            f"attention_batch_{batch_idx}_sample_{sample_idx}_{timestamp}.png",
        )
    elif vis_type == "ctc":
        return os.path.join(
            ctc_vis_dir,
            f"ctc_alignment_batch_{batch_idx}_sample_{sample_idx}_{timestamp}.png",
        )
    elif vis_type == "confusion":
        return os.path.join(confusion_matrix_dir, f"confusion_matrix_{timestamp}.png")
    elif vis_type == "confidence":
        return os.path.join(
            confidence_vis_dir,
            f"confidence_timeline_batch_{batch_idx}_sample_{sample_idx}_{timestamp}.png",
        )
    elif vis_type == "features":
        return os.path.join(
            feature_vis_dir,
            f"multilayer_features_batch_{batch_idx}_sample_{sample_idx}_{timestamp}.png",
        )
    else:
        return os.path.join(
            evaluation_base_dir,
            f"{vis_type}_batch_{batch_idx}_sample_{sample_idx}_{timestamp}.png",
        )


def get_evaluation_config():
    """評価設定を取得"""
    return evaluation_config


def get_all_visualization_dirs():
    """全ての可視化ディレクトリを取得"""
    return {
        "attention": attention_vis_dir,
        "ctc": ctc_vis_dir,
        "confusion": confusion_matrix_dir,
        "confidence": confidence_vis_dir,
        "features": feature_vis_dir,
        "base": evaluation_base_dir,
        "report": comprehensive_report_dir,
    }


# =============================================================================
# 互換性設定 (Backward Compatibility)
# =============================================================================

# 既存のplot_save_dirとの互換性を維持
if not os.path.exists(plot_save_dir):
    plot_save_dir = evaluation_base_dir
