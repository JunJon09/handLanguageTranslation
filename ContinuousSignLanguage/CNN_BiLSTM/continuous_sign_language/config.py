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

top_n = 50
word_error_distribution_save_path = "word_error_distribution.png"