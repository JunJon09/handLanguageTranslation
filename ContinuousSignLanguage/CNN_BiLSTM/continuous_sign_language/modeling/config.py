
# 時系列モデルの選択
cnn_model_type = "DualCNNWithCTC" # "DualCNNWithCTC" or "DualMultiScaleTemporalConv"
temporal_model_type = "bilstm"  # "transformer" or "bilstm"
# detail 1DCNN（WER低下のための調整）
cnn_out_channels = 512  
kernel_sizes = [10, 15, 20, 25, 30] 
cnn_dropout_rate = 0.25  
conv_type = 2
use_bn = True

num_layers = 4  
num_heads = 8  

dropout = 0.15

# 学習パラメータ（Loss改善とWER低下のための調整）
label_smoothing = 0.15  
lr = 1e-4 
epochs = 70
eval_every_n_epochs = 1

# 過学習対策の追加パラメータ（Loss安定化）
weight_decay = 5e-5  
grad_clip_norm = 1.0  
early_stopping_patience = 15 

# アンサンブル学習用パラメータ
ensemble_models = 3  
model_seed_base = 42 

# モデルの保存
model_save_path = "CNN_BiLSTM/models/CNN_BiLSTM_model.pth"

# モデル使用パス
model_use_path = "CNN_BiLSTM/models/CNN_BiLSTM_model_lstm_20.pth"
