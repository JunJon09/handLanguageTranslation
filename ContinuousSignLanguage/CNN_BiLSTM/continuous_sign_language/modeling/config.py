activation = "relu"

# 時系列モデルの選択
# "bilstm": 従来のBiLSTMモデル
# "transformer": 標準Transformerモデル
# "multiscale_transformer": マルチスケールTransformerモデル
temporal_model_type = "transformer"  # デフォルトはTransformerに変更

# detail 1DCNN（WER低下のための調整）
cnn_out_channels = 512  # 256から320に増加してより豊富な特徴抽出
kernel_size = 25  # 30から25に減らして細かい時間パターンを捉える
stride = 1
padding = kernel_size // 2
dropout_rate = 0.25  # 0.3から0.25に下げて学習を促進
bias = False
resNet = 0  # 0: なし, 1: restNet18, 2: restNet34 3: restNet50 restNet101 5: restNet152, 6: restNet152


# detail transformer encoder（WER改善のための調整）
tren_num_layers = 4  # 5から4に減らして過学習を防ぐ
tren_num_heads = 8  # 維持
tren_dim_ffw = 384  # 320から384に増やして表現力向上
tren_dropout_pe = 0.05  # 0.08から0.05に下げて学習促進
tren_dropout = 0.15  # 0.2から0.15に下げて学習促進
norm_type = "layer"
tren_norm_eps = 1e-5
tren_norm_first = True
tren_add_bias = True
tren_add_tailnorm = True
batch_first = True

# 学習パラメータ（Loss改善とWER低下のための調整）
label_smoothing = 0.15  # ラベルスムージングを少し下げて過度な正則化を防ぐ
lr = 1e-4  # 学習率をより保守的に設定してLoss安定化
epochs = 70  # エポック数を増やして十分な学習
eval_every_n_epochs = 1

# 過学習対策の追加パラメータ（Loss安定化）
weight_decay = 5e-5  # Weight decayを少し下げて学習を促進
grad_clip_norm = 1.0  # 勾配クリッピングを強化してLoss爆発を防ぐ
early_stopping_patience = 15  # 早期停止の忍耐度を上げる

# アンサンブル学習用パラメータ
ensemble_models = 3  # アンサンブルするモデル数
model_seed_base = 42  # 異なるシード値でのモデル訓練

# モデルの保存
model_save_path = "CNN_BiLSTM/models/CNN_BiLSTM_model.pth"

# CTC Beam Search パラメータ（第2回設定に戻す）
beam_width = 10  # 第2回の設定
alpha = 0.8  # 第2回の設定
beta = 1.2  # 第2回の設定

# デコードモード設定（第2回に戻す）
decode_mode = "max"  # SOVではなく標準のmaxデコードに戻す
use_sov_reordering = False  # SOV語順並べ替えを無効化

# データ拡張（WER改善のために控えめに調整）
data_augmentation = True
augment_noise_std = 0.005  # ノイズを半分に減らして精度向上
augment_scale_range = (0.99, 1.01)  # スケール変動を小さくして安定化
augment_probability = 0.5  # 確率を下げて過度な拡張を防ぐ
augment_rotation_range = 1.0  # 回転範囲を小さくして精度向上
augment_translation_range = 0.01  # 平行移動範囲を小さくして安定化

# モデル使用パス
model_use_path = "CNN_BiLSTM/models/CNN_BiLSTM_model.pth"
