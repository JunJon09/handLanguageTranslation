import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

# パラメータ設定
batch_size = 2  # テスト用にバッチサイズを2に設定
feature_dim = 512  # 特徴量の次元数
inter_channels = feature_dim
tren_num_heads = 8
tren_dim_ffw = 2048
tren_dropout = 0.1
activation = 'relu'
tren_num_layers = 6

# Transformerエンコーダレイヤーの定義
enlayer = nn.TransformerEncoderLayer(
    d_model=inter_channels,
    nhead=tren_num_heads,
    dim_feedforward=tren_dim_ffw,
    dropout=tren_dropout,
    activation=activation
)

# Transformerエンコーダの定義
tr_encoder = nn.TransformerEncoder(
    encoder_layer=enlayer,
    num_layers=tren_num_layers
)

# テストデータの作成
# シーケンス長が異なる2つのシーケンスを作成
sequence_lengths = [10, 15]  # シーケンス1は長さ10、シーケンス2は長さ15

# 各シーケンスをランダムなテンソルとして作成
# 各シーケンスの形状は [L, F]（長さL、特徴量F）
sequences = [torch.randn(seq_len, feature_dim) for seq_len in sequence_lengths]

# パディング処理
# pad_sequenceはシーケンスをリストからパディングされたテンソルに変換します
# 結果のテンソルの形状は [max_L, B, F] となります
padded_sequences = pad_sequence(sequences, batch_first=False, padding_value=0)  # [max_L, B, F]
max_length = padded_sequences.size(0)  # 最大シーケンス長

# マスク作成
# src_key_padding_maskは [B, S] の形状を持ち、パディング部分をTrueで示します
mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
for i, seq_len in enumerate(sequence_lengths):
    if seq_len < max_length:
        mask[i, seq_len:] = True  # パディング部分をマスク

# マスクの確認
print("マスク:\n", mask)

# Transformerエンコーダに入力
# nn.TransformerEncoderは入力テンソルの形状が [S, B, F] であることを期待します
encoder_output = tr_encoder(padded_sequences, src_key_padding_mask=mask)  # [S, B, F]

# 必要に応じて出力を元の形状に戻す: [S, B, F] → [B, F, S]
encoder_output = encoder_output.permute(1, 2, 0)  # [B, F, S]

# 結果の形状を確認
print("パディングされたシーケンスの形状:", padded_sequences.shape)  # [max_L, B, F]
print("Maskの形状:", mask.shape)  # [B, max_L]
print("エンコーダ出力の形状:", encoder_output.shape)  # [B, F, max_L]