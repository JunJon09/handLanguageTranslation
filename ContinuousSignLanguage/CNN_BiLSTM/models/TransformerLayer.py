import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """位置エンコーディング"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: (seq_len, batch_size, d_model)
        Returns:
            x + positional encoding: (seq_len, batch_size, d_model)
        """
        return x + self.pe[: x.size(0), :]


class TransformerLayer(nn.Module):
    """BiLSTMと同じインターフェースを持つTransformerLayer"""

    def __init__(
        self,
        input_size,
        hidden_size=512,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        max_len=5000,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 入力投影層（input_sizeをhidden_sizeに変換）
        self.input_projection = nn.Linear(input_size, hidden_size)

        # 位置エンコーディング
        self.pos_encoding = PositionalEncoding(hidden_size, max_len)

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,  # 通常は4倍
            dropout=dropout,
            activation="relu",
            batch_first=False,  # BiLSTMと同じ形式 (seq_len, batch_size, d_model)
        )

        # Transformer Encoder
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ドロップアウト
        self.dropout = nn.Dropout(dropout)

    def forward(self, src_feats, src_lens, hidden=None):
        """
        BiLSTMと同じインターフェース

        Args:
            src_feats: (max_src_len, batch_size, input_size)
            src_lens: (batch_size,) - 各シーケンスの実際の長さ
            hidden: 使用されない（Transformerでは状態を持たない）

        Returns:
            dict with:
                - predictions: (max_src_len, batch_size, hidden_size)
                - hidden: None (Transformerでは状態を持たない)
        """
        seq_len, batch_size, _ = src_feats.shape

        # 入力投影
        x = self.input_projection(src_feats)  # (seq_len, batch_size, hidden_size)

        # 位置エンコーディング追加
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # パディングマスクの作成
        # src_lens に基づいてマスクを作成
        src_key_padding_mask = self._create_padding_mask(
            src_lens, seq_len, batch_size, x.device
        )

        # Transformer処理
        # batch_first=False なので、入力は (seq_len, batch_size, d_model)
        transformer_output = self.transformer(
            x, src_key_padding_mask=src_key_padding_mask
        )

        return {
            "predictions": transformer_output,  # (seq_len, batch_size, hidden_size)
            "hidden": None,  # Transformerでは隠れ状態なし
        }

    def _create_padding_mask(self, src_lens, seq_len, batch_size, device):
        """
        パディング位置をマスクするためのマスクを作成

        Args:
            src_lens: (batch_size,) 各シーケンスの実際の長さ
            seq_len: int, 最大シーケンス長
            batch_size: int
            device: torch.device

        Returns:
            mask: (batch_size, seq_len) パディング位置がTrueのマスク
        """
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

        for i, length in enumerate(src_lens):
            if length < seq_len:
                mask[i, length:] = True  # パディング位置をTrueに

        return mask


class MultiScaleTransformerLayer(nn.Module):
    """マルチスケール特徴を考慮したTransformerLayer"""

    def __init__(
        self,
        input_size,
        hidden_size=512,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        max_len=5000,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 異なるスケールのTransformer
        self.short_term_transformer = TransformerLayer(
            input_size,
            hidden_size // 2,
            num_layers // 2,
            num_heads // 2,
            dropout,
            max_len,
        )

        self.long_term_transformer = TransformerLayer(
            input_size,
            hidden_size // 2,
            num_layers // 2,
            num_heads // 2,
            dropout,
            max_len,
        )

        # 特徴融合層
        self.fusion_layer = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, src_feats, src_lens, hidden=None):
        """
        Args:
            src_feats: (max_src_len, batch_size, input_size)
            src_lens: (batch_size,)
            hidden: 使用されない

        Returns:
            dict with predictions and hidden
        """
        # 短期的特徴抽出
        short_output = self.short_term_transformer(src_feats, src_lens, hidden)

        # 長期的特徴抽出（ダウンサンプリング）
        # 2つおきにサンプリングして長期依存関係を捉える
        long_feats = src_feats[::2]  # (max_src_len//2, batch_size, input_size)
        long_lens = (src_lens + 1) // 2  # 長さも調整
        long_output = self.long_term_transformer(long_feats, long_lens, hidden)

        # 長期特徴をアップサンプリング
        seq_len = src_feats.size(0)
        long_upsampled = F.interpolate(
            long_output["predictions"].permute(
                1, 2, 0
            ),  # (batch, hidden//2, seq_len//2)
            size=seq_len,
            mode="linear",
            align_corners=False,
        ).permute(
            2, 0, 1
        )  # (seq_len, batch, hidden//2)

        # 特徴融合
        fused_features = torch.cat(
            [
                short_output["predictions"],  # (seq_len, batch, hidden//2)
                long_upsampled,  # (seq_len, batch, hidden//2)
            ],
            dim=-1,
        )  # (seq_len, batch, hidden)

        # 最終的な特徴変換
        output = self.fusion_layer(fused_features)
        output = self.layer_norm(output)

        return {"predictions": output, "hidden": None}
