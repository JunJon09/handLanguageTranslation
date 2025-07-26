import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """位置エンコーディング - Transformerで時間情報を埋め込む"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerLayer(nn.Module):
    """Transformerエンコーダ層 - BiLSTMの代替として使用"""
    
    def __init__(self, input_size, hidden_size=512, num_layers=4, num_heads=8, 
                 dim_feedforward=2048, dropout=0.1, activation='relu'):
        super(TransformerLayer, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # 入力次元を調整するための線形層
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # 位置エンコーディング
        self.pos_encoder = PositionalEncoding(hidden_size, dropout=dropout)
        
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=False  # BiLSTMと同じ形式 (seq_len, batch, features)
        )
        
        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # 出力次元を調整（BiLSTMの双方向と同等の出力次元を確保）
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        
        # LayerNorm for final output
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def create_padding_mask(self, src_lens, max_len):
        """
        パディング用のマスクを作成
        Args:
            src_lens: (batch_size,) - 各サンプルの実際の長さ
            max_len: int - 最大長
        Returns:
            mask: (batch_size, max_len) - Trueの部分がパディング
        """
        batch_size = len(src_lens)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=src_lens.device)
        
        for i, length in enumerate(src_lens):
            if length < max_len:
                mask[i, length:] = True
                
        return mask
        
    def forward(self, src_feats, src_lens, hidden=None):
        """
        Args:
            src_feats: (max_src_len, batch_size, input_size)
            src_lens: (batch_size,) - 各サンプルの実際の長さ
            hidden: 未使用（BiLSTMとの互換性のため）
        Returns:
            dict: {
                "predictions": (max_src_len, batch_size, hidden_size),
                "hidden": None  # Transformerは隠れ状態を持たない
            }
        """
        max_src_len, batch_size, _ = src_feats.shape
        
        # 入力次元を調整
        x = self.input_projection(src_feats)  # (max_src_len, batch_size, hidden_size)
        
        # 位置エンコーディングを追加
        x = self.pos_encoder(x)
        
        # パディングマスクを作成
        src_key_padding_mask = self.create_padding_mask(src_lens, max_src_len)
        
        # Transformer処理
        transformer_out = self.transformer_encoder(
            x, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        # 出力次元を調整
        output = self.output_projection(transformer_out)
        output = self.layer_norm(output)
        
        return {
            "predictions": output,
            "hidden": None  # Transformerは隠れ状態を持たない
        }


class MultiScaleTransformerLayer(nn.Module):
    """マルチスケールTransformer - 異なる時間スケールでの特徴抽出"""
    
    def __init__(self, input_size, hidden_size=512, num_layers=4, num_heads=8, 
                 dim_feedforward=2048, dropout=0.1):
        super(MultiScaleTransformerLayer, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 入力投影層
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # 3つの異なるスケールのTransformer
        # 短期スケール（局所的な特徴）
        self.short_term_transformer = self._create_transformer_block(
            hidden_size, num_heads=num_heads, num_layers=2, dropout=dropout
        )
        
        # 中期スケール（中程度の時間依存性）
        self.mid_term_transformer = self._create_transformer_block(
            hidden_size, num_heads=num_heads//2, num_layers=3, dropout=dropout
        )
        
        # 長期スケール（長期的な時間依存性）
        self.long_term_transformer = self._create_transformer_block(
            hidden_size, num_heads=num_heads//4, num_layers=4, dropout=dropout
        )
        
        # 特徴統合層
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        # 位置エンコーディング
        self.pos_encoder = PositionalEncoding(hidden_size, dropout=dropout)
        
    def _create_transformer_block(self, d_model, num_heads, num_layers, dropout):
        """Transformerブロックを作成"""
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=False
        )
        return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def create_padding_mask(self, src_lens, max_len):
        """パディングマスクを作成"""
        batch_size = len(src_lens)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=src_lens.device)
        
        for i, length in enumerate(src_lens):
            if length < max_len:
                mask[i, length:] = True
                
        return mask
    
    def subsample_sequence(self, x, factor):
        """シーケンスをサブサンプリング"""
        if factor == 1:
            return x
        return x[::factor, :, :]
    
    def upsample_sequence(self, x, target_len):
        """シーケンスをアップサンプリング"""
        if x.size(0) == target_len:
            return x
        
        # 線形補間でアップサンプリング
        x_permuted = x.permute(1, 2, 0)  # (batch, features, seq_len)
        upsampled = F.interpolate(
            x_permuted, 
            size=target_len, 
            mode='linear', 
            align_corners=False
        )
        return upsampled.permute(2, 0, 1)  # (seq_len, batch, features)
    
    def forward(self, src_feats, src_lens, hidden=None):
        """
        Args:
            src_feats: (max_src_len, batch_size, input_size)
            src_lens: (batch_size,)
            hidden: 互換性のため（未使用）
        Returns:
            dict: {"predictions": tensor, "hidden": None}
        """
        max_src_len, batch_size, _ = src_feats.shape
        
        # 入力投影
        x = self.input_projection(src_feats)
        x = self.pos_encoder(x)
        
        # パディングマスク作成
        padding_mask = self.create_padding_mask(src_lens, max_src_len)
        
        # 短期スケール処理
        short_out = self.short_term_transformer(x, src_key_padding_mask=padding_mask)
        
        # 中期スケール処理（2倍サブサンプリング）
        mid_input = self.subsample_sequence(x, factor=2)
        mid_mask = self.create_padding_mask(src_lens//2, mid_input.size(0))
        mid_out = self.mid_term_transformer(mid_input, src_key_padding_mask=mid_mask)
        mid_out = self.upsample_sequence(mid_out, max_src_len)
        
        # 長期スケール処理（4倍サブサンプリング）
        long_input = self.subsample_sequence(x, factor=4)
        long_mask = self.create_padding_mask(src_lens//4, long_input.size(0))
        long_out = self.long_term_transformer(long_input, src_key_padding_mask=long_mask)
        long_out = self.upsample_sequence(long_out, max_src_len)
        
        # 特徴統合
        fused_features = torch.cat([short_out, mid_out, long_out], dim=-1)
        output = self.feature_fusion(fused_features)
        
        return {
            "predictions": output,
            "hidden": None
        }


class ConformerLayer(nn.Module):
    """Conformer - CNNとTransformerを組み合わせたアーキテクチャ"""
    
    def __init__(self, input_size, hidden_size=512, num_layers=4, num_heads=8, 
                 conv_kernel_size=31, dropout=0.1):
        super(ConformerLayer, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 入力投影
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # Conformerブロック
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                d_model=hidden_size,
                num_heads=num_heads,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # 位置エンコーディング
        self.pos_encoder = PositionalEncoding(hidden_size, dropout=dropout)
        
    def create_padding_mask(self, src_lens, max_len):
        """パディングマスクを作成"""
        batch_size = len(src_lens)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=src_lens.device)
        
        for i, length in enumerate(src_lens):
            if length < max_len:
                mask[i, length:] = True
        return mask
    
    def forward(self, src_feats, src_lens, hidden=None):
        """
        Args:
            src_feats: (max_src_len, batch_size, input_size)
            src_lens: (batch_size,)
        Returns:
            dict: {"predictions": tensor, "hidden": None}
        """
        max_src_len, batch_size, _ = src_feats.shape
        
        # 入力投影と位置エンコーディング
        x = self.input_projection(src_feats)
        x = self.pos_encoder(x)
        
        # パディングマスク
        padding_mask = self.create_padding_mask(src_lens, max_src_len)
        
        # Conformerブロック適用
        for conformer_block in self.conformer_blocks:
            x = conformer_block(x, padding_mask)
        
        return {
            "predictions": x,
            "hidden": None
        }


class ConformerBlock(nn.Module):
    """Conformerブロック - Feed Forward + Self-Attention + Convolution + Feed Forward"""
    
    def __init__(self, d_model, num_heads, conv_kernel_size=31, dropout=0.1):
        super(ConformerBlock, self).__init__()
        
        # 第1のFeed Forward
        self.ff1 = FeedForwardModule(d_model, dropout)
        
        # Multi-Head Self-Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False
        )
        self.attn_layer_norm = nn.LayerNorm(d_model)
        
        # Convolution Module
        self.conv_module = ConvolutionModule(d_model, conv_kernel_size, dropout)
        
        # 第2のFeed Forward
        self.ff2 = FeedForwardModule(d_model, dropout)
        
        self.final_layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, padding_mask=None):
        """
        Args:
            x: (seq_len, batch, d_model)
            padding_mask: (batch, seq_len)
        """
        # Feed Forward 1 (1/2 weight)
        x = x + 0.5 * self.ff1(x)
        
        # Self-Attention
        residual = x
        attn_out, _ = self.self_attn(
            x, x, x, 
            key_padding_mask=padding_mask
        )
        x = self.attn_layer_norm(residual + attn_out)
        
        # Convolution
        x = x + self.conv_module(x)
        
        # Feed Forward 2 (1/2 weight)
        x = x + 0.5 * self.ff2(x)
        
        return self.final_layer_norm(x)


class FeedForwardModule(nn.Module):
    """Feed Forward Module for Conformer"""
    
    def __init__(self, d_model, dropout=0.1):
        super(FeedForwardModule, self).__init__()
        
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = F.silu(x)  # Swish activation
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return self.layer_norm(residual + x)


class ConvolutionModule(nn.Module):
    """Convolution Module for Conformer"""
    
    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super(ConvolutionModule, self).__init__()
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Linear(d_model, d_model * 2)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, 
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise_conv2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (seq_len, batch, d_model)
        """
        residual = x
        x = self.layer_norm(x)
        
        # Pointwise convolution 1
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=-1)  # Gated Linear Unit
        
        # Depthwise convolution
        x = x.transpose(0, 2).transpose(1, 2)  # (seq, batch, dim) -> (batch, dim, seq)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = F.silu(x)  # Swish activation
        x = x.transpose(1, 2).transpose(0, 2)  # (batch, dim, seq) -> (seq, batch, dim)
        
        # Pointwise convolution 2
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        return residual + x
