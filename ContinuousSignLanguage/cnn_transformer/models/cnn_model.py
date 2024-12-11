import torch
from torch import nn

class CNNFeatureExtractor(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride
                 ):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        # 入力 x の形状: (batch_size, in_channels, sequence_length)
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pool(x)
        return x  # 形状: (batch_size, out_channels, reduced_sequence_length)

class CNNLocalFeatureExtractor(nn.Module):
    def __init__(self, cnn, window_size=5, stride=1):
        """
        cnn: 1D CNN モジュール（CNNFeatureExtractorのインスタンス）
        window_size: 固定長ウィンドウのフレーム数（例：5）
        stride: ウィンドウの移動ステップ
        """
        super().__init__()
        self.cnn = cnn
        self.window_size = window_size
        self.stride = stride

    def forward(self, x):
        """
        x: [N, C, T, J] の入力テンソル
        """


        #[N, C*J, T]

        # 2. 固定長ウィンドウへの分割
        # torch.Tensor.unfold(dimension, size, step) を使用
        # ここでは時間軸（dimension=2）に沿ってウィンドウを作成
        windows = x.unfold(dimension=2, size=self.window_size, step=self.stride)  # [N, C*J, num_windows, window_size]
        N, CJ, num_windows, window_size = windows.shape

        # 3. CNNへの入力整形: [N, C*J, num_windows, window_size] -> [N*num_windows, C*J, window_size]
        windows = windows.contiguous().view(N * num_windows, CJ, window_size)

        # 4. CNNで特徴抽出
        features = self.cnn(windows)  # [N*num_windows, out_channels, reduced_sequence_length]

        # 5. 特徴のプーリング: 時間次元を平均またはその他の手法で削減
        features = features.mean(dim=2)  # [N*num_windows, out_channels]

        # 6. 元のサンプルに戻す: [N*num_windows, out_channels] -> [N, num_windows, out_channels]
        features = features.view(N, num_windows, -1)  # [N, num_windows, out_channels]
        last_frame = features[:, -1:, :]  # 最後のタイムステップを保持するためにスライス

        # 最後のフレームを4回繰り返す: [N, 4, out_channels]
        repeated_frames = last_frame.repeat(1, 4, 1)

        # オリジナルの特徴量に繰り返されたフレームを追加: [N, num_windows + 4, out_channels]
        x_padded = torch.cat([features, repeated_frames], dim=1)

        # 7. 特徴量を連結: [N, num_windows, out_channels] -> [N, num_windows * out_channels]
        # concat_features = features.view(N, -1)  # [N, num_windows * out_channels]

        return x_padded  # 最終的な特徴量
