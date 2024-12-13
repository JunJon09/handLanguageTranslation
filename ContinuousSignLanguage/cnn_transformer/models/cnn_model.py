import torch
from torch import nn

class CNNFeatureExtractor(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 dropout_prob=0.2
                 ):
        super().__init__()
        # padding = kernel_size // 2
        padding = 0
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
        #self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.downsample = nn.Sequential(
            nn.AvgPool1d(kernel_size=5),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        # 入力 x の形状: (batch_size, in_channels, sequence_length)
        identity = x
        out = self.conv1(x)
        out = self.batch_norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        identity = self.downsample(x)
        out += identity
        out = self.activation(out)
        return out  # 形状: (batch_size, out_channels, reduced_sequence_length)

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
        x: [N, C*J, T] の入力テンソル
        """
        #

        #固定長ウィンドウへの分割, 時間軸に沿ってウィンドウを作成  # [N, C*J, T] -> [N, C*J, T-window_size+1(now_windows), window_size]
        windows = x.unfold(dimension=2, size=self.window_size, step=self.stride) 
        N, CJ, num_windows, window_size = windows.shape

        # 3. CNNへの入力整形: [N, C*J, T-window_size+1, window_size] -> [N*num_windows, C*J, window_size]
        windows = windows.contiguous().view(N * num_windows, CJ, window_size)

        # 4. CNNで特徴抽出
        # [N*num_windows, C*J, window_size] -> [N*num_windows, out_channels, 1]
        features = self.cnn(windows) 

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
