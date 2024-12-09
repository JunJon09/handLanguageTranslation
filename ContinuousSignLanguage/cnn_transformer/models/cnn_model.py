import torch
from torch import nn

class CNNFeatureExtractor(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding
                 ):
        super().__init__()
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

