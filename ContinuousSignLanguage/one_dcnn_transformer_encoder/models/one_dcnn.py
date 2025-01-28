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
        in_channels = in_channels + 24
        padding = 0
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(in_channels, out_channels, self.kernel_size, stride, padding)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
        #self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.downsample = nn.Sequential(
            nn.AvgPool1d(self.kernel_size),
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
        return out
