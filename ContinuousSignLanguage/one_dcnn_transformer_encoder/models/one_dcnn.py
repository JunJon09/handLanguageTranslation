import torch
import torch.nn as nn
import torch.nn.functional as F

# 基本ブロックの定義
class BasicBlock1D(nn.Module):
    expansion = 1  # 出力チャンネル数の倍率

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample  # 入力を合わせるための層

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # Residual Connection
        out = self.relu(out)

        return out

class Bottleneck1D(nn.Module):
    expansion = 4  # 出力チャンネル数の倍率（Bottleneckは4倍）

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # 入力を合わせるための層

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # Residual Connection
        out = self.relu(out)

        return out

# ResNet1D クラスの定義
class ResNet1D(nn.Module):
    def __init__(self, block, layers, num_classes=1000, in_channels=1):
        """
        Args:
            block: 使用するブロッククラス（BasicBlock1DまたはBottleneck1D）
            layers: 各層に含まれるブロックの数 [layer1, layer2, layer3, layer4]
            num_classes: 分類クラス数
            in_channels: 入力データのチャンネル数
        """
        super(ResNet1D, self).__init__()  # 引数なしで呼び出す
        self.in_channels = 64
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        # 各ResNetの層
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 重みの初期化
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """
        Args:
            block: ブロッククラス
            out_channels: 出力チャンネル数
            blocks: ブロックの数
            stride: ストライド
        """
        downsample = None
        # 出力と入力のチャネル数が異なる場合、ダウンサンプル層を定義
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 入力の形状: [batch_size, in_channels, sequence_length]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)  # layer1
        x = self.layer2(x)  # layer2
        x = self.layer3(x)  # layer3
        x = self.layer4(x)  # layer4

        x = self.avgpool(x)  # グローバル平均プーリング
        x = torch.flatten(x, 1)  # フラット化
        x = self.fc(x)  # 全結合層

        return x

# 各ResNetバージョン用のファクトリ関数
def resnet18_1d(num_classes=1000, in_channels=1):
    return ResNet1D(BasicBlock1D, [2, 2, 2, 2], num_classes=num_classes, in_channels=in_channels)

def resnet34_1d(num_classes=1000, in_channels=1):
    return ResNet1D(BasicBlock1D, [3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels)

def resnet50_1d(num_classes=1000, in_channels=1):
    return ResNet1D(Bottleneck1D, [3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels)

def resnet101_1d(num_classes=1000, in_channels=1):
    return ResNet1D(Bottleneck1D, [3, 4, 23, 3], num_classes=num_classes, in_channels=in_channels)

def resnet152_1d(num_classes=1000, in_channels=1):
    return ResNet1D(Bottleneck1D, [3, 8, 36, 3], num_classes=num_classes, in_channels=in_channels)

# 使用例
# if __name__ == "__main__":
#     # 入力テンソルの作成
#     N, C, T, J = 32, 3, 100, 25  # 例
#     input_tensor = torch.randn(N, C, T, J)  # 形状: [8, 3, 100, 25]
    
#     # 入力テンソルの形状を [N, C * J, T] に変換
#     input_tensor = input_tensor.permute(0, 3, 1, 2).contiguous().view(N, C * J, T)  # [8, 75, 100]
    
#     # モデルのインスタンス化
#     num_classes = 100  # 例: 100クラス分類
#     in_channels = C * J  # 3 * 25 = 75
#     model = resnet18_1d(num_classes=num_classes, in_channels=in_channels)
    
#     # フォワードパス
#     output = model(input_tensor)
#     print(output.shape)  # 期待される形状: [8, 100]
