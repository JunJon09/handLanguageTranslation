import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


# 基本ブロックの定義
class BasicBlock1D(nn.Module):
    expansion = 1  # 出力チャンネル数の倍率

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        print(in_channels, out_channels, stride)
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample  # 入力を合わせるための層

    def forward(self, x):
        identity = x
        print(f"input shape: {x.shape}")
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        print(f"after conv1 shape: {out.shape}")
        out = self.conv2(out)
        out = self.bn2(out)
        print(f"after conv2 shape: {out.shape}")

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
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(
            out_channels, out_channels * self.expansion, kernel_size=1, bias=False
        )
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
    def __init__(
        self,
        block,
        layers,
        kernel_size=3,
        stride=1,
        padding=0,
        num_classes=1000,
        in_channels=1,
        out_channels=64,
        dropout_rate=0.2,
        bias=False,
    ):
        """
        Args:
            block: 使用するブロッククラス（BasicBlock1DまたはBottleneck1D）
            layers: 各層に含まれるブロックの数 [layer1, layer2, layer3, layer4]
            num_classes: 分類クラス数
            in_channels: 入力データのチャンネル数
        """
        super(ResNet1D, self).__init__()  # 引数なしで呼び出す
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.conv1 = nn.Conv1d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=self.bias,
        )
        self.bn1 = nn.BatchNorm1d(self.out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        # 各ResNetの層
        self.in_channels = self.out_channels
        self.layer1 = self._make_layer(block, self.out_channels, layers[0])
        self.in_channels = self.out_channels
        self.layer2 = self._make_layer(
            block, self.out_channels * 2, layers[1], stride=self.stride
        )
        self.in_channels = self.out_channels * 2
        self.layer3 = self._make_layer(
            block, self.out_channels * 4, layers[2], stride=self.stride
        )
        self.in_channels = self.out_channels * 4
        self.layer4 = self._make_layer(
            block, self.out_channels * 8, layers[3], stride=self.stride
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 重みの初期化
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
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
                nn.Conv1d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels * block.expansion),
            )

        layers = []
        print(self.in_channels, "***************")
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 入力の形状: [batch_size, in_channels, sequence_length]
        print(f"ResNet1D input shape: {x.shape}")
        print(
            f"Conv1 params: in_channels={self.conv1.in_channels}, out_channels={self.conv1.out_channels}, kernel_size={self.conv1.kernel_size}, stride={self.conv1.stride}, padding={self.conv1.padding}"
        )

        x = self.conv1(x)
        print(f"After conv1 shape: {x.shape}")

        x = self.bn1(x)
        x = self.relu(x)
        print(f"After relu shape: {x.shape}")

        x = self.maxpool(x)
        print(f"After maxpool shape: {x.shape}")

        x = self.layer1(x)  # layer1
        print(f"After layer1 shape: {x.shape}")

        x = self.layer2(x)  # layer2
        print(f"After layer2 shape: {x.shape}")

        x = self.layer3(x)  # layer3
        print(f"After layer3 shape: {x.shape}")

        x = self.layer4(x)  # layer4
        print(f"After layer4 shape: {x.shape}")

        # x = self.avgpool(x)  # グローバル平均プーリング
        # x = torch.flatten(x, 1)  # フラット化
        # x = self.fc(x)  # 全結合層

        return x


class SimpleCNN1Layer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dropout_rate=0.2,
        bias=False,
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
            bias=bias,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # 入力テンソルのチェック
        self.check_tensor(x, "入力")

        x = self.conv(x)
        self.check_tensor(x, "畳み込み後")

        x = self.bn(x)
        self.check_tensor(x, "バッチ正規化後")

        x = self.relu(x)
        self.check_tensor(x, "ReLU後")

        x = self.dropout(x)
        self.check_tensor(x, "ドロップアウト後")

        return x

    def check_tensor(self, tensor, stage):
        if torch.isnan(tensor).any():
            print(f"{stage}でNaN値が検出されました")
            self.print_tensor_stats(tensor, stage)
            # オプション: トレースバックを表示
            import traceback

            traceback.print_stack()

        if torch.isinf(tensor).any():
            print(f"{stage}で無限大の値が検出されました")

    def print_tensor_stats(self, tensor, stage):
        print(f"{stage}のテンソル統計:")
        print(f"  形状: {tensor.shape}")
        print(f"  データ型: {tensor.dtype}")
        print(f"  NaN値の数: {torch.isnan(tensor).sum().item()}")
        print(f"  最小値: {tensor.min().item()}")
        print(f"  最大値: {tensor.max().item()}")
        print(f"  平均値: {tensor.mean().item()}")
        print(f"  標準偏差: {tensor.std().item()}")


class UltraStableCNN1Layer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=5,
        stride=1,
        padding="same",
        dropout_rate=0.2,
    ):
        super().__init__()

        # パディングの明示的計算
        if padding == "same":
            padding = kernel_size // 2

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )

        # 高度な重みの初期化
        self.reset_parameters()

        # より安定したバッチ正規化
        self.bn = nn.BatchNorm1d(
            out_channels,
            eps=1e-3,  # より小さなeps
            momentum=0.9,  # 安定化のためのmomentum調整
        )

        # 活性化関数の改善
        self.relu = nn.SiLU()  # ReLUの代わりにSiLU/Swishを使用

        # Dropoutの改善
        self.dropout = nn.Dropout(dropout_rate)

    def reset_parameters(self):
        # カスタム重み初期化
        nn.init.orthogonal_(self.conv.weight)

        # 重みのスケーリング
        with torch.no_grad():
            self.conv.weight.mul_(0.1)  # 小さな初期値

    def forward(self, x):
        # 追加の数値安定化
        x = self.safe_forward(x)
        return x

    def safe_forward(self, x):
        # 各ステップでの数値チェックと安定化
        x = self.conv(x)
        x = self.numerically_stable_normalize(x)
        x = self.bn(x)
        x = self.numerically_stable_normalize(x)
        x = self.relu(x)
        x = self.numerically_stable_normalize(x)
        x = self.dropout(x)
        return x

    def numerically_stable_normalize(self, x, eps=1e-5):
        # 高度な数値安定化正規化
        x_mean = x.mean(dim=(0, 2), keepdim=True)
        x_std = x.std(dim=(0, 2), keepdim=True)

        # クリッピング付きの正規化
        normalized_x = (x - x_mean) / (x_std + eps)

        # さらなる安定化のためのクリッピング
        normalized_x = torch.clamp(normalized_x, min=-5, max=5)

        return normalized_x


class SimpleCNN2Layer(nn.Module):
    """
    2層のシンプルなCNNモデル
    """

    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dropout_rate=0.2,
        bais=False,
    ):
        super(SimpleCNN2Layer, self).__init__()

        # 第1層
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bais,
        )
        self.bn1 = nn.BatchNorm1d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout_rate)

        # 第2層
        self.conv2 = nn.Conv1d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bais,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x の形状: [batch_size, in_channels, sequence_length]

        # 第1層
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        # 第2層
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        return x  # 出力形状: [batch_size, out_channels, new_sequence_length]


class SimpleCNN2LayerWithPooling(nn.Module):
    """
    プーリング層を含む2層のCNNモデル
    """

    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        pool_size=2,
        pool_stride=2,
        dropout_rate=0.2,
        bias=False,
    ):
        super(SimpleCNN2LayerWithPooling, self).__init__()

        # 第1層
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.bn1 = nn.BatchNorm1d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        self.dropout1 = nn.Dropout(dropout_rate)

        # 第2層
        self.conv2 = nn.Conv1d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x の形状: [batch_size, in_channels, sequence_length]

        # 第1層
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # 第2層
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        return x  # 出力形状: [batch_size, out_channels, reduced_sequence_length]


class SimpleCNN1LayerWithResidual(nn.Module):
    """
    残差接続を持つ1層のシンプルなCNNモデル
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dropout_rate=0.2,
        bias=False,
    ):
        super(SimpleCNN1LayerWithResidual, self).__init__()

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)

        # 入力チャンネル数と出力チャンネル数が異なる場合の調整用
        self.adjust_channels = None
        if in_channels != out_channels:
            self.adjust_channels = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            )

    def forward(self, x):
        # x の形状: [batch_size, in_channels, sequence_length]
        identity = x

        x = self.conv(x)
        x = self.bn(x)

        # チャンネル数の調整
        if self.adjust_channels is not None:
            identity = self.adjust_channels(identity)

        # 残差接続
        x = x + identity

        x = self.relu(x)
        x = self.dropout(x)

        return x  # 出力形状: [batch_size, out_channels, sequence_length]


# 各ResNetバージョン用のファクトリ関数
def resnet18_1d(
    num_classes=1000,
    in_channels=1,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=0,
    dropout_rate=0.2,
    bias=False,
):
    return ResNet1D(
        BasicBlock1D,
        [2, 2, 2, 2],
        num_classes=num_classes,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dropout_rate=0.2,
        bias=bias,
    )


def resnet34_1d(
    num_classes=1000, in_channels=1, kernel_size=3, stride=1, padding=0, bias=False
):
    return ResNet1D(
        BasicBlock1D,
        [3, 4, 6, 3],
        num_classes=num_classes,
        in_channels=in_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    )


def resnet50_1d(num_classes=1000, in_channels=1):
    return ResNet1D(
        Bottleneck1D, [3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels
    )


def resnet101_1d(num_classes=1000, in_channels=1):
    return ResNet1D(
        Bottleneck1D, [3, 4, 23, 3], num_classes=num_classes, in_channels=in_channels
    )


def resnet152_1d(num_classes=1000, in_channels=1):
    return ResNet1D(
        Bottleneck1D, [3, 8, 36, 3], num_classes=num_classes, in_channels=in_channels
    )


# 層が少ないファクトリ関数
def create_simple_cnn1layer(
    in_channels,
    out_channels,
    kernel_size=3,
    stride=1,
    padding=1,
    dropout_rate=0.2,
    bias=False,
):
    # return SimpleCNN1Layer(in_channels, out_channels, kernel_size, stride, 'same', dropout_rate, bias=bias)
    return UltraStableCNN1Layer(
        in_channels, out_channels, kernel_size, stride, padding, dropout_rate
    )


def create_simple_cnn2layer(
    in_channels,
    mid_channels,
    out_channels,
    kernel_size=3,
    stride=1,
    padding=1,
    dropout_rate=0.2,
    bias=False,
):
    return SimpleCNN2Layer(
        in_channels,
        mid_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dropout_rate,
        bias,
    )


def create_simple_cnn2layer_with_pooling(
    in_channels,
    mid_channels=32,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    pool_size=2,
    pool_stride=2,
    dropout_rate=0.2,
    bias=False,
):
    return SimpleCNN2LayerWithPooling(
        in_channels,
        mid_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        pool_size,
        pool_stride,
        dropout_rate,
        bias,
    )


def create_simple_cnn1layer_with_residual(
    in_channels,
    out_channels,
    kernel_size=3,
    stride=1,
    padding=1,
    dropout_rate=0.2,
    bias=False,
):
    return SimpleCNN1LayerWithResidual(
        in_channels, out_channels, kernel_size, stride, padding, dropout_rate, bias
    )


class MultiScaleTemporalConv(nn.Module):
    """
    複数層構造で異なる受容野（20, 30, 40フレーム）を実現するMulti-Scale Temporal Convolution
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        target_frames=[20, 30, 40],
        dropout_rate=0.2,
        use_parallel_processing=False,
    ):
        super(MultiScaleTemporalConv, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.target_frames = target_frames
        self.num_scales = len(target_frames)
        self.use_parallel_processing = use_parallel_processing

        # 各フレーム数に対応する多層畳み込みブランチを作成
        self.conv_branches = nn.ModuleList()

        for i, target_frame in enumerate(target_frames):
            # 各スケールで同じ出力チャンネル数になるよう調整
            scale_hidden_size = hidden_size // self.num_scales
            if i == len(target_frames) - 1:  # 最後の層で端数を調整
                scale_hidden_size = hidden_size - (hidden_size // self.num_scales) * (
                    self.num_scales - 1
                )

            # 目標フレーム数に応じた多層構造を作成
            branch = self._create_multi_layer_branch(
                input_size, scale_hidden_size, target_frame, dropout_rate
            )
            self.conv_branches.append(branch)

        # 特徴融合層
        self.fusion_layer = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
        )

    def _create_multi_layer_branch(
        self, input_size, output_size, target_frame, dropout_rate
    ):
        """目標フレーム数に達する多層ブランチを作成"""

        if target_frame == 20:
            # 20フレーム: K5 -> P2 -> K5 -> P2 (受容野: 1+4+2*(1+4) = 15程度)
            return nn.Sequential(
                nn.Conv1d(
                    input_size,
                    output_size,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    bias=False,
                ),
                nn.BatchNorm1d(output_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.MaxPool1d(kernel_size=2, stride=1, padding=0),  # 時間次元を少し縮小
                nn.Conv1d(
                    output_size,
                    output_size,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    bias=False,
                ),
                nn.BatchNorm1d(output_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.MaxPool1d(
                    kernel_size=2, stride=1, padding=0
                ),  # さらに時間次元を縮小
            )

        elif target_frame == 30:
            # 30フレーム: K7 -> P2 -> K7 -> P2 (より大きな受容野)
            return nn.Sequential(
                nn.Conv1d(
                    input_size,
                    output_size,
                    kernel_size=7,
                    stride=1,
                    padding=3,
                    bias=False,
                ),
                nn.BatchNorm1d(output_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.MaxPool1d(kernel_size=2, stride=1, padding=0),
                nn.Conv1d(
                    output_size,
                    output_size,
                    kernel_size=7,
                    stride=1,
                    padding=3,
                    bias=False,
                ),
                nn.BatchNorm1d(output_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.MaxPool1d(kernel_size=2, stride=1, padding=0),
            )

        elif target_frame == 40:
            # 40フレーム: K9 -> P2 -> K9 -> P2 -> K5 (最大の受容野)
            return nn.Sequential(
                nn.Conv1d(
                    input_size,
                    output_size,
                    kernel_size=9,
                    stride=1,
                    padding=4,
                    bias=False,
                ),
                nn.BatchNorm1d(output_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.MaxPool1d(kernel_size=2, stride=1, padding=0),
                nn.Conv1d(
                    output_size,
                    output_size,
                    kernel_size=9,
                    stride=1,
                    padding=4,
                    bias=False,
                ),
                nn.BatchNorm1d(output_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.MaxPool1d(kernel_size=2, stride=1, padding=0),
                nn.Conv1d(
                    output_size,
                    output_size,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    bias=False,
                ),
                nn.BatchNorm1d(output_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
            )

        else:
            # デフォルト: 汎用的な構造
            return nn.Sequential(
                nn.Conv1d(
                    input_size,
                    output_size,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    bias=False,
                ),
                nn.BatchNorm1d(output_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Conv1d(
                    output_size,
                    output_size,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    bias=False,
                ),
                nn.BatchNorm1d(output_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
            )

    def forward(self, x):
        # x: [B, C, T]
        scale_features = []

        if self.use_parallel_processing and torch.cuda.is_available():
            # GPU並列処理版
            device = x.device
            streams = [torch.cuda.Stream() for _ in range(self.num_scales)]

            # 各ブランチをCUDAストリームで並列実行
            for i, (branch, stream) in enumerate(zip(self.conv_branches, streams)):
                with torch.cuda.stream(stream):
                    scale_feat = branch(x)
                    scale_features.append(scale_feat)

            # 全てのストリームの完了を待機
            for stream in streams:
                stream.synchronize()

        else:
            # 通常の逐次処理版
            for branch in self.conv_branches:
                scale_feat = branch(x)
                scale_features.append(scale_feat)

        # 時間次元の長さを最小値に揃える
        min_T = min(feat.size(2) for feat in scale_features)
        aligned_features = []
        for feat in scale_features:
            if feat.size(2) > min_T:
                # 中央部分を切り出し
                start_idx = (feat.size(2) - min_T) // 2
                aligned_feat = feat[:, :, start_idx : start_idx + min_T]
            else:
                aligned_feat = feat
            aligned_features.append(aligned_feat)

        # チャンネル次元で結合
        fused_features = torch.cat(aligned_features, dim=1)  # [B, hidden_size, T]

        # 融合層で最終特徴を生成
        output = self.fusion_layer(fused_features)

        return output


class DualMultiScaleTemporalConv(nn.Module):
    """
    2つの入力（骨格座標と距離座標）を受け取る Multi-Scale Temporal Convolution
    DualCNNWithCTCと同様に2系統の特徴抽出を行い、MultiScaleTemporalConvの可変窓機能を組み合わせる
    """

    def __init__(
        self,
        skeleton_input_size,
        spatial_input_size,
        skeleton_hidden_size=256,
        spatial_hidden_size=256,
        fusion_hidden_size=512,
        skeleton_kernel_sizes=[20, 30, 40],
        spatial_kernel_sizes=[20, 30, 40],
        dropout_rate=0.2,
        num_classes=29,
        blank_idx=0,
        use_parallel_processing=False,
    ):
        super(DualMultiScaleTemporalConv, self).__init__()

        self.skeleton_input_size = skeleton_input_size
        self.spatial_input_size = spatial_input_size
        self.skeleton_hidden_size = skeleton_hidden_size
        self.spatial_hidden_size = spatial_hidden_size
        self.fusion_hidden_size = fusion_hidden_size
        self.num_classes = num_classes
        self.blank_id = blank_idx
        self.use_parallel_processing = use_parallel_processing

        # 骨格データ用のMultiScaleTemporalConv
        self.skeleton_multiscale = MultiScaleTemporalConv(
            input_size=skeleton_input_size,
            hidden_size=skeleton_hidden_size,
            target_frames=skeleton_kernel_sizes,
            dropout_rate=dropout_rate,
            use_parallel_processing=use_parallel_processing,
        )

        # 距離データ用のMultiScaleTemporalConv
        self.spatial_multiscale = MultiScaleTemporalConv(
            input_size=spatial_input_size,
            hidden_size=spatial_hidden_size,
            target_frames=spatial_kernel_sizes,
            dropout_rate=dropout_rate,
            use_parallel_processing=use_parallel_processing,
        )

        # 特徴融合層
        self.fusion_layer = nn.Sequential(
            nn.Conv1d(
                skeleton_hidden_size + spatial_hidden_size,
                fusion_hidden_size,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm1d(fusion_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

        # 分類層
        self.classifier = nn.Linear(fusion_hidden_size, num_classes)

        # 重みの初期化
        self._initialize_weights()

    def _initialize_weights(self):
        """重みの初期化"""
        # Classifier層の初期化
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0.5)
            # ブランクのバイアスを下げる
            with torch.no_grad():
                self.classifier.bias[self.blank_id] = -3.0

    def forward(self, skeleton_feat, spatial_feature, lgt):
        """
        順伝播処理

        Args:
            skeleton_feat: 骨格特徴量 [B, skeleton_input_size, T]
            spatial_feature: 距離特徴量 [B, spatial_input_size, T]
            lgt: 各サンプルの系列長 [B]

        Returns:
            tuple: (融合特徴量[T, B, C], 分類ロジット[T, B, num_classes], 更新長[B])
        """
        # 各入力をMultiScaleで処理
        skeleton_out = self.skeleton_multiscale(
            skeleton_feat
        )  # [B, skeleton_hidden_size, T]
        spatial_out = self.spatial_multiscale(
            spatial_feature
        )  # [B, spatial_hidden_size, T]

        # 時間次元の長さを揃える
        min_T = min(skeleton_out.size(2), spatial_out.size(2))
        skeleton_out = skeleton_out[:, :, :min_T]
        spatial_out = spatial_out[:, :, :min_T]

        # 特徴量の結合 (チャネル次元に沿って結合)
        combined_feat = torch.cat(
            [skeleton_out, spatial_out], dim=1
        )  # [B, skeleton_hidden+spatial_hidden, T]

        # 融合処理
        fused_feat = self.fusion_layer(combined_feat)  # [B, fusion_hidden_size, T]

        # 分類層適用
        # [B, C, T] -> [B, T, C] -> classifier -> [B, T, num_classes] -> [T, B, num_classes]
        logits = self.classifier(fused_feat.transpose(1, 2))  # [B, T, num_classes]
        logits = logits.transpose(0, 1)  # [T, B, num_classes]

        # 特徴量も同様に変換 [B, C, T] -> [T, B, C]
        fused_feat_output = fused_feat.transpose(0, 2).transpose(1, 2)  # [T, B, C]

        # 長さの更新（簡易版 - MultiScaleの畳み込みはsame paddingなので長さは基本的に保持される）
        updated_lgt = self.calculate_updated_lengths(lgt, skeleton_feat.size(2), min_T)

        return fused_feat_output, logits, updated_lgt

    def calculate_updated_lengths(self, input_lengths, original_T, output_T):
        """長さの更新計算"""
        if original_T == output_T:
            return input_lengths

        # 比例計算
        scale_ratio = output_T / original_T
        updated_lengths = []

        for length in input_lengths:
            updated_length = max(1, int(length.item() * scale_ratio))
            updated_lengths.append(updated_length)

        device = input_lengths.device
        return torch.tensor(updated_lengths, dtype=torch.long, device=device)


class AdaptiveTemporalConv(nn.Module):
    """
    フレーム数に応じて動的にカーネルサイズを調整するTemporal Convolution
    15, 25, 35フレームなど異なる受容野を持つ
    """

    def __init__(
        self, input_size, hidden_size, target_frames=[15, 25, 35], dropout_rate=0.2
    ):
        super(AdaptiveTemporalConv, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.target_frames = target_frames
        self.num_branches = len(target_frames)

        # 各目標フレーム数に対応するブランチを作成
        self.branches = nn.ModuleList()

        for target_frame in target_frames:
            # 目標フレーム数に達するためのカーネル設計
            branch_layers = self._design_branch(target_frame)
            self.branches.append(branch_layers)

        # 特徴融合
        branch_output_size = hidden_size // self.num_branches
        self.fusion_conv = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=1, bias=False
        )
        self.fusion_bn = nn.BatchNorm1d(hidden_size)
        self.fusion_relu = nn.ReLU(inplace=True)

    def _design_branch(self, target_frame):
        """目標フレーム数に達するブランチを設計"""
        branch_hidden = self.hidden_size // self.num_branches

        if target_frame <= 15:
            # 15フレーム: K5 -> K3 -> K3
            layers = nn.Sequential(
                nn.Conv1d(
                    self.input_size, branch_hidden, kernel_size=5, padding=2, bias=False
                ),
                nn.BatchNorm1d(branch_hidden),
                nn.ReLU(inplace=True),
                nn.Conv1d(
                    branch_hidden, branch_hidden, kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm1d(branch_hidden),
                nn.ReLU(inplace=True),
                nn.Conv1d(
                    branch_hidden, branch_hidden, kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm1d(branch_hidden),
                nn.ReLU(inplace=True),
            )
        elif target_frame <= 25:
            # 25フレーム: K5 -> P2 -> K5 -> P2 (現在の設定)
            layers = nn.Sequential(
                nn.Conv1d(
                    self.input_size, branch_hidden, kernel_size=5, padding=2, bias=False
                ),
                nn.BatchNorm1d(branch_hidden),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(
                    branch_hidden, branch_hidden, kernel_size=5, padding=2, bias=False
                ),
                nn.BatchNorm1d(branch_hidden),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2),
            )
        else:  # target_frame > 25
            # 35フレーム: K7 -> K5 -> K3 (より大きな受容野)
            layers = nn.Sequential(
                nn.Conv1d(
                    self.input_size, branch_hidden, kernel_size=7, padding=3, bias=False
                ),
                nn.BatchNorm1d(branch_hidden),
                nn.ReLU(inplace=True),
                nn.Conv1d(
                    branch_hidden, branch_hidden, kernel_size=5, padding=2, bias=False
                ),
                nn.BatchNorm1d(branch_hidden),
                nn.ReLU(inplace=True),
                nn.Conv1d(
                    branch_hidden, branch_hidden, kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm1d(branch_hidden),
                nn.ReLU(inplace=True),
            )

        return layers

    def forward(self, x):
        # x: [B, C, T]
        branch_outputs = []

        # 各ブランチで並列処理
        for branch in self.branches:
            branch_out = branch(x)
            branch_outputs.append(branch_out)

        # チャンネル次元で結合
        fused = torch.cat(branch_outputs, dim=1)  # [B, hidden_size, T]

        # 融合処理
        output = self.fusion_conv(fused)
        output = self.fusion_bn(output)
        output = self.fusion_relu(output)

        return output


class TemporalConv(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        conv_type=2,
        use_bn=False,
        num_classes=-1,
        use_multiscale=True,
        multiscale_kernels=[3, 5, 7, 9],
        target_frames=[15, 25, 35],
        use_parallel_processing=False,
    ):
        super(TemporalConv, self).__init__()
        self.use_bn = use_bn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.conv_type = conv_type
        self.use_multiscale = use_multiscale

        if self.use_multiscale:
            # Multi-Scale Temporal Convolutionを使用
            self.temporal_conv = MultiScaleTemporalConv(
                input_size=input_size,
                hidden_size=hidden_size,
                target_frames=target_frames,
                use_parallel_processing=use_parallel_processing,
            )
            # または Adaptive Temporal Convolutionを使用
            # self.temporal_conv = AdaptiveTemporalConv(
            #     input_size=input_size,
            #     hidden_size=hidden_size,
            #     target_frames=target_frames
            # )
        else:
            # 従来の固定カーネル方式
            if self.conv_type == 0:
                self.kernel_size = ["K3"]
            elif self.conv_type == 1:
                self.kernel_size = ["K5", "P2"]
            elif self.conv_type == 2:
                self.kernel_size = ["K5", "P2", "K5", "P2"]
            elif self.conv_type == 3:
                self.kernel_size = ["K5", "K5", "P2"]
            elif self.conv_type == 4:
                self.kernel_size = ["K5", "K5"]
            elif self.conv_type == 5:
                self.kernel_size = ["K5", "P2", "K5"]
            elif self.conv_type == 6:
                self.kernel_size = ["P2", "K5", "K5"]
            elif self.conv_type == 7:
                self.kernel_size = ["P2", "K7", "P2", "K7"]
            elif self.conv_type == 8:
                self.kernel_size = ["P2", "P2", "K5", "K5"]

            modules = []
            for layer_idx, ks in enumerate(self.kernel_size):
                input_sz = (
                    self.input_size
                    if layer_idx == 0
                    or self.conv_type == 6
                    and layer_idx == 1
                    or self.conv_type == 7
                    and layer_idx == 1
                    or self.conv_type == 8
                    and layer_idx == 2
                    else self.hidden_size
                )
                if ks[0] == "P":
                    modules.append(
                        nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False)
                    )
                elif ks[0] == "K":
                    modules.append(
                        nn.Conv1d(
                            input_sz,
                            self.hidden_size,
                            kernel_size=int(ks[1]),
                            stride=1,
                            padding=0,
                        )
                    )
                    modules.append(nn.BatchNorm1d(self.hidden_size))
                    modules.append(nn.ReLU(inplace=True))
            self.temporal_conv = nn.Sequential(*modules)

        if self.num_classes != -1:
            self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def update_lgt(self, lgt):
        if self.use_multiscale:
            # Multi-scaleの場合、paddingを使用しているので長さは変わらない
            return lgt
        else:
            # 従来の方法での長さ更新
            feat_len = copy.deepcopy(lgt)
            for ks in self.kernel_size:
                if ks[0] == "P":
                    feat_len = torch.div(feat_len, 2)
                else:
                    feat_len -= int(ks[1]) - 1
            return feat_len

    def forward(self, frame_feat, lgt):
        visual_feat = self.temporal_conv(frame_feat)
        lgt = self.update_lgt(lgt)

        if self.num_classes != -1:
            if self.use_multiscale:
                # Multi-scaleの場合、visual_featは既に[B, C, T]形式
                logits = self.fc(visual_feat.transpose(1, 2)).transpose(1, 2)
            else:
                # 従来の方法
                logits = self.fc(visual_feat.transpose(1, 2)).transpose(1, 2)
        else:
            logits = None

        return {
            "visual_feat": visual_feat.permute(2, 0, 1),
            "conv_logits": logits.permute(2, 0, 1) if logits is not None else None,
            "feat_len": lgt.cpu(),
        }


class DualFeatureTemporalConv(nn.Module):
    """
    骨格座標と手の空間的特徴を別々の畳み込み層で処理するモデル

    Args:
        skeleton_input_size: 骨格座標の入力チャンネル数 (C * J)
        hand_feature_size: 手の空間的特徴の入力チャンネル数
        skeleton_hidden_size: 骨格特徴の隠れ層サイズ
        hand_hidden_size: 手の特徴の隠れ層サイズ
        fusion_hidden_size: 融合後の隠れ層サイズ
        conv_type: 畳み込みタイプ (0-8)
        use_bn: バッチ正規化使用フラグ
        num_classes: 分類クラス数 (-1の場合は特徴抽出のみ)
    """

    def __init__(
        self,
        skeleton_input_size,
        hand_feature_size,
        skeleton_hidden_size=128,
        hand_hidden_size=64,
        fusion_hidden_size=192,  # 結合後の特徴量サイズ (デフォルトは両方の和)
        conv_type=2,
        use_bn=True,
        num_classes=-1,
    ):
        super(DualFeatureTemporalConv, self).__init__()
        self.use_bn = use_bn
        self.skeleton_input_size = skeleton_input_size
        self.hand_feature_size = hand_feature_size
        self.skeleton_hidden_size = skeleton_hidden_size
        self.hand_hidden_size = hand_hidden_size
        self.fusion_hidden_size = fusion_hidden_size
        self.num_classes = num_classes
        self.conv_type = conv_type

        # 骨格座標用の時間的畳み込み層
        self.skeleton_conv = self._create_temporal_conv(
            self.skeleton_input_size, self.skeleton_hidden_size
        )

        # 手の特徴量用の時間的畳み込み層
        self.hand_conv = self._create_temporal_conv(
            self.hand_feature_size, self.hand_hidden_size
        )

        # 特徴量融合後の処理層
        # 融合された特徴量に対する追加の変換層
        self.fusion_layer = nn.Sequential(
            nn.Conv1d(
                self.skeleton_hidden_size + self.hand_hidden_size,
                self.fusion_hidden_size,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm1d(self.fusion_hidden_size),
            nn.ReLU(inplace=True),
        )

        # 分類層 (num_classes > 0 の場合)
        if self.num_classes != -1:
            self.fc = nn.Linear(self.fusion_hidden_size, self.num_classes)

    def _create_temporal_conv(self, input_size, hidden_size):
        """畳み込み層の作成関数"""
        if self.conv_type == 0:
            kernel_size = ["K3"]
        elif self.conv_type == 1:
            kernel_size = ["K5", "P2"]
        elif self.conv_type == 2:
            kernel_size = ["K5", "P2", "K5", "P2"]
        elif self.conv_type == 3:
            kernel_size = ["K5", "K5", "P2"]
        elif self.conv_type == 4:
            kernel_size = ["K5", "K5"]
        elif self.conv_type == 5:
            kernel_size = ["K5", "P2", "K5"]
        elif self.conv_type == 6:
            kernel_size = ["P2", "K5", "K5"]
        elif self.conv_type == 7:
            kernel_size = ["P2", "K7", "P2", "K7"]
        elif self.conv_type == 8:
            kernel_size = ["P2", "P2", "K5", "K5"]

        modules = []
        for layer_idx, ks in enumerate(kernel_size):
            input_sz = (
                input_size
                if layer_idx == 0
                or self.conv_type == 6
                and layer_idx == 1
                or self.conv_type == 7
                and layer_idx == 1
                or self.conv_type == 8
                and layer_idx == 2
                else hidden_size
            )
            if ks[0] == "P":
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == "K":
                modules.append(
                    nn.Conv1d(
                        input_sz,
                        hidden_size,
                        kernel_size=int(ks[1]),
                        stride=1,
                        padding=0,
                    )
                )
                modules.append(nn.BatchNorm1d(hidden_size))
                modules.append(nn.ReLU(inplace=True))

        return nn.Sequential(*modules)

    def update_lgt(self, lgt):
        """系列長の更新"""
        feat_len = copy.deepcopy(lgt)
        for ks in (
            self.kernel_size
            if hasattr(self, "kernel_size")
            else (
                self.conv_type == 0
                and ["K3"]
                or self.conv_type == 1
                and ["K5", "P2"]
                or self.conv_type == 2
                and ["K5", "P2", "K5", "P2"]
                or self.conv_type == 3
                and ["K5", "K5", "P2"]
                or self.conv_type == 4
                and ["K5", "K5"]
                or self.conv_type == 5
                and ["K5", "P2", "K5"]
                or self.conv_type == 6
                and ["P2", "K5", "K5"]
                or self.conv_type == 7
                and ["P2", "K7", "P2", "K7"]
                or ["P2", "P2", "K5", "K5"]
            )
        ):
            if ks[0] == "P":
                feat_len = torch.div(feat_len, 2, rounding_mode="floor")
            else:
                feat_len -= int(ks[1]) - 1
        return feat_len

    def forward(self, skeleton_feat, hand_feat, lgt):
        """
        順伝播処理

        Args:
            skeleton_feat: 骨格特徴量 [batch_size, C*J, T]
            hand_feat: 手の特徴量 [batch_size, hand_feature_size, T]
            lgt: 各サンプルの系列長

        Returns:
            dict: 処理結果の辞書
        """
        # 骨格特徴の処理
        skeleton_visual_feat = self.skeleton_conv(skeleton_feat)

        # 手の特徴の処理
        hand_visual_feat = self.hand_conv(hand_feat)

        # 系列長の更新
        updated_lgt = self.update_lgt(lgt)

        # パディングが必要な場合は長さを揃える
        if skeleton_visual_feat.size(2) != hand_visual_feat.size(2):
            # より短い方を長い方に合わせる
            if skeleton_visual_feat.size(2) < hand_visual_feat.size(2):
                # パディングの計算
                pad_size = hand_visual_feat.size(2) - skeleton_visual_feat.size(2)
                skeleton_visual_feat = F.pad(skeleton_visual_feat, (0, pad_size))
            else:
                pad_size = skeleton_visual_feat.size(2) - hand_visual_feat.size(2)
                hand_visual_feat = F.pad(hand_visual_feat, (0, pad_size))

        # 特徴量の結合 (チャネル次元に沿って結合)
        combined_feat = torch.cat([skeleton_visual_feat, hand_visual_feat], dim=1)

        # 融合処理
        fused_feat = self.fusion_layer(combined_feat)

        # クラス分類 (必要な場合)
        if self.num_classes != -1:
            # [B, C, T] -> [B, T, C] -> 適用 -> [B, T, num_classes] -> [B, num_classes, T]
            logits = self.fc(fused_feat.transpose(1, 2)).transpose(1, 2)
        else:
            logits = None

        return {
            "visual_feat": fused_feat.permute(2, 0, 1),  # [T, B, C]
            "conv_logits": (
                None if logits is None else logits.permute(2, 0, 1)
            ),  # [T, B, num_classes]
            "feat_len": updated_lgt.cpu(),
            # 個別の特徴量も返すと分析に便利
            "skeleton_feat": skeleton_visual_feat.permute(
                2, 0, 1
            ),  # [T, B, C_skeleton]
            "hand_feat": hand_visual_feat.permute(2, 0, 1),  # [T, B, C_hand]
        }

    def __str__(self):
        """モデルの構造を表示するための文字列表現"""
        return (
            f"DualFeatureTemporalConv(\n"
            f"  skeleton_input: {self.skeleton_input_size}, skeleton_hidden: {self.skeleton_hidden_size}\n"
            f"  hand_input: {self.hand_feature_size}, hand_hidden: {self.hand_hidden_size}\n"
            f"  fusion_hidden: {self.fusion_hidden_size}, num_classes: {self.num_classes}\n"
            f"  conv_type: {self.conv_type}\n"
            f")"
        )


class DualCNNWithCTC(nn.Module):
    """
    骨格座標と手の特徴量のための二つの1D-CNNを使用し、
    CTC損失関数とビームサーチによる評価を行うモデル

    CorrNetを参考にした実装
    """

    def __init__(
        self,
        skeleton_input_size,
        hand_feature_size,
        skeleton_hidden_size=128,
        hand_hidden_size=64,
        fusion_hidden_size=192,
        dropout_rate=0.2,
        conv_type=2,
        num_classes=64,
        blank_idx=0,
    ):
        super(DualCNNWithCTC, self).__init__()

        self.blank_id = blank_idx
        self.num_classes = num_classes

        # 骨格データと手の特徴量のための二つの1D-CNN
        self.dual_feature_cnn = DualFeatureTemporalConv(
            skeleton_input_size=skeleton_input_size,
            hand_feature_size=hand_feature_size,
            skeleton_hidden_size=skeleton_hidden_size,
            hand_hidden_size=hand_hidden_size,
            fusion_hidden_size=fusion_hidden_size,
            conv_type=conv_type,
            use_bn=True,
            num_classes=-1,  # 特徴抽出のみを行う
        )

        # 融合特徴量からクラス予測を行う分類層
        self.classifier = nn.Linear(fusion_hidden_size, num_classes)

        # ログソフトマックス（CTC損失用）
        self.log_softmax = nn.LogSoftmax(dim=-1)

        # CTC損失関数
        # self.ctc_loss = nn.CTCLoss(
        #     blank=blank_idx, zero_infinity=True, reduction="mean"
        # )

        # モデルの初期化
        self._initialize_weights()

    def _initialize_weights(self):
        """
        重みの初期化を改良した関数
        """
        # Classifier層の初期化 - より均一な分布を目指す
        nn.init.xavier_uniform_(
            self.classifier.weight, gain=0.01
        )  # gainを小さくして初期値を抑制

        if self.classifier.bias is not None:
            # バイアスは最初はゼロに近い値に設定
            nn.init.constant_(self.classifier.bias, 0)

            # ブランク以外のクラスにわずかに正のバイアスを与える
            with torch.no_grad():
                for i in range(self.classifier.bias.size(0)):
                    if i != self.blank_id:  # ブランク以外のクラス
                        self.classifier.bias[i] += 0.1  # 小さな正のバイアス

    def forward(
        self,
        skeleton_feat,
        hand_feat,
        lgt,
        tgt_feature=None,
        target_lengths=None,
        mode="train",
        blank_id=None,
        current_epoch=None,
    ):
        """
        順伝播処理

        Args:
            skeleton_feat: 骨格特徴量 [batch_size, C*J, T]
            hand_feat: 手の特徴量 [batch_size, hand_feature_size, T]
            lgt: 各サンプルの系列長 [batch_size]
            tgt_feature: ターゲットラベル [batch, max_target_length]
            target_lengths: ターゲットの長さ [batch]
            mode: 'train', 'eval', 'test'のいずれか
            blank_id: ブランクインデックス（指定なしの場合はself.blank_id）
            current_epoch: 現在のエポック番号（ビームサーチのパラメータ調整用）

        Returns:
            hidden features and logits
        """
        # blank_idが指定されていない場合はクラス変数を使用
        if blank_id is None:
            blank_id = self.blank_id

        # 1D-CNN部分の処理（特徴抽出）
        cnn_output = self.dual_feature_cnn(skeleton_feat, hand_feat, lgt)

        # 融合された特徴量を取得 [T, B, C]
        fused_features = cnn_output["visual_feat"]

        # 特徴長の更新
        updated_lgt = cnn_output["feat_len"]

        # 分類層で各フレームのクラス予測を行う
        # [T, B, C] -> [T, B, num_classes]
        logits = self.classifier(fused_features)

        # # ログソフトマックスを適用
        # log_probs = self.log_softmax(logits)  # [T, B, num_classes]

        # 隠れ層の特徴量と出力のみを返す
        return fused_features, logits, updated_lgt
