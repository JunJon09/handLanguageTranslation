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


class TemporalConv(nn.Module):
    def __init__(
        self, input_size, hidden_size, conv_type=2, use_bn=False, num_classes=-1
    ):
        super(TemporalConv, self).__init__()
        self.use_bn = use_bn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.conv_type = conv_type

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
            self.kernel_size = ["P2", "K5", "P2", "K5"]
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
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == "K":
                modules.append(
                    nn.Conv1d(
                        input_sz,
                        self.hidden_size,
                        kernel_size=int(ks[1]),
                        stride=1,
                        padding=0,
                    )
                    # MultiScale_TemporalConv(input_sz, self.hidden_size)
                )
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*modules)

        if self.num_classes != -1:
            self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def update_lgt(self, lgt):
        feat_len = copy.deepcopy(lgt)
        for ks in self.kernel_size:
            if ks[0] == "P":
                feat_len = torch.div(feat_len, 2)
            else:
                feat_len -= int(ks[1]) - 1
                # pass
        return feat_len

    def forward(self, frame_feat, lgt):
        visual_feat = self.temporal_conv(frame_feat)
        lgt = self.update_lgt(lgt)
        logits = (
            None
            if self.num_classes == -1
            else self.fc(visual_feat.transpose(1, 2)).transpose(1, 2)
        )
        return {
            "visual_feat": visual_feat.permute(2, 0, 1),
            "conv_logits": logits.permute(2, 0, 1),
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
            kernel_size = ["P2", "K5", "P2", "K5"]
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
                and ["P2", "K5", "P2", "K5"]
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

        # 以下のCTC関連処理をコメントアウト
        """
        if mode == "train":
            # CTC損失を計算
            loss = self.ctc_loss(
                log_probs,  # [T, B, num_classes]
                tgt_feature,  # [batch, max_target_length]
                updated_lgt,  # [batch]
                target_lengths,  # [batch]
            )
            return loss, log_probs

        elif mode == "eval":
            # 評価モード: 損失計算と復号化
            loss = self.ctc_loss(
                log_probs,
                tgt_feature,
                updated_lgt,
                target_lengths,
            )
            # ビームサーチによる復号化
            from one_dcnn_transformer_encoder.models.beam_search import (
                beam_search_decode,
            )

            decoded_sequences = beam_search_decode(
                log_probs,
                beam_width=10,
                blank_id=self.blank_id,
                current_epoch=current_epoch,
            )
            return loss, decoded_sequences

        elif mode == "test":
            # テストモード: 復号化のみ
            from one_dcnn_transformer_encoder.models.beam_search import (
                beam_search_decode,
            )

            decoded_sequences = beam_search_decode(
                log_probs,
                beam_width=10,
                blank_id=self.blank_id,
                current_epoch=current_epoch,
            )
            return decoded_sequences
        """

    # 残りのメソッドもコメントアウト
    """
    def evaluate_individual_features(
        self,
        skeleton_feat=None,
        hand_feat=None,
        lgt=None,
        tgt_feature=None,
        target_lengths=None,
        mode="eval",
    ):
        \"""
        個別の特徴量を評価するためのメソッド（アブレーション実験用）

        Args:
            skeleton_feat: 骨格特徴量 [batch_size, C*J, T]（Noneの場合は手の特徴量のみ使用）
            hand_feat: 手の特徴量 [batch_size, hand_feature_size, T]（Noneの場合は骨格特徴量のみ使用）
            lgt: 各サンプルの系列長 [batch_size]
            tgt_feature: ターゲットラベル [batch, max_target_length]
            target_lengths: ターゲットの長さ [batch]
            mode: 'eval' または 'test'

        Returns:
            mode='eval': (loss, decoded_sequences)
            mode='test': decoded_sequences
        \"""
        batch_size = lgt.size(0)
        device = lgt.device
        seq_length = None

        # 骨格データまたは手の特徴量がNoneの場合、ゼロテンソルで代用
        if skeleton_feat is None and hand_feat is not None:
            # 手の特徴量のみを使用
            seq_length = hand_feat.size(2)
            skeleton_feat = torch.zeros(
                batch_size,
                self.dual_feature_cnn.skeleton_input_size,
                seq_length,
                device=device,
            )

        elif hand_feat is None and skeleton_feat is not None:
            # 骨格データのみを使用
            seq_length = skeleton_feat.size(2)
            hand_feat = torch.zeros(
                batch_size,
                self.dual_feature_cnn.hand_feature_size,
                seq_length,
                device=device,
            )

        # 通常の順伝播処理を実行
        if mode == "eval":
            return self.forward(
                skeleton_feat, hand_feat, lgt, tgt_feature, target_lengths, mode=mode
            )
        else:
            return self.forward(skeleton_feat, hand_feat, lgt, mode=mode)

    def ablation_study(
        self,
        skeleton_feat,
        hand_feat,
        lgt,
        tgt_feature,
        target_lengths,
    ):
        \"""
        アブレーション実験（特徴量の寄与度分析）を行う

        Args:
            skeleton_feat: 骨格特徴量 [batch_size, C*J, T]
            hand_feat: 手の特徴量 [batch_size, hand_feature_size, T]
            lgt: 各サンプルの系列長 [batch_size]
            tgt_feature: ターゲットラベル [batch, max_target_length]
            target_lengths: ターゲットの長さ [batch]

        Returns:
            dict: 3種類の実験結果（骨格のみ、手のみ、両方）
        \"""
        # 骨格特徴量のみ
        skel_loss, skel_decoded = self.evaluate_individual_features(
            skeleton_feat=skeleton_feat,
            hand_feat=None,
            lgt=lgt,
            tgt_feature=tgt_feature,
            target_lengths=target_lengths,
        )

        # 手の特徴量のみ
        hand_loss, hand_decoded = self.evaluate_individual_features(
            skeleton_feat=None,
            hand_feat=hand_feat,
            lgt=lgt,
            tgt_feature=tgt_feature,
            target_lengths=target_lengths,
        )

        # 両方の特徴量
        both_loss, both_decoded = self.forward(
            skeleton_feat=skeleton_feat,
            hand_feat=hand_feat,
            lgt=lgt,
            tgt_feature=tgt_feature,
            target_lengths=target_lengths,
            mode="eval",
        )

        return {
            "skeleton_only": {
                "loss": skel_loss.item(),
                "decoded": skel_decoded,
            },
            "hand_only": {
                "loss": hand_loss.item(),
                "decoded": hand_decoded,
            },
            "both_features": {
                "loss": both_loss.item(),
                "decoded": both_decoded,
            },
        }
    """


# # テスト用コード
if __name__ == "__main__":
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # モデルパラメータ
    skeleton_input_size = 75  # 骨格座標の入力サイズ (C*J, 例えば 3*25)
    hand_feature_size = 20  # 手の特徴量の入力サイズ
    num_classes = 64  # 分類クラス数

    # サンプルデータの作成
    batch_size = 16
    seq_length = 100  # シーケンス長
    max_target_length = 20  # ターゲットの最大長

    # 骨格データの作成 [batch_size, skeleton_input_size, seq_length]
    skeleton_tensor = torch.randn(batch_size, skeleton_input_size, seq_length).to(
        device
    )

    # 手の特徴量データの作成 [batch_size, hand_feature_size, seq_length]
    hand_tensor = torch.randn(batch_size, hand_feature_size, seq_length).to(device)

    # 系列長の作成（入力系列長）
    lengths = torch.full((batch_size,), seq_length, dtype=torch.long).to(device)
    # ダミーのターゲット系列を作成（0からnum_classes-1の整数）
    target_tensor = torch.randint(
        1, num_classes, (batch_size, max_target_length), dtype=torch.long
    ).to(device)

    # ターゲット長（実際のラベル長）- 最大長よりも短いランダムな長さ
    target_lengths = torch.randint(
        5, max_target_length + 1, (batch_size,), dtype=torch.long
    ).to(device)

    # モデル初期化
    model = DualCNNWithCTC(
        skeleton_input_size=skeleton_input_size,
        hand_feature_size=hand_feature_size,
        skeleton_hidden_size=128,
        hand_hidden_size=64,
        fusion_hidden_size=192,
        num_classes=num_classes,
        blank_idx=0,
    ).to(device)

    # トレーニングモードでテスト
    model.train()
    fused_features, log_probs, updated_lgt = model(
        skeleton_tensor,
        hand_tensor,
        lengths,
        target_tensor,
        target_lengths,
        mode="train",
    )
    print(f"Fused features shape: {fused_features.shape}")
    print(f"Log probs shape: {log_probs.shape}")
    print(f"Updated lengths: {updated_lgt}")
    print(f"Fused features: {fused_features[0]}")
    print(skeleton_tensor.shape)
    # print(f"\n===== トレーニングモード =====")
    # print(f"Loss: {loss.item():.4f}")
    # print(f"Log probs shape: {log_probs.shape}")

    # # 評価モードでテスト
    # model.eval()
    # with torch.no_grad():
    #     eval_loss, decoded_seqs = model(
    #         skeleton_tensor,
    #         hand_tensor,
    #         lengths,
    #         target_tensor,
    #         target_lengths,
    #         mode="eval",
    #     )
    #     print(f"\n===== 評価モード =====")
    #     print(f"Eval Loss: {eval_loss.item():.4f}")
    #     print(f"デコード結果サンプル (バッチ0): {decoded_seqs[0]}")
    #     print(target_tensor[0], target_tensor.shape)
    #     # アブレーション実験
    #     ablation_results = model.ablation_study(
    #         skeleton_tensor, hand_tensor, lengths, target_tensor, target_lengths
    #     )

    #     print("\n===== アブレーション実験 =====")
    #     print(f"骨格のみ - 損失: {ablation_results['skeleton_only']['loss']:.4f}")
    #     print(f"手のみ - 損失: {ablation_results['hand_only']['loss']:.4f}")
    #     print(f"両方 - 損失: {ablation_results['both_features']['loss']:.4f}")

    #     # テストモードでテスト
    #     test_decoded = model(skeleton_tensor, hand_tensor, lengths, mode="test")
    #     print(f"\n===== テストモード =====")
    #     print(f"デコード結果サンプル (バッチ0): {test_decoded[0]}")
    # print(f"Lengths: {lengths}")
