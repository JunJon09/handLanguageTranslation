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
    def __init__(self, input_size, hidden_size, conv_type=2, use_bn=False, num_classes=-1):
        super(TemporalConv, self).__init__()
        self.use_bn = use_bn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.conv_type = conv_type

        if self.conv_type == 0:
            self.kernel_size = ['K3']
        elif self.conv_type == 1:
            self.kernel_size = ['K5', "P2"]
        elif self.conv_type == 2:
            self.kernel_size = ['K5', "P2", 'K5', "P2"]
        elif self.conv_type == 3:
            self.kernel_size = ['K5', 'K5', "P2"]
        elif self.conv_type == 4:
            self.kernel_size = ['K5', 'K5']
        elif self.conv_type == 5:
            self.kernel_size = ['K5', "P2", 'K5']
        elif self.conv_type == 6:
            self.kernel_size = ["P2", 'K5', 'K5']
        elif self.conv_type == 7:
            self.kernel_size = ["P2", 'K5', "P2", 'K5']
        elif self.conv_type == 8:
            self.kernel_size = ["P2", "P2", 'K5', 'K5']

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 or self.conv_type == 6 and layer_idx == 1 or self.conv_type == 7 and layer_idx == 1 or self.conv_type == 8 and layer_idx == 2 else self.hidden_size
            if ks[0] == 'P':
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                modules.append(
                    nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0)
                    #MultiScale_TemporalConv(input_sz, self.hidden_size)
                )
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*modules)

        if self.num_classes != -1:
            self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def update_lgt(self, lgt):
        feat_len = copy.deepcopy(lgt)
        for ks in self.kernel_size:
            if ks[0] == 'P':
                feat_len = torch.div(feat_len, 2)
            else:
                feat_len -= int(ks[1]) - 1
                #pass
        return feat_len

    def forward(self, frame_feat, lgt):
        visual_feat = self.temporal_conv(frame_feat)
        lgt = self.update_lgt(lgt)
        logits = None if self.num_classes == -1 \
            else self.fc(visual_feat.transpose(1, 2)).transpose(1, 2)
        return {
            "visual_feat": visual_feat.permute(2, 0, 1),
            "conv_logits": logits.permute(2, 0, 1),
            "feat_len": lgt.cpu(),
        }


# テスト用コード
if __name__ == "__main__":
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # モデルパラメータ
    input_size = 75  # 入力特徴量のサイズ
    hidden_size = 128  # 隠れ層のサイズ
    conv_type = 7  # 畳み込みタイプ
    
    # サンプルデータの作成
    batch_size = 16
    seq_length = 100  # シーケンス長
    
    # 入力テンソル作成 [batch_size, input_size, seq_length]
    input_tensor = torch.randn(batch_size, input_size, seq_length).to(device)
    
    # 系列長の作成（バッチ内の各サンプルのシーケンス長）
    lengths = torch.full((batch_size,), seq_length, dtype=torch.long).to(device)
    
    # モデルのインスタンス化と設定
    model = TemporalConv(
        input_size=input_size,
        hidden_size=hidden_size,
        conv_type=conv_type,
        use_bn=True,
        num_classes=64  # 分類するクラス数（-1の場合は特徴抽出のみ）
    ).to(device)
    
    print(f"Model structure: {model}")
    
    # 入力データの形状確認
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Initial lengths: {lengths}")
    
    # モデルを評価モードに設定
    model.eval()
    
    # 推論実行
    with torch.no_grad():
        output = model(input_tensor, lengths)
    
    # 出力結果の表示
    print("\nOutput:")
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            print(f"{key} shape: {value.shape}")
        else:
            print(f"{key}: {value}")
    
    # 更新された系列長の確認
    updated_lengths = output["feat_len"]
    print(f"Updated lengths: {updated_lengths}")
    print("conv_logits: ", output["conv_logits"].shape)
    
    # メモリ情報（GPUを使用している場合）
    if torch.cuda.is_available():
        print(f"\nGPU Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU Memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")


# # 使用例
# if __name__ == "__main__":
#     # 入力テンソルの作成
#     N, C, T, J = 32, 3, 100, 25  # 例
#     input_tensor = torch.randn(N, C, T, J)  # 形状: [32, 3, 100, 25]

#     # 入力テンソルの形状を [N, C * J, T] に変換
#     input_tensor = input_tensor.permute(0, 3, 1, 2).contiguous().view(N, C * J, T)  # [32, 75, 100]

#     # モデルのインスタンス化
#     num_classes = 100  # 例: 100クラス分類
#     in_channels = C * J  # 3 * 25 = 75
#     out_channels = 64  # 出力チャンネル数

#     # RestNet系
#     model = resnet18_1d(num_classes=num_classes, in_channels=in_channels, kernel_size=3, stride=1 , padding=0, bias=False)
#     output = model(input_tensor)
#     print(output.shape, input_tensor.shape)  # 期待される形状: [batch_size, out_channels, new_sequence_length]

#     # シンプルなCNNモデルのインスタンス化
#     model1 = create_simple_cnn1layer(in_channels, out_channels, kernel_size=25, stride=1, padding=1, dropout_rate=0.2, bias=False)
#     output1 = model1(input_tensor)
#     print(f"SimpleCNN1Layer output shape: {output1.shape}, ", input_tensor.shape)  # 期待される形状: [32, out_channels, new_sequence_length]

#      # 2層CNN
#     model2 = create_simple_cnn2layer(in_channels=in_channels, mid_channels=32, out_channels=64)
#     output2 = model2(input_tensor)
#     print("2層CNN出力形状:", output2.shape)

#     # プーリング付き2層CNN
#     model3 = create_simple_cnn2layer_with_pooling(in_channels=in_channels)
#     output3 = model3(input_tensor)
#     print("プーリング付き2層CNN出力形状:", output3.shape)

#     # 残差接続付き1層CNN
#     model4 = create_simple_cnn1layer_with_residual(in_channels=in_channels, out_channels=out_channels)
#     output4 = model4(input_tensor)
#     print("残差接続付き1層CNN出力形状:", output4.shape)
