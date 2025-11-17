import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


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
        print(target_frame, "***************")
        if target_frame == 10:
            # 10フレーム: K3 -> K3 (小さな受容野)
            return nn.Sequential(
                nn.Conv1d(
                    input_size,
                    output_size,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm1d(output_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Conv1d(
                    output_size,
                    output_size,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm1d(output_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
            )

        elif target_frame == 15:
            # 15フレーム: K3 -> P2 -> K5 (中小受容野)
            return nn.Sequential(
                nn.Conv1d(
                    input_size,
                    output_size,
                    kernel_size=3,
                    stride=1,
                    padding=1,
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

        elif target_frame == 20:
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

        elif target_frame == 25:
            # 25フレーム: K5 -> P2 -> K7 (中程度の受容野)
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

        elif target_frame == 35:
            # 35フレーム: K7 -> P2 -> K9 -> K5 (大きな受容野)
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
                    kernel_size=9,
                    stride=1,
                    padding=4,
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
        skeleton_kernel_sizes=[10, 15, 20, 25, 30],
        spatial_kernel_sizes=[10, 15, 20, 25, 30],
        dropout_rate=0.2,
        num_classes=29,
        blank_id=0,
        use_parallel_processing=False,
    ):
        super(DualMultiScaleTemporalConv, self).__init__()

        self.skeleton_input_size = skeleton_input_size
        self.spatial_input_size = spatial_input_size
        self.skeleton_hidden_size = skeleton_hidden_size
        self.spatial_hidden_size = spatial_hidden_size
        self.fusion_hidden_size = fusion_hidden_size
        self.num_classes = num_classes
        self.blank_id = blank_id
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
        dropout_rate=0.2,
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
        self.dropout_rate = dropout_rate

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
            nn.Dropout1d(p=self.dropout_rate),  # CNNの特徴マップにDropout1dを適用
        )

        

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
                # 畳み込み層の後にDropout1dを追加
                modules.append(nn.Dropout1d(p=self.dropout_rate))

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


        return {
            "visual_feat": fused_feat.permute(2, 0, 1),  # [T, B, C]
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
        use_bn=True,
        num_classes=64,
        blank_id=0,
    ):
        super(DualCNNWithCTC, self).__init__()

        self.blank_id = blank_id
        self.num_classes = num_classes

        # 骨格データと手の特徴量のための二つの1D-CNN
        self.dual_feature_cnn = DualFeatureTemporalConv(
            skeleton_input_size=skeleton_input_size,
            hand_feature_size=hand_feature_size,
            skeleton_hidden_size=skeleton_hidden_size,
            hand_hidden_size=hand_hidden_size,
            fusion_hidden_size=fusion_hidden_size,
            conv_type=conv_type,
            use_bn=use_bn,
            num_classes=-1,  # 特徴抽出のみを行う
            dropout_rate=dropout_rate,
        )

        # Dropout層を追加 - 特徴抽出後の正則化
        self.dropout = nn.Dropout(p=dropout_rate)

        # 融合特徴量からクラス予測を行う分類層
        self.classifier = nn.Linear(fusion_hidden_size, num_classes)

        # ログソフトマックス（CTC損失用）
        self.log_softmax = nn.LogSoftmax(dim=-1)

        # モデルの初期化
        self._initialize_weights()

    def _initialize_weights(self):
        """
        重みの初期化を改良した関数 - logits値の安定化
        """

        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            # バイアスは最初は小さな負の値に設定
            nn.init.constant_(self.classifier.bias, 0.1)

            # ブランクトークンのバイアスを他より少し低くする
            with torch.no_grad():
                self.classifier.bias[self.blank_id] = -2.0  # ブランクを抑制
                

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

        cnn_output = self.dual_feature_cnn(skeleton_feat, hand_feat, lgt)

        # 融合された特徴量を取得 [T, B, C]
        fused_features = cnn_output["visual_feat"]

        # 特徴長の更新
        updated_lgt = cnn_output["feat_len"]

        # Dropoutを適用してから分類層へ
        # [T, B, C] -> Dropout -> [T, B, C]
        dropped_features = self.dropout(fused_features)

        # 分類層で各フレームのクラス予測を行う
        # [T, B, C] -> [T, B, num_classes]
        logits = self.classifier(dropped_features)
        
        # 適応的logits制御：訓練の進行に応じて制限を緩和
        with torch.no_grad():
            logits_std = logits.std()
            logits_max_abs = logits.abs().max()
            
        # 標準偏差が大きすぎる場合のみソフトクリッピング適用
        if logits_std > 3.0 or logits_max_abs > 8.0:
            # 動的な制限範囲（最大値に応じて調整）
            clip_range = min(8.0, max(5.0, logits_max_abs.item() * 0.8))
            logits = clip_range * torch.tanh(logits / clip_range)
        
        # 軽微な正規化（分布を少し平滑化）
        logits = logits * 0.95  # 5%の縮小で過信を防ぐ


        # 隠れ層の特徴量と出力のみを返す
        return fused_features, logits, updated_lgt
