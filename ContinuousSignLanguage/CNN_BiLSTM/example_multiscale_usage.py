"""
Multi-Scale Temporal Convolutionの使用例
異なるフレーム範囲（15, 25, 35フレーム）を同時に処理する方法
"""

import torch
from models.one_dcnn import TemporalConv, MultiScaleTemporalConv, AdaptiveTemporalConv, DualMultiScaleTemporalConv


def example_multiscale_temporal_conv():
    batch_size = 2
    skeleton_channels = 75  # 25関節 * 3次元
    spatial_channels = 50   # 距離特徴量
    time_steps = 100

    skeleton_feat = torch.randn(batch_size, skeleton_channels, time_steps)
    spatial_feat = torch.randn(batch_size, spatial_channels, time_steps)
    lengths = torch.tensor([90, 80])

    # モデル作成とテスト
    model = DualMultiScaleTemporalConv(
        skeleton_input_size=skeleton_channels,
        spatial_input_size=spatial_channels,
        skeleton_hidden_size=256,
        spatial_hidden_size=256,
        fusion_hidden_size=512,
        num_classes=29
    )

    print('モデル作成成功!')
    print(f'入力形状: skeleton={skeleton_feat.shape}, spatial={spatial_feat.shape}')

    # 順伝播テスト
    with torch.no_grad():
        fused_feat, logits, updated_lengths = model(skeleton_feat, spatial_feat, lengths)
        print(f'出力形状: fused_feat={fused_feat.shape}, logits={logits.shape}')
        print(f'更新された長さ: {updated_lengths}')
        print('テスト成功!')

    """Multi-Scale Temporal Convolutionの使用例"""

    # パラメータ設定
    batch_size = 8
    input_channels = 1662  # MediaPipeの座標数
    sequence_length = 100
    hidden_size = 128

    # 入力データ（ダミー）
    input_data = torch.randn(batch_size, input_channels, sequence_length)
    sequence_lengths = torch.tensor([90, 80, 100, 75, 85, 95, 70, 88])

    print("=== Multi-Scale Temporal Convolution ===")

    # 方法1: MultiScaleTemporalConvを直接使用
    print("\n1. MultiScaleTemporalConv (並列カーネル)")
    multiscale_conv = MultiScaleTemporalConv(
        input_size=input_channels,
        hidden_size=hidden_size,
        kernel_sizes=[3, 5, 7, 9],  # 異なるカーネルサイズを並列処理
        dropout_rate=0.2,
    )

    output1 = multiscale_conv(input_data)
    print(f"入力形状: {input_data.shape}")
    print(f"出力形状: {output1.shape}")
    print(f"受容野: カーネル3,5,7,9を並列処理 → 様々な時間スケールを捉える")

    # 方法2: AdaptiveTemporalConvを使用
    print("\n2. AdaptiveTemporalConv (目標フレーム指定)")
    adaptive_conv = AdaptiveTemporalConv(
        input_size=input_channels,
        hidden_size=hidden_size,
        target_frames=[15, 25, 35],  # 目標とする受容野フレーム数
        dropout_rate=0.2,
    )

    output2 = adaptive_conv(input_data)
    print(f"入力形状: {input_data.shape}")
    print(f"出力形状: {output2.shape}")
    print(f"受容野: 15,25,35フレームを見る3つのブランチを並列処理")

    # 方法3: TemporalConvでmulti-scaleを有効化
    print("\n3. TemporalConv (multi-scale有効)")
    temporal_conv = TemporalConv(
        input_size=input_channels,
        hidden_size=hidden_size,
        conv_type=2,  # 従来の設定も保持
        num_classes=64,
        use_multiscale=True,  # Multi-scaleを有効化
        multiscale_kernels=[3, 5, 7, 9],  # カーネルサイズ指定
        target_frames=[15, 25, 35],  # または目標フレーム指定
    )

    output3 = temporal_conv(input_data, sequence_lengths)
    print(f"入力形状: {input_data.shape}")
    print(f"Visual特徴形状: {output3['visual_feat'].shape}")
    print(f"系列長: {output3['feat_len']}")

    return output3


def analyze_receptive_field():
    """受容野の計算と比較"""

    print("\n=== 受容野分析 ===")

    # 従来の固定カーネル方式
    print("\n従来方式 (conv_type=2):")
    print("K5 -> P2 -> K5 -> P2")
    print("受容野計算:")
    print("  K5: 5フレーム")
    print("  P2: /2 (ダウンサンプリング)")
    print("  K5: 5フレーム (実際には10フレーム分)")
    print("  P2: /2 (ダウンサンプリング)")
    print("  → 最終受容野: 約20-25フレーム")

    # Multi-scale方式
    print("\nMulti-scale方式:")
    print("並列処理:")
    print("  ブランチ1 (K3): 3フレーム受容野")
    print("  ブランチ2 (K5): 5フレーム受容野")
    print("  ブランチ3 (K7): 7フレーム受容野")
    print("  ブランチ4 (K9): 9フレーム受容野")
    print("  → 全ての時間スケールを同時に捉える")

    # Adaptive方式
    print("\nAdaptive方式:")
    print("目標フレーム別:")
    print("  15フレーム用: K5->K3->K3 (受容野11フレーム)")
    print("  25フレーム用: K5->P2->K5->P2 (受容野20-25フレーム)")
    print("  35フレーム用: K7->K5->K3 (受容野15フレーム)")
    print("  → 短・中・長期の時間パターンを捉える")


def performance_comparison():
    """パフォーマンス比較"""

    print("\n=== パフォーマンス比較 ===")

    input_channels = 1662
    hidden_size = 128
    batch_size = 8
    seq_len = 100

    input_data = torch.randn(batch_size, input_channels, seq_len)

    # 従来方式
    conv_traditional = TemporalConv(
        input_channels, hidden_size, conv_type=2, use_multiscale=False
    )

    # Multi-scale方式
    conv_multiscale = TemporalConv(
        input_channels,
        hidden_size,
        conv_type=2,
        use_multiscale=True,
        multiscale_kernels=[3, 5, 7, 9],
    )

    # パラメータ数比較
    traditional_params = sum(p.numel() for p in conv_traditional.parameters())
    multiscale_params = sum(p.numel() for p in conv_multiscale.parameters())

    print(f"従来方式のパラメータ数: {traditional_params:,}")
    print(f"Multi-scale方式のパラメータ数: {multiscale_params:,}")
    print(f"パラメータ比率: {multiscale_params/traditional_params:.2f}x")

    # 推論時間比較（簡易）
    import time

    # Warm-up
    for _ in range(10):
        _ = conv_traditional(input_data, torch.tensor([seq_len] * batch_size))
        _ = conv_multiscale(input_data, torch.tensor([seq_len] * batch_size))

    # 従来方式の時間測定
    start_time = time.time()
    for _ in range(100):
        _ = conv_traditional(input_data, torch.tensor([seq_len] * batch_size))
    traditional_time = time.time() - start_time

    # Multi-scale方式の時間測定
    start_time = time.time()
    for _ in range(100):
        _ = conv_multiscale(input_data, torch.tensor([seq_len] * batch_size))
    multiscale_time = time.time() - start_time

    print(f"従来方式の推論時間: {traditional_time:.4f}秒 (100回)")
    print(f"Multi-scale方式の推論時間: {multiscale_time:.4f}秒 (100回)")
    print(f"速度比率: {multiscale_time/traditional_time:.2f}x")


if __name__ == "__main__":
    # 使用例実行
    example_multiscale_temporal_conv()

    # 受容野分析
    analyze_receptive_field()

    # パフォーマンス比較
    performance_comparison()

    print("\n=== 実装のポイント ===")
    print("1. Multi-Scale: 複数のカーネルサイズを並列処理")
    print("2. Adaptive: 目標フレーム数に応じてブランチを設計")
    print("3. 系列長維持: paddingを使用して入力と同じ長さを保持")
    print("4. 特徴融合: 各スケールの特徴を結合して最終出力を生成")
    print("5. 柔軟性: use_multiscaleフラグで従来方式との切り替え可能")
