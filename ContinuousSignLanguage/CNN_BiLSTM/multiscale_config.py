"""
Multi-Scale Temporal Convolutionの設定例
config.pyに追加すべき設定項目
"""

# Multi-Scale Temporal Convolution設定
# =============================================================================

# Multi-Scaleを使用するかどうか
USE_MULTISCALE_TEMPORAL_CONV = True

# Multi-Scale設定: 方法1 - 並列カーネル方式
MULTISCALE_CONFIG = {
    "method": "parallel_kernels",  # "parallel_kernels" or "adaptive_frames"
    "kernel_sizes": [3, 5, 7, 9],  # 並列処理するカーネルサイズ
    "dropout_rate": 0.2,
    "use_fusion_layer": True,  # 特徴融合層を使用するか
}

# Multi-Scale設定: 方法2 - 適応フレーム方式
ADAPTIVE_CONFIG = {
    "method": "adaptive_frames",
    "target_frames": [15, 25, 35],  # 目標とする受容野フレーム数
    "dropout_rate": 0.2,
    "branch_architectures": {
        15: ["K5", "K3", "K3"],  # 15フレーム用アーキテクチャ
        25: ["K5", "P2", "K5", "P2"],  # 25フレーム用アーキテクチャ (従来)
        35: ["K7", "K5", "K3"],  # 35フレーム用アーキテクチャ
    },
}

# Temporal Convolution設定 (既存の設定を拡張)
TEMPORAL_CONV_CONFIG = {
    "conv_type": 2,  # 従来の設定も保持
    "hidden_size": 128,
    "use_bn": True,
    # Multi-Scale拡張設定
    "use_multiscale": USE_MULTISCALE_TEMPORAL_CONV,
    "multiscale_config": (
        MULTISCALE_CONFIG
        if MULTISCALE_CONFIG["method"] == "parallel_kernels"
        else ADAPTIVE_CONFIG
    ),
    # 受容野分析用設定
    "analyze_receptive_field": False,  # 受容野を分析するか
    "save_feature_maps": False,  # 特徴マップを保存するか
    "feature_map_save_dir": "CNN_BiLSTM/reports/feature_maps",
}

# デュアル特徴量処理の設定 (既存のDualFeatureTemporalConvを使用する場合)
DUAL_FEATURE_CONFIG = {
    "skeleton_input_size": 1662,  # MediaPipe座標数
    "hand_feature_size": 50,  # 手の空間的特徴サイズ
    "skeleton_hidden_size": 96,  # 骨格特徴の隠れ層サイズ
    "hand_hidden_size": 32,  # 手特徴の隠れ層サイズ
    "fusion_hidden_size": 128,  # 融合後のサイズ
    # Multi-Scale設定をデュアル特徴量にも適用
    "use_multiscale_skeleton": True,
    "use_multiscale_hand": True,
    "skeleton_kernel_sizes": [3, 5, 7],  # 骨格用カーネル
    "hand_kernel_sizes": [3, 5],  # 手用カーネル (より小さな受容野)
}

# パフォーマンス監視設定
PERFORMANCE_CONFIG = {
    "monitor_receptive_field": True,  # 受容野を監視
    "log_feature_statistics": True,  # 特徴統計をログ出力
    "save_attention_maps": False,  # アテンションマップを保存 (今後の拡張用)
    "benchmark_inference_time": True,  # 推論時間をベンチマーク
}

# 実験設定 (A/B テスト用)
EXPERIMENT_CONFIG = {
    "experiment_name": "multiscale_temporal_conv_v1",
    "compare_methods": {
        "baseline": {"use_multiscale": False, "conv_type": 2},
        "multiscale_parallel": {
            "use_multiscale": True,
            "method": "parallel_kernels",
            "kernel_sizes": [3, 5, 7, 9],
        },
        "multiscale_adaptive": {
            "use_multiscale": True,
            "method": "adaptive_frames",
            "target_frames": [15, 25, 35],
        },
    },
}

# ログ設定
LOGGING_CONFIG = {
    "log_receptive_field_analysis": True,
    "log_feature_fusion_weights": True,
    "log_multiscale_contributions": True,  # 各スケールの寄与度をログ
    "save_detailed_metrics": True,
    "metrics_save_path": "CNN_BiLSTM/reports/multiscale_metrics.json",
}

# バックワード互換性のための設定
BACKWARD_COMPATIBILITY = {
    "fallback_to_traditional": True,  # エラー時に従来方式にフォールバック
    "validate_output_shapes": True,  # 出力形状を検証
    "warn_on_config_mismatch": True,  # 設定不一致時に警告
}


# 使用例関数
def get_temporal_conv_config(method="multiscale_parallel"):
    """
    指定された方法のTemporal Convolution設定を取得

    Args:
        method: "traditional", "multiscale_parallel", "multiscale_adaptive"

    Returns:
        dict: 設定辞書
    """
    base_config = {
        "input_size": 1662,  # MediaPipe特徴量
        "hidden_size": TEMPORAL_CONV_CONFIG["hidden_size"],
        "conv_type": TEMPORAL_CONV_CONFIG["conv_type"],
        "num_classes": 64,  # 手話語彙数
        "use_bn": TEMPORAL_CONV_CONFIG["use_bn"],
    }

    if method == "traditional":
        base_config.update({"use_multiscale": False})
    elif method == "multiscale_parallel":
        base_config.update(
            {
                "use_multiscale": True,
                "multiscale_kernels": MULTISCALE_CONFIG["kernel_sizes"],
                "target_frames": None,
            }
        )
    elif method == "multiscale_adaptive":
        base_config.update(
            {
                "use_multiscale": True,
                "multiscale_kernels": None,
                "target_frames": ADAPTIVE_CONFIG["target_frames"],
            }
        )

    return base_config


# 設定バリデーション関数
def validate_multiscale_config():
    """Multi-Scale設定の整合性をチェック"""

    if USE_MULTISCALE_TEMPORAL_CONV:
        if MULTISCALE_CONFIG["method"] == "parallel_kernels":
            assert (
                len(MULTISCALE_CONFIG["kernel_sizes"]) > 1
            ), "カーネルサイズは2つ以上指定してください"
            assert all(
                k % 2 == 1 for k in MULTISCALE_CONFIG["kernel_sizes"]
            ), "カーネルサイズは奇数にしてください"

        elif ADAPTIVE_CONFIG["method"] == "adaptive_frames":
            assert (
                len(ADAPTIVE_CONFIG["target_frames"]) > 1
            ), "目標フレーム数は2つ以上指定してください"
            assert all(
                f > 0 for f in ADAPTIVE_CONFIG["target_frames"]
            ), "目標フレーム数は正の値にしてください"

    print("✅ Multi-Scale設定の検証が完了しました")


if __name__ == "__main__":
    # 設定例の表示
    print("=== Multi-Scale Temporal Convolution設定例 ===")

    print("\n1. 並列カーネル方式:")
    parallel_config = get_temporal_conv_config("multiscale_parallel")
    for key, value in parallel_config.items():
        print(f"  {key}: {value}")

    print("\n2. 適応フレーム方式:")
    adaptive_config = get_temporal_conv_config("multiscale_adaptive")
    for key, value in adaptive_config.items():
        print(f"  {key}: {value}")

    print("\n3. 従来方式 (比較用):")
    traditional_config = get_temporal_conv_config("traditional")
    for key, value in traditional_config.items():
        print(f"  {key}: {value}")

    # 設定検証
    validate_multiscale_config()
