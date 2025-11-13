"""
連続手話認識システム用 共通ユーティリティパッケージ

このパッケージは以下の機能を提供します:
- visualization: 各種可視化機能の統合パッケージ
- visualization_utils: 汎用的な可視化関数（レガシー）
- 他のプロジェクトからも利用可能な汎用的な実装
"""

# 新しい視覚化パッケージからインポート
from .visualization import (
    visualize_attention_weights,
    visualize_ctc_alignment,
    process_attention_visualization,
    process_confidence_visualization,
    generate_confusion_matrix_analysis,
    collect_prediction_labels,
    process_multilayer_feature_visualization,
    setup_visualization_environment,
    finalize_visualization,
    calculate_wer_metrics,
)

# レガシーサポート
try:
    from .visualization_utils import (
        process_attention_visualization as utils_process_attention_visualization,
        process_ctc_visualization as utils_process_ctc_visualization,
        process_confidence_visualization as utils_process_confidence_visualization,
        process_multilayer_feature_visualization as utils_process_multilayer_feature_visualization,
        process_confusion_matrix_visualization as utils_process_confusion_matrix_visualization,
    )

    LEGACY_UTILS_AVAILABLE = True
except ImportError:
    LEGACY_UTILS_AVAILABLE = False

__all__ = [
    # 新しい視覚化パッケージ
    "visualize_attention_weights",
    "visualize_ctc_alignment",
    "process_attention_visualization",
    "process_confidence_visualization",
    "generate_confusion_matrix_analysis",
    "collect_prediction_labels",
    "process_multilayer_feature_visualization",
    "setup_visualization_environment",
    "finalize_visualization",
    "calculate_wer_metrics",
    # レガシーサポート（条件付き）
    *(
        [
            "utils_process_attention_visualization",
            "utils_process_ctc_visualization",
            "utils_process_confidence_visualization",
            "utils_process_multilayer_feature_visualization",
            "utils_process_confusion_matrix_visualization",
        ]
        if LEGACY_UTILS_AVAILABLE
        else []
    ),
]
