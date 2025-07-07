"""
視覚化ユーティリティパッケージ

このパッケージは連続手話認識システム用の各種可視化機能を提供します:
- Attention重みとCTC Alignmentの可視化
- 信頼度可視化
- 混同行列分析
- 多層特徴量可視化
- 統合プロセス関数
"""

from .attention_visualization import (
    visualize_attention_weights,
    visualize_ctc_alignment,
)

from .confidence_visualization import (
    process_confidence_visualization,
)

from .confusion_matrix_visualization import (
    generate_confusion_matrix_analysis,
    collect_prediction_labels,
)

from .feature_visualization import (
    process_multilayer_feature_visualization,
)

from .visualization_integration import (
    process_attention_visualization,
    setup_visualization_environment,
    finalize_visualization,
    calculate_wer_metrics,
)

from .test_loop_integration import (
    TestLoopVisualizer,
    run_all_test_visualizations,
    run_complete_test_loop_visualizations,
    process_batch_visualizations_integrated,
    finalize_all_visualizations,
    collect_confusion_matrix_labels,
)

__all__ = [
    # Attention関連
    "visualize_attention_weights",
    "visualize_ctc_alignment",
    "process_attention_visualization",
    # 信頼度関連
    "process_confidence_visualization",
    # 混同行列関連
    "generate_confusion_matrix_analysis",
    "collect_prediction_labels",
    # 特徴量可視化関連
    "process_multilayer_feature_visualization",
    # 統合プロセス関数
    "setup_visualization_environment",
    "finalize_visualization",
    "calculate_wer_metrics",
    # test_loop統合関数
    "TestLoopVisualizer",
    "run_all_test_visualizations",
]
