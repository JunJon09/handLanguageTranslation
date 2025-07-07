# 視覚化ユーティリティパッケージ使用ガイド

## 概要

`/ContinuousSignLanguage/utils/`パッケージは、連続手話認識プロジェクト間で共通利用できる視覚化ツールを提供します。

## 特徴

- **プロジェクト間共通利用**: CNN_BiLSTM、Transformer、One_DCNN_Transformer など、どのプロジェクトからでも利用可能
- **統一されたインターフェース**: 一貫したパラメータ設定と関数呼び出し
- **エラーハンドリング**: 堅牢なエラー処理とロギング
- **フォールバック機能**: utils パッケージが利用できない場合は既存実装を使用

## 使用方法

### 基本的なインポート

```python
import sys
import os

# utilsパッケージを追加
utils_path = os.path.join(os.path.dirname(__file__), '..', '..', 'utils')
if utils_path not in sys.path:
    sys.path.append(utils_path)

try:
    from utils.visualization_utils import (
        process_attention_visualization,
        process_ctc_visualization,
        process_confidence_visualization,
        process_multilayer_feature_visualization,
        process_confusion_matrix_visualization,
    )
    UTILS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"視覚化ユーティリティのインポートに失敗: {e}")
    UTILS_AVAILABLE = False
```

### 使用例

#### 1. Attention 可視化

```python
if UTILS_AVAILABLE:
    config_attention = {
        'enabled': True,
        'sample_rate': 0.1,  # 10%のサンプルを可視化
        'save_path': 'attention_visualizations'
    }

    success = process_attention_visualization(
        model=model,
        reference_text=reference_text,
        hypothesis_text=hypothesis_text,
        batch_idx=batch_idx,
        config=config_attention,
        output_base_dir="outputs"
    )
```

#### 2. CTC 可視化

```python
if UTILS_AVAILABLE:
    config_ctc = {
        'enabled': True,
        'sample_rate': 0.1,
        'save_path': 'ctc_alignments',
        'show_path': True
    }

    success = process_ctc_visualization(
        sequence_logits=sequence_logits,
        reference_text=reference_text,
        hypothesis_text=hypothesis_text,
        batch_idx=batch_idx,
        config=config_ctc,
        vocab=vocab_dict,
        output_base_dir="outputs"
    )
```

#### 3. 信頼度可視化

```python
if UTILS_AVAILABLE:
    config_confidence = {
        'enabled': True,
        'sample_rate': 1.0,
        'save_path': 'confidence_analysis',
        'threshold': 0.5,
        'show_entropy': True
    }

    success = process_confidence_visualization(
        log_probs=log_probs,
        reference_text=reference_text,
        hypothesis_text=hypothesis_text,
        batch_idx=batch_idx,
        config=config_confidence,
        vocab=vocab_dict,
        output_base_dir="outputs"
    )
```

#### 4. 多層特徴量可視化

```python
if UTILS_AVAILABLE:
    config_features = {
        'enabled': True,
        'save_path': 'feature_visualizations',
        'method': 'umap',  # 'tsne' or 'umap'
        'layers': ['cnn', 'bilstm', 'attention', 'final'],
        'perplexity': 30,    # t-SNEパラメータ
        'n_neighbors': 15    # UMAPパラメータ
    }

    success = process_multilayer_feature_visualization(
        model=model,
        data_loader=data_loader,
        config=config_features,
        output_base_dir="outputs"
    )
```

#### 5. 混同行列可視化

```python
if UTILS_AVAILABLE:
    config_confusion = {
        'enabled': True,
        'save_path': 'confusion_analysis',
        'normalize': 'true',
        'show_metrics': True
    }

    success = process_confusion_matrix_visualization(
        all_predictions=all_predictions,
        all_references=all_references,
        config=config_confusion,
        output_base_dir="outputs"
    )
```

## プロジェクト別の統合例

### CNN_BiLSTM プロジェクトでの使用

```python
# functions.pyで
def process_attention_visualization(*args, **kwargs):
    if UTILS_AVAILABLE:
        return utils_process_attention_visualization(*args, **kwargs)
    else:
        # 既存の実装を使用（フォールバック）
        return legacy_attention_visualization(*args, **kwargs)
```

### Transformer プロジェクトでの使用

```python
# 新しいプロジェクトでの利用例
import sys
import os

# utilsパッケージパスの追加
project_root = os.path.dirname(os.path.dirname(__file__))
utils_path = os.path.join(project_root, 'utils')
sys.path.append(utils_path)

from utils.visualization_utils import process_attention_visualization

# 直接利用
success = process_attention_visualization(
    model, ref_text, hyp_text, batch_idx, config, "outputs"
)
```

## 設定パラメータ詳細

### 共通パラメータ

- **enabled**: 可視化の有効/無効フラグ
- **sample_rate**: サンプリング率（0.0-1.0）
- **save_path**: 保存ディレクトリ名
- **output_base_dir**: ベース出力ディレクトリ

### 可視化別パラメータ

#### Attention 可視化

- 追加パラメータなし

#### CTC 可視化

- **show_path**: アライメントパスの表示

#### 信頼度可視化

- **threshold**: 信頼度閾値
- **show_entropy**: エントロピー表示フラグ

#### 特徴量可視化

- **method**: 次元削減手法（'tsne', 'umap', 'both'）
- **layers**: 可視化対象層のリスト
- **perplexity**: t-SNE の perplexity パラメータ
- **n_neighbors**: UMAP の n_neighbors パラメータ

#### 混同行列可視化

- **normalize**: 正規化方法（'true', 'pred', 'all', None）
- **show_metrics**: メトリクス表示フラグ

## ファイル構造

```
/ContinuousSignLanguage/utils/
├── __init__.py                 # パッケージ初期化
├── visualization_utils.py      # 主要な可視化関数
└── README_utils_usage.md      # このファイル
```

## エラーハンドリング

utils パッケージは以下のエラーに対して適切に対応します：

1. **インポートエラー**: `UTILS_AVAILABLE`フラグでフォールバック
2. **ファイルパスエラー**: 自動的にディレクトリ作成
3. **データエラー**: 適切なログ出力とスキップ処理
4. **メモリエラー**: サンプリング率による負荷軽減

## 依存関係

- `torch`: PyTorch モデル操作
- `numpy`: 数値計算
- `logging`: ログ出力
- `os`, `sys`: ファイルシステム操作

plots.py の可視化関数も動的にインポートします：

- `plot_attention_matrix`, `plot_attention_focus`, `plot_attention_stats`
- `plot_ctc_heatmap`, `plot_ctc_prob_time`, `plot_ctc_statistics`
- `plot_confidence_timeline`, `plot_word_confidence`
- `extract_multilayer_features`, `plot_multilayer_feature_visualization`
- `plot_confusion_matrix_analysis`

## メリット

1. **再利用性**: プロジェクト間での関数の共有
2. **保守性**: 単一場所での視覚化ロジック管理
3. **一貫性**: 統一されたインターフェース
4. **拡張性**: 新しい可視化機能の追加が容易
5. **後方互換性**: 既存コードとの互換性維持

## 注意事項

- utils パッケージが利用できない場合は、既存の実装が使用されます
- パフォーマンスを考慮して、`sample_rate`パラメータを適切に設定してください
- 大量のデータを扱う場合は、メモリ使用量に注意してください
