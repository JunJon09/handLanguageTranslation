# 視覚化ユーティリティパッケージ 使用ガイド

## 概要

`utils/visualization`パッケージは、連続手話認識システム用の包括的な視覚化機能を提供します。このパッケージは他のプロジェクト（CNN_BiLSTM、Transformer、One_DCNN_Transformer など）からも利用可能な汎用的な実装です。

## パッケージ構成

```
utils/visualization/
├── __init__.py                          # パッケージエントリポイント
├── attention_visualization.py          # Attention重みとCTC可視化
├── confidence_visualization.py         # 信頼度可視化
├── confusion_matrix_visualization.py   # 混同行列可視化
├── feature_visualization.py           # 多層特徴量可視化
└── visualization_integration.py       # 統合プロセス関数
```

## 提供機能

### 1. Attention 重みと CTC Alignment 可視化

- **モジュール**: `attention_visualization.py`
- **主要関数**:
  - `visualize_attention_weights()`: Attention 重み可視化
  - `visualize_ctc_alignment()`: CTC Alignment パス可視化

### 2. 信頼度可視化

- **モジュール**: `confidence_visualization.py`
- **主要関数**:
  - `process_confidence_visualization()`: 時系列・単語レベル信頼度可視化

### 3. 混同行列分析

- **モジュール**: `confusion_matrix_visualization.py`
- **主要関数**:
  - `generate_confusion_matrix_analysis()`: 混同行列生成・分析
  - `collect_prediction_labels()`: ラベル収集

### 4. 多層特徴量可視化

- **モジュール**: `feature_visualization.py`
- **主要関数**:
  - `process_multilayer_feature_visualization()`: CNN/BiLSTM/Attention 層の特徴量可視化

### 5. 統合プロセス関数

- **モジュール**: `visualization_integration.py`
- **主要関数**:
  - `process_attention_visualization()`: Attention 可視化統合処理
  - `setup_visualization_environment()`: 可視化環境セットアップ
  - `finalize_visualization()`: 可視化後処理
  - `calculate_wer_metrics()`: WER 評価指標計算

## 使用方法

### 基本的なインポート

```python
# 任意のプロジェクトから
import sys
import os

# utilsパッケージパスを追加
utils_path = os.path.join(os.path.dirname(__file__), '..', '..', 'utils')
if utils_path not in sys.path:
    sys.path.append(utils_path)

# 必要な関数をインポート
from utils.visualization import (
    process_attention_visualization,
    process_confidence_visualization,
    process_multilayer_feature_visualization,
    generate_confusion_matrix_analysis,
    setup_visualization_environment,
    finalize_visualization,
)
```

### 統合的な使用例

```python
def run_visualization_enabled_test(model, test_loader):
    \"\"\"可視化機能を有効にしたテスト実行例\"\"\"

    # 1. 可視化環境のセットアップ
    output_dir, visualize_count = setup_visualization_environment(
        visualize_attention=True,
        max_visualize_samples=10
    )

    all_predictions = []
    all_references = []

    for batch_idx, batch in enumerate(test_loader):
        # モデル推論処理
        feature, spatial_feature, tokens, feature_pad_mask, \
        input_lengths, target_lengths, reference_text, hypothesis_text = batch

        # 推論実行
        pred, conv_pred, sequence_logits = model.forward(
            src_feature=feature,
            spatial_feature=spatial_feature,
            tgt_feature=tokens,
            src_causal_mask=None,
            src_padding_mask=feature_pad_mask,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            mode="test"
        )

        # 2. Attention可視化（サンプリング制御付き）
        if visualize_count < 10 and np.random.random() < 0.1:  # 10%の確率
            success_attention, success_ctc = process_attention_visualization(
                model=model,
                batch_idx=batch_idx,
                feature=feature,
                spatial_feature=spatial_feature,
                tokens=tokens,
                feature_pad_mask=feature_pad_mask,
                input_lengths=input_lengths,
                target_lengths=target_lengths,
                reference_text=reference_text,
                hypothesis_text=hypothesis_text,
                output_dir=output_dir
            )

            if success_attention or success_ctc:
                visualize_count += 1

        # 3. 信頼度可視化
        if sequence_logits is not None:
            log_probs = torch.nn.functional.log_softmax(sequence_logits, dim=-1)
            process_confidence_visualization(
                log_probs=log_probs,
                predictions=hypothesis_text,
                batch_idx=batch_idx,
                output_dir=output_dir
            )

        # 4. 多層特徴量可視化（限定的に実行）
        if batch_idx < 5:  # 最初の5バッチのみ
            process_multilayer_feature_visualization(
                model=model,
                feature=feature,
                spatial_feature=spatial_feature,
                tokens=tokens,
                feature_pad_mask=feature_pad_mask,
                input_lengths=input_lengths,
                target_lengths=target_lengths,
                pred=pred,
                batch_idx=batch_idx,
                output_dir=output_dir,
                method="both"  # t-SNEとUMAP両方
            )

        # 予測結果を収集
        all_predictions.extend(hypothesis_text)
        all_references.extend(reference_text)

    # 5. 混同行列分析
    generate_confusion_matrix_analysis(
        prediction_labels=all_predictions,
        ground_truth_labels=all_references,
        save_dir=output_dir
    )

    # 6. 可視化処理の終了
    finalize_visualization(
        model=model,
        visualize_attention=True,
        visualize_count=visualize_count,
        max_visualize_samples=10,
        output_dir=output_dir
    )

    return all_predictions, all_references
```

### 個別機能の使用例

#### Attention 可視化のみ

```python
from utils.visualization import visualize_attention_weights

success = visualize_attention_weights(
    model=model,
    batch_idx=0,
    reference_text=["正解文"],
    hypothesis_text=["予測文"],
    output_dir="outputs/attention"
)
```

#### 信頼度可視化のみ

```python
from utils.visualization import process_confidence_visualization

success_conf, success_word = process_confidence_visualization(
    log_probs=log_probs,
    predictions=["予測文1", "予測文2"],
    batch_idx=0,
    output_dir="outputs/confidence"
)
```

#### 混同行列分析のみ

```python
from utils.visualization import generate_confusion_matrix_analysis

success = generate_confusion_matrix_analysis(
    prediction_labels=["予測1", "予測2", "予測3"],
    ground_truth_labels=["正解1", "正解2", "正解3"],
    save_dir="outputs/confusion"
)
```

## 他のプロジェクトでの利用

### Transformer プロジェクトでの使用例

```python
# ContinuousSignLanguage/transformer/modeling/functions.py

import sys
import os
utils_path = os.path.join(os.path.dirname(__file__), '..', '..', 'utils')
sys.path.append(utils_path)

from utils.visualization import process_attention_visualization

def transformer_test_with_visualization(model, test_loader):
    \"\"\"Transformerモデル用の可視化付きテスト\"\"\"

    for batch_idx, batch in enumerate(test_loader):
        # Transformer特有の処理...

        # 共通の可視化機能を利用
        process_attention_visualization(
            model=model,
            batch_idx=batch_idx,
            # Transformer用のパラメータを渡す
            feature=feature,
            spatial_feature=None,  # Transformerでは使用しない場合
            tokens=tokens,
            feature_pad_mask=padding_mask,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            reference_text=reference_text,
            hypothesis_text=hypothesis_text,
            output_dir="outputs/transformer_attention"
        )
```

### One_DCNN_Transformer_Encoder プロジェクトでの使用例

```python
# ContinuousSignLanguage/one_dcnn_transformer_encoder/modeling/functions.py

from utils.visualization import (
    process_multilayer_feature_visualization,
    generate_confusion_matrix_analysis
)

def one_dcnn_test_with_analysis(model, test_loader):
    \"\"\"One DCNN Transformer Encoder用の分析付きテスト\"\"\"

    all_preds = []
    all_refs = []

    for batch_idx, batch in enumerate(test_loader):
        # One DCNN特有の処理...

        # 特徴量可視化（One DCNNの特徴を分析）
        process_multilayer_feature_visualization(
            model=model,
            feature=feature,
            spatial_feature=spatial_feature,
            tokens=tokens,
            feature_pad_mask=feature_pad_mask,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            pred=pred,
            batch_idx=batch_idx,
            output_dir="outputs/one_dcnn_features",
            method="umap"  # UMAPを使用
        )

        all_preds.extend(hypothesis_text)
        all_refs.extend(reference_text)

    # 最終的な混同行列分析
    generate_confusion_matrix_analysis(
        prediction_labels=all_preds,
        ground_truth_labels=all_refs,
        save_dir="outputs/one_dcnn_analysis"
    )
```

## パフォーマンス考慮事項

### メモリ使用量の最適化

```python
# サンプリング率を調整してメモリ使用量を削減
visualization_config = {
    'attention_sample_rate': 0.05,  # 5%のサンプルのみ可視化
    'feature_sample_rate': 0.02,   # 2%のサンプルのみ特徴量可視化
    'max_batches': 10,              # 最大10バッチまで
}

# 条件付き可視化
if batch_idx < visualization_config['max_batches'] and \
   np.random.random() < visualization_config['attention_sample_rate']:
    process_attention_visualization(...)
```

### 大規模データセット対応

```python
# バッチごとに可視化を制限
def selective_visualization(batch_idx, total_batches):
    \"\"\"選択的可視化実行判定\"\"\"
    # 最初と最後の数バッチのみ可視化
    return batch_idx < 5 or batch_idx > total_batches - 5

for batch_idx, batch in enumerate(test_loader):
    if selective_visualization(batch_idx, len(test_loader)):
        # 可視化処理を実行
        process_attention_visualization(...)
```

## トラブルシューティング

### よくある問題と解決方法

1. **インポートエラー**

   ```
   ImportError: cannot import name 'process_attention_visualization'
   ```

   - utils パッケージのパスが正しく設定されているか確認
   - `__init__.py`ファイルが存在するか確認

2. **メモリ不足エラー**

   ```
   RuntimeError: CUDA out of memory
   ```

   - サンプリング率を下げる (0.01-0.05)
   - バッチサイズを小さくする
   - 可視化する層を限定する

3. **ファイル出力エラー**

   ```
   PermissionError: cannot create directory
   ```

   - 出力ディレクトリの権限を確認
   - 絶対パスを使用する

4. **plots.py のインポートエラー**
   ```
   ImportError: cannot import name 'plot_attention_matrix' from 'plots'
   ```
   - plots.py が存在することを確認
   - 必要な依存パッケージ（matplotlib, seaborn 等）がインストールされているか確認

## 拡張方法

新しい可視化機能を追加する場合：

1. 適切なモジュールに新しい関数を追加
2. `__init__.py`の`__all__`リストに追加
3. 必要に応じて統合プロセス関数を更新
4. 使用例をこのガイドに追加

このパッケージにより、連続手話認識の各プロジェクト間で一貫した可視化・分析機能を共有できます。
