# Attention 可視化機能の使用方法

## 概要

CNN-BiLSTM モデルの SpatialCorrelationModule（Self-Attention）の重みを可視化する機能が追加されました。

## 機能

1. **Attention Matrix 可視化**: 時間ステップ間の注意重みをヒートマップ表示
2. **Attention 統計情報**: 注意重みの分布、最大値、エントロピーなどを可視化
3. **時間的変化分析**: 注意の焦点が時間とともにどう変化するかを可視化

## 使用方法

### 1. テスト時に Attention 可視化を実行

`continuous_sign_language/modeling/predict.py`を編集して以下のように使用：

```python
# Attention可視化を有効にする
wer, test_times = functions.test_loop(
    dataloader=test_dataloader,
    model=load_model,
    device=device,
    return_pred_times=True,
    blank_id=VOCAB-1,
    visualize_attention=True,        # 可視化を有効化
    max_visualize_samples=10,        # 可視化するサンプル数
    output_dir=None                  # 出力ディレクトリ（Noneで自動設定）
)
```

### 2. パラメータ説明

- `visualize_attention`: Attention 可視化を有効/無効
- `max_visualize_samples`: 可視化する最大サンプル数（計算時間を考慮して設定）
- `output_dir`: 可視化結果の保存先（None の場合は`config.plot_save_dir/attention_test`）

### 3. 出力ファイル

各サンプルに対して以下のファイルが生成されます：

```
reports/figures/attention_test/
├── attention_matrix_batch_0_sample_0_correct.png     # Attentionマトリックス
├── attention_stats_batch_0_sample_0_correct.png      # 統計情報
├── attention_focus_batch_0_sample_0_correct.png      # 時間的変化
├── attention_matrix_batch_1_sample_0_incorrect.png   # 誤認識サンプル
└── ...
```

### 4. 可視化の種類

#### 4.1 Attention Matrix（注意行列）

- 各時間ステップがどの時間ステップに注目しているかをヒートマップで表示
- 対角線が強い場合：自己注意が強い
- 帯状パターン：局所的な注意
- 散在パターン：グローバルな注意

#### 4.2 Attention Statistics（統計情報）

- 注意重みの分布ヒストグラム
- 各クエリ位置での最大注意重み
- 自己注意の強度（対角線要素）
- 注意の集中度（エントロピー）

#### 4.3 Attention Focus Over Time（時間的変化）

- 各時間ステップで最も注目している位置の軌跡
- 移動平均による平滑化
- 自己注意ライン（対角線）との比較

## 分析のポイント

### 正常な Attention パターン

- 適度な自己注意（対角線要素）
- 近隣時間ステップへの注意
- 滑らかな注意の移り変わり

### 問題のある Attention パターン

- 過度に集中した注意（低エントロピー）
- 不規則な注意の飛び方
- 遠距離の時間ステップへの過度な注意

### WER との関連分析

- 正解サンプルと誤認識サンプルの注意パターンの違い
- 特定の手話語彙での注意パターンの特徴
- 系列長による注意パターンの変化

## 注意事項

1. **計算コスト**: 可視化は計算時間を増加させるため、`max_visualize_samples`を適切に設定
2. **メモリ使用量**: Attention 重みを保存するため、メモリ使用量が増加
3. **ファイルサイズ**: 高解像度の画像ファイルが大量生成される可能性

## トラブルシューティング

### よくあるエラー

1. **"Attention 重みが取得できませんでした"**

   - モデルの`enable_attention_visualization()`が呼ばれていない
   - SpatialCorrelationModule が正常に動作していない

2. **可視化ファイルが生成されない**

   - 出力ディレクトリの権限を確認
   - 十分なディスク容量があるか確認

3. **メモリエラー**
   - `max_visualize_samples`を小さくする
   - バッチサイズを小さくする

## 応用例

### 誤認識の原因分析

```python
# WERが高いサンプルのみを可視化
wer, test_times = functions.test_loop(
    dataloader=test_dataloader,
    model=load_model,
    device=device,
    return_pred_times=True,
    blank_id=VOCAB-1,
    visualize_attention=True,
    max_visualize_samples=20,  # 多めに設定して誤認識サンプルを確保
    output_dir="reports/figures/error_analysis"
)
```

### 特定の手話語彙の分析

データローダーを特定の語彙のみにフィルタリングして可視化を実行

## 更新履歴

- 2025/07/01: 初版作成、基本的な可視化機能を実装
