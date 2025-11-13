# 混同行列（Confusion Matrix）分析機能

## 概要

手話認識モデルの**単語レベルでの誤認識パターン**を詳細に分析するための混同行列可視化機能です。どの手話単語がどの単語と間違えられやすいかを視覚的に把握できます。

## 機能

### 1. 混同行列可視化

- **正規化混同行列**: 各クラスの正解数で正規化した確率表示
- **Raw Count 混同行列**: 実際の誤分類回数を表示
- **美しいヒートマップ**: seaborn を使用した見やすい可視化

### 2. 詳細分析レポート

- **クラスごとの評価指標**:
  - Precision（適合率）
  - Recall（再現率）
  - F1 スコア
- **主な誤分類パターン**: 間違いの多い上位 5 パターンを特定
- **全体精度**: Overall Accuracy

## 使用方法

### 1. 基本設定

```python
# predict.py での設定
GENERATE_CONFUSION_MATRIX = True  # 混同行列生成を有効化
```

### 2. 実行

```bash
cd /path/to/CNN_BiLSTM
python continuous_sign_language/modeling/predict.py
```

### 3. 出力ファイル

- **混同行列画像**: `plots/word_level_confusion_matrix.png`
- **分析ログ**: ログファイルに詳細な分析結果が出力

## 出力例

### 混同行列画像の内容

- **X 軸**: 予測ラベル（Predicted Label）
- **Y 軸**: 正解ラベル（True Label）
- **色の濃さ**: 誤分類の頻度（濃いほど頻繁に間違える）
- **数値**: 正規化された確率または実際の回数

### ログ出力例

```
=== 混同行列分析結果 ===
全体精度: 0.847
friend: Precision=0.923, Recall=0.857, F1=0.889
station: Precision=0.812, Recall=0.891, F1=0.850
meet: Precision=0.889, Recall=0.823, F1=0.855

=== 主な誤分類パターン ===
friend -> station: 12回
station -> meet: 8回
meet -> friend: 6回
```

## 技術詳細

### 混同行列の計算方法

1. **データ収集**: テストループで予測結果と正解ラベルを収集
2. **単語分割**: 文章を単語レベルに分割
3. **ラベル整合**: 予測と正解の長さを合わせる
4. **行列計算**: scikit-learn の confusion_matrix 関数を使用

### 正規化方式

- **行正規化**: 各行（真のクラス）の合計で割る
- **計算式**: `normalized[i,j] = raw[i,j] / sum(raw[i,:])`
- **意味**: 真のクラス i のサンプルが予測クラス j に分類される確率

## 必要パッケージ

### Docker 環境

```dockerfile
# requirements.txtに追加済み
seaborn>=0.11.0
scikit-learn>=1.0.0
```

### 手動インストール

```bash
pip install seaborn scikit-learn pandas
```

## カスタマイズ

### 混同行列の見た目

```python
# plots.py の plot_confusion_matrix 関数で調整可能
figsize=(15, 12)  # 図のサイズ
cmap='Blues'      # カラーマップ
fmt='.3f'         # 数値フォーマット
```

### 分析対象の制限

```python
# functions.py の test_loop 関数で調整可能
min_len = min(len(ref_words), len(pred_words))  # 短い方に合わせる
```

## トラブルシューティング

### よくある問題

1. **パッケージエラー**

   ```
   ImportError: No module named 'seaborn'
   ```

   → `pip install seaborn scikit-learn`でインストール

2. **空の混同行列**

   ```
   混同行列生成用のデータが不足しています
   ```

   → テストデータが少ない、または全て予測失敗

3. **メモリエラー**
   → 大量のクラスがある場合、バッチサイズを小さくする

### デバッグ情報

ログレベルを DEBUG に設定すると詳細情報が出力されます：

```python
logging.getLogger().setLevel(logging.DEBUG)
```

## 応用例

### 1. 類似手話の特定

混同行列から、形が似ている手話同士の誤認識パターンを特定

### 2. データセット改善

頻繁に間違えられる単語ペアのデータを追加収集

### 3. モデル改善指針

F1 スコアの低いクラスに対する特別な学習戦略の検討

## 次のステップ

1. **時系列予測確率の可視化** - 各時間ステップでの予測信頼度
2. **Feature Visualization** - t-SNE/UMAP による特徴空間分析
3. **エラー分析** - 誤分類サンプルの詳細調査
