# 🔍 Attention 可視化機能 - 簡単使用ガイド

## ✅ 実装完了！

Attention 可視化機能が実装されました。**True/False**で簡単に切り替えができます。

## 🚀 使い方

### 1. 基本的な使い方

`continuous_sign_language/modeling/predict.py`の以下の行を変更するだけ：

```python
# ========================================
# 🔍 Attention可視化設定
# ========================================
VISUALIZE_ATTENTION = True  # True: 可視化する, False: 可視化しない
```

### 2. 設定の切り替え

```python
# 🔍 可視化を有効にする場合
VISUALIZE_ATTENTION = True

# 📊 通常のテストのみ実行する場合
VISUALIZE_ATTENTION = False
```

### 3. 出力結果

可視化を有効にすると、以下のディレクトリに結果が保存されます：

```
reports/figures/attention_test/
├── attention_matrix_batch_0_sample_0_correct.png     # ✅ 正解サンプル
├── attention_stats_batch_0_sample_0_correct.png
├── attention_focus_batch_0_sample_0_correct.png
├── attention_matrix_batch_1_sample_0_incorrect.png   # ❌ 誤認識サンプル
├── attention_stats_batch_1_sample_0_incorrect.png
└── attention_focus_batch_1_sample_0_incorrect.png
```

### 4. 可視化の種類

#### 📊 Attention Matrix

- 各時間ステップがどこに注目しているかのヒートマップ
- 青色が濃いほど注意度が高い

#### 📈 Attention Statistics

- 注意重みの分布とエントロピー
- 自己注意の強度分析

#### 📉 Attention Focus Over Time

- 注意の焦点が時間とともにどう変化するか
- 移動平均による平滑化表示

## 🎯 分析のポイント

### ✅ 正常パターン

- 対角線（自己注意）が適度に強い
- 近隣の時間ステップへの滑らかな注意

### ❌ 問題パターン

- 過度に集中した注意（低エントロピー）
- 不規則な注意の飛び方
- 遠距離への過度な注意

## ⚙️ 設定変更

デフォルトでは最大 10 サンプルまで可視化します。変更したい場合は：

`functions.py`の`max_visualize_samples = 10`を変更

## 🔧 トラブルシューティング

### よくあるエラー

1. **"Attention 重みが取得できませんでした"**

   - モデルの Attention 機能が正常に動作していない可能性

2. **ファイルが生成されない**

   - `reports/figures`ディレクトリの権限を確認
   - 十分なディスク容量があるか確認

3. **メモリエラー**
   - バッチサイズを小さくする
   - 可視化サンプル数を減らす

## 📝 実装済み機能

- ✅ SpatialCorrelationModule（Self-Attention）の修正
- ✅ test_loop 関数への可視化パラメータ追加
- ✅ predict.py での簡単 True/False 切り替え
- ✅ 3 種類の可視化グラフ生成
- ✅ 正解/誤認識の自動判定と分類
- ✅ 自動ファイル命名と保存

## 🎉 これで完了！

`VISUALIZE_ATTENTION = True`にして実行するだけで、Attention パターンの分析ができます！
