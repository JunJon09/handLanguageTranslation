# 手話認識 多層特徴量可視化システム

## 概要

手話認識モデル（CNN-BiLSTM）の各層から特徴量を抽出し、以下の4つの観点から包括的に可視化・分析するシステムです：

### 1. CNN出力 → 空間的パターン
- **特徴**: 手の形、位置関係、顔の表情、体の姿勢
- **解釈**: 静的な空間情報、形状認識、姿勢パターン
- **可視化**: t-SNE/UMAPによる2次元埋め込み、クラスタリング分析

### 2. BiLSTM隠れ状態 → 時系列ダイナミクス
- **特徴**: 動きの軌跡、速度、手話の文脈情報
- **解釈**: 動的な時系列パターン、モーション認識、文脈理解
- **可視化**: 時系列特徴量の次元削減、動き軌跡の可視化

### 3. Attention重み → 重要度マップ
- **特徴**: どの時刻・部位に注目しているか、手話認識の判断根拠
- **解釈**: モデルの注意機構、判断の透明性、重要箇所の特定
- **可視化**: 注意重みのヒートマップ、重要度統計分析

### 4. 最終層直前 → 統合的判断
- **特徴**: 全情報を統合した最終特徴量、分類に直結する表現
- **解釈**: 高レベル抽象化、決定境界に近い表現、分類性能
- **可視化**: 決定境界分析、分類信頼度マップ

## 主要機能

### 特徴量抽出
- `extract_multilayer_features()`: 各層からの特徴量抽出
- フック機能による中間層出力の取得
- 形状変換とデータ正規化

### 可視化機能
- `plot_multilayer_feature_visualization()`: 多層特徴量の統合可視化
- t-SNE/UMAPによる次元削減
- 層別クラスタリング分析
- 相関分析・分布分析

### 分析機能
- `analyze_feature_separation()`: 特徴量分離度分析
- シルエット係数による分離性能評価
- 最適層の特定と推奨事項
- クラス内・クラス間距離分析

### 統合処理
- `process_multilayer_feature_visualization()`: 全体処理の統合
- test_loopからの呼び出し対応
- パラメータ制御とエラーハンドリング

## 使用方法

### 1. 基本的な使用方法

```python
# test_loopでの使用例
success = process_multilayer_feature_visualization(
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
    vocab_dict=vocab_dict,
    method="both"  # "tsne", "umap", "both"
)
```

### 2. 設定可能なパラメータ

- `method`: 可視化手法選択
  - `"tsne"`: t-SNEのみ
  - `"umap"`: UMAPのみ
  - `"both"`: 両方実行（推奨）

### 3. 出力ファイル

以下のファイルが生成されます：

```
output_dir/multilayer_features_batch_X/
├── multilayer_features_tsne_sample_0.png    # t-SNE可視化
├── multilayer_features_umap_sample_0.png    # UMAP可視化
├── feature_correlation_sample_0.png         # 特徴量相関分析
├── feature_distribution_sample_0.png        # 特徴量分布分析
└── feature_separation_analysis_sample_0.png # 分離度分析
```

## 解釈ガイド

### 1. 空間的パターン (CNN)
- **良好な例**: 手話の形状ごとにクラスタが明確に分離
- **問題の例**: 異なる手話の形状が混在、境界が曖昧
- **改善案**: データ前処理、空間正規化の見直し

### 2. 時系列ダイナミクス (BiLSTM)
- **良好な例**: 動きパターンごとの明確な軌跡分離
- **問題の例**: 異なる動きが同じ領域に集中
- **改善案**: 時系列モデリングの強化、文脈長の調整

### 3. 重要度マップ (Attention)
- **良好な例**: 重要箇所への集中的注目
- **問題の例**: 注意が散漫、重要でない部分への過度の注目
- **改善案**: Attention機構の調整、正則化

### 4. 統合的判断 (最終層)
- **良好な例**: クラス境界が明確、高い分離度
- **問題の例**: クラスが混在、低い分離度
- **改善案**: 特徴量統合方法の改善、分類器の調整

## 分離度分析の見方

### シルエット係数
- **0.7以上**: 優秀な分離性能
- **0.5-0.7**: 良好な分離性能
- **0.25-0.5**: 中程度の分離性能
- **0.25未満**: 改善が必要

### 推奨事項
- 最高スコア層の特徴量を重点活用
- 低スコア層の改善検討
- 層間の相関関係の活用

## 技術仕様

### 依存ライブラリ
```bash
pip install scikit-learn umap-learn seaborn
```

### 処理フロー
1. モデルの各層にフックを登録
2. Forward処理で中間層出力を取得
3. 特徴量の形状変換と正規化
4. 次元削減による可視化
5. 統計分析と分離度評価
6. 結果の保存とログ出力

### エラーハンドリング
- ライブラリ不足時の適切な警告
- データ不足時のスキップ処理
- 詳細なエラー情報の出力

## 今後の拡張予定

1. **リアルタイム可視化**: 学習過程での特徴量変化の動的表示
2. **インタラクティブ可視化**: Plotlyを使った対話的な分析
3. **特徴量重要度分析**: SHAP値による特徴量貢献度分析
4. **異常検出**: 外れ値や異常パターンの自動検出
5. **比較分析**: 異なるモデル間の特徴量比較

## 参考文献

- t-SNE: van der Maaten & Hinton (2008)
- UMAP: McInnes et al. (2018)
- Silhouette Analysis: Rousseeuw (1987)
