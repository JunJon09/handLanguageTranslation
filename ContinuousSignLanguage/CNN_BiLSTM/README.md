# 1DCNN and BiLSTM による連続手話認識

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## プロジェクト概要

このプロジェクトは、1D CNNとBiLSTMを組み合わせた連続手話認識システムです。MediaPipeで抽出した骨格座標を入力し、CTC損失を用いて連続手話単語列を認識します。

### モデルアーキテクチャ
- **DualCNNWithCTC**: 単一の1D-CNN
- **DualMultiScaleTemporalConv**: 複数の1D-CNN
- **特徴抽出**:
  - 骨格座標系: MediaPipeの顔・手・姿勢のランドマーク座標（x, y）
- **BiLSTM**: 双方向LSTMで前後の文脈情報を統合
- **CTC**: Connectionist Temporal Classification による系列アライメント

### データセット

- **RWTH-PHOENIX-Weather 2014 T**: 連続手話認識用のベンチマークデータセット
- **前処理**:
  - MediaPipeによる骨格座標抽出(hdf5に入っている前提)
  - PartsBasedNormalization（部位ごとの正規化）
  - 空間特徴量の付加（指先間距離、角度など）


### 学習結果

- **テスト精度**: 0.4135 (41.35%)
- **評価指標**: WER (Word Error Rate) - 連続手話認識の標準指標

### 主要なファイル

- `models/one_dcnn.py`: DualMultiScaleTemporalConvモデルの実装
- `continuous_sign_language/features.py`: 特徴量抽出・正規化処理
- `continuous_sign_language/config.py`: 使用する骨格座標，使用するデータセット，モデルの設定
- `continuous_sign_language/modeling/config.py`: ハイパーパラメータ設定
- `continuous_sign_language/modeling/train.py`: 学習スクリプト
- `continuous_sign_language/modeling/predict.py`: テスト・推論スクリプト

## 実行コマンド

### 学習の実行

以下のディレクトリで実行してください:

```bash
cd /path/to/ContinuousSignLanguage
```

```bash
python -m CNN_BiLSTM.continuous_sign_language.modeling.train
```

**主要なハイパーパラメータ（config.py）**:

- `batch_size`: 16
- `skeleton_hidden_size`: 256
- `spatial_hidden_size`: 256
- `kernel_sizes`: [10, 15, 20, 25, 30]
- `dropout_rate`: 0.2

### テスト・推論の実行

以下のディレクトリで実行してください:

```bash
cd /path/to/ContinuousSignLanguage
```

```bash
python -m CNN_BiLSTM.continuous_sign_language.modeling.predict
```

**入力データ形式**:

- 骨格座標: (Batch, Channels, Time) - MediaPipeの顔・手・姿勢ランドマーク
- 空間特徴量: (Batch, 12\*2, Time) - 指先間距離・角度など

**出力**:

- 認識された手話単語列
- WER (Word Error Rate)
- 各種評価指標のグラフ（`reports/figures/`に保存）

### ログとモデルの保存先

- **学習ログ**: `CNN_BiLSTM/logs/train_YYYYMMDD_HHMMSS.log`
- **モデル**: `models/` ディレクトリに保存
- **評価結果**: `reports/figures/` および `reports/results/`

## 引き継ぎのための技術メモ

### 重要な実装のポイント

#### 1. データの前処理パイプライン

`features.py`に実装されている主要なクラス:

- **SelectLandmarksAndFeature**: 必要なランドマークと特徴（x, y, z）を選択
- **ReplaceNan**: 欠損値（NaN）を0.0で置換
- **PartsBasedNormalization**: 部位ごとに原点と単位を設定して正規化
  - 顔: 鼻を原点、両耳の距離を単位
  - 手: 手首を原点、手のひらサイズを単位
  - 空間特徴量の付加: 指先間距離、角度など12次元

#### 2. モデルの系列長更新

畳み込み層を通過するごとに系列長が変化するため、`update_lgt()`メソッドで追跡:

```python
def update_lgt(self, lgt):
    feat_len = lgt.clone()  # 元の系列長を保持
    for ks in self.conv_kernel_sizes:
        if ks[0] == "P":  # Pooling層
            feat_len = torch.div(feat_len, 2, rounding_mode="floor")
        else:  # Conv層
            feat_len -= int(ks[1]) - 1  # kernel_size - 1
    return feat_len
```

#### 3. データセットの構造

- **入力**: HDF5ファイル（`hdf5/phoenix_2014_t/`）
  - `sign`: 骨格座標 (T, C, J) - Time, Channels(x,y), Joints
  - `label`: 手話単語のインデックス列
  - `signer`: 話者ID
- **出力**: `character_to_prediction_index.json` で単語とIDの対応

#### 4. 既知の問題と注意点
- **メモリ**: `load_into_ram=True`の場合、全データをRAMにロード（大規模データセットでは注意）。
- **系列長**: CTC損失計算時、出力系列長が入力系列長より短くならないよう注意。


### トラブルシューティング

**Q: 学習が進まない（損失が下がらない）**

- A: 学習率を調整（0.001 → 0.0001など）、バッチサイズを増やす、データ正規化を確認

**Q: OOM (Out of Memory) エラー**

- A: `batch_size`を減らす、`load_into_ram=False`に設定、`hidden_size`を削減

**Q: WERが高い（精度が低い）**

- A: エポック数を増やす、モデルアーキテクチャの見直し、データ前処理の確認

**Q: logファイル等のエラー
- A: 最初エラーが出る場合がある．その場合、logs/train or logs/testのフォルダを作成してください

### 関連リンク

- [RWTH-PHOENIX-2014-T データセット](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)
- [MediaPipe Holistic](https://google.github.io/mediapipe/solutions/holistic.html)
- [CTC論文](https://www.cs.toronto.edu/~graves/icml_2006.pdf)

## ファイル構成

```
CNN_BiLSTM/
├── README.md                   <- このファイル
│
├── models/                     <- 学習済みモデルの保存先
│   └── cnn_bilstm_model.py             <- ここがモデルの元
│
├── logs/                       <- 学習・テストのログファイル
│   ├── train/
│   └── test/
│
├── reports/                    <- 実験結果・評価レポート
│   ├── figures/                <- 生成されたグラフ・図表
│   │   ├── cnn_transformer_loss.png
│   │   ├── cnn_transformer_wer.png
│   │   └── attention_test/     <- アテンション可視化
│   └── results/                <- 評価結果（CSV、JSONなど）
│
├── analysis_graphs/            <- データ分析用のグラフ
│
└── continuous_sign_language/   <- メインのソースコード
    ├── __init__.py
    ├── config.py               <- ハイパーパラメータ・パス設定
    ├── dataset.py              <- データセット読み込み
    ├── features.py             <- 特徴量抽出・正規化
    ├── init_log.py             <- ログ設定
    ├── plots.py                <- 可視化ユーティリティ
    │
    └── modeling/               <- 学習・推論スクリプト
        ├── __init__.py
        ├── train.py            <- 学習実行
        └── predict.py          <- テスト・推論実行
```

---
