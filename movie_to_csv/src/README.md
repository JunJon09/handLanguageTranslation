# Movie to CSV Converter

## Overview
このプロジェクトは動画ファイルからMediaPipeを用いて特徴量を抽出です。

## Features
- 動画ファイルの処理
- CSVファイルへのデータ出力

## File Structure
```
movie_to_csv/
├── src/
│   ├── movie_to_csv_skelton.py
│   ├── config.py
│   ├── mediapip_relation.py
│   └── utils.py
├── data/
│   ├── dataset_name/
│   └── dataset_name/
└── README.md
```

## Requirements
- Python 3.10.10 (推奨、他のVersionはためしていない)
```bash
pip install -r requirement
```
## Usage
config.pyでdataset_nameを指定
movie_ti_csv_skelton.pyに各データに対して特有の分割の仕方を記述。
```bash
python3 movie_to_csv_skeleton.py
```

## Installation
```bash
pip install -r requirements.txt
```

