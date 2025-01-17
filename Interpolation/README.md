# Interpolation
骨格座標の補間手法の比較評価

## リポジトリの内容
本研究では手話映像から画像処理で手話を認識するために話者の骨格を検出している．骨格検出で得られる時系列手話動作データにはコマ落ちのような欠損が存在する．そこでこのリポジトリでは，骨格検出により取得した時系列手話動作データにおける欠損値補間手法の比較評価を行った．欠損値を補間する手法には，線形補間，カルマンフィルタを採用して，補間なしを含めた3手法で比較した．その結果，線形補間を用いた認識精度は91.6%であり，カルマンフィルタが86.5%の認識精度となった．一方，補間なしでは80.0%となり，認識精度が低かった．以上より，骨格検出による手話動作データの欠損値補間には線形補間が有効であることが示唆された．


## 実行コマンド
以下のディレクトリで実行して下さい
```bash
/src/Interpolation/
```

```bash
python -m xxxxx_interpolation.xxx.py
```
xxxはそれぞれの補間方法の名前になっている。


## ファイル構成
<pre>
Interpolation/
├── kalmanFilter_interpolation カルマンフィルタ
│   ├── config.py 固定値
│   ├── csv_function.py csvファイルの関数
│   └── kalman_filter.py カルマンフィルタのメイン関数
├── liner_interpolation 線形補間
│   ├── config.py 固定値
│   ├── csv_function.py csvファイルの関数
│   └── liner.py 線形補間のメイン関数
└── README.md
</pre>
