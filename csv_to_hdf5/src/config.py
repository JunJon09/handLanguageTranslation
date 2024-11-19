index_file_path = "../../csv/minimum_continuous_hand_language/index.csv"
out_dir = "../../hdf5/minimum_continuous_hand_language/"
import pandas as pd

# # CSVファイルの読み込み
# df = pd.read_csv(index_file_path)

# # 'path'カラムの最初のドットを削除
# df['path'] = df['path'].str.lstrip('.')

# # 結果を確認
# print(df.head())

# # 必要に応じてCSVファイルに書き出す
# df.to_csv(index_file_path, index=False)