index_file_path = "../../csv/lsa64_liner/index.csv"
out_dir = "../../hdf5/lsa64_liner/"
import pandas as pd

# # CSVファイルの読み込み
# df = pd.read_csv(index_file_path)

# # 'path'カラムの最初のドットを削除
# df['path'] = df['path'].str.lstrip('.')

# # 結果を確認
# print(df.head())

# # 必要に応じてCSVファイルに書き出す
# df.to_csv(index_file_path, index=False)