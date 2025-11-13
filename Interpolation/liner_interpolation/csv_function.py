import liner_interpolation.config as config
import pandas as pd
import os

def get_csv():
    files = pd.read_csv(config.index_file_path)
    return files

def create_index_csv(index_records: list, index_csv_path: str):

    index_df = pd.DataFrame(index_records)
    output_dir = os.path.dirname(index_csv_path)
    
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"フォルダ '{output_dir}' を作成しました。")
        except Exception as e:
            print(f"フォルダ '{output_dir}' の作成中にエラーが発生しました: {e}")
            return

    # 必要な列のみを選択
    index_df = index_df[['path', 'person_number', 'file_name', 'sign']].drop_duplicates()

    # インデックスCSVに書き出し
    index_df.to_csv(index_csv_path, index=False)
    print(f"インデックス CSV ファイル '{index_csv_path}' を作成しました。")
