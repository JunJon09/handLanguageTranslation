import Interpolation.liner_interpolation.config as config
import Interpolation.liner_interpolation.csv_function as csv_function
import pandas as pd
import os

# 補間を行う関数を定義
def interpolate_landmarks(group):
    # x, y, zを数値型に変換（欠損値はNaNになる）
    group['x'] = pd.to_numeric(group['x'], errors='coerce')
    group['y'] = pd.to_numeric(group['y'], errors='coerce')
    group['z'] = pd.to_numeric(group['z'], errors='coerce')
    
    # 線形補間を適用
    group[['x', 'y', 'z']] = group[['x', 'y', 'z']].interpolate(method='linear', limit_direction='both')
    
    return group

if __name__ == "__main__":
    files = csv_function.get_csv()
    index_records = []
    for column_name, row in files.iterrows():
        file_path = "./" + row['path']
        df = pd.read_csv(file_path)
        df['frame_index'] = pd.to_numeric(df['frame_index'], errors='coerce')

        df['original_index'] = df.index

        df = df.sort_values(by=['person_number', 'landmark_label', 'landmark_id', 'frame_index'])
        pd.set_option('display.max_rows', None)

        df_interpolated = df.groupby(['person_number', 'landmark_label', 'landmark_id']).apply(interpolate_landmarks)

        df_interpolated = df_interpolated.reset_index(drop=True)
        
        df_interpolated = df_interpolated.sort_values(by='original_index').drop(columns='original_index').reset_index(drop=True)

        output_csv_dir = config.output_csv_directory_path + str(row['sign'])
        os.makedirs(output_csv_dir, exist_ok=True)

        output_csv_path = output_csv_dir +  "/" + str(row['file_name']) + ".csv"
        df_interpolated.to_csv(output_csv_path, index=False)
        print(f"補間したデータを {output_csv_path} に保存しました。")

        index_record = {
            'path': output_csv_path,
            'person_number': row['person_number'],
            'file_name': row['file_name'],
            'sign': row['sign']
        }
        index_records.append(index_record)
        
    
    csv_function.create_index_csv(index_records, config.restore_index_file_path)
