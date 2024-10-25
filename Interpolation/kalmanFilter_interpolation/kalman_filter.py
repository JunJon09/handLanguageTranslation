import kalmanFilter_interpolation.config as config
import kalmanFilter_interpolation.csv_function as csv_function
import pandas as pd
from filterpy.kalman import KalmanFilter
import numpy as np
import os

def initialize_kalman_filter():
    # 1次元のカルマンフィルタをx, y, zそれぞれに適用する
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.F = np.array([[1, 1],
                     [0, 1]])  # 状態遷移行列
    kf.H = np.array([[1, 0]])  # 観測行列
    kf.P *= 1000.           # 初期不確実性
    kf.R = 5                # 観測ノイズ
    kf.Q = np.array([[1, 1],
                     [1, 1]])  # プロセスノイズ
    return kf

def apply_kalman_filter(data):
    kf_x = initialize_kalman_filter()
    kf_y = initialize_kalman_filter()
    kf_z = initialize_kalman_filter()

    filtered_x = []
    filtered_y = []
    filtered_z = []

    '''MediaPipeの都合上xだけ取得できてyが取れない場合がないからflagは一つ'''
    has_valid_observation = False

    for x, y, z in zip(data['x'], data['y'], data['z']):
        # x座標のカルマンフィルタ
        if not np.isnan(x):
            kf_x.update(x)
            has_valid_observation = True
        if has_valid_observation:
            kf_x.predict()
            filtered_x.append(kf_x.x[0][0])
        else:
            filtered_x.append(np.nan)

        # y座標のカルマンフィルタ
        if not np.isnan(y):
            kf_y.update(y)
        if has_valid_observation:
            kf_y.predict()
            filtered_y.append(kf_y.x[0][0])
        else:
            filtered_y.append(np.nan)
    
        # z座標のカルマンフィルタ
        if not np.isnan(z):
            kf_z.update(z)
        if has_valid_observation:
            kf_z.predict()
            filtered_z.append(kf_z.x[0][0])
        else:
            filtered_z.append(np.nan)
    if not has_valid_observation:
        filtered_x = [np.nan] * len(data)
        filtered_y = [np.nan] * len(data)
        filtered_z = [np.nan] * len(data)

    data['x'] = filtered_x
    data['y'] = filtered_y
    data['z'] = filtered_z

    return data


def kalman_filter_landmarks(group):
    # x, y, zを数値型に変換（欠損値はNaNになる）
    group['x'] = pd.to_numeric(group['x'], errors='coerce')
    group['y'] = pd.to_numeric(group['y'], errors='coerce')
    group['z'] = pd.to_numeric(group['z'], errors='coerce')

    # カルマンフィルタを適用
    group = apply_kalman_filter(group)
    return group

if __name__ == "__main__":
    files = csv_function.get_csv()
    index_records = []
    for column_name, row in files.iterrows():
        file_path = ".." + row['path']
        df = pd.read_csv(file_path)
        df['frame_index'] = pd.to_numeric(df['frame_index'], errors='coerce')

        df['original_index'] = df.index

        df = df.sort_values(by=['person_number', 'landmark_label', 'landmark_id', 'frame_index'])
        pd.set_option('display.max_rows', None)

        df_filtered = df.groupby(['person_number', 'landmark_label', 'landmark_id']).apply(kalman_filter_landmarks)

        df_filtered = df_filtered.reset_index(drop=True)
        
        df_filtered = df_filtered.sort_values(by='original_index').drop(columns='original_index').reset_index(drop=True)

        output_csv_dir = config.output_csv_directory_path + str(row['sign'])
        os.makedirs(output_csv_dir, exist_ok=True)

        output_csv_path = os.path.join(output_csv_dir, str(row['file_name']))
        df_filtered.to_csv(output_csv_path, index=False)
        print(f"カルマンフィルタを適用したデータを {output_csv_path} に保存しました。")

        index_record = {
            'path': output_csv_path,
            'person_number': row['person_number'],
            'file_name': row['file_name'],
            'sign': row['sign']
        }
        index_records.append(index_record)
        
    csv_function.create_index_csv(index_records, config.restore_index_file_path)