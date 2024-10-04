import pandas as pd

def read_one_data():
    file_path = "../../data/asl-signs/train_landmark_files/2044/635217.parquet"
    df = pd.read_parquet(file_path)
    start_indices = [0, 460, 480, 500, 520, 540, 560, 580, ]

    # 各スライスのサイズ
    slice_size = 20

    for start in start_indices:
        end = start + slice_size
        slice_df = df.iloc[start:end]
        print(f"\n=== Rows {start} to {end - 1} ===")
        print(slice_df)

read_one_data()