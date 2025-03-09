import os
import pandas as pd
import h5py
import numpy as np
import config
import lsa64_relation

ROWS_PER_FRAME = 478 + 33 + 21 + 21  # Number of landmarks per frame.


def get_index_csv_files():
    try:
       track_info = pd.read_csv(config.index_file_path)
       print(track_info)
    except FileNotFoundError:
        print(f"ディレクトリ '{config.index_file_path}' が存在しません。")
    return track_info


def load_relevant_data_subset(pq_path):
    #(T, J, C)
    names = ['x', 'y', 'z']
    data = pd.read_csv(pq_path, usecols=names)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(names))
    print(data[0][478], pq_path)
    return data.astype(np.float32)

def landmarks_csv_to_hdf5(track_info:list , output_dir: str, dictionary: dict, convert_to_channel_first: bool=False):
    
    pids = np.array(track_info["person_number"])
    upids = np.unique(pids)
    upids = track_info['person_number'].unique()
    root_dir = "../../"
    for upid in upids:
        print(upid)
        temp_info = track_info[track_info["person_number"] == upid]
        out_path = os.path.join(output_dir, f"{str(upid).zfill(3)}.hdf5")
        print(f"person_number={upid} のデータを '{out_path}' に保存します。")
        if not os.path.exists(out_path):
            try:
                os.makedirs(output_dir, exist_ok=True)
                print(f"フォルダ '{output_dir}' を作成しました。")
            except Exception as e:
                print(f"フォルダ '{output_dir}' の作成中にエラーが発生しました: {e}")
                return
        with h5py.File(out_path, "w") as f:
            for info in temp_info.itertuples(index=False):
                path = info[0]
                pid = info[1]
                sid = info[2]

                #lsa64
                # sign = lsa64_relation.lsa64_nan_to_none(info[3])
                # token = np.array([dictionary[sign]])

                #minimum_continuous_hand_language
                sign = [data for data in info[3].split(",")]
                token = np.array(sign, dtype=int)
                
                assert pid == upid, f"{pid}:{upid}"
                track_path = "../.." + path
               
                if not os.path.exists(track_path):
                    continue

                track = load_relevant_data_subset(track_path)
                # `[T, J, C] -> [C, T, J]`
                if convert_to_channel_first:
                    track = track.transpose([2, 0, 1])

                # Create group.
                grp = f.create_group(str(sid))
                grp.create_dataset("feature", data=track)
                grp.create_dataset("token", data=token)



if __name__ == "__main__":
    track_info = get_index_csv_files()
    landmarks_csv_to_hdf5(track_info, output_dir=config.out_dir, dictionary=lsa64_relation.lsa64_key_value, convert_to_channel_first=True)