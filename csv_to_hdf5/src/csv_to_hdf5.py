import os
import glob
import pandas as pd
import h5py
import numpy as np

ROWS_PER_FRAME = 478 + 33 + 21 + 21  # Number of landmarks per frame.


def get_index_csv_files():
    file_path = "../../csv/nhk/index.csv"
    try:
       track_info = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"ディレクトリ '{file_path}' が存在しません。")
    return track_info


def load_relevant_data_subset(pq_path):
    names = ['x', 'y', 'z']
    data = pd.read_csv(pq_path, usecols=names)
    print(data)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(names))
    return data.astype(np.float32)

def landmarks_csv_to_hdf5(track_info:list , output_dir: str, dictionary: dict, convert_to_channel_first: bool=False):
    
    pids = np.array(track_info["person_number"])
    upids = np.unique(pids)
    upids = track_info['person_number'].unique()
    root_dir = "../../"
    for upid in upids:
        temp_info = track_info[track_info["person_number"] == upid]
        out_path = os.path.join(output_dir, f"{upid}.hdf5")
        print(f"person_number={upid} のデータを '{out_path}' に保存します。")

        with h5py.File(out_path, "w") as f:
            for info in temp_info.itertuples(index=False):
                path = info[0]
                pid = info[1]
                sid = info[2]
                token = np.array([dictionary[info[3]]])
                assert pid == upid, f"{pid}:{upid}"
                track_path = "../.." + path
                print(track_path)
                if not os.path.exists(track_path):
                    continue

                track = load_relevant_data_subset(track_path)
                print(track)
                # `[T, J, C] -> [C, T, J]`
                if convert_to_channel_first:
                    track = track.transpose([2, 0, 1])

                # Create group.
                grp = f.create_group(str(sid))
                grp.create_dataset("feature", data=track)
                grp.create_dataset("token", data=token)



if __name__ == "__main__":
    track_info = get_index_csv_files()
    output_dir = "../../hdf5/nhk/"
    dictionary = {
        "America": 0,
        "china": 1,
        "commitee": 2,
        "confirm": 3,
        "corona": 4,
        "emergency": 5,
        "expand": 6,
        "gorvement": 7,
        "infection": 8,
        "jpan": 9,
        "measure": 10,
        "minister": 11,
        "osaka": 12,
        "person": 13,
        "policy": 14,
        "prezen": 15,
        "state": 16,
        "today": 17,
        "tokyo": 18,
        "vaccination": 19,
    }
    landmarks_csv_to_hdf5(track_info, output_dir=output_dir, dictionary=dictionary, convert_to_channel_first=True)