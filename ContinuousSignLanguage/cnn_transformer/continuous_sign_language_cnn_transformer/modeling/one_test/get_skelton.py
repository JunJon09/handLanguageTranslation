import os
import pandas as pd
import random
import cnn_transformer.continuous_sign_language_cnn_transformer.modeling.one_test.mediapipe_relation as mediapipe_relation

def write_csv(landmarks_list: list, output_csv: str, person_number: int):
    records = []

    for frame in landmarks_list:
        frame_index = frame.get('frame_index', None)
        if frame_index is None:
            continue  # フレームインデックスがない場合はスキップ
        for landmark_label in ['face', 'pose', 'left_hand', 'right_hand']:
            landmarks = frame.get(landmark_label, [])
            for idx, lm in enumerate(landmarks):
                record = {
                    'frame_index': frame_index,
                    'landmark_label': landmark_label,
                    'landmark_id': idx,
                    'x': lm.get('x', None),
                    'y': lm.get('y', None),
                    'z': lm.get('z', None),
                    'person_number': person_number
                }
                records.append(record)
    
    # DataFrame を作成
    df = pd.DataFrame(records)
    
    output_dir = os.path.dirname(output_csv)
    
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"フォルダ '{output_dir}' を作成しました。")
        except Exception as e:
            return
    # CSV に書き出し
    df.to_csv(output_csv, index=False)


def restore_nhk(file, i, j):
    file_split = file.split("/")
    folder_name = file_split[4] #ex:America
    file_name = str((i+1) * 1000 + j)
    csv_path = "/csv/nhk/" + folder_name + "/" + file_name + ".csv"
    output_csv_path = "../../" + csv_path
    person_number = str(random.randint(10000, 10010))
    
    return output_csv_path, person_number, csv_path, file_name, folder_name

def restore_lsa64(file):
    file_split = file.split("/")
    folder_name = file_split[4] #ex:America
    file_name = file_split[5].replace(".mp4", ".csv") #001_001_001.mp4
    csv_path = "/csv/lsa64/" + folder_name + "/" + file_name
    output_csv_path = "../.." + csv_path
    person_number = file_split[5].split("_")[1] #001
    print( output_csv_path, person_number, csv_path, file_name, folder_name)
    return output_csv_path, person_number, csv_path, file_name, folder_name

def restore_minimum_continuous_hand_language(file, i, j):
    file_split = file.split("/")
    folder_name = file_split[4] #ex:morning
    file_name = str((i+1) * 1000 + j)
    csv_path = "/csv/minimum_continuous_hand_language/" + folder_name + "/" + file_name + ".csv"
    output_csv_path = "../../" + csv_path
    """
    動画は、0~4, 5~9, 10~14, 15~19と4人の人で構成されている
    """
    if 0<=j and j<=4: 
        person_number = "001"
    elif 5<= j and j<= 9:
        person_number = "002"
    elif 10<=j and j<= 14:
        person_number = "003"
    elif 15<=j and j<=19:
        person_number = "004"
    else:
        person_number = "005"

    return output_csv_path, person_number, csv_path, file_name, folder_name

# if __name__ == "__main__":
#     file_path = "../data/one_test/test.mp4"
#     MediaPipeClass = mediapipe_relation.MediaPipeClass()
#     landmarks_list = MediaPipeClass.get_skeleton_by_mediapipe(file_path)
#     print(len(landmarks_list), len(landmarks_list[0]))
