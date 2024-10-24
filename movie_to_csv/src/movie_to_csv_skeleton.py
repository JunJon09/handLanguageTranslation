import os
import pandas as pd
import mediapipe_relation
import random
import config


def get_movie_files():
    movie_files_list = []
    try:
        files = os.listdir(config.movie_directory_path)
        directory_list = [file for file in files if file!=".DS_Store" and file!= "訓練" and file!="テスト"]
        for word in directory_list:
            full_path = os.path.join(config.movie_directory_path, word)
            files = os.listdir(full_path)
            movie_files = [os.path.join(full_path, file) for file in files if file != ".DS_Store" and file.lower().endswith('.mp4')]
            movie_files_list.append(movie_files)
        return sorted(movie_files_list)
    except FileNotFoundError:
        print(f"ディレクトリ '{config.movie_directory_path}' が存在しません。")

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

if __name__ == "__main__":
    movie_files_list = get_movie_files()
    MediaPipeClass = mediapipe_relation.MediaPipeClass()
    index_records = []
    

    for i, word_directory in enumerate(movie_files_list):
        for j, file in enumerate(word_directory):
            landmarks_list = MediaPipeClass.get_skeleton_by_mediapipe(file)
            if len(landmarks_list) <= 5 :
                continue
            #NHK
            #output_csv_path, person_number, csv_path, file_name, folder_name = restore_nhk(file, i, j)
            
            #LSA64
            output_csv_path, person_number, csv_path, file_name, folder_name = restore_lsa64(file)
            
            write_csv(landmarks_list, output_csv_path, person_number)
            index_record = {
                'path': csv_path,
                'person_number': person_number,
                'file_name': file_name,
                'sign': folder_name
            }
            index_records.append(index_record)
    index_csv_path = "../../csv/lsa64/index.csv"
    create_index_csv(index_records, index_csv_path)
