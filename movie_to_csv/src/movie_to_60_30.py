import subprocess
import config
import os

def get_movie_files():
    movie_files_list = []
    try:
        files = os.listdir(config.movie_directory_path)
        directory_list = [file for file in files if file!=".DS_Store" and file!= "訓練" and file!="テスト"]
        for word in directory_list:
            full_path = os.path.join(config.movie_directory_path, word)
            files = os.listdir(full_path)
            movie_files = [os.path.join(full_path, file) for file in files if file != ".DS_Store" and file.lower().endswith('.mp4')]
            movie_files_list.append(sorted(movie_files))
        return sorted(movie_files_list)
    except FileNotFoundError:
        print(f"ディレクトリ '{config.movie_directory_path}' が存在しません。")

def change_video_fps(input_path, output_path, target_fps=30):
    """
    FFmpegを使用して動画のFPSを変更
    
    Parameters:
    - input_path: 入力動画のパス
    - output_path: 出力動画のパス
    - target_fps: 目標のFPS（デフォルト30）
    """
    try:
        # FFmpegコマンドを構築
        command = [
            'ffmpeg', 
            '-i', input_path,      # 入力ファイル
            '-filter:v', f'fps={target_fps}',  # FPSフィルター 
            output_path             # 出力ファイル
        ]
        
        # サブプロセスの実行
        result = subprocess.run(command, capture_output=True, text=True)
        
        # エラーチェック
        if result.returncode != 0:
            print("Error occurred:")
            print(result.stderr)
        else:
            print(f"Successfully converted to {target_fps} FPS")
        
    except Exception as e:
        print(f"An error occurred: {e}")

# 使用例
if __name__ == "__main__":
    movie_files = get_movie_files()
    for movie_file in movie_files:
        for file in movie_file:
            # output_file = file.replace('one_word', 'one_word_30fps')
            # print(output_file)
            change_video_fps(file, file.replace('.mp4', '_30fps.mp4'), target_fps=30)
    # change_video_fps('input_video.mp4', 'output_video.mp4', 30)