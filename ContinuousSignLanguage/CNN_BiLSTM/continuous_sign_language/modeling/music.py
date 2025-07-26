from playsound import playsound
import threading

def play_sound_simple(file_path):
    """
    指定された音声ファイルを再生します。
    再生中は処理がブロックされます。

    Args:
        file_path (str): 再生する音声ファイルのパス。
                         例: 'path/to/your/sound.wav', 'path/to/your/sound.mp3'
    """
    print(f"Playing {file_path}...")
    try:
        playsound(file_path)
        print(f"Finished playing {file_path}.")
    except Exception as e:
        print(f"Error playing sound: {e}")

def play_sound_async(file_path):
    """
    指定された音声ファイルを非同期で再生します。
    再生中にメインの処理をブロックしません。

    Args:
        file_path (str): 再生する音声ファイルのパス。
                         例: 'path/to/your/sound.wav', 'path/to/your/sound.mp3'
    """
    print(f"Playing {file_path} asynchronously...")
    try:
        # スレッドを作成し、その中でplaysoundを呼び出す
        thread = threading.Thread(target=playsound, args=(file_path,))
        thread.start()
        print(f"Started playing {file_path} in a separate thread.")
    except Exception as e:
        print(f"Error starting sound playback: {e}")
