import os
import cnn_transformer.continuous_sign_language_cnn_transformer.modeling.one_test.functions as functions
import cnn_transformer.continuous_sign_language_cnn_transformer.modeling.one_test.chatgpt as chatgpt
import cnn_transformer.continuous_sign_language_cnn_transformer.modeling.config as model_config
import torch
import cnn_transformer.continuous_sign_language_cnn_transformer.modeling.one_test.mediapipe_relation as mediapipe_relation
import tkinter as tk
import time
import threading
import locale

def close_windows():
    # 10秒後にウィンドウを閉じる
    time.sleep(60)
    root1.destroy()
    root2.destroy()

root1 = ""
root2 = ""

if __name__ == "__main__":
    file_path = "../data/one_test/test.mp4"
    tokens_list = {1: "挨拶", 2: "朝", 3: "昼", 4: "夜", 5: "私", 7: "バナナ", 8: "好き", 9: "食べ物"}
    MediaPipeClass = mediapipe_relation.MediaPipeClass()
    landmarks_list = MediaPipeClass.get_skeleton_by_mediapipe(file_path)
    save_path = os.path.join(model_config.model_save_dir, model_config.model_save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_loaded, optimizer_loaded, epoch_loaded, test_dataloader, key2token = functions.load_model(save_path, device)

    max_seqlen = model_config.max_seqlen

    sos_token = key2token["<sos>"]
    eos_token = key2token["<eos>"]
    tokens, pred_ids = functions.test_loop_csir_s2s(
                test_dataloader, model_loaded, device,
                sos_token, eos_token,
                max_seqlen=max_seqlen,
                return_pred_times=True,
                verbose_num=0)

    words = [tokens_list[token] for token in tokens]
    result = ' '.join(words)
    print("認識結果: {}".format(result))
    replay = chatgpt.word_translate(result)
    
    print("日本語の翻訳結果: {}".format(replay))

    # システムのデフォルトエンコーディングを取得
    default_encoding = locale.getpreferredencoding()

    root1 = tk.Tk()
    root1.title("認識結果")
    root1.geometry("800x200")

    # フォントを日本語対応のフォントに変更
    label1 = tk.Label(root1, text=result, font=("Noto Sans CJK JP", 50))
    label1.pack(padx=20, pady=20)

    # 2つ目のウィンドウ
    root2 = tk.Toplevel(root1)
    root2.title("日本語翻訳結果")
    root2.geometry("1200x200")
    label2 = tk.Label(root2, text=replay, font=("Noto Sans CJK JP", 50))
    label2.pack(padx=20, pady=20)
    # タイマースレッドを開始
    timer_thread = threading.Thread(target=close_windows)
    timer_thread.daemon = True
    timer_thread.start()

    root1.mainloop()
    
