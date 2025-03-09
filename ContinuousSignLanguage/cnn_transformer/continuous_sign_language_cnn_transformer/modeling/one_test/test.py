import os
import cnn_transformer.continuous_sign_language_cnn_transformer.modeling.one_test.functions as functions
import cnn_transformer.continuous_sign_language_cnn_transformer.modeling.one_test.chatgpt as chatgpt
import cnn_transformer.continuous_sign_language_cnn_transformer.modeling.config as model_config
import torch
import cnn_transformer.continuous_sign_language_cnn_transformer.modeling.one_test.mediapipe_relation as mediapipe_relation

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
    
    