import cv2
from mmpose.apis import MMPoseInferencer

# ===== 設定 =====
# 入力を選べます
# Webカメラの場合: 0 (通常は内蔵カメラ)
# 動画ファイルの場合: "path/to/your/video.mp4"
INPUT_SOURCE = "/Users/jonmac/jon/研究/手話/handLangageTranslation/data/middle_dataset_30fps/friend_with_station_meet/01_30fps.mp4"

# 使用するモデルのエイリアスを指定します
# 'human' は一般的な人間向けの高精度モデル(HRNet)です
MODEL_ALIAS = 'human'
# =================

def main():
    # 1. MMPose推論器を初期化
    print(f"'{MODEL_ALIAS}' モデルを読み込んでいます...")
    # 'cpu' or 'cuda:0' などデバイスを指定できます
    inferencer = MMPoseInferencer(MODEL_ALIAS, device='cpu') 
    print("モデルの読み込みが完了しました。")

    # 2. 動画ソースを準備
    cap = cv2.VideoCapture(INPUT_SOURCE)
    if not cap.isOpened():
        print(f"エラー: {INPUT_SOURCE} を開けませんでした。")
        return

    print("推論を開始します。'q'キーを押すと終了します。")

    # 3. フレームごとにループ処理
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("動画の終端に達したか、フレームを読み込めませんでした。")
            break

        # 4. MMPoseで推論を実行
        # result_generatorはジェネレータを返すので、next()で結果を取得
        result_generator = inferencer(frame, return_vis=True, show=False)
        result = next(result_generator)

        # 5. 結果画像を取得して表示
        visualized_frame = result['visualization'][0]
        cv2.imshow('MMPose Real-time Demo', visualized_frame)

        # 6. 'q'キーが押されたら終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 7. 後処理
    cap.release()
    cv2.destroyAllWindows()
    print("処理を終了しました。")

if __name__ == '__main__':
    main()