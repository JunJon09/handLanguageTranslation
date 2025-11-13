import cv2
import mediapipe as mp
import numpy as np
from filterpy.kalman import KalmanFilter

# Mediapipeの初期化
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

# モデルごとのランドマーク数
LANDMARK_COUNTS = {
    'hands': 21,        # 片手あたり21点
    'pose': 33,
    'face_mesh': 478
}

def create_kalman_filter_3d():
    kf = KalmanFilter(dim_x=6, dim_z=3)
    # 状態ベクトル [x, y, z, vx, vy, vz]
    kf.F = np.array([
        [1, 0, 0, 1, 0, 0],  # x
        [0, 1, 0, 0, 1, 0],  # y
        [0, 0, 1, 0, 0, 1],  # z
        [0, 0, 0, 1, 0, 0],  # vx
        [0, 0, 0, 0, 1, 0],  # vy
        [0, 0, 0, 0, 0, 1]   # vz
    ])

    # 観測行列
    kf.H = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0]
    ])

    # 観測ノイズ
    kf.R = np.eye(3) * 0.01

    # プロセスノイズ
    kf.Q = np.eye(6) * 0.001

    # 初期状態
    kf.x = np.zeros(6)

    # 共分散行列
    kf.P = np.eye(6) * 1.0

    return kf

# 各モジュールにカルマンフィルタを割り当てる
def initialize_kalman_filters():
    kalman_filters = {
        'left_hand': [create_kalman_filter_3d() for _ in range(LANDMARK_COUNTS['hands'])],
        'right_hand': [create_kalman_filter_3d() for _ in range(LANDMARK_COUNTS['hands'])],
        'pose': [create_kalman_filter_3d() for _ in range(LANDMARK_COUNTS['pose'])],
        'face_mesh': [create_kalman_filter_3d() for _ in range(LANDMARK_COUNTS['face_mesh'])]
    }
    return kalman_filters

kalman_filters = initialize_kalman_filters()

# カメラのキャプチャを開始
cap = cv2.VideoCapture(0)

# 描画用の色設定
COLOR_FILTERED = (0, 0, 0)  # グリーン
COLOR_PREDICTED = (255, 0, 0) # ブルー

# Mediapipeの各モデルを初期化
with mp_hands.Hands(
    max_num_hands=2,       # 左手と右手
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands, \
     mp_pose.Pose(
         static_image_mode=False,
         model_complexity=1,
         enable_segmentation=False,
         min_detection_confidence=0.5
     ) as pose, \
     mp_face_mesh.FaceMesh(
         static_image_mode=False,
         max_num_faces=1,
         refine_landmarks=True,
         min_detection_confidence=0.5,
         min_tracking_confidence=0.5
     ) as face_mesh:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("カメラから映像を取得できません。")
            break

        # 画像を反転（鏡像表示）およびRGB変換
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # パフォーマンス向上のためにフラグを設定
        image.flags.writeable = False

        # 手、ポーズ、フェイスを検出
        results_hands = hands.process(image)
        results_pose = pose.process(image)
        results_face = face_mesh.process(image)

        # 描画用に画像を再度BGRに変換
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 画像サイズ
        h, w, _ = image.shape

        # スムーズなランドマークを保持するリスト
        smoothed_landmarks = {
            'left_hand': [],
            'right_hand': [],
            'pose': [],
            'face_mesh': []
        }

        # === 手のランドマークの処理 ===
        if results_hands.multi_hand_landmarks and results_hands.multi_handedness:
            for hand_landmarks, handedness in zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness):
                # 手の左右を判定
                hand_label = handedness.classification[0].label.lower()  # 'left' or 'right'
                if hand_label == 'left':
                    key = 'left_hand'
                else:
                    key = 'right_hand'

                for idx, lm in enumerate(hand_landmarks.landmark):
                    # 座標をスケール
                    x = lm.x * w
                    y = lm.y * h
                    z = lm.z * w  # zもwでスケーリング

                    # カルマンフィルタに予測と更新を適用
                    kf = kalman_filters[key][idx]
                    kf.predict()
                    kf.update([x, y, z])

                    # フィルタされた座標を取得
                    smoothed_x, smoothed_y, smoothed_z = kf.x[:3]
                    smoothed_landmarks[key].append((int(smoothed_x), int(smoothed_y), smoothed_z))

        else:
            # 手が検出されなかった場合、各手に対して予測のみを行う
            for key in ['left_hand', 'right_hand']:
                for idx in range(LANDMARK_COUNTS['hands']):
                    kf = kalman_filters[key][idx]
                    kf.predict()

                    smoothed_x, smoothed_y, smoothed_z = kf.x[:3]
                    smoothed_landmarks[key].append((int(smoothed_x), int(smoothed_y), int(smoothed_z)))

        # === ポーズのランドマークの処理 ===
        if results_pose.pose_landmarks:
            for idx, lm in enumerate(results_pose.pose_landmarks.landmark):
                x = lm.x * w
                y = lm.y * h
                z = lm.z * w  # zもwでスケーリング

                kf = kalman_filters['pose'][idx]
                kf.predict()
                kf.update([x, y, z])

                smoothed_x, smoothed_y, smoothed_z = kf.x[:3]
                smoothed_landmarks['pose'].append((int(smoothed_x), int(smoothed_y), int(smoothed_z)))
        else:
            # ポーズが検出されなかった場合
            for idx in range(LANDMARK_COUNTS['pose']):
                kf = kalman_filters['pose'][idx]
                kf.predict()

                smoothed_x, smoothed_y, smoothed_z = kf.x[:3]
                smoothed_landmarks['pose'].append((int(smoothed_x), int(smoothed_y), int(smoothed_z)))

        # === フェイスのランドマークの処理 ===
        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    x = lm.x * w
                    y = lm.y * h
                    z = lm.z * w  # zもwでスケーリング

                    kf = kalman_filters['face_mesh'][idx]
                    kf.predict()
                    kf.update([x, y, z])

                    smoothed_x, smoothed_y, smoothed_z = kf.x[:3]
                    smoothed_landmarks['face_mesh'].append((int(smoothed_x), int(smoothed_y), int(smoothed_z)))
        else:
            # フェイスが検出されなかった場合
            for idx in range(LANDMARK_COUNTS['face_mesh']):
                kf = kalman_filters['face_mesh'][idx]
                kf.predict()

                smoothed_x, smoothed_y, smoothed_z = kf.x[:3]
                smoothed_landmarks['face_mesh'].append((int(smoothed_x), int(smoothed_y), int(smoothed_z)))

        # === フィルタされたランドマークの描画 ===
        # 左手
        for point in smoothed_landmarks['left_hand']:
            cv2.circle(image, (point[0], point[1]), 3, COLOR_FILTERED, -1)

        # 右手
        for point in smoothed_landmarks['right_hand']:
            cv2.circle(image, (point[0], point[1]), 3, COLOR_FILTERED, -1)

        # ポーズ
        for point in smoothed_landmarks['pose']:
            cv2.circle(image, (point[0], point[1]), 3, (0, 255, 255), -1)  # シアン

        # フェイス
        for point in smoothed_landmarks['face_mesh']:
            cv2.circle(image, (point[0], point[1]), 1, (255, 0, 255), -1)  # マゼンタ

        # Mediapipeのデフォルトの描画も行う
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if results_pose.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))

        # 結果を表示
        cv2.imshow('MediaPipe with Kalman Filter', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

# リソースの解放
cap.release()
cv2.destroyAllWindows()