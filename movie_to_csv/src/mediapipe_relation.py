import cv2
import mediapipe as mp
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import config


class MediaPipeClass:

    def __init__(self):
        # FaceMeshの鼻先のランドマークインデックス
        self.nose_tip_index = 4
        # 前フレームの情報を保存
        self.previous_ear_distance = None
        self.previous_face_center = None
        self.previous_face_landmarks = None

    def validate_nose_position(
        self,
        face_landmarks,
        pose_nose_x,
        pose_nose_y,
        face_region_x1,
        face_region_y1,
        face_region_width,
        face_region_height,
        threshold=0.3,
    ):
        """
        FaceMeshの鼻の位置とPoseの鼻の位置を比較して妥当性を検証する

        Args:
            face_landmarks: FaceMeshのランドマーク
            pose_nose_x, pose_nose_y: Poseで検出された鼻の画像座標
            face_region_x1, face_region_y1: 顔領域の左上座標
            face_region_width, face_region_height: 顔領域のサイズ
            threshold: 許容誤差の閾値（顔領域サイズに対する比率）

        Returns:
            bool: 妥当性（True: 有効, False: 無効）
        """
        try:
            if (
                not face_landmarks
                or len(face_landmarks.landmark) <= self.nose_tip_index
            ):
                return False

            # FaceMeshの鼻先座標を取得（顔領域内の相対座標）
            face_nose = face_landmarks.landmark[self.nose_tip_index]

            # FaceMeshの鼻の座標を画像座標系に変換
            face_nose_x = face_region_x1 + (face_nose.x * face_region_width)
            face_nose_y = face_region_y1 + (face_nose.y * face_region_height)

            # 距離を計算
            distance = np.sqrt(
                (face_nose_x - pose_nose_x) ** 2 + (face_nose_y - pose_nose_y) ** 2
            )

            # 顔領域のサイズに基づく許容範囲を計算
            tolerance = min(face_region_width, face_region_height) * threshold

            # 距離が許容範囲内かチェック
            is_valid = distance <= tolerance

            if not is_valid:
                print(f"鼻の位置が不整合: 距離={distance:.2f}, 許容値={tolerance:.2f}")

            return is_valid

        except Exception as e:
            print(f"鼻の位置検証エラー: {e}")
            return False

    def validate_face_size(self, ear_distance, min_face_size=50, max_face_size=300):
        """
        耳までの長さで顔の大きさが適切かを判定する

        Args:
            ear_distance: 耳と耳の間の距離（ピクセル）
            min_face_size: 最小の顔サイズ（ピクセル）
            max_face_size: 最大の顔サイズ（ピクセル）

        Returns:
            bool: 顔サイズの妥当性（True: 適切, False: 不適切）
        """
        try:
            is_valid = min_face_size <= ear_distance <= max_face_size

            if not is_valid:
                if ear_distance < min_face_size:
                    print(
                        f"顔が小さすぎます: 耳間距離={ear_distance}px (最小値: {min_face_size}px)"
                    )
                elif ear_distance > max_face_size:
                    print(
                        f"顔が大きすぎます: 耳間距離={ear_distance}px (最大値: {max_face_size}px)"
                    )
            else:
                print(f"顔サイズ適切: 耳間距離={ear_distance}px")

            return is_valid

        except Exception as e:
            print(f"顔サイズ検証エラー: {e}")
            return False

    def validate_face_stability(
        self,
        current_ear_distance,
        current_face_center,
        current_face_landmarks,
        size_change_threshold=0.3,
        position_change_threshold=50,
        landmark_change_threshold=0.1,
    ):
        """
        前フレームと比較して顔の形や位置が急激に変化していないかを検証する

        Args:
            current_ear_distance: 現在フレームの耳間距離
            current_face_center: 現在フレームの顔の中心座標 (x, y)
            current_face_landmarks: 現在フレームの顔ランドマーク
            size_change_threshold: 顔サイズ変化の許容閾値（比率）
            position_change_threshold: 顔位置変化の許容閾値（ピクセル）
            landmark_change_threshold: ランドマーク変化の許容閾値（比率）

        Returns:
            bool: 安定性（True: 安定, False: 不安定）
        """
        try:
            # 初回フレームは常に有効
            if (
                self.previous_ear_distance is None
                or self.previous_face_center is None
                or self.previous_face_landmarks is None
            ):
                return True

            # 顔サイズの変化をチェック
            size_change_ratio = (
                abs(current_ear_distance - self.previous_ear_distance)
                / self.previous_ear_distance
            )
            if size_change_ratio > size_change_threshold:
                print(
                    f"顔サイズが急激に変化: 変化率={size_change_ratio:.3f} (閾値: {size_change_threshold})"
                )
                return False

            # 顔位置の変化をチェック
            position_distance = np.sqrt(
                (current_face_center[0] - self.previous_face_center[0]) ** 2
                + (current_face_center[1] - self.previous_face_center[1]) ** 2
            )
            if position_distance > position_change_threshold:
                print(
                    f"顔位置が急激に変化: 移動距離={position_distance:.1f}px (閾値: {position_change_threshold}px)"
                )
                return False

            # 主要ランドマークの変化をチェック（鼻、左右の口角、顎先など）
            key_landmark_indices = [
                self.nose_tip_index,
                61,
                291,
                17,
            ]  # 鼻、左口角、右口角、顎先

            for idx in key_landmark_indices:
                if idx < len(current_face_landmarks.landmark) and idx < len(
                    self.previous_face_landmarks.landmark
                ):

                    current_lm = current_face_landmarks.landmark[idx]
                    previous_lm = self.previous_face_landmarks.landmark[idx]

                    # ランドマークの移動距離を計算
                    landmark_distance = np.sqrt(
                        (current_lm.x - previous_lm.x) ** 2
                        + (current_lm.y - previous_lm.y) ** 2
                    )

                    if landmark_distance > landmark_change_threshold:
                        print(
                            f"主要ランドマーク{idx}が急激に変化: 移動距離={landmark_distance:.3f} (閾値: {landmark_change_threshold})"
                        )
                        return False

            return True

        except Exception as e:
            print(f"顔安定性検証エラー: {e}")
            return True  # エラーの場合は通す

    def update_previous_face_info(self, ear_distance, face_center, face_landmarks):
        """
        前フレーム情報を更新する
        """
        self.previous_ear_distance = ear_distance
        self.previous_face_center = face_center
        self.previous_face_landmarks = face_landmarks

    def extract_face_region(self, image, pose_landmarks) -> Optional[np.ndarray]:
        """
        Poseの肩幅を基準に顔領域を動的に抽出する

        Args:
            image: 入力画像 (numpy array)
            pose_landmarks: Poseのランドマーク

        Returns:
            face_image: 抽出された顔画像（失敗時はNone）
        """
        try:
            if pose_landmarks is None:
                return None

            # 画像のサイズを取得
            h, w = image.shape[:2]

            # Poseランドマークのインデックス
            # 7: 左耳, 8: 右耳, 0: 鼻
            left_ear_idx = 7
            right_ear_idx = 8
            nose_idx = 0

            # MediaPipeの新しいAPIに対応した処理
            if hasattr(pose_landmarks, "landmark"):
                # 従来のAPI
                landmarks = pose_landmarks.landmark
            elif hasattr(pose_landmarks, "pose_landmarks"):
                # 新しいAPI
                landmarks = pose_landmarks.pose_landmarks
            else:
                print(f"Pose landmarks type: {type(pose_landmarks)}")
                print(f"Pose landmarks attributes: {dir(pose_landmarks)}")
                return None

            # ランドマークがリストでない場合の処理
            if not landmarks:
                return None

            # リストの長さチェック
            if len(landmarks) <= max(left_ear_idx, right_ear_idx, nose_idx):
                return None

            # 耳の座標を取得（画像座標に変換）
            left_ear = landmarks[left_ear_idx]
            right_ear = landmarks[right_ear_idx]
            nose = landmarks[nose_idx]

            # 座標を画像座標系に変換
            left_ear_x = int(left_ear.x * w)
            left_ear_y = int(left_ear.y * h)
            right_ear_x = int(right_ear.x * w)
            right_ear_y = int(right_ear.y * h)
            nose_x = int(nose.x * w)
            nose_y = int(nose.y * h)

            # 耳間距離（X）を計算
            ear_distance = abs(left_ear_x - right_ear_x)

            # 顔が小さすぎる場合はスキップ
            if ear_distance < 20:
                print("顔が小さすぎて抽出できません。")
                return None

            # 顔領域のサイズを計算
            face_width = int(ear_distance * 1.7)  # は耳間距離の1.1倍
            face_height = int(ear_distance * 2.0)  # 縦幅は耳間距離の1.5倍

            center_x = nose_x
            center_y = nose_y

            # 抽出領域の座標を計算
            x1 = max(0, center_x - face_width // 2)
            y1 = max(0, center_y - face_height // 2)
            x2 = min(w, center_x + face_width // 2)
            y2 = min(h, center_y + face_height // 2)

            # 顔領域を抽出
            face_region = image[y1:y2, x1:x2]

            # 抽出された領域が有効かチェック
            if face_region.size == 0:
                return None

            return face_region

        except Exception as e:
            print(f"顔領域抽出エラー: {e}")
            return None

    def get_skeleton_by_mediapipe(self, input_file: str) -> List[Dict[str, Any]]:
        mp_face_mesh = mp.solutions.face_mesh
        mp_pose = mp.solutions.pose
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        landmarks_list = []

        cap = cv2.VideoCapture(input_file)

        if not cap.isOpened():
            print(f"Error: 動画ファイル '{input_file}' を開くことができません。")
            return landmarks_list

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames to process: {frame_count}")

        fps = cap.get(cv2.CAP_PROP_FPS)

        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=0.3,  # 0.5から0.3に下げる
            min_tracking_confidence=0.3,
        ) as face_mesh, mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # 1から0に変更してパフォーマンスを改善
            enable_segmentation=False,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        ) as pose, mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands:

            frame_idx = 0
            while True:
                success, frame = cap.read()
                if not success:
                    break

                # 元のフレームをコピーして保存（骨格描画前の状態）
                original_frame = frame.copy()

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # face_results = face_mesh.process(image_rgb)
                pose_results = pose.process(image_rgb)
                hands_results = hands.process(image_rgb)

                # 顔領域を抽出（元のフレームから）
                face_region = None
                if pose_results.pose_landmarks:
                    face_region = self.extract_face_region(
                        original_frame, pose_results.pose_landmarks
                    )

                frame_landmarks = {
                    "frame_index": frame_idx,
                    "face": [],
                    "pose": [],
                    "left_hand": [],
                    "right_hand": [],
                }

                # if face_results.multi_face_landmarks:
                #     for face_landmark in face_results.multi_face_landmarks:
                #         for lm in face_landmark.landmark:
                #             frame_landmarks['face'].append({
                #                 'x': lm.x,
                #                 'y': lm.y,
                #                 'z': lm.z
                #             })
                #     mp_drawing.draw_landmarks(
                #         image=frame,
                #         landmark_list=face_landmark,
                #         connections=mp_face_mesh.FACEMESH_TESSELATION,
                #         landmark_drawing_spec=None,
                #         connection_drawing_spec=mp_drawing_styles
                #             .get_default_face_mesh_tesselation_style())
                # else:
                #     empty_landmarks = [{'x': None, 'y': None, 'z': None} for _ in range(config.face_landmark_number)]
                #     frame_landmarks['face'].extend(empty_landmarks)

                if pose_results.pose_landmarks:
                    for lm in pose_results.pose_landmarks.landmark:
                        frame_landmarks["pose"].append(
                            {"x": lm.x, "y": lm.y, "z": lm.z}
                        )
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=pose_results.pose_landmarks,
                        connections=mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                    )
                else:
                    empty_landmarks = [
                        {"x": None, "y": None, "z": None}
                        for _ in range(config.pose_landmark_number)
                    ]
                    frame_landmarks["pose"].extend(empty_landmarks)

                if (
                    hands_results.multi_hand_landmarks
                    and hands_results.multi_handedness
                ):
                    for hand_landmark, handedness in zip(
                        hands_results.multi_hand_landmarks,
                        hands_results.multi_handedness,
                    ):
                        # 手の種類（左/右）を判定
                        hand_label = handedness.classification[
                            0
                        ].label.lower()  # 'left' または 'right'
                        hand_key = "left_hand" if hand_label == "left" else "right_hand"

                        # ランドマークを追加
                        for lm in hand_landmark.landmark:
                            frame_landmarks[hand_key].append(
                                {"x": lm.x, "y": lm.y, "z": lm.z}
                            )
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=hand_landmark,
                            connections=mp_hands.HAND_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                            connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
                        )

                # フレームにテキスト情報を追加
                cv2.putText(
                    frame,
                    f"Frame: {frame_idx}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

                # フレームを表示
                cv2.imshow("MediaPipe Landmarks", frame)

                face_only_result = None
                if (
                    face_region is not None
                    and face_region.size > 0
                    and pose_results.pose_landmarks
                ):
                    # BGRからRGBに変換
                    face_region_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
                    # FaceMeshで処理
                    face_only_result = face_mesh.process(face_region_rgb)

                    if face_only_result.multi_face_landmarks:
                        # Poseの鼻の座標を取得
                        h, w = original_frame.shape[:2]
                        pose_nose = pose_results.pose_landmarks.landmark[
                            0
                        ]  # 鼻のインデックス
                        pose_nose_x = pose_nose.x * w
                        pose_nose_y = pose_nose.y * h

                        # 顔領域の情報を取得（extract_face_regionから）
                        landmarks = pose_results.pose_landmarks.landmark
                        left_ear = landmarks[7]
                        right_ear = landmarks[8]
                        nose = landmarks[0]

                        left_ear_x = int(left_ear.x * w)
                        right_ear_x = int(right_ear.x * w)
                        nose_x = int(nose.x * w)
                        nose_y = int(nose.y * h)

                        ear_distance = abs(left_ear_x - right_ear_x)
                        face_center = (nose_x, nose_y)

                        # 顔のサイズが適切かチェック
                        if not self.validate_face_size(ear_distance):
                            print(
                                "顔サイズが不適切なため、FaceMeshの処理をスキップします"
                            )
                            face_only_result = None
                        else:
                            face_landmark = face_only_result.multi_face_landmarks[
                                0
                            ]  # 最初の顔を直接取得

                            if self.validate_face_stability(
                                ear_distance,
                                face_center,
                                face_landmark,
                                size_change_threshold=0.3,
                                position_change_threshold=50,
                                landmark_change_threshold=0.1,
                            ):
                                face_width = int(ear_distance * 1.7)
                                face_height = int(ear_distance * 2.0)

                                x1 = max(0, nose_x - face_width // 2)
                                y1 = max(0, nose_y - face_height // 2)

                                # 鼻の位置も検証
                                if self.validate_nose_position(
                                    face_landmark,
                                    pose_nose_x,
                                    pose_nose_y,
                                    x1,
                                    y1,
                                    face_width,
                                    face_height,
                                ):
                                    # 検証が通った場合のみランドマークを追加
                                    for lm in face_landmark.landmark:
                                        frame_landmarks["face"].append(
                                            {"x": lm.x, "y": lm.y, "z": lm.z}
                                        )
                                    # 有効な場合は前フレーム情報を更新
                                    self.update_previous_face_info(
                                        ear_distance, face_center, face_landmark
                                    )
                                else:
                                    print("鼻の位置検証により、顔を無効化しました")
                                    face_only_result = None
                            else:
                                print("前フレームとの比較により、顔を無効化しました")
                                face_only_result = None

                # face_landmarksが空の場合は空のランドマークで埋める
                if face_only_result is None:
                    empty_landmarks = [
                        {"x": None, "y": None, "z": None}
                        for _ in range(config.face_landmark_number)
                    ]
                    frame_landmarks["face"].extend(empty_landmarks)

                # 抽出された顔領域を別ウィンドウで表示
                if face_only_result is not None and face_region is not None:
                    # FaceMeshの結果も表示（検証が通った場合のみ）
                    if (
                        face_only_result
                        and face_only_result.multi_face_landmarks
                        and len(frame_landmarks["face"]) > 0
                        and frame_landmarks["face"][0]["x"] is not None
                    ):
                        # FaceMeshのランドマークを顔領域に描画
                        face_with_landmarks = face_region.copy()
                        mp_drawing.draw_landmarks(
                            image=face_with_landmarks,
                            landmark_list=face_only_result.multi_face_landmarks[0],
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                        )
                        display_face_landmarks = cv2.resize(
                            face_with_landmarks, (200, 300)
                        )
                        cv2.imshow("Face Region with Landmarks", display_face_landmarks)
                else:
                    print("顔領域が抽出できなかったため、ランドマークを表示しません。")
                    cv2.imshow("Face Region with Landmarks", face_region)

                # # ウィンドウを閉じるために 'q' キーを押す
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("ユーザーによって処理が中断されました。")
                    break

                if len(frame_landmarks["left_hand"]) == 0:
                    empty_landmarks = [
                        {"x": None, "y": None, "z": None}
                        for _ in range(config.left_landmark_number)
                    ]
                    frame_landmarks["left_hand"].extend(empty_landmarks)
                if len(frame_landmarks["right_hand"]) == 0:
                    empty_landmarks = [
                        {"x": None, "y": None, "z": None}
                        for _ in range(config.right_landmark_number)
                    ]
                    frame_landmarks["right_hand"].extend(empty_landmarks)

                # フレームのランドマークデータをリストに追加
                if (
                    len(frame_landmarks["face"]) == config.face_landmark_number
                    and len(frame_landmarks["pose"]) == config.pose_landmark_number
                    and len(frame_landmarks["left_hand"]) == config.left_landmark_number
                    and len(frame_landmarks["right_hand"])
                    == config.right_landmark_number
                ):
                    landmarks_list.append(frame_landmarks)
                else:
                    print(
                        len(frame_landmarks["face"]),
                        len(frame_landmarks["pose"]),
                        len(frame_landmarks["left_hand"]),
                        len(frame_landmarks["right_hand"]),
                    )

                frame_idx += 1
                if frame_idx % 100 == 0:
                    print(f"Processed {frame_idx} / {frame_count} frames...")
        cap.release()

        print("ランドマークの抽出が完了しました。")
        return landmarks_list
