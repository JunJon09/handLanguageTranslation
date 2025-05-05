import cv2
import mediapipe as mp
from typing import List, Dict, Any
import config

class MediaPipeClass:

    def get_skeleton_by_mediapipe(self, input_file:str) -> List[Dict[str, Any]]:
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

        with mp_face_mesh.FaceMesh(static_image_mode=False,
                        max_num_faces=10,
                        refine_landmarks=True,
                        min_detection_confidence=0.05,  # 0.5から0.3に下げる
                        min_tracking_confidence=0.05) as face_mesh, \
            mp_pose.Pose(static_image_mode=False,
                        model_complexity=0,  # 1から0に変更してパフォーマンスを改善
                        enable_segmentation=False,
                        min_detection_confidence=0.3,
                        min_tracking_confidence=0.3) as pose, \
            mp_hands.Hands(static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.3,
                        min_tracking_confidence=0.3) as hands:
        
            frame_idx = 0
            while True:
                success, frame = cap.read()
                if not success:
                    break

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                face_results = face_mesh.process(image_rgb)
                pose_results = pose.process(image_rgb)
                hands_results = hands.process(image_rgb)

                frame_landmarks = {
                    'frame_index': frame_idx,
                    'face': [],
                    'pose': [],
                    'left_hand': [],
                    'right_hand': []
                }

                if face_results.multi_face_landmarks:
                    for face_landmark in face_results.multi_face_landmarks:
                        for lm in face_landmark.landmark:
                            frame_landmarks['face'].append({
                                'x': lm.x,
                                'y': lm.y,
                                'z': lm.z
                            })
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmark,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_tesselation_style())
                else:
                    empty_landmarks = [{'x': None, 'y': None, 'z': None} for _ in range(config.face_landmark_number)]
                    frame_landmarks['face'].extend(empty_landmarks)

                if pose_results.pose_landmarks:
                    for lm in pose_results.pose_landmarks.landmark:
                        frame_landmarks['pose'].append({
                            'x': lm.x,
                            'y': lm.y,
                            'z': lm.z
                        })
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=pose_results.pose_landmarks,
                        connections=mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                else:
                    empty_landmarks = [{'x': None, 'y': None, 'z': None} for _ in range(config.pose_landmark_number)]
                    frame_landmarks['pose'].extend(empty_landmarks)

                if hands_results.multi_hand_landmarks and hands_results.multi_handedness:
                    for hand_landmark, handedness in zip(hands_results.multi_hand_landmarks, hands_results.multi_handedness):
                        # 手の種類（左/右）を判定
                        hand_label = handedness.classification[0].label.lower()  # 'left' または 'right'
                        hand_key = 'left_hand' if hand_label == 'left' else 'right_hand'

                        # ランドマークを追加
                        for lm in hand_landmark.landmark:
                            frame_landmarks[hand_key].append({
                                'x': lm.x,
                                'y': lm.y,
                                'z': lm.z
                            })
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=hand_landmark,
                            connections=mp_hands.HAND_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                            connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
                        )
                

                 # フレームにテキスト情報を追加
                cv2.putText(frame, f'Frame: {frame_idx}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                # フレームを表示
                cv2.imshow('MediaPipe Landmarks', frame)

                # # ウィンドウを閉じるために 'q' キーを押す
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("ユーザーによって処理が中断されました。")
                    break

                if len(frame_landmarks['left_hand']) == 0:
                    empty_landmarks = [{'x': None, 'y': None, 'z': None} for _ in range(config.left_landmark_number)]
                    frame_landmarks['left_hand'].extend(empty_landmarks)
                if len(frame_landmarks['right_hand']) == 0:
                    empty_landmarks = [{'x': None, 'y': None, 'z': None} for _ in range(config.right_landmark_number)]
                    frame_landmarks['right_hand'].extend(empty_landmarks)


                # フレームのランドマークデータをリストに追加
                if len(frame_landmarks['face']) == config.face_landmark_number and len(frame_landmarks['pose'])==config.pose_landmark_number and len(frame_landmarks['left_hand'])==config.left_landmark_number and len(frame_landmarks['right_hand'])==config.right_landmark_number:
                    landmarks_list.append(frame_landmarks)
                else:
                    print(len(frame_landmarks['face']), len(frame_landmarks['pose']), len(frame_landmarks['left_hand']), len(frame_landmarks['right_hand']))

                frame_idx += 1
                if frame_idx % 100 == 0:
                    print(f"Processed {frame_idx} / {frame_count} frames...")
        cap.release()

        print("ランドマークの抽出が完了しました。")
        return landmarks_list