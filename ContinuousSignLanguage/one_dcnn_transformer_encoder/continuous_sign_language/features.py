import one_dcnn_transformer_encoder.continuous_sign_language.config as config
import numpy as np
from typing import Dict, Any
import torch
import math

def get_fullbody_landmarks():
    USE_FACE = np.sort(
        np.unique(config.USE_LIP + config.USE_NOSE + config.USE_REYE + config.USE_LEYE)
    )
    use_landmarks = np.concatenate([USE_FACE, config.USE_LHAND, config.USE_POSE, config.USE_RHAND])
    use_landmarks_filtered = np.arange(len(use_landmarks))
    return use_landmarks_filtered, use_landmarks

class SelectLandmarksAndFeature:
    """Select joint and feature."""

    def __init__(self, landmarks, features=["x", "y", "z"]):
        self.landmarks = landmarks
        _features = []
        if "x" in features:
            _features.append(0)
        if "y" in features:
            _features.append(1)
        if "z" in features:
            _features.append(2)
        self.features = np.array(_features, dtype=np.int32)
        assert self.features.shape[0] > 0, f"{self.features}"

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        feature = data["feature"]
        # `[C, T, J]`
        feature = feature[self.features]
        feature = feature[:, :, self.landmarks]
        data["feature"] = feature
        return data
    
class ReplaceNan:
    """ NaNのデータを違う値に置き換える。"""

    def __init__(self, replace_val=0.0) -> None:
        self.replace_val = replace_val

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        feature = data["feature"]
        feature[np.isnan(feature)] = self.replace_val
        data["feature"] = feature
        return data
    


class PartsBasedNormalization:
    def __init__(
        self,
        face_head=0,
        face_num=76,
        face_origin=[0, 2],
        face_unit1=[7],
        face_unit2=[42],
        lhand_head=76 + 12,
        lhand_num=21,
        lhand_origin=[0, 2, 5, 9, 13, 17],
        lhand_unit1=[0],
        lhand_unit2=[2, 5, 9, 13, 17],
        pose_head=76,
        pose_num=12,
        pose_origin=[0, 1],
        pose_unit1=[0],
        pose_unit2=[1],
        rhand_head=76 + 21 + 12,
        rhand_num=21,
        rhand_origin=[0, 2, 3, 9, 13, 17],
        rhand_unit1=[0],
        rhand_unit2=[2, 5, 9, 13, 17],
        align_mode="framewise",
        scale_mode="framewise",
    ) -> None:
        assert align_mode in ["framewise", "unique"]
        assert scale_mode in ["framewise", "unique", "none"]
        self.align_mode = align_mode
        self.scale_mode = scale_mode

        self.face_head = face_head
        self.face_num = face_num
        self.face_origin = face_origin
        self.face_unit1 = face_unit1
        self.face_unit2 = face_unit2

        self.lhand_head = lhand_head
        self.lhand_num = lhand_num
        self.lhand_origin = lhand_origin
        self.lhand_unit1 = lhand_unit1
        self.lhand_unit2 = lhand_unit2

        self.pose_head = pose_head
        self.pose_num = pose_num
        self.pose_origin = pose_origin
        self.pose_unit1 = pose_unit1
        self.pose_unit2 = pose_unit2

        self.rhand_head = rhand_head
        self.rhand_num = rhand_num
        self.rhand_origin = rhand_origin
        self.rhand_unit1 = rhand_unit1
        self.rhand_unit2 = rhand_unit2

    def _gen_tmask(self, feature):
        tmask = feature == 0.0
        tmask = np.all(tmask, axis=(0, 2))
        tmask = np.logical_not(tmask.reshape([1, -1, 1]))
        return tmask

    def _calc_origin(self, feature, origin_lm):
        # `[C, T, J] -> [C, T, 1]`
        origin = feature[:, :, origin_lm].mean(axis=-1, keepdims=True)
        if self.align_mode == "unique":
            # `[C, T, 1] -> [C, 1, 1]`
            mask = self._gen_tmask(origin)
            mask = mask.reshape([mask.shape[1]])
            if mask.any():
                origin = origin[:, mask, :].mean(axis=1, keepdims=True)
            else:
                origin = np.array([0.0] * feature.shape[0]).reshape([-1, 1, 1])
        return origin

    def _calc_unit(self, feature, unit_lm1, unit_lm2, unit_range):
        if self.scale_mode == "none":
            return 1.0
        # The frame-wise unit lengths are unstable.
        # So, we calculate average unit length.
        # Extract.
        # `[C, T, J] -> [C, T, 1]`
        unit1 = feature[:, :, unit_lm1].mean(axis=-1)
        unit2 = feature[:, :, unit_lm2].mean(axis=-1)
        # Mean square between target points.
        unit = np.sqrt((unit1 - unit2) ** 2)
        # Norm.
        # `[C, T, J] -> [1, T, 1]`
        unit = np.linalg.norm(unit, axis=0)
        if self.scale_mode == "framewise":
            unit = unit.reshape([1, unit.shape[0], 1])
            unit[unit <= 0] = 1.0
            unit[np.isnan(unit)] = 1.0
        else:
            # Calculate average removing undetected frame.
            mask = unit > 0
            if mask.sum() > 0:
                unit = unit[unit > 0].mean()
            else:
                unit = 1.0
            unit = 1.0 if np.isnan(unit).any() else unit
        # Finally, clip extreme values.
        unit = np.clip(unit, a_min=unit_range[0], a_max=unit_range[1])
        return unit

    def _normalize(self, feature, origin_lm, unit_lm1, unit_lm2, unit_range=[1.0e-3, 5.0]):
        tmask = self._gen_tmask(feature)
        origin = self._calc_origin(feature, origin_lm)
        unit = self._calc_unit(feature, unit_lm1, unit_lm2, unit_range)

        _feature = feature - origin
        _feature = _feature / unit
        _feature = _feature * tmask
        return _feature

    def __normalize_spatial__(
        self, feature, spatial, unit_lm1, unit_lm2, unit_range=[1.0e-3, 5.0]
    ):
        unit = self._calc_unit(feature, unit_lm1, unit_lm2, unit_range)
        angle_indices = list(range(12, 25))
        distance_feature = spatial[:, :12]
        deg_feature = spatial[:, 12:]
        distance_feature = distance_feature / unit
        _feature = np.concatenate((distance_feature, deg_feature), axis=1)
        
        return _feature

    def append_spatial_feature(self, feature):
        def calculate_basis_distances(basis, ip, tip):
            return math.sqrt((tip[0] - basis[0]) ** 2 + (tip[1] - basis[1]) ** 2 + (tip[2] - basis[2]) ** 2) - math.sqrt(
                (ip[0] - basis[0]) ** 2 + (ip[1] - basis[1]) ** 2 + (ip[2] - basis[2]) ** 2)

        def calculate_adjacent_distances(tip1, tip2):
            return math.sqrt((tip1[0] - tip2[0]) ** 2 + (tip1[1] - tip2[1]) ** 2 + (tip1[2] - tip2[2]) ** 2)
        
        def calculate_angles_with_axes(base1, base2, axis):
            """
            与えられた3次元ベクトルと任意の軸ベクトルとの成す角度を計算します。
            """
            vector = base2 - base1
            # ベクトルの大きさ
            V = np.array(vector, dtype=float)
            V_magnitude = np.linalg.norm(V)
            if V_magnitude == 0:
                return 0
            
            # 内積の計算
            dot_product = np.dot(V, axis)
            cos_theta = dot_product / V_magnitude

            if cos_theta > 1.0 and cos_theta < 0:
                raise ValueError("不正な値になりました")
            
            # 角度の計算（ラジアン）
            theta_rad = math.acos(cos_theta)

            # 角度の計算（度）
            theta_deg = math.degrees(theta_rad)
            return theta_deg
        """
            dataの配列: (C, T, J)
            基準点から5 本の指先と第一関節の距離の差
            親指の先と残りの指先の距離
            それぞれの指先同士の距離
            手首と中指の付け根を結ぶベクトルとx軸との成す角
            手首と親指の先を結ぶベクトルとz軸との成す角
            手首と小指の先を結ぶベクトルとz軸との成す角
            図1 の点9 と指先の線分とx 軸との成す角
            指の第二関節と指先の線分とx 軸との成す角
            上記を追加する。
        """
        spatial_feature = []
        C, T, J = feature.shape
        for frame in range(T):
            wrist = feature[:, frame, 0]
            thumb_basis = feature[:, frame, 1]
            thumb_dip = feature[:, frame, 2]
            thumb_ip = feature[:, frame, 3]
            thumb_tip = feature[:, frame, 4]
            index_basis = feature[:, frame, 5]
            index_dip = feature[:, frame, 6]
            index_ip = feature[:, frame, 7]
            index_tip = feature[:, frame, 8]
            middle_basis = feature[:, frame, 9]
            middle_dip = feature[:, frame, 10]
            middle_ip = feature[:, frame, 11]
            middle_tip = feature[:, frame, 12]
            ring_basis = feature[:, frame, 13]
            ring_dip = feature[:, frame, 14]
            ring_ip = feature[:, frame, 15]
            ring_tip = feature[:, frame, 16]
            pinky_basis = feature[:, frame, 17]
            pinky_dip = feature[:, frame, 18]
            pinky_ip = feature[:, frame, 19]
            pinky_tip = feature[:, frame, 20]

            # 基準点から5 本の指先と第一関節の距離の差
            break_thumb_feature = calculate_basis_distances(wrist, thumb_ip, thumb_tip)
            break_index_feature = calculate_basis_distances(wrist, index_ip, index_tip)
            break_middle_feature = calculate_basis_distances(wrist, middle_ip, middle_tip)
            break_ring_feature = calculate_basis_distances(wrist, ring_ip, ring_tip)
            break_pinky_feature = calculate_basis_distances(wrist, pinky_ip, pinky_tip)

            #親指の先から残りの指先の距離
            tip_distance_index_feature = calculate_adjacent_distances(tip1=thumb_ip, tip2=index_tip)
            tip_distance_middle_feature = calculate_adjacent_distances(tip1=thumb_ip, tip2=middle_tip)
            tip_distance_ring_feature = calculate_adjacent_distances(tip1=thumb_tip, tip2=ring_tip)
            tip_distance_pinky_feature = calculate_adjacent_distances(tip1=thumb_tip, tip2=pinky_tip)

            #それぞれの指の距離、(親指は前で行っため省く)
            tip_distance_index_to_middle_feature = calculate_adjacent_distances(tip1=index_tip, tip2=middle_tip)
            tip_distance_middle_to_ring_feature = calculate_adjacent_distances(tip1=middle_tip, tip2=ring_tip)
            tip_distance_ring_to_pinky_feature = calculate_adjacent_distances(tip1=ring_tip, tip2=pinky_tip)

            x_axis = np.array([1, 0, 0])
            y_axis = np.array([0, 1, 0])
            z_axis = np.array([0, 0, 1])
            #手首と中指の付け根を結ぶベクトルとx軸との成す角
            deg_wrist_with_middle_basis = calculate_angles_with_axes(base1=wrist, base2=middle_basis, axis=x_axis)

            #手首と親指の先を結ぶベクトルとz軸との成す角
            deg_wrist_with_thumb_tip = calculate_angles_with_axes(base1=wrist, base2=thumb_tip, axis=z_axis)

            #手首と小指の先を結ぶベクトルとz軸との成す角
            deg_wrist_with_pinky_tip = calculate_angles_with_axes(base1=wrist, base2=pinky_tip, axis=z_axis)

            #中指の付け根から各先を結ぶベクトルとx軸との成す角
            deg_middle_basis_with_thumb_tip = calculate_angles_with_axes(base1=middle_basis, base2=thumb_tip, axis=x_axis)
            deg_middle_basis_with_index_tip = calculate_angles_with_axes(base1=middle_basis, base2=index_tip, axis=x_axis)
            deg_middle_basis_with_middle_tip = calculate_angles_with_axes(base1=middle_basis, base2=middle_tip, axis=x_axis)
            deg_middle_basis_with_ring_tip = calculate_angles_with_axes(base1=middle_basis, base2=ring_tip, axis=x_axis)
            deg_middle_basis_with_pinky_tip = calculate_angles_with_axes(base1=middle_basis, base2=pinky_tip, axis=x_axis)

            #各指の第二関節から指先の線分とx軸との成す角
            deg_thumb_dip_with_thumb_tip = calculate_angles_with_axes(base1=thumb_dip, base2=thumb_tip, axis=x_axis)
            deg_index_dip_with_index_tip = calculate_angles_with_axes(base1=index_dip, base2=index_tip, axis=x_axis)
            deg_middle_dip_with_middle_tip = calculate_angles_with_axes(base1=middle_dip, base2=middle_tip, axis=x_axis)
            deg_ring_dip_with_ring_tip = calculate_angles_with_axes(base1=ring_dip, base2=ring_tip, axis=x_axis)
            deg_pinky_dip_with_pinky_tip = calculate_angles_with_axes(base1=pinky_dip, base2=pinky_tip, axis=x_axis)
            spatial_feature.append(
                [
                    break_thumb_feature,
                    break_index_feature,
                    break_middle_feature,
                    break_ring_feature,
                    break_pinky_feature,
                    tip_distance_index_feature,
                    tip_distance_middle_feature,
                    tip_distance_ring_feature,
                    tip_distance_pinky_feature,
                    tip_distance_index_to_middle_feature,
                    tip_distance_middle_to_ring_feature,
                    tip_distance_ring_to_pinky_feature,
                    deg_wrist_with_middle_basis,
                    deg_wrist_with_thumb_tip,
                    deg_wrist_with_pinky_tip,
                    deg_middle_basis_with_thumb_tip,
                    deg_middle_basis_with_index_tip,
                    deg_middle_basis_with_middle_tip,
                    deg_middle_basis_with_ring_tip,
                    deg_middle_basis_with_pinky_tip,
                    deg_thumb_dip_with_thumb_tip,
                    deg_index_dip_with_index_tip,
                    deg_middle_dip_with_middle_tip,
                    deg_ring_dip_with_ring_tip,
                    deg_pinky_dip_with_pinky_tip,
                ]
            )
        return np.array(spatial_feature)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        feature = data["feature"]
        if self.face_num > 0:
            face = feature[:, :, self.face_head : self.face_head + self.face_num]
            face = self._normalize(face, self.face_origin, self.face_unit1, self.face_unit2)
            feature[:, :, self.face_head : self.face_head + self.face_num] = face

        if self.lhand_num > 0:
            lhand = feature[:, :, self.lhand_head : self.lhand_head + self.lhand_num]
            l_spatial_feature = self.append_spatial_feature(lhand)
            l_spatial_feature = self.__normalize_spatial__(
                feature, l_spatial_feature, self.lhand_unit1, self.lhand_unit2
            )
            lhand = self._normalize(lhand, self.lhand_origin, self.lhand_unit1, self.lhand_unit2)
            feature[:, :, self.lhand_head : self.lhand_head + self.lhand_num] = lhand

        if self.pose_num > 0:
            pose = feature[:, :, self.pose_head : self.pose_head + self.pose_num]
            pose = self._normalize(pose, self.pose_origin, self.pose_unit1, self.pose_unit2)
            feature[:, :, self.pose_head : self.pose_head + self.pose_num] = pose
        if self.rhand_num > 0:
            rhand = feature[:, :, self.rhand_head : self.rhand_head + self.rhand_num]
            r_spatial_feature = self.append_spatial_feature(rhand)
            r_spatial_feature = self.__normalize_spatial__(
                feature, r_spatial_feature, self.rhand_unit1, self.rhand_unit2
            )
            rhand = self._normalize(rhand, self.rhand_origin, self.rhand_unit1, self.rhand_unit2)
            feature[:, :, self.rhand_head : self.rhand_head + self.rhand_num] = rhand
        spatial_feature = np.concatenate((l_spatial_feature, r_spatial_feature), axis=1)
        data["feature"] = feature
        data["spatial"] = torch.tensor(spatial_feature)
        return data

class ToTensor:
    """Convert data to torch.Tensor."""

    def __init__(self) -> None:
        pass

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        new_data = {}
        for key, val in data.items():
            if val is not None:
                if isinstance(val, list):
                    for i, subval in enumerate(val):
                        if subval.dtype in [float, np.float64]:
                            # pylint: disable=no-member
                            val[i] = torch.from_numpy(subval.astype(np.float32))
                        else:
                            val[i] = torch.from_numpy(subval)  # pylint: disable=no-member
                elif isinstance(val, np.ndarray):
                    if val.dtype in [float, np.float64]:
                        # pylint: disable=no-member
                        val = torch.from_numpy(val.astype(np.float32))
                    else:
                        val = torch.from_numpy(val)  # pylint: disable=no-member
            new_data[key] = val
        return new_data

    def __str__(self):
        return f"{self.__class__.__name__}:{self.__dict__}"