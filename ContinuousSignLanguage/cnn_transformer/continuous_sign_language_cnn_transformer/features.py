import cnn_transformer.continuous_sign_language_cnn_transformer.config as config
from typing import Dict, Any
import numpy as np
import torch

class SelectLandmarksAndFeature():
    """ Select joint and feature.
    """
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

    def __call__(self,
                 data: Dict[str, Any]) -> Dict[str, Any]:
        feature = data["feature"]
        # `[C, T, J]`
        feature = feature[self.features]
        feature = feature[:, :, self.landmarks]
        data["feature"] = feature
        return data


def get_fullbody_landmarks():
    USE_FACE = np.sort(np.unique(config.USE_LIP + config.USE_NOSE + config.USE_REYE + config.USE_LEYE))
    use_landmarks = np.concatenate([USE_FACE, config.USE_LHAND, config.USE_POSE, config.USE_RHAND])
    use_landmarks_filtered = np.arange(len(use_landmarks))
    return use_landmarks_filtered, use_landmarks

class ReplaceNan():
    """ Replace NaN value in the feature.
    """
    def __init__(self, replace_val=0.0) -> None:
        self.replace_val = replace_val

    def __call__(self,
                 data: Dict[str, Any]) -> Dict[str, Any]:
        feature = data["feature"]
        feature[np.isnan(feature)] = self.replace_val
        data["feature"] = feature
        return data

class PartsBasedNormalization():
    def __init__(self,
                 face_head=0,
                 face_num=76,
                 face_origin=[0, 2],
                 face_unit1=[7],
                 face_unit2=[42],
                 lhand_head=76+12,
                 lhand_num=21,
                 lhand_origin=[0, 2, 5, 9, 13, 17],
                 lhand_unit1=[0],
                 lhand_unit2=[2, 5, 9, 13, 17],
                 pose_head=76,
                 pose_num=12,
                 pose_origin=[0, 1],
                 pose_unit1=[0],
                 pose_unit2=[1],
                 rhand_head=76+21+12,
                 rhand_num=21,
                 rhand_origin=[0, 2, 3, 9, 13, 17],
                 rhand_unit1=[0],
                 rhand_unit2=[2, 5, 9, 13, 17],
                 align_mode="framewise",
                 scale_mode="framewise") -> None:
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
                origin = np.array([0.] * feature.shape[0]).reshape([-1, 1, 1])
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

    def _normalize(self, feature, origin_lm, unit_lm1, unit_lm2,
                   unit_range=[1.0e-3, 5.0]):
        tmask = self._gen_tmask(feature)
        origin = self._calc_origin(feature, origin_lm)
        unit = self._calc_unit(feature, unit_lm1, unit_lm2, unit_range)

        _feature = feature - origin
        _feature = _feature / unit
        _feature = _feature * tmask
        return _feature
    
    def append_spatial_feature():
        """
            dataの配列: (C, T, J)
            基準点から5 本の指先と第一関節の差:
            親指の先と残りの指先の距離
            それぞれの指先同士の距離
            基準点と図1 の点9 とx 軸との成す角
            基準点と図1 の点4 とz 軸との成す角
            基準点と図1 の点20 とz 軸との成す角
            図1 の点9 と指先の線分とx 軸との成す角
            指の第二関節と指先の線分とx 軸との成す角
            上記を追加する。
        """
        


    def __call__(self,
                 data: Dict[str, Any]) -> Dict[str, Any]:
        feature = data["feature"]
        print(feature.shape)
        if self.face_num > 0:
            face = feature[:, :, self.face_head: self.face_head+self.face_num]
            face = self._normalize(face, self.face_origin,
                                   self.face_unit1, self.face_unit2)
            feature[:, :, self.face_head: self.face_head+self.face_num] = face
        if self.lhand_num > 0:
            print(feature.shape, self.lhand_num, self.lhand_head)
            lhand = feature[:, :, self.lhand_head: self.lhand_head+self.lhand_num]
            print(lhand.shape, "lhand", )
            lhand = self._normalize(lhand, self.lhand_origin,
                                    self.lhand_unit1, self.lhand_unit2)
            feature[:, :, self.lhand_head: self.lhand_head+self.lhand_num] = lhand
        if self.pose_num > 0:
            pose = feature[:, :, self.pose_head: self.pose_head+self.pose_num]
            pose = self._normalize(pose, self.pose_origin,
                                   self.pose_unit1, self.pose_unit2)
            feature[:, :, self.pose_head: self.pose_head+self.pose_num] = pose
        if self.rhand_num > 0:
            rhand = feature[:, :, self.rhand_head: self.rhand_head+self.rhand_num]
            rhand = self._normalize(rhand, self.rhand_origin,
                                    self.rhand_unit1, self.rhand_unit2)
            feature[:, :, self.rhand_head: self.rhand_head+self.rhand_num] = rhand
        data["feature"] = feature
        return data

class ToTensor():
    """ Convert data to torch.Tensor.
    """
    def __init__(self) -> None:
        pass

    def __call__(self,
                 data: Dict[str, Any]) -> Dict[str, Any]:
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
    

class InsertTokensForS2S():
    def __init__(self,
                 sos_token,
                 eos_token,
                 error_at_exist=False):
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.error_at_exist = error_at_exist

    def check_format(self, tokens):
        insert_sos = False
        if tokens[0] != self.sos_token:
            insert_sos = True
        elif self.error_at_exist:
            message = f"The sos_token:{self.sos_token} is exist in {tokens}." \
                + "Please check the format."
            raise ValueError(message)
        insert_eos = False
        if tokens[-1] != self.eos_token:
            insert_eos = True
        elif self.error_at_exist:
            message = f"The eos_token:{self.eos_token} is exist in {tokens}." \
                + "Please check the format."
            raise ValueError(message)
        return insert_sos, insert_eos

    def __call__(self,
                 data: Dict[str, Any]) -> Dict[str, Any]:

        tokens = data["token"]
        dtype = tokens.dtype

        insert_sos, insert_eos = self.check_format(tokens)
        # Insert.
        new_tokens = []
        if insert_sos:
            new_tokens.append(self.sos_token)
        new_tokens += tokens.tolist()
        if insert_eos:
            new_tokens.append(self.eos_token)
        new_tokens = np.array(new_tokens, dtype=dtype)
        data["token"] = new_tokens
        return data
