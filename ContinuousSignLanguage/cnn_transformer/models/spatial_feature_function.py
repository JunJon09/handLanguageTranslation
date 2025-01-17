import math
import torch

def get_spatial(feature):
    """
        featureの配列: (B, C, T, J)
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
    spatial_feature = []
    
    left_start_point = 88
    right_start_point = 88 + 21
    B, C, T, J = feature.shape
    for i in range(B):
        left_spatial_feature = append_spatial_feature(feature=feature[i, :, :, left_start_point:right_start_point])
        right_spatial_feature = append_spatial_feature(feature=feature[i, :, :, right_start_point:])
        spatial_feature.append([a+ b for a, b in zip(left_spatial_feature, right_spatial_feature)])

    spatial_feature = torch.tensor(spatial_feature)
    return spatial_feature

def append_spatial_feature(feature):
        def calculate_basis_distances(basis, ip, tip):
            return math.sqrt((tip[0] - basis[0]) ** 2 + (tip[1] - wrist[1]) ** 2) - math.sqrt(
                (ip[0] - basis[0]) ** 2 + (ip[1] - basis[1]) ** 2
            )

        def calculate_adjacent_distances(tip1, tip2):
            return math.sqrt((tip1[0] - tip2[0]) ** 2 + (tip1[1] - tip2[1]) ** 2)

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
        spatial_feature = []
        C, T, J = feature.shape
        for frame in range(T):
            wrist = feature[:, frame, 0]
            thumb_basis = feature[:, frame, 1]
            thumb_ip = feature[:, frame, 3]
            thumb_tip = feature[:, frame, 4]
            index_basis = feature[:, frame, 5]
            index_ip = feature[:, frame, 7]
            index_tip = feature[:, frame, 8]
            middle_basis = feature[:, frame, 9]
            middle_ip = feature[:, frame, 11]
            middle_tip = feature[:, frame, 12]
            ring_basis = feature[:, frame, 13]
            ring_ip = feature[:, frame, 15]
            ring_tip = feature[:, frame, 16]
            pinky_basis = feature[:, frame, 17]
            pinky_ip = feature[:, frame, 19]
            pinky_tip = feature[:, frame, 20]

            # 6~10 手首から5本の指先と第一関節の差
            break_thumb_feature = calculate_basis_distances(wrist, thumb_ip, thumb_tip)
            break_index_feature = calculate_basis_distances(wrist, index_ip, index_tip)
            break_middle_feature = calculate_basis_distances(wrist, middle_ip, middle_tip)
            break_ring_feature = calculate_basis_distances(wrist, ring_ip, ring_tip)
            break_pinky_feature = calculate_basis_distances(wrist, pinky_ip, pinky_tip)

            # 11~14 親指の先から残りの指先の距離
            tip_distance_index_feature = calculate_basis_distances(
                basis=thumb_tip, ip=index_ip, tip=index_tip
            )
            tip_distance_middle_feature = calculate_basis_distances(
                basis=thumb_tip, ip=middle_ip, tip=middle_tip
            )
            tip_distance_ring_feature = calculate_basis_distances(
                basis=thumb_tip, ip=ring_ip, tip=ring_tip
            )
            tip_distance_pinky_feature = calculate_basis_distances(
                basis=thumb_tip, ip=pinky_ip, tip=pinky_tip
            )

            # 15~17 それぞれの指の距離、(親指は前で行っため省く)
            tip_distance_index_to_middle_feature = calculate_adjacent_distances(
                index_tip, middle_tip
            )
            tip_distance_middle_to_ring_feature = calculate_adjacent_distances(
                middle_tip, ring_tip
            )
            tip_distance_ring_to_pinky_feature = calculate_adjacent_distances(ring_tip, pinky_tip)
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
                ]
            )
            if len(spatial_feature[-1]) != 12:
                raise
        return spatial_feature