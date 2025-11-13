import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def detailed_ctc_loss_calculation():
    """CTC損失の具体的な計算過程を示す"""

    print("=== CTC損失の具体的な計算例 ===\n")

    # 入力データの準備
    T = 10  # 時間ステップ数（実際の例では100だが、可視化のため10に縮小）
    N = 2  # バッチサイズ（実際の例では32だが、可視化のため2に縮小）
    C = 5  # クラス数（実際の例では29だが、可視化のため5に縮小）

    # ログits（モデルの生出力）
    conv_logits = torch.randn(T, N, C)  # [時間, バッチ, クラス数]
    print(f"1. 入力ロジットの形状: {conv_logits.shape}")
    print(f"   ロジット値の例（最初の時間ステップ、最初のバッチ）:")
    print(f"   {conv_logits[0, 0, :].numpy()}")

    # 正解ラベル（バッチ内の各サンプルのラベル系列）
    label = torch.tensor([[1, 2], [3, 1]])  # 2つのサンプル
    print(f"\n2. 正解ラベル: {label}")
    print(f"   - サンプル1: クラス1 → クラス2")
    print(f"   - サンプル2: クラス3 → クラス1")

    # 特徴量の実際の長さ（パディングを除いた長さ）
    feat_len = torch.tensor([T, T - 2])  # サンプル1は全長、サンプル2は8ステップ
    print(f"\n3. 特徴量の実際の長さ: {feat_len}")

    # ラベルの実際の長さ
    label_lgt = torch.tensor([2, 2])  # 両方とも2つのクラス
    print(f"4. ラベルの実際の長さ: {label_lgt}")

    # ステップ1: ログソフトマックスの適用
    log_probs = F.log_softmax(conv_logits, dim=-1)
    print(f"\n5. ログソフトマックス適用後:")
    print(f"   形状: {log_probs.shape}")
    print(f"   最初の時間ステップ、最初のバッチの確率:")
    probs = F.softmax(conv_logits[0, 0, :], dim=-1)
    log_probs_sample = F.log_softmax(conv_logits[0, 0, :], dim=-1)
    for i in range(C):
        print(
            f"     クラス{i}: 確率={probs[i]:.4f}, ログ確率={log_probs_sample[i]:.4f}"
        )

    # ステップ2: CTC損失の計算
    ctc_loss_fn = nn.CTCLoss(reduction="none", zero_infinity=False, blank=0)
    losses = ctc_loss_fn(log_probs, label, feat_len, label_lgt)

    print(f"\n6. CTC損失の計算結果:")
    print(f"   各サンプルの損失: {losses}")
    print(f"   平均損失: {losses.mean():.4f}")

    # ステップ3: CTC損失の内部動作の説明
    print(f"\n7. CTC損失の内部動作:")
    print(f"   - ブランクトークン（クラス0）を使用してアライメント")
    print(f"   - 可能なすべてのアライメントパスを考慮")

    # サンプル1のアライメント例を示す
    print(f"\n   サンプル1のラベル [1, 2] の可能なアライメント例:")
    print(f"   - 0011122200... (ブランクで区切り)")
    print(f"   - 0001122000... (異なるタイミング)")
    print(f"   - 1100222000... (連続する同じクラス)")

    return losses


def demonstrate_ctc_alignment():
    """CTCアライメントの具体例を示す"""

    print("\n=== CTCアライメントの具体例 ===")

    # 簡単な例：ラベル [1, 2] を時間長5にアライメント
    T = 5
    target = [1, 2]
    blank = 0

    print(f"ターゲットラベル: {target}")
    print(f"時間長: {T}")
    print(f"ブランクトークン: {blank}")

    # 可能なアライメントパスを列挙
    possible_paths = [
        [0, 1, 0, 2, 0],  # ブランク-1-ブランク-2-ブランク
        [1, 1, 0, 2, 2],  # 1-1-ブランク-2-2
        [0, 1, 1, 2, 0],  # ブランク-1-1-2-ブランク
        [1, 0, 0, 2, 0],  # 1-ブランク-ブランク-2-ブランク
        [0, 0, 1, 2, 2],  # ブランク-ブランク-1-2-2
    ]

    print(f"\n可能なアライメントパス:")
    for i, path in enumerate(possible_paths):
        print(f"  パス{i+1}: {path}")

    # 各パスの確率を仮定（実際にはモデルの出力から計算）
    print(f"\n各パスの確率計算（仮定値）:")
    path_probs = []
    for i, path in enumerate(possible_paths):
        # 仮の確率値
        prob = np.random.uniform(0.01, 0.1)
        path_probs.append(prob)
        print(f"  パス{i+1}の確率: {prob:.4f}")

    # 全パスの確率の合計（CTC損失の計算に使用）
    total_prob = sum(path_probs)
    print(f"\n全パスの確率合計: {total_prob:.4f}")
    print(f"CTC損失 = -log({total_prob:.4f}) = {-np.log(total_prob):.4f}")


def step_by_step_example():
    """ステップバイステップの詳細な例"""

    print("\n=== ステップバイステップの詳細例 ===")

    # 非常に小さな例
    T = 3  # 時間ステップ3
    N = 1  # バッチサイズ1
    C = 3  # クラス数3（0=blank, 1=class1, 2=class2）

    # 手動で設定したロジット値
    logits = torch.tensor(
        [
            [
                [2.0, 1.0, 0.5],  # t=0: blank有利
                [0.5, 2.5, 1.0],  # t=1: class1有利
                [1.0, 0.5, 2.0],
            ]
        ],  # t=2: class2有利
        dtype=torch.float32,
    )
    logits = logits.transpose(0, 1)  # [T, N, C]に変換

    print(f"ロジット値:")
    for t in range(T):
        print(f"  時間{t}: {logits[t, 0, :].numpy()}")

    # 確率値の計算
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)

    print(f"\n確率値:")
    for t in range(T):
        print(f"  時間{t}:")
        for c in range(C):
            class_name = "blank" if c == 0 else f"class{c}"
            print(
                f"    {class_name}: {probs[t, 0, c]:.4f} (log: {log_probs[t, 0, c]:.4f})"
            )

    # ターゲットラベル [1] （class1のみ）
    target = torch.tensor([[1]])
    target_lengths = torch.tensor([1])
    input_lengths = torch.tensor([T])

    # CTC損失の計算
    ctc_loss = nn.CTCLoss(reduction="none", blank=0)
    loss = ctc_loss(log_probs, target, input_lengths, target_lengths)

    print(f"\nターゲット: class1")
    print(f"CTC損失: {loss.item():.4f}")

    # 手動でのアライメント確率計算（簡略版）
    print(f"\n可能なアライメントパス:")
    print(f"1. [0, 1, 0] - blank, class1, blank")
    print(f"2. [1, 1, 0] - class1, class1, blank")
    print(f"3. [0, 1, 1] - blank, class1, class1")
    print(f"4. [1, 1, 1] - class1, class1, class1")

    # 各パスの確率を手動計算
    path1_prob = probs[0, 0, 0] * probs[1, 0, 1] * probs[2, 0, 0]
    path2_prob = probs[0, 0, 1] * probs[1, 0, 1] * probs[2, 0, 0]
    path3_prob = probs[0, 0, 0] * probs[1, 0, 1] * probs[2, 0, 1]
    path4_prob = probs[0, 0, 1] * probs[1, 0, 1] * probs[2, 0, 1]

    total_path_prob = path1_prob + path2_prob + path3_prob + path4_prob
    manual_loss = -torch.log(total_path_prob)

    print(f"\n手動計算:")
    print(f"パス1確率: {path1_prob:.6f}")
    print(f"パス2確率: {path2_prob:.6f}")
    print(f"パス3確率: {path3_prob:.6f}")
    print(f"パス4確率: {path4_prob:.6f}")
    print(f"合計確率: {total_path_prob:.6f}")
    print(f"手動計算損失: {manual_loss:.4f}")
    print(f"PyTorch CTC損失: {loss.item():.4f}")


if __name__ == "__main__":
    # 具体的な計算例を実行
    detailed_ctc_loss_calculation()
    demonstrate_ctc_alignment()
    step_by_step_example()
