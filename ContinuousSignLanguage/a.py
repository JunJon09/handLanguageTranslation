import numpy as np


def extend_with_blanks(target, blank=0):
    # e.g. target=[3,1,4,2] → [0,3,0,1,0,4,0,2,0]
    ext = [blank]
    for t in target:
        ext += [t, blank]
    return np.array(ext, dtype=int)


def ctc_forward(log_probs, target, blank=0):
    """
    log_probs: (T, C) ログ確率（自然対数）
    target: [u1, …, uU] 正解ラベル系列
    """
    T, C = log_probs.shape
    label = extend_with_blanks(target, blank)  # 長さ L=2U+1
    L = len(label)

    # α をゼロ初期化
    alpha = np.zeros((T, L), dtype=float)

    # t=0 の初期条件
    alpha[0, 0] = np.exp(log_probs[0, blank])
    alpha[0, 1] = np.exp(log_probs[0, label[1]])

    # 前向き確率の再帰
    for t in range(1, T):
        for s in range(L):
            p = np.exp(log_probs[t, label[s]])
            a = alpha[t - 1, s]
            b = alpha[t - 1, s - 1] if s - 1 >= 0 else 0.0
            c = (
                alpha[t - 1, s - 2]
                if s - 2 >= 0 and label[s] != blank and label[s] != label[s - 2]
                else 0.0
            )
            alpha[t, s] = p * (a + b + c)

    # P(y|x) は最後の2 状態の和
    prob = alpha[-1, -1] + alpha[-1, -2]
    loss = -np.log(prob + 1e-20)
    return loss


# 例
# T=5, C=5, targets=[3,1,4,2]
# ダミーの log_probs を用意
np.random.seed(0)
T, C = 5, 5
logits = np.random.randn(T, C)
log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
print(logits)
print("Log probabilities:\n", log_probs)
target = [3, 1, 4, 2]

loss = ctc_forward(log_probs, target, blank=0)
print(f"Manual CTC Loss: {loss:.4f}")

# 追加例: 2行のlogitsで log-softmax を計算
logits2 = np.array([[1, 5, 7, 3], [2, 10, 6, 14]], dtype=float)
# 正規化定数 (log-sum-exp)
log_norm = np.log(np.exp(logits2).sum(axis=1, keepdims=True))
# ログソフトマックス
log_probs2 = logits2 - log_norm
exp_logits2 = np.exp(logits2)

print("\nExample logits2:\n", exp_logits2)
print("\nExample logits2:\n", logits2)
print("Log-sum-exp term:\n", log_norm)
print("Log-softmax (log_probs2):\n", log_probs2)
print("Softmax probabilities:\n", np.exp(log_probs2))
