def scores_to_rankvec(scores, higher_is_better=True):
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=higher_is_better)
    rank = [0] * len(scores)
    for r, idx in enumerate(order):
        rank[idx] = r
    return rank

def inversion_score(p1, p2):
    assert len(p1) == len(p2), f"{len(p1)}, {len(p2)}"
    n = len(p1)
    if n < 2:
        return 1.0
    cnt = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            # if p1[i] == p1[j]:
            #     cnt += 1  # 跳过 pred tie
            if (p1[i] >= p1[j] and p2[i] < p2[j]) or (p1[i] < p1[j] and p2[i] > p2[j]):
                cnt += 1
    return 1 - cnt / (n * (n - 1) / 2)