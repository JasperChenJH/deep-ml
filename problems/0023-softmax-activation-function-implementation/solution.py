import math

def softmax(scores: list[float]) -> list[float]:
    m = max(scores)
    ls2 = [math.exp(x - m) for x in scores]
    s = sum(ls2)
    res = []
    for i in ls2:
        res.append(i / s)
    return res
