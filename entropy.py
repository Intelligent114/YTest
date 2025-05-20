import math
from collections import Counter
from multi import compositions


def get_entropy(values: list):
    counter = Counter(values)
    h = 0
    for v in counter.values():
        p = v/sum(counter.values())
        h += -p*math.log(p, 2)
    return h

def entropy_dict(n: int, digits: int = 2):
    es = set()
    for k in range(1, n + 1):
        for comp in compositions(n, k):
            probs = [c / n for c in comp]
            h = -sum(p * math.log(p, 2) for p in probs)
            es.add(round(h, digits))
    return {h: 0 for h in sorted(es)}

print(entropy_dict(6))