import re

import Levenshtein
import numpy as np


def to_char_sentence(words):
    return list(''.join(words))

def wer1(r, h, char_level=False):   
    if char_level:
        r = to_char_sentence(r)
        h = to_char_sentence(h)

    d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint8)
    d = d.reshape((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)] / len(r)

def wer2(r, h, char_level=False):
    error = 0
    for i in range(min(len(r), len(h))):
        error += Levenshtein.distance(r[i], h[i])

    if char_level:
        r = to_char_sentence(r)
        h = to_char_sentence(h)

    return (error + abs(len(r) - len(r))) / len(r)
