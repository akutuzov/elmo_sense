#!/bin/env/ python3

from os import path
import sys
from smart_open import open
import numpy as np


file0 = sys.argv[1]
file1 = sys.argv[2


cos_scores = {}
apd_scores = {}
frequencies = {}

for line in open(file0, 'r').readlines():
    res = line.strip().split('\t')
    word, score, freq = res
    cos_scores[word.strip()] = float(score)
    frequencies[word.strip()] = float(freq)

for line in open(file1, 'r').readlines():
    res = line.strip().split('\t')
    word, score, freq = res
    apd_scores[word.strip()] = float(score)


ensemble_scores = {}

for word in cos_scores:
    score = np.average([cos_scores[word], apd_scores[word]])
    ensemble_scores[word] = score

ensemble_file = sys.argv[3]
with open(ensemble_file, 'w') as f:
    for word in ensemble_scores:
        f.write('\t'.join([word, str(ensemble_scores[word], str(frequneices[word])) + '\n']))

