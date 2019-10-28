# python3
# coding: utf-8

import sys
from scipy.stats import spearmanr, pearsonr
from pandas import read_csv
from smart_open import open

if __name__ == '__main__':
    goldfile = sys.argv[1]
    files2process = sys.argv[2:]
    gold_scores = read_csv(goldfile, sep="\t", encoding="utf-8")
    all_coeffs = {}

    for f in files2process:
        name = f.split('/')[-1].split('_')[0]
        all_coeffs[name] = {}
        with open(f) as inp:
            for l in inp.readlines():
                res = l.strip().split('\t')
                word, coeff = res
                all_coeffs[name][word] = float(coeff)

    words = set.intersection(*[set(all_coeffs[d].keys()) for d in all_coeffs])

    years = sorted(all_coeffs.keys())
    coeffs = {}
    for word in words:
        coeffs[word] = [all_coeffs[year][word] for year in years]

    deltas = []
    gold_shifts = []

    for word in coeffs:
        diff = abs(coeffs[word][0] - coeffs[word][1])
        deltas.append(diff)
        gold_shifts.append(gold_scores[gold_scores['word'] == word]['average'].values[0])

    print('Spearman correlation between diversity changes and gold standard: %.3f, %.3f'
          % spearmanr(deltas, gold_shifts))
    print('Pearson correlation between diversity changes and gold standard: %.3f, %.3f'
          % pearsonr(deltas, gold_shifts))
