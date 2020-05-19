# python3
# coding: utf-8

import csv
from scipy.stats import spearmanr
from smart_open import open
import sys
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = {}
    with open(sys.argv[1], 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            word = row['Word']
            data[word] = row

    limits = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]

    figure, ax = plt.subplots()

    diversity_scores = []
    jsd_scores = []
    freq_scores = []

    for l in limits:
        words = [w for w in data if int(data[w]['Frequency']) > l]
        print('Words to test: %d, frequency limit: %d' % (len(words), l), file=sys.stderr)

        diversity_degrees = [float(data[w]['Diversity']) for w in words]
        cluster_degrees = [float(data[w]['Clusters']) for w in words]
        frequencies = [float(data[w]['Frequency']) for w in words]
        wfrequencies = [float(data[w]['WikiFreq']) for w in words]
        nr_synsets = [float(data[w]['Synsets']) for w in words]

        print('Spearman correlation between diversity and the number of WordNet synsets: %.4f, %.4f'
              % spearmanr(diversity_degrees, nr_synsets))
        diversity_scores.append(spearmanr(diversity_degrees, nr_synsets)[0])
        print('Spearman correlation between cluster number and the number of WordNet synsets: '
              '%.4f, %.4f' % spearmanr(cluster_degrees, nr_synsets))
        jsd_scores.append(spearmanr(cluster_degrees, nr_synsets)[0])
        print('Spearman correlation between SensEval frequency and the number of WordNet synsets: '
              '%.4f, %.4f' % spearmanr(frequencies, nr_synsets))
        freq_scores.append(spearmanr(frequencies, nr_synsets)[0])
        print('Spearman correlation between Wikipedia frequency and the number of WordNet synsets: '
              '%.4f, %.4f' % spearmanr(wfrequencies, nr_synsets))

    ax.plot(limits, diversity_scores, label='DIV', marker='o')
    ax.plot(limits, jsd_scores, label='JSD', marker='o')
    ax.plot(limits, freq_scores, label='Frequency', marker='o')
    ax.set(xlabel='Minimal word frequency', ylabel='Spearman correlation',
           title='Ambiguity correlations')
    ax.grid()
    ax.legend(loc='best')
    figure.savefig('ambiguity.png', dpi=300)
    plt.close()
