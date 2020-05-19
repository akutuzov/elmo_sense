# python3
# coding: utf-8

from smart_open import open
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import random

if __name__ == '__main__':
    filesdir = sys.argv[1]

    vocfiles = [f for f in os.listdir(filesdir) if f.endswith('_vocab.txt.gz')]

    vocs = []
    for f in vocfiles:
        vocabulary = {}
        for line in open(os.path.join(filesdir, f), 'r'):
            res = line.strip().split('\t')
            word, freq = res
            vocabulary[word.strip()] = int(freq)
        vocs.append(vocabulary)

    for nr, i in enumerate(vocs):
        print(nr, len(i))

    dicts = [set(v) for v in vocs]
    valid_words = set.intersection(*dicts)

    print('Found %d shared words' % len(valid_words))

    valid_dictionary = {}

    for w in valid_words:
        if len(w.strip()) < 2:
            continue
        freq = []
        for voc in vocs:
            freq.append(voc[w])
        valid_dictionary[w] = np.median(freq)

    mid_threshold = 10000
    low_threshold = 30000
    sample = 1000

    frequencies = sorted([valid_dictionary[w] for w in valid_dictionary], reverse=True)
    figure, ax = plt.subplots()
    ax.scatter(range(len(frequencies)), frequencies, marker='o')
    ax.set_yscale('log')
    plt.axvline(x=mid_threshold, color="red")
    plt.axvline(x=low_threshold, color="red")
    ax.set(xlabel='Word rank', ylabel='Median frequency across 5 time bins',
           title='Word frequency distribution in COHA')
    ax.grid()
    ax.legend(loc='best')
    plt.show()
    plt.close()

    sort_word = sorted(valid_dictionary, key=valid_dictionary.get, reverse=True)
    high = sort_word[:mid_threshold]
    mid = sort_word[mid_threshold:low_threshold]
    low = sort_word[low_threshold:]

    print('High frequency words:', len(high))
    print('Mid frequency words:', len(mid))
    print('Low frequency words:', len(low))

    high = random.sample(high, sample)
    mid = random.sample(mid, sample)
    low = random.sample(low, sample)

    with open('coha_high.txt', 'a') as f:
        for w in high:
            f.write(w + '\n')

    with open('coha_mid.txt', 'a') as f:
        for w in mid:
            f.write(w + '\n')

    with open('coha_low.txt', 'a') as f:
        for w in low:
            f.write(w + '\n')
