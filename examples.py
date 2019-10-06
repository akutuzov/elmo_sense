# python3
# coding: utf-8

import sys
import pickle
from smart_open import open

if __name__ == '__main__':
    word_file = sys.argv[1]
    out_file = sys.argv[2]

    words = {}
    for line in open(word_file).readlines():
        res = line.strip().split('\t')
        word = res[0].strip()
        words[word] = []

    for line in sys.stdin:
        sentence = line.strip()
        for w in sentence.split():
            if w in words:
                words[w].append(sentence)

    print('Reading complete', file=sys.stderr)
    for word in words:
        print(word, len(words[word]), file=sys.stderr)

    with open(out_file, 'wb') as w:
        pickle.dump(words, w)
    print('Examples saved to', out_file, file=sys.stderr)
