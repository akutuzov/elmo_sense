# python3
# coding: utf-8

import sys
import pickle
from smart_open import open

if __name__ == '__main__':
    word_file = sys.argv[1]
    out_file = sys.argv[2]

    WORD_LIMIT = 400
    WINDOW = 10

    words = {}
    for line in open(word_file).readlines():
        word = line.strip()
        words[word] = []

    for line in sys.stdin:
        res = line.strip().split()[:WORD_LIMIT]
        for nr, word in enumerate(res):
            if word in words:
                if nr < WINDOW:
                    context = ' '.join(res[:nr+WINDOW])
                else:
                    context = ' '.join(res[nr-WINDOW:nr+WINDOW])
                words[word].append(context)

    print('Reading complete', file=sys.stderr)
    for word in words:
        print(word, len(words[word]), file=sys.stderr)

    with open(out_file, 'wb') as w:
        pickle.dump(words, w)
    print('Examples saved to', out_file, file=sys.stderr)
