#!/projects/ltg/python3/bin/python3
import sys
from nltk.corpus import wordnet as wn
from collections import Counter

words = Counter()

for line in sys.stdin:
    res = line.strip().split()
    for word in res:
        synsets = len(wn.synsets(word))
        if synsets > 0:
            words.update([word])

print('Vocabulary:', len(words), file=sys.stderr)

a = sorted(words, key=words.get, reverse=True)
for w in a:
    if words[w] > 2:
        print(w + '\t' + str(words[w]))
