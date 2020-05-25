# python3
# coding: utf-8

import sys
import pickle
from smart_open import open

file2load = sys.argv[1]
word = sys.argv[2]

with open(file2load, 'rb') as f:
    corpus = pickle.load(f)

print('Number of {} occurrences in {}: {}'.format(word, file2load, len(corpus[word])))

while True:
    query = input("Enter occurrence number:")
    word, nr = query.strip().split()
    nr = int(nr)
    try:
        print(corpus[word][nr])
    except IndexError:
        print('Wrong number!')

