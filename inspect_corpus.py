# python3
# coding: utf-8

import sys
import pickle
from smart_open import open

file2load = sys.argv[1]

with open(file2load, 'rb') as f:
    corpus = pickle.load(f)

print(file2load)

while True:
    query = input("Enter word and occurrence number:")
    word, nr = query.strip().split()
    nr = int(nr)
    try:
        print(corpus[word][nr])
    except IndexError:
        print('Wrong number!')

