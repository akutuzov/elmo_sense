# /bin/env python3

import sys
import csv
from collections import OrderedDict

data = {}
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        word = row['Word']
        data[word] = row

for line in sys.stdin:
    res = line.strip().split('\t')
    word, freq = res
    word = word.lower().strip()
    if word in data:
        new = OrderedDict(WikiFreq=int(freq))
        data[word].update(new)

with open(sys.argv[2], 'w', newline='\n') as csvfile:
    fieldnames = ['Word', 'Frequency', 'Diversity', 'Clusters', 'Synsets', 'WikiFreq']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t', dialect='unix',
                            quoting=csv.QUOTE_NONE)
    writer.writeheader()
    for word in data:
        writer.writerow(data[word])
