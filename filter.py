# /bin/env python3

import sys

for line in sys.stdin:
    res = line.strip().split('\t')
    word, freq = res
    if word.strip().isdigit() or len(word.strip()) < 2:
        continue
    print(line.strip())
