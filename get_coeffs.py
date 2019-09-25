# python3
# coding: utf-8

import argparse
import sys
import numpy as np
import os
from elmo_helpers import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--input', '-i', help='Path to directory with npz files', required=True)

    args = parser.parse_args()
    data_path = args.input
    files = [f for f in os.listdir(data_path) if f.endswith('.npz')]

    array = {}

    for f in files:
        print('Processing', f, file=sys.stderr)
        word = f.split('.')[0]
        cur_array = np.load(os.path.join(data_path, f))
        array[word] = cur_array['arr_0']

    print('Loaded an array of %d entries' % len(array), file=sys.stderr)

    for word in array:
        var_coeff = diversity(array[word])
        print(word+'\t', var_coeff)

    print('Variation coefficients produced', file=sys.stderr)
