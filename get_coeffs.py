# python3
# coding: utf-8

import argparse
from nltk.corpus import wordnet as wn
from elmo_helpers import *

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--input', '-i', help='Path to npz file with the embeddings', required=True)
    arg('--target', '-t', help='Path to target words', required=True)
    arg('--output', '-o', help='Output path (csv)', required=False)
    args = parser.parse_args()

    data_path = args.input
    target_words = {w.split('\t')[0].strip(): int(w.split('\t')[1]) for w in
                    open(args.target, 'r', encoding='utf-8').readlines()}

    array = np.load(data_path)
    logger.info('Loaded an array of %d entries from %s' % (len(array), data_path))

    try:
        f_out = open(args.output, 'w', encoding='utf-8')
    except TypeError:
        f_out = None

    print('\t'.join(['Word', 'Frequency', 'Diversity', 'Clusters', 'Synsets']), file=f_out)
    for word in target_words:
        logger.info('Processing %s' % word)
        if array[word].shape[0] < 3:
            logger.info('%s omitted because of low frequency' % word)
            continue
        vectors = array[word]
        diversity_coeff = diversity(vectors)
        nr_clusters = cluster(vectors, 'affinity', word)[0]
        synsets = len(wn.synsets(word))
        print('\t'.join([word, str(target_words[word]), str(diversity_coeff), str(nr_clusters),
                         str(synsets)]), file=f_out)

