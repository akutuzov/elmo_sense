# python3
# coding: utf-8

import numpy as np
import argparse
import sys
from gensim.models import LsiModel, word2vec
from gensim.matutils import unitvec
from scipy.spatial.distance import cdist
import logging


def lsi_vector(query, lsimodel):
    word2index = lsimodel.id2word.token2id
    vectors = lsimodel.projection.u
    vector = vectors[word2index[query]]
    return vector


def text_vector(text, lsimodel):
    lexicon = [w for w in text if w in lsimodel.id2word.token2id]
    lw = len(lexicon)
    if lw < 1:
        print('Empty lexicon in', text, file=sys.stderr)
    vectors = np.zeros((lw, len(lsimodel.projection.s)))
    for i in list(range(lw)):
        query = lexicon[i]
        vectors[i, :] = lsi_vector(query, lsimodel)
    semantic_fingerprint = np.sum(vectors, axis=0)
    normalized_sum = unitvec(semantic_fingerprint)
    return normalized_sum


def extract_context(target_word, text, lsimodel, cwindow):
    position = text.index(target_word)
    if position >= cwindow:
        text = text[position - cwindow:]
    position = text.index(target_word)
    if (len(text) - position) > cwindow:
        text = text[:position + cwindow + 1]
    context_vector = text_vector(text, lsimodel)
    return context_vector


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--input', '-i', help='Path to corpus', required=True)
    arg('--target', '-t', help='Path to target words', required=True)
    arg('--model', '-m', help='Path to LSI model', required=True)
    arg('--output', '-o', help='Output path (tsv)', required=False)

    args = parser.parse_args()

    target_words = set([w.strip() for w in open(args.target, 'r', encoding='utf-8').readlines()])

    try:
        f_out = open(args.output, 'w', encoding='utf-8')
    except TypeError:
        f_out = None

    model = LsiModel.load(args.model)

    texts = word2vec.LineSentence(args.input)

    data = {w: {0: [], 'density0': 0} for w in target_words}

    WINDOW = 15
    logger.info('Reading corpus...')
    for line in texts:
        words = set(line)
        found = words.intersection(target_words)
        if found:
            for target in found:
                cv = extract_context(target, line, model, WINDOW)
                data[target][0].append(cv)

    logger.info('Calculating densities...')
    for word in target_words:
        if len(data[word][0]) < 1:
            logger.info('%s omitted because of low frequency' % word)
            print('{}\t10'.format(word), file=f_out)
            continue
        w_vectors = np.stack(data[word][0], axis=0)
        distances = cdist(w_vectors, w_vectors, metric='cosine')
        similarities = [1 - x for x in distances]
        data[word]['density0'] = np.mean(similarities)
        print('{}\t{}'.format(word, data[word]['density0']), file=f_out)

    if f_out:
        f_out.close()
