# python3
# coding: utf-8

import argparse
import pickle
from gensim.matutils import unitvec
from scipy.stats import ttest_ind as test
from scipy.stats import spearmanr
from smart_open import open
from elmo_helpers import *
import sys
from nltk.corpus import wordnet as wn



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--input', '-i', help='Path to output pickle', required=True)
    arg('--norm', '-n', help='Normalize vectors?', default=False, action='store_true')
    # arg('--tagger', '-t', help='Path to UDPipe model', required=True)

    parser.set_defaults(norm=False)
    args = parser.parse_args()
    data_path = args.input

    # model = Model.load(args.tagger)
    # pipeline = Pipeline(model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')

    with open(args.input, 'rb') as f:
        sentences, vectors, freq_dict, ambig_words = pickle.load(f)

    print('Words to test:', len(freq_dict), file=sys.stderr)
    vector_size = vectors[0].shape[1]

    representations = {w: np.zeros((freq_dict[w], vector_size)) for w in freq_dict}
    counters = {w: 0 for w in freq_dict}

    for sent, matrix in zip(sentences, vectors):
        for word, vector in zip(sent, matrix):
            if word in representations:
                if args.norm:
                    representations[word][counters[word], :] = unitvec(vector)
                else:
                    representations[word][counters[word], :] = vector
                counters[word] += 1

    vectors = None

    diversities = {w: diversity(representations[w]) for w in representations}

    ambig_freqs = [freq_dict[w] for w in ambig_words]
    mean_amb_freq = np.mean(ambig_freqs)
    std_amb_freq = np.std(ambig_freqs)
    print('Ambiguous frequencies: %.4f, %.4f' % (float(mean_amb_freq), float(std_amb_freq)))
    min_freq = int(mean_amb_freq - std_amb_freq)
    max_freq = int(mean_amb_freq + std_amb_freq)

    diversities_ambig = [diversities[w] for w in diversities if w in ambig_words]
    # print(ambig_words)

    common_words_test = [w for w in diversities if w not in ambig_words]
                         # and 100 < freq_dict[w] < 400]
    # common_tagged = [tag(pipeline, w) for w in common_words_test]
    # common_words_test = [w.split('_')[0] for w in common_tagged if w.endswith('_NOUN')]
    # common_words_test = {w for w in common_words_test if w in diversities}
    # print(common_words_test)

    diversities_common = [diversities[w] for w in common_words_test]
    common_freqs = [freq_dict[w] for w in common_words_test]
    print('Common frequencies: %.4f, %.4f' %
          (float(np.mean(common_freqs)), float(np.std(common_freqs))))

    for nr, dataset in enumerate([diversities_ambig, diversities_common]):
        print(nr, 'Words:', len(dataset))
        print('Average diversity: %.4f' % np.mean(dataset))
        print('Diversity std: %.4f' % np.std(dataset))

    print('T-test (difference and p-value): %.4f, %.4f' %
          test(diversities_ambig, diversities_common, equal_var=False))

    diversity_degrees = []
    frequencies = []
    nr_synsets = []

    print('===================')
    common_diversities = {w: diversities[w] for w in common_words_test}
    for word in sorted(common_diversities, key=common_diversities.get, reverse=True):
        # print(word, round(common_diversities[word], 3), len(wn.synsets(word)))
        synsets = len(wn.synsets(word))
        if synsets > 0:
            diversity_degrees.append(common_diversities[word])
            frequencies.append(freq_dict[word])
            nr_synsets.append(synsets)

    print('===================')
    amb_diversities = {w: diversities[w] for w in ambig_words}
    for word in sorted(amb_diversities, key=amb_diversities.get, reverse=True):
        # print(word, round(amb_diversities[word], 3), len(wn.synsets(word)))
        synsets = len(wn.synsets(word))
        if synsets > 0:
            diversity_degrees.append(amb_diversities[word])
            frequencies.append(freq_dict[word])
            nr_synsets.append(synsets)

    print('Words with at least 1 synset:', len(diversity_degrees))
    print('Spearman correlation between diversity and the number of WordNet synsets: %.4f, %.4f'
          % spearmanr(diversity_degrees, nr_synsets))
    print('Spearman correlation between frequency and the number of WordNet synsets: %.4f, %.4f'
          % spearmanr(frequencies, nr_synsets))
