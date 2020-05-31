# python3
# coding: utf-8

import sys
import re
import os
import numpy as np
import tensorflow as tf
from bilm import Batcher, BidirectionalLanguageModel, weight_layers
from sklearn import preprocessing
import json
from scipy.spatial.distance import cosine
from sklearn.cluster import DBSCAN, AffinityPropagation
import logging
from scipy.spatial.distance import pdist

logger = logging.getLogger(__name__)


def tokenize(string):
    """
    :param string: well, text string
    :return: list of tokens
    """
    token_pattern = re.compile('(?u)\w+')
    tokens = [t.lower() for t in token_pattern.findall(string)]
    return tokens


def get_elmo_vectors(sess, texts, batcher, sentence_character_ids, elmo_sentence_input):
    """
    :param sess: TensorFlow session
    :param texts: list of sentences (lists of words)
    :param batcher: ELMo batcher object
    :param sentence_character_ids: ELMo character id placeholders
    :param elmo_sentence_input: ELMo op object
    :return: embedding matrix for all sentences (max word count by vector size)
    """

    # Create batches of data.
    sentence_ids = batcher.batch_sentences(texts)
    print('Sentences in this batch:', len(texts), file=sys.stderr)

    # Compute ELMo representations.
    elmo_sentence_input_ = sess.run(elmo_sentence_input['weighted_op'],
                                    feed_dict={sentence_character_ids: sentence_ids})

    return elmo_sentence_input_


def get_elmo_vector_average(sess, texts, batcher, sentence_character_ids, elmo_sentence_input):
    vectors = []

    # Create batches of data.
    sentence_ids = batcher.batch_sentences(texts)
    print('Sentences in this chunk:', len(texts), file=sys.stderr)
    # Compute ELMo representations.
    elmo_sentence_input_ = sess.run(elmo_sentence_input['weighted_op'],
                                    feed_dict={sentence_character_ids: sentence_ids})
    print('ELMo sentence input shape:', elmo_sentence_input_.shape, file=sys.stderr)
    for sentence in range(len(texts)):
        sent_vec = np.zeros((elmo_sentence_input_.shape[1], elmo_sentence_input_.shape[2]))
        for word_vec in enumerate(elmo_sentence_input_[sentence, :, :]):
            sent_vec[word_vec[0], :] = word_vec[1]
        semantic_fingerprint = np.sum(sent_vec, axis=0)
        semantic_fingerprint = np.divide(semantic_fingerprint, sent_vec.shape[0])
        query_vec = preprocessing.normalize(semantic_fingerprint.reshape(1, -1), norm='l2')
        vectors.append(query_vec.reshape(-1))
    return vectors


def load_elmo_embeddings(directory, top=False):
    """
    :param directory: directory with an ELMo model ('model.hdf5', 'options.json' and 'vocab.txt.gz')
    :param top: use ony top ELMo layer
    :return: ELMo batcher, character id placeholders, op object
    """
    if os.path.isfile(os.path.join(directory, 'vocab.txt.gz')):
        vocab_file = os.path.join(directory, 'vocab.txt.gz')
    elif os.path.isfile(os.path.join(directory, 'vocab.txt')):
        vocab_file = os.path.join(directory, 'vocab.txt')
    else:
        raise SystemExit('Error: no vocabulary file found in the directory.')
    options_file = os.path.join(directory, 'options.json')
    weight_file = os.path.join(directory, 'model.hdf5')
    with open(options_file, 'r') as f:
        m_options = json.load(f)
    max_chars = m_options['char_cnn']['max_characters_per_token']
    vector_size = m_options['lstm']['projection_dim'] * 2

    # Create a Batcher to map text to character ids.
    batcher = Batcher(vocab_file, max_chars)

    # Input placeholders to the biLM.
    sentence_character_ids = tf.placeholder('int32', shape=(None, None, max_chars))

    # Build the biLM graph.
    bilm = BidirectionalLanguageModel(options_file, weight_file, max_batch_size=128)

    # Get ops to compute the LM embeddings.
    sentence_embeddings_op = bilm(sentence_character_ids)

    # Get an op to compute ELMo (weighted average of the internal biLM layers)
    elmo_sentence_input = weight_layers('input', sentence_embeddings_op, use_top_only=top)
    return batcher, sentence_character_ids, elmo_sentence_input, vector_size


def divide_chunks(data, n):
    for i in range(0, len(data), n):
        yield data[i:i + n]


def calc_diversity(embfile, method="centroid"):
    array = np.load(embfile)
    logger.info('Loaded an array of %d entries from %s' % (len(array), embfile))
    words = {}
    for word in array:
        if array[word].shape[0] < 3:
            logger.info('%s omitted because of low frequency: %d' % (word, array[word].shape[0]))
            continue
        if method == 'pairwise':
            var_coeff = pairwise_diversity(array[word])
        else:
            var_coeff = diversity(array[word])
        words[word] = var_coeff
    logger.info('Variation coefficients produced')
    return words


def diversity(matrix):
    mean = np.average(matrix, axis=0)
    distances = [cosine(v, mean) for v in matrix]
    return np.mean(distances)


def pairwise_diversity(matrix):
    distances = pdist(matrix, metric='cosine')
    return np.mean(distances)


def diversity_ax(matrix):
    stds = np.std(matrix, axis=0)
    return np.mean(stds)

# Jensen-Shannon divergence (JSD) algorithm

def cluster(usage_matrix, algorithm, word, max_examples=10000):
    """
    :param word: target word
    :param usage_matrix: a matrix of contextualised word representations of shape
    (num_usages, model_dim)
    :param algorithm:the clustering algorithm: DBSCAN or Affinity Propagation
    :param args_dict: the sklearn parameters of the chosen clustering algorithm
    :param max_examples: maximum number of usages to cluster (if higher, will be downsampled)
    :return: n_clusters - the number of clusters in the given usage matrix
             labels - a list of labels, one for each contextualised representation
    """
    if algorithm.lower() not in ['dbscan', 'db', 'affinity', 'affinity propagation', 'ap']:
        raise ValueError('Invalid clustering method:', algorithm)

    if usage_matrix.shape[0] > max_examples:
        prev = usage_matrix.shape[0]
        rand_indices = np.random.choice(prev, max_examples, replace=False)
        usage_matrix = usage_matrix[rand_indices]
        logger.info('Choosing {} random rows from {} for {}'.format(max_examples, prev, word))
    usage_matrix = preprocessing.StandardScaler().fit_transform(usage_matrix)

    if algorithm.lower() in ['dbscan', 'db']:
        clustering = DBSCAN(**args_dict).fit(usage_matrix)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    else:
        clustering = AffinityPropagation().fit(usage_matrix)
        labels = clustering.labels_
        n_clusters = len(set(labels))

    logger.info('{} has {} clusters'.format(word, n_clusters))
    return n_clusters, labels