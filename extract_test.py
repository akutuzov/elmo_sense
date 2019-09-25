# python3
# coding: utf-8

import sys
import argparse
from elmo_helpers import *
from smart_open import open
import csv
from collections import Counter
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--input', '-i', help='Path to input text, tab-separated', required=True)
    arg('--elmo', '-e', help='Path to ELMo model', required=True)
    arg('--outfile', '-o', help='Output pickle to save embeddings and sentences', required=True)
    arg('--batch', '-b', help='ELMo batch size', default=30, type=int)

    args = parser.parse_args()
    data_path = args.input
    batch_size = args.batch

    ambig_words = set()
    freq_dict = Counter()
    sentences = []
    vectors = []

    data = csv.DictReader(open(data_path), delimiter='\t')
    for i in data:
        word = i['word'].strip()
        left = i['left'].strip().split()
        right = i['right'].strip().split()
        ambig_words.add(word)
        sentence = left + [word] + right
        freq_dict.update(sentence)
        sentences.append(sentence)

    freq_dict = {w: freq_dict[w] for w in freq_dict if freq_dict[w] > 2 and len(w) > 2}
    print('Words to test:', len(freq_dict), file=sys.stderr)
    print('Ambiguous words:', len(ambig_words), file=sys.stderr)

    print('=====')
    print('%d sentences total' % len(sentences))
    print('=====')

    # Loading a pre-trained ELMo model:
    # You can call load_elmo_embeddings() with top=True to use only the top ELMo layer
    batcher, sentence_character_ids, elmo_sentence_input = load_elmo_embeddings(args.elmo, top=True)

    # Actually producing ELMo embeddings for our data:
    with tf.Session() as sess:
        # It is necessary to initialize variables once before running inference.
        sess.run(tf.global_variables_initializer())
        for chunk in divide_chunks(sentences, batch_size):
            elmo_vectors = get_elmo_vectors(sess, chunk, batcher, sentence_character_ids,
                                            elmo_sentence_input)

            # Due to batch processing, the above code produces for each sentence
            # the same number of token vectors, equal to the length of the longest sentence
            # (the 2nd dimension of the elmo_vector tensor).
            # If a sentence is shorter, the vectors for non-existent words are filled with zeroes.
            # Let's make a version without these redundant vectors:
            for vect, sent in zip(elmo_vectors, chunk):
                cropped_vector = vect[:len(sent), :]
                vectors.append(cropped_vector)

    print('ELMo embeddings for your input are ready')
    print('Vectors:', len(vectors))

    with open(args.outfile, 'wb') as w:
        pickle.dump([sentences, vectors, freq_dict, ambig_words], w)
    print('Vectors saved to', args.outfile, file=sys.stderr)
