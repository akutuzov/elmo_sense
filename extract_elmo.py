# python3
# coding: utf-8

import sys
import argparse
from smart_open import open
from elmo_helpers import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--input', '-i', help='Path to input text', required=True)
    arg('--elmo', '-e', help='Path to ELMo model', required=True)
    arg('--outfile', '-o', help='Output pickle to save embeddings and sentences', required=True)
    arg('--vocab', '-v', help='Path to vocabulary file', required=True)
    arg('--batch', '-b', help='ELMo batch size', default=30, type=int)

    args = parser.parse_args()
    data_path = args.input
    batch_size = args.batch
    vocab_path = args.vocab
    vector_size = 1024

    vect_dict = {}
    with open(vocab_path, 'r') as f:
        # for line in f.readlines():
        for line in f.readlines()[100:1500]:
            (word, freq) = line.strip().split('\t')
            if len(word) > 2 and word.isalpha():
                vect_dict[word] = np.zeros((int(freq), vector_size))

    print('Words to test:', len(vect_dict), file=sys.stderr)

    counters = {w: 0 for w in vect_dict}

    # Loading a pre-trained ELMo model:
    # You can call load_elmo_embeddings() with top=True to use only the top ELMo layer
    batcher, sentence_character_ids, elmo_sentence_input = load_elmo_embeddings(args.elmo, top=True)

    # Actually producing ELMo embeddings for our data:
    lines_processed = 0
    with tf.Session() as sess:
        # It is necessary to initialize variables once before running inference.
        sess.run(tf.global_variables_initializer())

        lines_cache = []
        with open(data_path, 'r') as dataset:
            for line in dataset:
                lines_cache.append(line.strip().split()[:400])
                lines_processed += 1
                if len(lines_cache) == batch_size:
                    elmo_vectors = get_elmo_vectors(sess, lines_cache, batcher,
                                                    sentence_character_ids, elmo_sentence_input)

                    for sent, matrix in zip(lines_cache, elmo_vectors):
                        for word, vector in zip(sent, matrix):
                            if word in vect_dict:
                                vect_dict[word][counters[word], :] = vector
                                counters[word] += 1

                    lines_cache = []
                    print('Lines processed:', lines_processed, file=sys.stderr)

    print('Vector extracted. Pruning zeros...', file=sys.stderr)
    vect_dict = {w: vect_dict[w][~(vect_dict[w] == 0).all(1)] for w in vect_dict}

    print('ELMo embeddings for your input are ready', file=sys.stderr)

    words2save = sorted([w for w in vect_dict.keys() if vect_dict[w].size != 0])
    arrnames = {word: vect_dict[word] for word in words2save}

    np.savez_compressed(args.outfile, **arrnames)

    print('Vectors saved to', args.outfile, file=sys.stderr)
