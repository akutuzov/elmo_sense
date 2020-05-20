# python3
# coding: utf-8

import sys
from gensim.corpora import Dictionary
from gensim.models import LsiModel, word2vec

if __name__ == '__main__':
    textfile = sys.argv[1]

    texts = word2vec.LineSentence(textfile)

    print('Building dictionary...', file=sys.stderr)
    dictionary = Dictionary(texts, prune_at=40000)
    dictionary.save('lsi.dic')

    texts = word2vec.LineSentence(textfile)
    print('Vectorizing lines...', file=sys.stderr)
    corpus = [dictionary.doc2bow(line) for line in texts]

    print('Building LSI model...', file=sys.stderr)
    model = LsiModel(corpus, id2word=dictionary, num_topics=100)
    model.save('lsi.model')
