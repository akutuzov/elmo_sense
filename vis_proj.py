# python3
# coding: utf-8

import sys
import pylab as plot
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

if __name__ == '__main__':
    file2process = sys.argv[1]
    out = sys.argv[2]

    array = np.load(file2process)['arr_0']
    word = file2process.split('/')[-1].split('.')[0]
    year = file2process.split('/')[-2]
    print('Number of points:', array.shape[0], file=sys.stderr)

    perplexity = 10.0  # Should be smaller than the number of points!
    embedding_pca = PCA(n_components=2)
    embedding_tsne = TSNE(n_components=2, perplexity=perplexity, metric='cosine',
                          n_iter=500, init='pca')

    for name, e in zip(['PCA', 't-SNE'], [embedding_pca, embedding_tsne]):
        y = e.fit_transform(array)

        print(name, '2-d embedding finished', file=sys.stderr)

        xpositions = y[:, 0]
        ypositions = y[:, 1]

        plot.clf()

        # for x, y in zip(xpositions, ypositions):
        plot.scatter(xpositions, ypositions, 5, marker='*', color='green')

        # plot.annotate(lemma, xy=(x - mid, y), size='x-large', weight='bold')

        plot.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plot.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        plot.title(', '.join([word, year, name]))

        plot.savefig(out + '_' + name + '.png', dpi=300, bbox_inches='tight')
        plot.close()
        plot.clf()
