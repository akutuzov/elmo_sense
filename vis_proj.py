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

    labels = False

    array = np.load(file2process)['arr_0']
    word = file2process.split('/')[-1].split('.')[0]
    year = file2process.split('/')[-2]
    print('Number of points:', array.shape[0], file=sys.stderr)

    embedding = PCA(n_components=2)
    # perplexity = 1000.0  # Should be smaller than the number of points!
    # embedding = TSNE(n_components=2, perplexity=perplexity, metric='cosine',
    #                      n_iter=500, init='pca')

    y = embedding.fit_transform(array)

    print('2-d embedding finished', file=sys.stderr)

    xpositions = y[:, 0]
    ypositions = y[:, 1]

    plot.clf()

    if labels:
        for x, y, nr in zip(xpositions, ypositions, range(len(xpositions))):
            plot.scatter(x, y, 2, marker='*', color='green')
            plot.annotate(nr, xy=(x+1, y), size=2, color='green')
    else:
        plot.scatter(xpositions, ypositions, 5, marker='*', color='green')
    plot.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plot.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    plot.title(' '.join([word, 'in', year+ "'s"]))

    plot.savefig(out + '_' + 'PCA.png', dpi=600, bbox_inches='tight')
    plot.close()
    plot.clf()
