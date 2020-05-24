# python3
# coding: utf-8

import sys
import pylab as plot
import numpy as np
from sklearn.decomposition import PCA
import logging

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    file2process = sys.argv[1]

    LABELS = False

    embeddings = np.load(file2process)
    words = embeddings.files

    year = file2process.split('/')[-1].split('.')[0]

    for word in words:
        array = embeddings[word]
        logger.info('{}, number of points: {}'.format(word, array.shape[0]))

        embedding = PCA(n_components=2)
        y = embedding.fit_transform(array)

        xpositions = y[:, 0]
        ypositions = y[:, 1]

        plot.clf()

        if LABELS:
            for x, y, nr in zip(xpositions, ypositions, range(len(xpositions))):
                plot.scatter(x, y, 2, marker='*', color='green')
                plot.annotate(nr, xy=(x+1, y), size=2, color='green')
        else:
            plot.scatter(xpositions, ypositions, 5, marker='*', color='green')
        plot.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plot.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        plot.title("{} in {}'s".format(word, year))
        out = "{}_{}".format(word, year)

        plot.savefig(out + 'PCA.png', dpi=600, bbox_inches='tight')
        plot.close()
        plot.clf()
