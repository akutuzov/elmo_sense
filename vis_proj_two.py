import numpy as np
import matplotlib.pyplot as plt

import argparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def main():
    """
    Produce PCA plot of a word's contextualised embeddings.
    """

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--input0', '-i0', help='Path to the npz file 0', required=True)
    arg('--input1', '-i1', help='Path to the npz file 1', required=True)
    arg('--target', '-t', help='the target word', required=True)
    arg('--out', '-o', help='Output path for the PCA plot', action="store_true")
    args = parser.parse_args()

    npz1 = args.input0
    npz2 = args.input1
    target = args.target
    if args.out:
        out_path = args.out
    else:
        out_path = args.targer + '.png'

    usages1 = np.load(npz1)
    usages2 = np.load(npz2)

    try:
        usages1 = usages1[target]
    except KeyError:
        print('No usages of "{}" in corpus 1.'.format(target))
        usages1 = None

    try:
        usages2 = usages2[target]
    except KeyError:
        print('No usages of "{}" in corpus 2.'.format(target))
        usages2 = None

    if usages1 is None and usages2 is None:
        print('No usages of "{}". Stop.'.format(target))
        return

    x = np.concatenate([usages1, usages2], axis=0)

    x = StandardScaler().fit_transform(x)
    x_2d = PCA(n_components=2).fit_transform(x)

    plt.figure(figsize=(15, 15))
    plt.xticks([]), plt.yticks([])
    plt.title("'{}'\n".format(target), fontsize=20)

    usages = [x_2d[:len(usages1), :], x_2d[len(usages1):, :]]
    labels = ['Corpus 1', 'Corpus 2']
    colors = ['b', 'g']

    for matrix, label, color in zip(usages, labels, colors):
        plt.scatter(matrix[:, 0], matrix[:, 1], c=color, s=20)

    plt.legend(labels, prop={'size': 15}, location="best")

    plt.savefig(out_path)
    print('Saved plot to file: {}'.format(out_path))


if __name__ == '__main__':
    main()
