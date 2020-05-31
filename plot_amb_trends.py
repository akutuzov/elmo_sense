# python3
# coding: utf-8

import argparse
from elmo_helpers import *
import pylab as plot
from matplotlib.ticker import StrMethodFormatter
from operator import itemgetter
from scipy.stats import spearmanr
from smart_open import open

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--input', '-i', help='Path to the directory with npz files or to the csv', required=True)
    arg('--vocs', '-v', help='Path to the directory with decades frequency vocabularies')
    arg('--model', '-m', help='Name of the model', default='wikipedia')
    arg('--calc', '-c', help='Calculate from npz or tak from csv?', choices=['npz', 'csv'],
        default='npz')
    arg('--outfile', '-o', help='Name of the output file')
    args = parser.parse_args()

    int_years = [1960, 1970, 1980, 1990, 2000]
    coeffs = {}

    if args.calc == 'npz':

        e_60 = np.load(os.path.join(args.input, '1960.npz'))
        logger.info('Loaded an array of {} entries from 1960.npz'.format(len(e_60)))
        e_70 = np.load(os.path.join(args.input, '1970.npz'))
        logger.info('Loaded an array of {} entries from 1970.npz'.format(len(e_70)))
        e_80 = np.load(os.path.join(args.input, '1980.npz'))
        logger.info('Loaded an array of {} entries from 1980.npz'.format(len(e_80)))
        e_90 = np.load(os.path.join(args.input, '1990.npz'))
        logger.info('Loaded an array of {} entries from 1990.npz'.format(len(e_90)))
        e_00 = np.load(os.path.join(args.input, '2000.npz'))
        logger.info('Loaded an array of {} entries from 2000.npz'.format(len(e_00)))

        words = e_00.files

        logger.info('Calculating diversities...')
        for word in words:
            valid = True
            for decade in [e_60, e_70, e_80, e_90, e_00]:
                if decade[word].shape[0] < 2:
                    valid = False
                    break
            if not valid:
                logger.info('{} too rare! Skipping.'.format(word))
                continue
            coeffs[word] = []
            for decade in [e_60, e_70, e_80, e_90, e_00]:
                coeff = diversity(decade[word])
                coeffs[word].append(coeff)
        logger.info('Finished!')

    else:
        logger.info('Loading diversities from {}...'.format(args.input))
        with open(args.input, 'r') as f:
            for line in f:
                res = line.strip().split('\t')
                word, w_60, w_70, w_80, w_90, w_00 = res
                coeffs[word.strip()] = [float(d) for d in [w_60, w_70, w_80, w_90, w_00]]

    for word in coeffs:
        plot.plot(int_years, coeffs[word], label=word)
    plot_title = 'Changes in word ambiguity over time, {} frequency'.format(
        args.input.split('/')[-1].split('_')[2].split('.')[0])
    plot.title(plot_title)
    plot.xticks(int_years)
    plot.xlabel('Decades')
    plot.ylabel('ELMo diversity coefficients')
    plot.savefig('{0}_{1}_elmo_dia_all.png'.format(
        args.model, args.input.split('/')[-1].split('_')[2].split('.')[0]), dpi=300)
    plot.close()
    plot.clf()

    changes = []

    vocs = None
    if args.vocs:
        logger.info('Loading frequency vocabularies...')
        vocfiles = [f for f in os.scandir(args.vocs) if f.name.endswith('_vocab.txt.gz')]
        vocs = {}
        for f in vocfiles:
            year = f.name.split('_')[0]
            vocabulary = {}
            for line in open(os.path.join(args.vocs, f), 'r'):
                res = line.strip().split('\t')
                word, wordfreq = res
                vocabulary[word.strip()] = int(wordfreq)
            vocs[year] = vocabulary

    for word in coeffs:
        for i in range(len(coeffs[word]) - 1):
            diff = round(coeffs[word][i + 1] - coeffs[word][i], 4)
            if args.vocs:
                freqdiff = abs(vocs[str(int_years[i + 1])][word] - vocs[str(int_years[i])][word])
                out = (word, (str(int_years[i]), str(int_years[i + 1])), diff, freqdiff)
            else:
                out = (word, (str(int_years[i]), str(int_years[i + 1])), diff)
            changes.append(out)

    sorted_changes = [d for d in sorted(changes, key=itemgetter(2), reverse=True)]
    logger.info('Top 5 with increased ambiguity:')
    for i in sorted_changes[:5]:
        logger.info(i)

    logger.info('=============')
    logger.info('Top 5 with decreased ambiguity:')
    for i in sorted_changes[-5:]:
        logger.info(i)

    logger.info('=============')
    year_changes = {}
    for el in changes:
        if el[1] not in year_changes:
            year_changes[el[1]] = []
        year_changes[el[1]].append(el[2])

    for year in year_changes:
        logger.info('{} change: {}'.format(year, np.mean(year_changes[year])))
    sorted_doubles = sorted(year_changes.keys())
    double_labels = [el[0] + '-' + el[1] for el in sorted_doubles]

    if args.vocs:
        for pair in sorted_doubles:
            div_changes = [el[2] for el in changes if el[1] == pair]
            freq_changes = [el[3] for el in changes if el[1] == pair]
            correlation = spearmanr(div_changes, freq_changes)
            print('Spearman correlation between diversity changes and frequency changes '
                  'in {0}: {1:.4f}, {2:.4f}'.format(pair, correlation[0], correlation[1]))

        exit()

    plot.clf()
    plot.bar(range(len(sorted_doubles)), [np.mean(year_changes[year]) for year in sorted_doubles],
             tick_label=double_labels, color='red')
    plot_title = 'Average changes in lexical ambiguity per decade, {} frequency'.format(
        args.input.split('/')[-1].split('_')[2].split('.')[0])
    plot.title(plot_title)
    plot.xticks()
    plot.xlabel('Transitions between decades')
    plot.ylabel('Change in the average ELMo variation coefficients')
    plot.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.4f}'))
    plot.savefig('{0}_{1}_elmo_dia_years.png'.format(
        args.model, args.input.split('/')[-1].split('_')[2].split('.')[0]), dpi=300)
    plot.close()
    plot.clf()

    if args.outfile:
        with open(args.outfile, 'a') as f:
            for word in coeffs:
                coeff = '\t'.join([str(d) for d in coeffs[word]])
                f.write('\t'.join([word, coeff]) + '\n')
