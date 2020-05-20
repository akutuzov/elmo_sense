# python3
# coding: utf-8

import argparse
from elmo_helpers import *
import pylab as plot
from matplotlib.ticker import StrMethodFormatter
from operator import itemgetter

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--input', '-i', help='Path to the directory with npz files', required=True)
    arg('--model', '-m', help='Name of the model', default='Wikipedia')
    arg('--outfile', '-o', help='Name of the output file')
    args = parser.parse_args()

    e_60 = np.load(os.path.join(args.input, '1960.npz'))
    logger.info('Loaded an array of %d entries from %s' % (len(e_60), '1960.npz'))
    e_70 = np.load(os.path.join(args.input, '1970.npz'))
    logger.info('Loaded an array of %d entries from %s' % (len(e_70), '1970.npz'))
    e_80 = np.load(os.path.join(args.input, '1980.npz'))
    logger.info('Loaded an array of %d entries from %s' % (len(e_80), '1980.npz'))
    e_90 = np.load(os.path.join(args.input, '1990.npz'))
    logger.info('Loaded an array of %d entries from %s' % (len(e_90), '1990.npz'))
    e_00 = np.load(os.path.join(args.input, '2000.npz'))
    logger.info('Loaded an array of %d entries from %s' % (len(e_00), '2000.npz'))

    int_years = [1960, 1970, 1980, 1990, 2000]

    words = e_00.files

    coeffs = {}
    logger.info('Calculating diversities...')
    for word in words:
        valid = True
        for decade in [e_60, e_70, e_80, e_90, e_00]:
            if decade[word].shape[0] < 2:
                valid = False
                break
        if not valid:
            print('%s too rare! Skipping.' % word)
            continue
        coeffs[word] = []
        for decade in [e_60, e_70, e_80, e_90, e_00]:
            coeff = diversity(decade[word])
            coeffs[word].append(coeff)
    logger.info('Finished!')

    for word in coeffs:
        plot.plot(int_years, coeffs[word], label=word)
    plot_title = 'Changes in word ambiguity over time, %s frequency' % args.input.split('/')[-1]
    plot.title(plot_title)
    plot.xticks(int_years)
    plot.xlabel('Decades')
    plot.ylabel('ELMo diversity coefficients')
    plot.savefig('%s_%s_elmo_dia_all.png' % (args.model, args.input.split('/')[-1]), dpi=300)
    plot.close()
    plot.clf()

    changes = []

    for word in coeffs:
        for i in range(len(coeffs[word]) - 1):
            diff = round(coeffs[word][i + 1] - coeffs[word][i], 4)
            out = (word, (str(int_years[i]), str(int_years[i + 1])), diff)
            changes.append(out)

    sorted_changes = [d for d in sorted(changes, key=itemgetter(2), reverse=True)]
    logger.info('Top 5 most changed:')
    for i in sorted_changes[:5]:
        logger.info(i)
        # interesting_words.add(i[0])
    logger.info('=============')
    logger.info('Top 5 least changed:')
    for i in sorted_changes[-5:]:
        logger.info(i)
        # interesting_words.add(i[0])
    logger.info('=============')
    year_changes = {}
    for el in changes:
        if el[1] not in year_changes:
            year_changes[el[1]] = []
        year_changes[el[1]].append(el[2])

    for year in year_changes:
        print(year, np.mean(year_changes[year]))
    sorted_doubles = sorted(year_changes.keys())
    double_labels = [el[0] + '-' + el[1] for el in sorted_doubles]

    plot.clf()
    plot.bar(range(len(sorted_doubles)), [np.mean(year_changes[year]) for year in sorted_doubles],
             tick_label=double_labels, color='red')
    plot_title = 'Average changes in lexical ambiguity per decade, %s frequency' \
                 % args.input.split('/')[-1]
    plot.title(plot_title)
    plot.xticks()
    plot.xlabel('Transitions between decades')
    plot.ylabel('Change in the average ELMo variation coefficients')
    plot.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.4f}'))
    plot.savefig('%s_%s_elmo_dia_years.png' % (args.model, args.input.split('/')[-1]), dpi=300)
    plot.close()
    plot.clf()

    if args.outfile:
        with open(args.outfile, 'a') as f:
            for word in coeffs:
                coeff = '\t'.join([str(d) for d in coeffs[word]])
                f.write('\t'.join([word, coeff]) + '\n')
