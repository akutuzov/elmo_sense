# python3
# coding: utf-8

import argparse
from elmo_helpers import *
import pylab as plot

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--input', '-i', help='Path to the directory with npz files', required=True)
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
        coeffs[word] = []
        for decade in [e_60, e_70, e_80, e_90, e_00]:
            coeff = diversity(decade[word])
            coeffs[word].append(coeff)
    logger.info('Finished!')

    for word in words:
        plot.plot(int_years, coeffs[word], label=word)
    plot_title = 'Changes in word ambiguity over time, %s' % args.input.split('/')[-1]
    plot.title(plot_title)
    plot.xticks(int_years)
    plot.xlabel('Decades')
    plot.ylabel('ELMo diversity coefficients')
    plot.savefig('%s_elmo_dia_all.png' % args.input.split('/')[-1], dpi=300)
    plot.close()
    plot.clf()
