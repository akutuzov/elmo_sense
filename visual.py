# python3
# coding: utf-8

import sys
import pylab as plot
from smart_open import open


if __name__ == '__main__':
    files2process = sys.argv[1:]

    all_coeffs = {}

    for f in files2process:
        name = f.split('/')[1].split('_')[0]
        all_coeffs[name] = {}
        with open(f) as inp:
            for l in inp.readlines():
                res = l.strip().split('\t')
                word, coeff = res
                all_coeffs[name][word] = float(coeff)


    words = set.intersection(*[set(all_coeffs[d].keys()) for d in all_coeffs])

    years = sorted(all_coeffs.keys())
    coeffs = {}
    for word in words:
        coeffs[word] = [all_coeffs[year][word] for year in years]
    int_years = [int(y) for y in years]

    plot.clf()
    for word in coeffs:
        plot.plot(int_years, coeffs[word])
    plot_title = 'Diachronic changes in the degree of ambiguity'
    plot.title(plot_title)
    plot.xticks(int_years)
    plot.xlabel('Decades')
    plot.ylabel('ELMo variation coefficients')
    # plot.show()
    plot.savefig('elmo_dia.png', dpi=300)
    plot.close()
    plot.clf()
