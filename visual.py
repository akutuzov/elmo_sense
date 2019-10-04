# python3
# coding: utf-8

import sys
import pylab as plot
from smart_open import open
from operator import itemgetter
import numpy as np


if __name__ == '__main__':
    files2process = sys.argv[1:]

    all_coeffs = {}

    for f in files2process:
        name = f.split('/')[-1].split('_')[0]
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

    ranks = {word: [] for word in words}
    for nr, year in enumerate(years):
        year_coeffs = []
        for word in words:
            year_coeffs.append((word, coeffs[word][nr]))
        ranked_words = [w[0] for w in sorted(year_coeffs, key=itemgetter(1))]
        for word in words:
            rank = ranked_words.index(word)
            ranks[word].append(rank)

    int_years = [int(y) for y in years]

    for data in [coeffs]:
        diversified = [w for w in data.keys()
                       if data[w][0] < data[w][1] < data[w][2] < data[w][3] < data[w][4]]
        specialized = [w for w in data.keys()
                       if data[w][0] > data[w][1] > data[w][2] > data[w][3] > data[w][4]]
        # for i in diversified:
        #    print(i, data[i])
        # print('==========')
        # for i in specialized:
        #    print(i, data[i])
        # print('=============')

    changes = []

    for word in coeffs:
        for i in range(len(coeffs[word]) - 1):
            diff = round(coeffs[word][i + 1] - coeffs[word][i], 4)
            out = (word, (years[i], years[i + 1]), diff)
            changes.append(out)

    interesting_words = set()
    sorted_changes = [d for d in sorted(changes, key=itemgetter(2), reverse=True)]
    for i in sorted_changes[:5]:
        print(i)
        interesting_words.add(i[0])
    print('=============')
    for i in sorted_changes[-5:]:
        print(i)
        interesting_words.add(i[0])

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
             tick_label=double_labels)
    plot_title = 'Average changes in lexical ambiguity per decade'
    plot.title(plot_title)
    plot.xticks()
    plot.xlabel('Transitions between decades')
    plot.ylabel('Change in the average ELMo variation coefficients')
    plot.savefig('elmo_dia_years.png', dpi=300)
    plot.close()
    plot.clf()

    plot.clf()
    for word in sorted(interesting_words):
        plot.plot(int_years, coeffs[word], label=word)
    plot_title = 'Sharpest changes in words ambiguity'
    plot.title(plot_title)
    plot.xticks(int_years)
    plot.xlabel('Decades')
    plot.ylabel('ELMo variation coefficients')
    plot.legend(loc='best')
    # plot.show()
    plot.savefig('elmo_dia_interesting.png', dpi=300)
    plot.close()
    plot.clf()

    plot.clf()
    for word in words:
        plot.plot(int_years, coeffs[word], label=word)
    plot_title = 'Sharpest changes in words ambiguity'
    plot.title(plot_title)
    plot.xticks(int_years)
    plot.xlabel('Decades')
    plot.ylabel('ELMo variation coefficients')
    # plot.show()
    plot.savefig('elmo_dia_all.png', dpi=300)
    plot.close()
    plot.clf()
