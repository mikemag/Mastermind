# Copyright (c) Michael M. Magruder (https://github.com/mikemag)
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import json
import os
from collections import OrderedDict


def load_json(filename, results, metric):
    with open(os.path.join(filename), 'r') as f:
        r = json.load(f)
        runs = [x['run'] for x in r if 'run' in x]

        for run in runs:
            ar = results.setdefault(run['Strategy'], {})
            pr = ar.setdefault(int(run['Pin Count']), {})
            gr = pr.setdefault(int(run['Color Count']), {
                'best': {
                    'avg': 9999.9,
                    'ig': '',
                    'max': 9999,
                    'time': 99999.9,
                    'scores': 999_999_999_999_999,
                },
            })

            d = {
                'ig': run['Initial Guess'],
                'avg': float(run['Average Turns']),
                'max': int(run['Max Turns']),
                'time': float(run['Elapsed (s)']),
                'scores': int(run['Scores']),
            }

            if d[metric] < gr['best'][metric] or run['Solver'] == 'CUDA':
                gr['best'] = d


def process_results(metric, metric_format, header):
    results = OrderedDict()
    result_files = glob.glob(
        '/Users/mike/dev/Mastermind/results/2022_i7-10700K_CUDA_3070_ubuntu22/*.json')

    for f in result_files:
        load_json(f, results, metric)

    print('##', header)
    print()

    for a, pd in results.items():
        print('### ' + a)
        print()
        print('', ' ', *[str(c) + 'c' for c in range(2, 16)], '', sep='|')
        print('|---:' * 15, '|', sep='')

        for p in range(2, 9):
            print('|' + str(p) + 'p', end='')
            if p in pd:
                cd = pd[p]
                for c in range(2, 16):
                    print('|', end='')
                    if c in cd:
                        gd = cd[c]
                        ba = gd['best']
                        print(metric_format.format(ba[metric]), end='')
            print('|')
        print()


if __name__ == '__main__':
    process_results('avg', '{:,.4f}', 'Average turns over all games')
    process_results('max', '{:,d}', 'Max turns over all games')
    process_results('time', '{:,.5f}s', 'Run time')
    process_results('ig', '{}', 'Initial guess')
    process_results('scores', '{:,d}', 'Codeword scores performed')
