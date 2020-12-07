# Copyright (c) Michael M. Magruder (https://github.com/mikemag)
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import csv
import os
import glob


def load_csv(filename, results, metric):
    with open(os.path.join(filename), 'r') as f:
        r = csv.reader(f)
        header = next(r)
        p_col = header.index('Pin Count')
        c_col = header.index('Color Count')
        algo_col = header.index('Strategy')
        gpu_mode_col = header.index('GPU Mode')
        ig_col = header.index('Initial Guess')
        avg_turns_col = header.index('Average Turns')
        max_turns_col = header.index('Max Turns')
        time_col = header.index('Elapsed (s)')

        for row in r:
            ar = results.setdefault(row[algo_col], {})
            pr = ar.setdefault(int(row[p_col]), {})
            gr = pr.setdefault(int(row[c_col]), {
                'best': {
                    'avg': 9999.9,
                    'ig': '',
                    'max': 9999,
                    'time': 99999.9
                },
            })

            d = {
                'ig': row[ig_col],
                'avg': float(row[avg_turns_col]),
                'max': int(row[max_turns_col]),
                'time': float(row[time_col]),
            }

            if d[metric] < gr['best'][metric] or row[gpu_mode_col] == 'GPU':
                gr['best'] = d


def process_results(metric, metric_format, header):
    results = {}
    result_files = glob.glob(
        '/Users/mike/dev/Mastermind/results/2019_mbp/*.csv')

    for f in result_files:
        load_csv(f, results, metric)

    print('##', header)
    print()

    for a, pd in results.items():
        print('### ' + a)
        print()
        print('', ' ', *[str(c) + 'c' for c in range(2, 16)], '', sep='|')
        print('|:---:' * 15, '|', sep='')

        for p in range(2, 9):
            if p in pd:
                cd = pd[p]
                print('|' + str(p) + 'p', end='')
                for c in range(2, 16):
                    if c in cd:
                        gd = cd[c]
                        ba = gd['best']
                        print('|' + metric_format % ba[metric], end='')
            print('|')
        print()


if __name__ == '__main__':
    process_results('avg', '%0.4f', 'Average turns over all games')
    # process_results('ig', '%s', 'Best initial guess')
    process_results('max', '%d', 'Max turns over all games')
    process_results('time', '%0.5fs', 'Run time')
