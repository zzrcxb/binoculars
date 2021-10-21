#! /usr/bin/env python3

import logging
import numpy as np
import subprocess

from time import sleep
from random import sample
from typing import Iterable
from subprocess import PIPE
from matplotlib import pyplot as plt
from collections import defaultdict

SMT_CONTEXTS_CNT = 2
BINO_BIN = 'bin/bino'


def reject_outliers(data, m=2):
    mean = np.mean(data)
    std = np.std(data)
    return list(filter(lambda d: abs(d - mean) < m * std, data))


def scan_ht_cores():
    results = defaultdict(set)
    line_of_interests = ['processor', 'physical id', 'core id']
    ids = []
    with open('/proc/cpuinfo') as f:
        for line in f:
            if any([line.startswith(loi) for loi in line_of_interests]):
                _, data = line.split(':')
                ids.append(int(data))

    assert(len(ids) % 3 == 0)
    for i in range(0, len(ids), 3):
        # key = (physical id, core id); value = processor
        results[(ids[i+1], ids[i+2])].add(str(ids[i]))

    return [','.join(list(c)) for c in
            filter(lambda t: len(t) == SMT_CONTEXTS_CNT, results.values())]


def plotter(name, offsets: Iterable[int], means: Iterable[float], stds: Iterable[float]):
    fig, ax = plt.subplots()
    ax.plot(offsets, means, color='blue')

    min_x, max_x = min(offsets), max(offsets)
    interval_x = (max_x - min_x) // 16
    max_x += .001
    ax.set_xlim(min_x, max_x)
    ax.set_xticks(np.arange(min_x, max_x, interval_x))
    ax.set_xticklabels([f'{int(x):#x}' for x in ax.get_xticks()])
    ax.set_xlabel('offset')
    ax.set_ylabel('measure')
    ax.tick_params(labelsize=9)

    fig.text(.5, 1, name, ha='center', rotation='horizontal', fontsize=12)
    fig.tight_layout()
    fig.set_size_inches(10.0, 4)
    fig.savefig(f'{name}.pdf', bbox_inches='tight')
    plt.clf()


def quality_checker(data: Iterable[int], avg, std) -> bool:
    if avg is not None and np.mean(data) < avg:
        return False

    if std is not None and np.std(data) > std:
        return False
    return True


def repeat_runner(name: str, cores: str, args):
    counts = defaultdict(list)
    valid_cnt, run_cnt = 0, 0
    while valid_cnt < args.iter:
        logging.info(f'Run: {run_cnt}...')
        p = subprocess.run(['taskset', '-c', cores, BINO_BIN, name],
                           stderr=PIPE, stdout=PIPE)
        offsets, hist = [], []
        out = p.stdout.decode('utf8')
        for line in out.split('\n'):
            try:
                offset, cnt = line.strip().split('\t')
            except ValueError:
                pass
            else:
                offsets.append(int(offset, base=16))
                hist.append(int(cnt))

        if not quality_checker(hist, args.avg_thresh, args.std_thresh):
            logging.warning(f'Poor quality. Discarded run {run_cnt}')
        else:
            for offset, cnt in zip(offsets, hist):
                counts[offset].append(cnt)
            valid_cnt += 1
        run_cnt += 1

    offsets = []
    means = []
    stds = []
    for offset in sorted(counts.keys()):
        mean = np.mean(counts[offset])
        std = np.std(counts[offset])
        offsets.append(offset)
        means.append(mean)
        stds.append(std)
        print(f'{offset:#x}\t{mean:.2f}\t{std:.2f}')

    plotter(name, offsets, means, stds)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('runner.py')
    parser.add_argument('-c', '--cores', type=str, default=None,
                        help='Pin processes to these cores (e.g., "1,9")'\
                             'Will do auto detection if not specified')
    parser.add_argument('-i', '--iter', type=int, default=10)

    subparsers = parser.add_subparsers(dest='command')
    store_sender = subparsers.add_parser('store_offset')
    store_sender.add_argument('--avg-thresh', type=float, default=75,
                              help='A threshold of avg(hist) to accept the data')
    store_sender.add_argument('--std-thresh', type=float, default=20,
                              help='A threshold of std(hist) to reject the data')
    store_sender.set_defaults(name='store_offset')

    load_trp = subparsers.add_parser('load_page_throughput')
    load_trp.add_argument('--avg-thresh', type=float, default=None,
                              help='A threshold of avg(hist) to accept the data')
    load_trp.add_argument('--std-thresh', type=float, default=None,
                              help='A threshold of std(hist) to reject the data')
    load_trp.set_defaults(name='load_page_throughput')

    load_ctt = subparsers.add_parser('load_page_contention')
    load_ctt.add_argument('--avg-thresh', type=float, default=None,
                              help='A threshold of avg(hist) to accept the data')
    load_ctt.add_argument('--std-thresh', type=float, default=None,
                              help='A threshold of std(hist) to reject the data')
    load_ctt.set_defaults(name='load_page_contention')


    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='[%(levelname)s] %(message)s')

    if args.cores:
        cores = args.cores
    else:
        cores_list = scan_ht_cores()
        if not cores_list:
            logging.error('Cannot find SMT-enabled processor pairs')
            exit(1)
        cores = sample(cores_list, 1)[0]
    logging.info(f'Going to pin processes to core {cores}')
    repeat_runner(getattr(args, 'command'), cores, args)
