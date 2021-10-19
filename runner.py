#! /usr/bin/env python3

import sys
import logging
import numpy as np
import subprocess

from random import sample
from subprocess import PIPE
from collections import defaultdict

SMT_CONTEXTS = 2
BINO_BIN = 'bin/bino'


def reject_outliers(data, m=2):
    mean = np.mean(data)
    std = np.std(data)
    return list(filter(lambda d: abs(d - mean) < m * std, data))


def scan_ht_cores():
    results = defaultdict(set)
    pid = None
    with open('/proc/cpuinfo') as f:
        for line in f:
            if pid is None and line.startswith('processor'):
                _, pid = line.split(':')
                pid = pid.strip()
            elif pid is not None and line.startswith('core id'):
                _, cid = line.split(':')
                results[int(cid)].add(pid)
                pid = None
    return [','.join(list(c)) for c in filter(lambda t: len(t) == SMT_CONTEXTS,
                                              results.values())]


def run_store_sender(args, cores: str):
    counts = defaultdict(list)
    valid_cnt = 0
    run_cnt = 0
    while valid_cnt < args.iter:
        logging.info(f'Run: {run_cnt}...')
        p = subprocess.run(['taskset', '-c', cores, BINO_BIN, '0'],
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

        # measure this run's quality
        if np.average(hist) < 75 or np.std(hist) > 20:
            logging.warning(f'Poor quality. Discarded run {run_cnt}')
        else:
            for offset, cnt in zip(offsets, hist):
                counts[offset].append(cnt)
            valid_cnt += 1
        run_cnt += 1

    print('', file=sys.stderr)
    for disp in range(512):
        offset = disp << 3
        mean = np.mean(counts[offset])
        std = np.std(counts[offset])
        print(f'{offset:#x}\t{mean:.2f}\t{std:.2f}')


def run_load_timing(args, cores: str):
    counts = defaultdict(list)
    valid_cnt = 0
    while valid_cnt < args.iter:
        logging.info(f'Run: {valid_cnt}...')
        p = subprocess.run(['taskset', '-c', cores, BINO_BIN, args.num],
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

        for offset, cnt in zip(offsets, hist):
            counts[offset].append(cnt)
        valid_cnt += 1

    print('', file=sys.stderr)
    for disp in range(512):
        offset = disp
        mean = np.mean(counts[offset])
        std = np.std(counts[offset])
        print(f'{offset:#x}\t{mean:.2f}\t{std:.2f}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('runner.py')
    parser.add_argument('-c', '--cores', type=str, default=None,
                        help='Pin processes to these cores (e.g., "1,9")')

    subparsers = parser.add_subparsers()
    store_sender = subparsers.add_parser('store_sender')
    store_sender.add_argument('-i', '--iter', type=int, default=10)
    store_sender.set_defaults(func=run_store_sender)

    load_timing = subparsers.add_parser('load_timing')
    load_timing.add_argument('-i', '--iter', type=int, default=10)
    load_timing.add_argument('-n', '--num', type=str, default='1')
    load_timing.set_defaults(func=run_load_timing)

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

    args.func(args, cores)
