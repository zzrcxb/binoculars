#! /usr/bin/env python3

import logging
import numpy as np
import subprocess

from random import sample
from typing import Iterable, List, Optional
from subprocess import PIPE
from matplotlib import pyplot as plt
from collections import defaultdict
from scipy.signal import find_peaks
from scipy import ndimage
from operator import gt, lt

SMT_CONTEXTS_CNT = 2
BINO_BIN = 'bin/bino'


def scan_ht_cores() -> List[str]:
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


def top_n_peaks(peaks: np.ndarray, means: np.ndarray, npeaks: int,
                reverse: bool) -> np.ndarray:
    peaks = sorted(peaks, key=lambda x: means[x], reverse=reverse)
    return peaks[-npeaks:]


def peak_seeking_scipy(means: np.ndarray, npeaks: Optional[int]=None,
                       reverse: bool=True, thresh_r: Optional[float]=None):
    data = -means if reverse else means
    thresh = np.mean(means) * thresh_r if thresh_r is not None else None
    peaks, _ = find_peaks(data, threshold=thresh)
    if npeaks is None:
        return peaks
    else:
        return top_n_peaks(peaks, means, npeaks, reverse)


def peak_seeking_std(means: np.ndarray, npeaks: Optional[int]=None, sigma: int=2,
                     reverse: bool=True):
    data = -means if reverse else means
    mean = np.mean(data)
    std = np.std(data)
    image = data > (mean + sigma * std)
    regions = ndimage.find_objects(ndimage.label(image)[0])
    peaks = np.array([np.argmax(data[r[0]]) + r[0].start for r in regions])
    if npeaks is None:
        return peaks

    if len(regions) < npeaks:
        extra_peaks = list(peaks)
        for r in regions:
            extras = peak_seeking_scipy(means[r[0]])
            extra_peaks.extend([r[0].start + e for e in extras])
        peaks = np.array(list(set(extra_peaks)))
    return top_n_peaks(peaks, means, npeaks, reverse)


def plotter(name, offsets: np.ndarray, means: np.ndarray,
            stds: np.ndarray, args, plot_std: bool = True):
    fig, ax = plt.subplots()
    ax.plot(offsets, means, color='blue')

    avg = np.mean(means)
    std = np.std(means)
    ax.axhline(y=avg, color='orange', linestyle='dashed', linewidth=1)
    ax.axhline(y=avg + args.sigma * std, color='grey', linestyle='dashed', linewidth=.5)
    ax.axhline(y=avg - args.sigma * std, color='grey', linestyle='dashed', linewidth=.5)

    if plot_std:
        ub = means + stds
        lb = means - stds
        ax.fill_between(offsets, lb, ub, alpha=.5)

    peaks = peak_seeking_std(means, args.npeaks, sigma=args.sigma,
                             reverse=args.reverse)
    ax.plot(offsets[peaks], means[peaks], 'rx')
    for p in peaks:
        y_offset = -15 if args.reverse else 15
        ax.annotate(f'{offsets[p]:#x}', (offsets[p], means[p]),
                    textcoords="offset points", xytext=(0, y_offset),
                    ha='center', fontsize=9)

    min_x, max_x = min(offsets), max(offsets)
    interval_x = (max_x - min_x) // 16
    max_x += .001
    ax.set_xlim(min_x, max_x)
    ax.set_xticks(np.arange(min_x, max_x, interval_x))
    ax.set_xticklabels([f'{int(x):#x}' for x in ax.get_xticks()])
    ax.set_xlabel('offset')
    ax.set_ylabel(f'avg. measure ({args.iter} runs)')
    ax.tick_params(labelsize=9)

    fig.text(.5, 1, name, ha='center', rotation='horizontal', fontsize=12)
    fig.tight_layout()
    fig.set_size_inches(10.0, 4)
    fig.savefig(f'{name}.pdf', bbox_inches='tight')
    plt.clf()


def quality_checker(data: Iterable[int], avg_thresh, std_thresh,
                    reverse: bool=True) -> bool:
    avg_checker = lt if reverse else gt
    if avg_thresh is not None and avg_checker(np.mean(data), avg_thresh):
        return False

    if std_thresh is not None and np.std(data) > std_thresh * abs(np.mean(data)):
        return False
    return True


def repeat_runner(name: str, cores: str, args):
    counts = defaultdict(list)
    valid_cnt, run_cnt = 0, 0
    while valid_cnt < args.iter:
        logging.info(f'Run: {run_cnt}...')
        cmd = ['taskset', '-c', cores, BINO_BIN, name]
        if args.sudo:
            cmd = ['sudo'] + cmd
        p = subprocess.run(cmd, stderr=PIPE, stdout=PIPE)
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

        if not quality_checker(hist, args.avg_thresh, args.std_thresh, reverse=args.reverse):
            logging.warning(f'Poor quality. Discarded run {run_cnt}')
        else:
            for offset, cnt in zip(offsets, hist):
                counts[offset].append(cnt)
            valid_cnt += 1
        run_cnt += 1

    offsets = np.array(sorted(counts.keys()))
    means = np.array([np.mean(counts[o]) for o in offsets])
    stds = np.array([np.std(counts[o]) for o in offsets])
    plotter(name, offsets, means, stds, args)
    for offset, mean, std in zip(offsets, means, stds):
        print(f'{offset:#x}\t{mean:.2f}\t{std:.2f}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('runner.py')
    parser.add_argument('-c', '--cores', type=str, default=None,
                        help='Pin processes to these cores (e.g., "1,9")'\
                             'Will do auto detection if not specified')
    parser.add_argument('-i', '--iter', type=int, default=10)
    parser.add_argument('-n', '--npeaks', type=int,
                        default=4, help='# of peaks to find')
    parser.add_argument('--avg-thresh', type=float, default=None,
                              help='A threshold of avg(hist) to accept the data')
    parser.add_argument('--std-thresh', type=float, default=None,
                              help='A threshold of std(hist) to reject the data')
    parser.add_argument('-S', '--sudo', action='store_true',
                        help='Running it as sudo so it can set priorty to 0,'
                             'make sure you don\'t need passwd for sudo, or it'
                             'has been cached')
    parser.add_argument('--sigma', type=int, default=2, help='Sigma for the filter')

    subparsers = parser.add_subparsers(dest='command')
    store_sender = subparsers.add_parser('store_offset')
    store_sender.set_defaults(npeaks=1, avg_thresh=75, std_thresh=20, reverse=True)

    store_sender_lat = subparsers.add_parser('store_offset_latency')
    store_sender_lat.set_defaults(npeaks=1, avg_thresh=100, reverse=False)

    load_trp = subparsers.add_parser('load_page_throughput')
    load_trp.set_defaults(reverse=True)

    load_ctt = subparsers.add_parser('load_page_contention')
    load_ctt.set_defaults(reverse=True)

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
