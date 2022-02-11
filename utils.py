import io
import re
import subprocess
import pandas as pd
import numpy as np

from pathlib import Path
from scipy import ndimage
from functools import lru_cache
from scipy.signal import find_peaks
from collections import defaultdict
from typing import List, Tuple, Optional


SMT_CONTEXTS_CNT = 2
BUILD_DIR = Path('build')


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

    return [sorted(list(c)) for c in
            filter(lambda t: len(t) == SMT_CONTEXTS_CNT, results.values())]


@lru_cache
def get_uarch():
    p = subprocess.run(['gcc', '-march=native', '-Q', '--help=target'],
                       stdout=subprocess.PIPE)
    assert(p.returncode == 0)
    out = p.stdout.decode()
    match = re.findall(r'-march=\s+(\w+)', out)
    if match:
        return match[0]
    else:
        raise ValueError('Failed to determine the uarch')


@lru_cache
def check_pti():
    with open('/sys/devices/system/cpu/vulnerabilities/meltdown') as f:
        status = f.read()
        return 'PTI' in status


def run_once(cmd: List[str], output: str, oracle: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise ValueError(f'Failed to execute {cmd}, return code: {p.returncode}')

    def read_tsv(filename: str):
        if filename is None:
            return None
        src = io.StringIO(p.stdout.decode()) if filename == '-' else filename
        data = pd.read_csv(src, sep='\t', header=None).to_numpy()
        Path(filename).unlink(missing_ok=True)
        return data

    return read_tsv(output), read_tsv(oracle)


def rebuild(build: Path=BUILD_DIR):
    # check ninja
    ninja_path = build / 'build.ninja'
    use_ninja = ninja_path.exists()

    cmd = 'ninja' if use_ninja else 'make'
    subprocess.run([cmd], cwd=build)


def get_pls(addr):
    page = addr >> 12
    pl1 = page & 0x1ff
    pl2 = (page >> 9) & 0x1ff
    pl3 = (page >> 18) & 0x1ff
    pl4 = (page >> 27) & 0x1ff
    return pl1, pl2, pl3, pl4


def top_n_peaks(peaks: np.ndarray, means: np.ndarray, npeaks: int,
                reverse: bool) -> np.ndarray:
    peaks = sorted(peaks, key=lambda x: means[x], reverse=reverse)
    return peaks[-npeaks:]


def peak_seeking_scipy(means: np.ndarray, npeaks: Optional[int]=None,
                       reverse: bool=False, thresh_r: Optional[float]=None):
    data = -means if reverse else means
    thresh = np.mean(means) * thresh_r if thresh_r is not None else None
    peaks, _ = find_peaks(data, threshold=thresh)
    if npeaks is None:
        return peaks
    else:
        return top_n_peaks(peaks, means, npeaks, reverse)


def peak_seeking_std(means: np.ndarray, npeaks: Optional[int]=None, sigma: int=2,
                     reverse: bool=False):
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


# may raise IndexError
def find_indexes(data: np.array, ts: int) -> int:
    assert(len(data.shape) == 1 and 'Only support 1d array')
    return np.where(data >= ts)[0][0]


# filters
def filter_outliers(data: np.array, sigma: int, iters: int=3, thresh=1e-4) -> np.array:
    if len(data) == 0:
        return data

    for _ in range(iters):
        mean = np.mean(data)
        stdev = np.std(data)
        if stdev < thresh:
            break
        f_arr = abs(data - mean) <= sigma * stdev
        data = data[f_arr]
    return data
