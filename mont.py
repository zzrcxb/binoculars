import random
import numpy as np

from scipy import ndimage
from traces import TimeSeries
from typing import Optional, List
from utils import find_indexes, filter_outliers


class TimeTrace(np.ndarray):
    def __new__(cls, data: np.ndarray, ts_slice=0, data_slice=1, **kwargs) -> None:
        obj = np.asarray(data, **kwargs).view(cls)
        obj._ts_slice = ts_slice
        obj._data_slice = data_slice
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self._ts_slice = getattr(obj, '_ts_slice', None)
        self._data_slice = getattr(obj, '_data_slice', None)

    @property
    def tss(self):
        return self[:, self._ts_slice]

    @tss.setter
    def tss(self, update):
        self[:, self._ts_slice] = update

    @property
    def vals(self):
        return self[:, self._data_slice]

    @vals.setter
    def vals(self, update):
        self[:, self._data_slice] = update


def get_tick_rate(tss: np.array, sigma=3):
    diffs = filter_outliers(tss[1:] - tss[:-1], sigma=sigma)
    return np.mean(diffs), np.std(diffs)


def normalize_data(vals: np.ndarray):
    return (vals - np.mean(vals)) / np.std(vals)


class DataTrace:
    def __init__(self, r_data: np.ndarray, oracle: np.ndarray, filter_func=None, margin=100,
                 period=200, zero_index=1, correct_anomaly=True, anomaly_thresh=1000) -> None:
        self.period = period
        self.margin = margin

        # make a copy, we don't want to mess up the original data
        r_data = TimeTrace(np.copy(r_data))
        oracle = TimeTrace(np.copy(oracle))

        self._calibrate_and_check(r_data, oracle, zero_index)
        r_data = r_data.astype('float64')
        oracle = oracle.astype('float64')

        self._detect_anomaly(r_data, oracle, anomaly_thresh, correct_anomaly)
        self._calibrate_and_check(r_data, oracle, zero_index) # needs a re-calibration
        self.raw_data = TimeTrace(r_data) # save calibrated data

        # new oracle has a format of (t_start, t_end, bit)
        new_oracle = list(zip(oracle.tss[:-1], oracle.tss[1:], oracle.vals[:-1]))
        last_ts, last_val = oracle[-1, :]
        new_oracle.append([last_ts, last_ts + self.tick_rate, last_val])
        self.oracle = TimeTrace(new_oracle, ts_slice=slice(0, 2), data_slice=2)

        self.filtered_data = np.copy(self.raw_data)
        if callable(filter_func):
            self.filtered_data = filter_func(self.filtered_data)

        ts = TimeSeries(self.filtered_data)
        resampled = ts.sample(period, self.t_start, self.t_end, interpolate='linear')
        self.resampled = TimeTrace(resampled, dtype='float64')

        # normalize resampled data
        self.normalized = TimeTrace(np.copy(self.resampled))
        self.normalized.vals = normalize_data(self.normalized.vals)

    def _calibrate_and_check(self, r_data: TimeTrace, oracle: TimeTrace, zero_index: int):
        # calibrate the second oracle timestamp to zero, improve FP's precision
        r_data.tss -= oracle.tss[zero_index]
        oracle.tss -= oracle.tss[zero_index]

        margin_in_cycle = self.margin * self.period
        self.tick_rate, self.tick_std = get_tick_rate(oracle.tss)
        self.t_start = int(oracle.tss[0] - margin_in_cycle)
        self.t_end = int(oracle.tss[-1] + self.tick_rate + 3 * self.tick_std + margin_in_cycle)
        if self.t_start < r_data.tss[0] or self.t_end > r_data.tss[-1]:
            raise ValueError(f'TimeSeries do not fully overlap.'
                             f't_start: {self.t_start}, first: {r_data.tss[0]}; '
                             f't_end: {self.t_end}, last: {r_data.tss[-1]}')

    def _detect_anomaly(self, raw_data: TimeTrace, oracle: TimeTrace, anomaly_thresh: int,
                        correct_anomaly: bool):
        self.lats_anomalies = []

        start_idx = find_indexes(raw_data.tss, self.t_start)
        end_idx = find_indexes(raw_data.tss, self.t_end)
        tss = raw_data.tss[start_idx:end_idx]
        tick_rate, tick_std = get_tick_rate(tss)
        assert(anomaly_thresh > tick_rate + 3 * tick_std)

        diffs = tss[1:] - tss[:-1]
        anomalies = diffs > anomaly_thresh
        features = ndimage.find_objects(ndimage.label(anomalies)[0])
        for f in features:
            s_idx, e_idx = f[0].start, f[0].stop
            if correct_anomaly:
                for i in range(s_idx + 1, e_idx + 1):
                    # tss[i] is the second (right) point of the gap
                    cor = tss[i] - tss[i - 1] - tick_rate
                    raw_data.tss[start_idx + i:] -= cor
                    try:
                        o_idx = find_indexes(oracle.tss, tss[i - 1])
                    except IndexError:
                        pass
                    else:
                        oracle.tss[o_idx:] -= cor
            self.lats_anomalies.append((tss[s_idx], tss[e_idx]))

        for ts, te in zip(oracle.tss[:-1], oracle.tss[1:]):
            if ts > te:
                raise ValueError('Data trace contains too extreme anomalies')

    def split(self, data_src: TimeTrace, regions=None, n_samples=None) -> List[np.ndarray]:
        results = []
        if n_samples:
            tss = TimeSeries(data_src)

        regions = self.oracle.tss if regions is None else regions
        for ts, te in regions:
            if n_samples:
                # resample the region into n_samples, period is calculated on the fly
                period = (te - ts) / n_samples
                resample = tss.sample(period, ts, te - period * .1, interpolate='linear')
                slc = TimeTrace(resample, dtype='float64')
                assert(len(slc) == n_samples)
            else:
                slc = data_src[np.logical_and(data_src.tss >= ts, data_src.tss < te)]
            results.append(slc)
        return results

    def slide_boundary_data(self, width: int=80, drop_rate: Optional[float]=None):
        tss, vals = self.normalized.tss, self.normalized.vals
        oracle_indexes = [find_indexes(tss, ts) for ts in self.iter_boundaries]
        Xs, ys = [], []
        start_idx = width
        end_idx = len(vals) - width
        assert(start_idx < oracle_indexes[0])
        for idx in range(start_idx, end_idx):
            vec = vals[idx - width:idx + width]
            if oracle_indexes and oracle_indexes[0] in [idx, idx + 1]:
                Xs.append(vec)
                ys.append(1)
                if idx == oracle_indexes[0]:
                    oracle_indexes.pop(0)
            elif drop_rate is None or random.random() > drop_rate:
                Xs.append(vec)
                ys.append(-1)
        return tss[start_idx:end_idx], np.array(Xs), np.array(ys)

    @property
    def iter_boundaries(self):
        boundaries = [ts for ts, _ in self.oracle.tss] + [self.oracle.tss[-1, 1]]
        return boundaries

    @property
    def ground_truth(self) -> np.array:
        return self.oracle.vals


# predict signal
best_signal_params = dict(n_estimators=400, min_samples_split=5, min_samples_leaf=1,
                          max_features='log2', max_depth=30, bootstrap=True)

def std_filter_fact(sigma: int=3, iters: int=1, index=1):
    def std_filter(data: np.array):
        for _ in range(iters):
            mean = np.mean(data[:, index])
            stdev = np.std(data[:, index])
            data = data[abs(data[:, index] - mean) < sigma * stdev]
        return data
    return std_filter

