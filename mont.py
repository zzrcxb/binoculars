import random
import pickle
import logging
import numpy as np

from scipy import ndimage
from pathlib import Path
from traces import TimeSeries
from collections import Counter
from typing import Optional, List
from utils import find_indexes, filter_outliers

from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


DATASET_ROOT = Path('results')
MODEL_ROOT = Path('results')

MONT_LOGGER = logging.getLogger('mont')

# trace related data structure
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
        tick_rate, _ = get_tick_rate(tss)

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


def get_raw_filename(ver: str, use: str):
    data_f = DATASET_ROOT / f'{use}-{ver}-data.raw'
    oracle_f = DATASET_ROOT / f'{use}-{ver}-oracle.raw'
    return data_f, oracle_f


def save_raw_data(ver, use, data_l: List[np.ndarray], oracle_l: List[np.ndarray]):
    data_f, oracle_f = get_raw_filename(ver, use)
    with data_f.open('wb') as d, oracle_f.open('wb') as o:
        pickle.dump(data_l, d)
        pickle.dump(oracle_l, o)


def load_raw_data(ver, use):
    data_f, oracle_f = get_raw_filename(ver, use)
    with data_f.open('rb') as d, oracle_f.open('rb') as o:
        data_l = pickle.load(d)
        oracle_l = pickle.load(o)
    return data_l, oracle_l


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


# predict boundaries
best_boundary_params = dict(n_estimators=400, min_samples_split=5, min_samples_leaf=1,
                            max_features='log2', max_depth=30, bootstrap=True)


def boundary_dataset_names(ver, width=80, period=200):
    suffix = f'-boundary-{ver}-{width}-{period}.npy'
    return DATASET_ROOT / ('X' + suffix), DATASET_ROOT / ('y' + suffix)


def boundary_model_name(ver, width=80, period=200):
    return MODEL_ROOT / f'boundary-{ver}-{width}-{period}.pickle'


def gen_boundary_training(iters, ver: str, width=80, period=200, drop_rate=.9,
                          raw_results=None, oracles=None, correct_anomaly=True, **kwargs):
    def get_len(d):
        if isinstance(d, list):
            return len(d)
        else:
            return 0

    Xs, ys = [], []
    cnt, i = 0, 0
    max_iters = max(iters, min(get_len(raw_results), get_len(oracles)))
    while cnt < iters and i < max_iters:
        print(f'\r{i}', end='')
        data = raw_results[i]
        oracle = oracles[i]
        i += 1

        try:
            dt = DataTrace(data, oracle, std_filter_fact(), margin=width * 2,
                           period=period, correct_anomaly=correct_anomaly, **kwargs)
        except ValueError as e:
            print(e)
            continue

        _, X, y = dt.slide_boundary_data(width, drop_rate)
        Xs.append(X)
        ys.append(y)
        cnt += 1

    Xs = np.concatenate(Xs)
    ys = np.concatenate(ys)
    Xs_file, tests_file = boundary_dataset_names(ver, width, period)
    np.save(Xs_file, Xs)
    np.save(tests_file, ys)
    print('')
    print(Xs.shape, ys.shape)
    for k, c in Counter(ys).items():
        print(f'{k}: {c / len(ys): .2%}')
    return Xs, ys


def train_boundary_model(params, ver, width=50, period=200, X=None, y=None, split=.2):
    if X is None or y is None:
        X_file, y_file = boundary_dataset_names(ver, width, period)
        X = np.load(X_file)
        y = np.load(y_file)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split)

    model = RandomForestClassifier(n_jobs=-1, **params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(accuracy_score(y_test, y_pred))

    model_name = boundary_model_name(ver, width, period)
    with open(model_name, 'wb') as f:
        pickle.dump(model, f)
    return model


def load_boundary_model(ver, width=50, period=200):
    model_name = boundary_model_name(ver, width, period)
    with open(model_name, 'rb') as f:
        model = pickle.load(f)
    return model


class BoundaryPredictor:
    def __init__(self, dt: DataTrace, model: RandomForestClassifier,
                 proba_thresh=.6, n_ref=10, score_thresh=.9, gap_thresh=.2,
                 width=80, target_range=(9000, 18000), n_boundaries=571):
        self.dt = dt
        self.model = model
        self.proba_thresh = proba_thresh
        self.n_ref = n_ref
        self.score_thresh = score_thresh
        self.gap_thresh = gap_thresh
        self.width = width
        self.target_range = target_range
        self.n_boundaries = n_boundaries
        self.tss, self.X, _ = self.dt.slide_boundary_data(width=width)

    def predict(self):
        self.pred_proba = self.model.predict_proba(self.X)[:, 1]
        self.pred = self.pred_proba > self.proba_thresh
        slices = ndimage.find_objects(ndimage.label(self.pred)[0])
        self.raw_centers = np.array([np.mean(self.tss[s]) for s in slices])

        diffs = self.raw_centers[1:] - self.raw_centers[:-1]
        filtered_diffs = diffs[np.logical_and(diffs > self.target_range[0], diffs < self.target_range[1])]
        filtered_diffs = filter_outliers(filtered_diffs, 3, iters=1)
        if len(filtered_diffs) == 0:
            raise ValueError('Failed to predict a tick rate')
        else:
            self.tick_rate = np.mean(filtered_diffs)
            print(f'Predicted tick rate: {self.tick_rate}; oracle: {self.dt.tick_rate}')

        # use scores to filter obvious false positives
        l_scores, r_scores = [], []
        for i, ts in enumerate(self.raw_centers):
            right = (self.raw_centers[i+1:i+1+self.n_ref] - ts) / self.tick_rate
            r_score = 1 - np.mean(abs(right - np.round(right))) if len(right) else 0
            r_scores.append(r_score)

            left = (self.raw_centers[i-self.n_ref:i] - ts) / self.tick_rate
            l_score = 1 - np.mean(abs(left - np.round(left))) if len(left) else 0
            l_scores.append(l_score)

        self.l_scores = np.array(l_scores)
        self.r_scores = np.array(r_scores)
        score_mask = np.logical_or(self.l_scores > self.score_thresh, self.r_scores > self.score_thresh)
        self.scored_centers = self.raw_centers[score_mask]
        if len(self.scored_centers) < 3:
            raise ValueError(f'Boundaries are highly unreliable. {len(self.scored_centers)}')

        # filter false positives that are close to a true boundary
        s_scores = np.max(np.concatenate([[self.l_scores[score_mask]], [self.r_scores[score_mask]]]), axis=0)
        gap_ratio = (self.scored_centers[1:] - self.scored_centers[:-1]) / self.tick_rate
        gap_mask = gap_ratio < self.gap_thresh
        gap_mask = np.append(gap_mask, False)
        for s in ndimage.find_objects(ndimage.label(gap_mask)[0]):
            slc = slice(s[0].start, s[0].stop + 1)
            idx = max(range(slc.start, slc.stop), key=s_scores.__getitem__)
            gap_mask[slc] = True
            gap_mask[idx] = False
        self.gap_centers = self.scored_centers[np.logical_not(gap_mask)]

        # finally, interpolate boundaries, assuming the first two boundaries are known
        boundaries = [self.dt.iter_boundaries[0], self.dt.iter_boundaries[1]]
        for c in self.gap_centers:
            last_c = boundaries[-1]
            ratio = int(round((c - last_c) / self.tick_rate))
            # check anomaly
            anoms = [anom for anom in self.dt.lats_anomalies if anom[1] >= last_c and anom[0] <= c]
            if ratio:
                if anoms:
                    anom = anoms[0]
                    if boundaries[-1] < anom[0]:
                        while True:
                            next_ts = boundaries[-1] + self.tick_rate
                            if next_ts < anom[0]:
                                boundaries.append(next_ts)
                            else:
                                break

                    rev = [c, ]
                    if c > (anom[1] - self.tick_rate * .2) and (c - boundaries[-1]) / self.tick_rate > 0.8:
                        while True:
                            next_ts = rev[-1] - self.tick_rate
                            if next_ts > (anom[1] - self.tick_rate * .2) and (next_ts - boundaries[-1]) / self.tick_rate > 0.8:
                                rev.append(next_ts)
                            else:
                                break
                    elif (c - boundaries[-1]) / self.tick_rate <= 0.8:
                        boundaries.pop(-1)
                    boundaries.extend(rev[::-1])
                else:
                    step = (c - last_c) / ratio
                    step_ratio = step / self.tick_rate
                    step_score = abs(step_ratio - round(step_ratio))
                    if step_score > 0.1:
                        MONT_LOGGER.info(f'Large step ratio! {step_ratio}; {last_c} -> {c}')

                    for i in range(1, ratio + 1):
                        boundaries.append(last_c + step * i)

        while len(boundaries) < self.n_boundaries:
            boundaries.append(boundaries[-1] + self.tick_rate)
        self.boundaries = np.array(boundaries)

        if len(self.boundaries) > self.n_boundaries:
            raise ValueError('Too many iteration boundaries!')

        return self.boundaries


# predict signal
best_signal_params = dict(n_estimators=400, min_samples_split=5, min_samples_leaf=1,
                          max_features='log2', max_depth=30, bootstrap=True)

def signal_dataset_names(ver, n_samples, period):
    suffix = f'-signal-{ver}-{n_samples}-{period}.npy'
    return DATASET_ROOT / ('X' + suffix), DATASET_ROOT / ('y' + suffix)


def signal_model_name(ver, n_samples, period):
    return MODEL_ROOT / f'signal-{ver}-{n_samples}-{period}.pickle'


def gen_signal_training(ver, n_samples=70, period=200, iters=20,
                        raw_results=None, oracles=None, **kwargs):
    def get_len(d):
        if isinstance(d, list):
            return len(d)
        else:
            return 0

    Xs, ys = [], []
    cnt, i = 0, 0
    max_iters = max(iters, min(get_len(raw_results), get_len(oracles)))
    while cnt < iters and i < max_iters:
        print(f'\r{cnt}', end='')
        if raw_results is None or oracles is None:
            data, oracle = run_mont(ver, True)
        else:
            data = raw_results[i]
            oracle = oracles[i]
            i += 1

        try:
            dt = DataTrace(data, oracle, std_filter_fact(), margin=100, period=period, **kwargs)
        except ValueError as e:
            print(e)
            continue

        splitted = dt.split(dt.normalized, n_samples=n_samples)
        feat = np.array([s[:, 1] for s in splitted])
        assert(len(feat) == len(dt.ground_truth))
        Xs.append(feat)
        ys.append(dt.ground_truth)
        cnt += 1

    Xs = np.concatenate(Xs)
    ys = np.concatenate(ys)
    Xs_file, ys_file = signal_dataset_names(ver, n_samples, period)
    np.save(Xs_file, Xs)
    np.save(ys_file, ys)
    print(Xs.shape, ys.shape)
    for k, c in Counter(ys).items():
        print(f'{k}: {c / len(ys): .2%}')
    return Xs, ys


def train_signal_model(params, ver, n_samples=70, period=200, X=None, y=None, split=.2):
    if X is None or y is None:
        X_file, y_file = signal_dataset_names(ver, n_samples, period)
        X = np.load(X_file)
        y = np.load(y_file)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split)
    model = RandomForestClassifier(n_jobs=-1, **params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(accuracy_score(y_test, y_pred))

    with open(signal_model_name(ver, n_samples, period), 'wb') as f:
        pickle.dump(model, f)
    return model


def load_signal_model(ver, n_samples=70, period=200):
    with open(signal_model_name(ver, n_samples, period), 'rb') as f:
        model = pickle.load(f)
    return model


class SignalPredictor:
    def __init__(self, dt: DataTrace, model: RandomForestClassifier, boundaries: List[float],
                 n_samples=70) -> None:
        self.dt = dt
        self.model = model
        self.n_samples = n_samples
        self.regions = list(zip(boundaries[:-1], boundaries[1:]))

    def predict(self):
        nbits = len(self.dt.ground_truth)

        self.splitted = self.dt.split(self.dt.normalized, regions=self.regions,
                                      n_samples=self.n_samples)
        X = np.array([s.vals for s in self.splitted])
        self.pred_probs = self.model.predict_proba(X)
        assert(len(self.pred_probs) == nbits)

        self.pred = np.array([max(range(len(p)), key=p.__getitem__) for p in self.pred_probs])
        self.acc = accuracy_score(self.dt.ground_truth, self.pred)
        self.confs = {i: (b, ps[b]) for i, (b, ps) in enumerate(zip(self.pred, self.pred_probs))}
        self.ranks = {b: r for r, b in enumerate(sorted(self.confs, key=lambda x: self.confs[x][1]))}
        self.wrongs = np.arange(len(self.pred), dtype='int')[self.pred != self.dt.ground_truth]

        def has_anomaly(anom, region):
            return anom[1] >= region[0] and anom[0] <= region[1]

        self.anomaly_mask = np.zeros(nbits, dtype='bool')
        for anom in self.dt.lats_anomalies:
            idx = list(filter(lambda i: has_anomaly(anom, self.regions[i]), range(nbits)))
            if idx:
                self.anomaly_mask[idx[0]] = True

        return self.pred

    @property
    def wrong_stats(self):
        return [(*self.confs[w], self.ranks[w]) for w in self.wrongs]

    @property
    def off_by_one(self):
        ratio = abs(self.regions[-1][1] - self.dt.iter_boundaries[-1]) / self.dt.tick_rate
        return ratio > .2
