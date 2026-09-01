import os
import sys
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(proj_dir)
from util import config, file_dir

from datetime import datetime
import numpy as np
import arrow
import metpy.calc as mpcalc
from metpy.units import units
from torch.utils import data
import torch


# Fixed feature layout written by data/prepare_sensor_dataset.py into
# SensorAir.npy (order matters - see _process_feature's 'sensor' branch
# below). Unlike KnowAir's metero_use ablation list, this dataset has no
# selectable subset - all 5 raw variables are always used.
SENSOR_METERO_VAR = ['temperature', 'relative_humidity', 'rain', 'wind_speed10', 'wind_direction10']


class HazeData(data.Dataset):

    def __init__(self, graph,
                       hist_len=1,
                       pred_len=24,
                       dataset_num=1,
                       flag='Train',
                       ):

        if flag == 'Train':
            start_time_str = 'train_start'
            end_time_str = 'train_end'
        elif flag == 'Val':
            start_time_str = 'val_start'
            end_time_str = 'val_end'
        elif flag == 'Test':
            start_time_str = 'test_start'
            end_time_str = 'test_end'
        else:
            raise Exception('Wrong Flag!')

        ds_cfg = config['dataset'][dataset_num]
        # 'family' distinguishes the original KnowAir-derived datasets
        # (1-3: same 184-city, 3-hourly array, just different date-range
        # subsets of it) from a differently-shaped dataset like 4 (a
        # hourly low-cost sensor network - see data/prepare_sensor_dataset.py).
        # Every key below falls back to the KnowAir-family default when
        # absent, so config.yaml's existing 1/2/3 entries don't change.
        self.family = ds_cfg.get('family', 'knowair')
        self.freq_hours = ds_cfg.get('freq_hours', 3)

        self.start_time = self._get_time(ds_cfg[start_time_str])
        self.end_time = self._get_time(ds_cfg[end_time_str])
        self.data_start = self._get_time(ds_cfg.get('data_start', config['dataset']['data_start']))
        self.data_end = self._get_time(ds_cfg.get('data_end', config['dataset']['data_end']))

        if 'data_fp' in ds_cfg:
            # NB: proj_dir (module-level, above) is this file's *parent*
            # directory, not the repo root dataset.py itself lives in
            # (pre-existing oddity - harmless there since nothing else
            # joins paths against it) - resolve data_fp against the repo
            # root directly instead of reusing that constant.
            repo_dir = os.path.dirname(os.path.abspath(__file__))
            self.knowair_fp = os.path.join(repo_dir, ds_cfg['data_fp'])
        else:
            self.knowair_fp = file_dir['knowair_fp']

        self.graph = graph

        self._load_npy()
        self._gen_time_arr()
        self._process_time()
        self._process_feature()
        self.feature = np.float32(self.feature)
        self.pm25 = np.float32(self.pm25)
        self._calc_mean_std()
        seq_len = hist_len + pred_len
        self._add_time_dim(seq_len)
        self._norm()

    def _norm(self):
        self.feature = (self.feature - self.feature_mean) / self.feature_std
        self.pm25 = (self.pm25 - self.pm25_mean) / self.pm25_std

    def _add_time_dim(self, seq_len):

        def _add_t(arr, seq_len):
            t_len = arr.shape[0]
            assert t_len > seq_len
            arr_ts = []
            for i in range(seq_len, t_len):
                arr_t = arr[i-seq_len:i]
                arr_ts.append(arr_t)
            arr_ts = np.stack(arr_ts, axis=0)
            return arr_ts

        self.pm25 = _add_t(self.pm25, seq_len)
        self.feature = _add_t(self.feature, seq_len)
        self.time_arr = _add_t(self.time_arr, seq_len)

    def _calc_mean_std(self):
        self.feature_mean = self.feature.mean(axis=(0,1))
        self.feature_std = self.feature.std(axis=(0,1))
        self.wind_mean = self.feature_mean[-2:]
        self.wind_std = self.feature_std[-2:]
        self.pm25_mean = self.pm25.mean()
        self.pm25_std = self.pm25.std()

    def _process_feature(self):
        if self.family == 'sensor':
            # SensorAir.npy already stores exactly SENSOR_METERO_VAR, in
            # that order (see data/prepare_sensor_dataset.py) - no
            # metero_use-style selection to do. Unlike the earlier
            # placeholder-direction dataset this repo tried, this source
            # has a genuine measured wind_speed10 (m/s) and
            # wind_direction10 (degrees) - just needs the same km/h unit
            # convention as the KnowAir branch below for wind_speed, and
            # direction is used as-is (no derivation needed, it's not a
            # pair of u/v components to resolve).
            speed = 3.6 * self.feature[:, :, SENSOR_METERO_VAR.index('wind_speed10')]
            direc = self.feature[:, :, SENSOR_METERO_VAR.index('wind_direction10')]
        else:
            metero_var = config['data']['metero_var']
            metero_use = config['experiments']['metero_use']
            metero_idx = [metero_var.index(var) for var in metero_use]
            self.feature = self.feature[:,:,metero_idx]

            u = self.feature[:, :, -2] * units.meter / units.second
            v = self.feature[:, :, -1] * units.meter / units.second
            speed = 3.6 * mpcalc.wind_speed(u, v)._magnitude
            direc = mpcalc.wind_direction(u, v)._magnitude

        h_arr = []
        w_arr = []
        for i in self.time_arrow:
            h_arr.append(i.hour)
            w_arr.append(i.isoweekday())
        h_arr = np.stack(h_arr, axis=-1)
        w_arr = np.stack(w_arr, axis=-1)
        h_arr = np.repeat(h_arr[:, None], self.graph.node_num, axis=1)
        w_arr = np.repeat(w_arr[:, None], self.graph.node_num, axis=1)

        self.feature = np.concatenate([self.feature, h_arr[:, :, None], w_arr[:, :, None],
                                       speed[:, :, None], direc[:, :, None]
                                       ], axis=-1)

    def _process_time(self):
        start_idx = self._get_idx(self.start_time)
        end_idx = self._get_idx(self.end_time)
        self.pm25 = self.pm25[start_idx: end_idx+1, :]
        self.feature = self.feature[start_idx: end_idx+1, :]
        self.time_arr = self.time_arr[start_idx: end_idx+1]
        self.time_arrow = self.time_arrow[start_idx: end_idx + 1]

    def _gen_time_arr(self):
        self.time_arrow = []
        self.time_arr = []
        for time_arrow in arrow.Arrow.interval('hour', self.data_start, self.data_end.shift(hours=+self.freq_hours), self.freq_hours):
            self.time_arrow.append(time_arrow[0])
            # use timestamp() method to get numeric value (avoid method object)
            self.time_arr.append(time_arrow[0].timestamp())
        self.time_arr = np.stack(self.time_arr, axis=-1)

    def _load_npy(self):
        self.knowair = np.load(self.knowair_fp)
        self.feature = self.knowair[:,:,:-1]
        self.pm25 = self.knowair[:,:,-1:]

    def _get_idx(self, t):
        t0 = self.data_start
        return int((t.timestamp() - t0.timestamp()) / (60 * 60 * self.freq_hours))

    def _get_time(self, time_yaml):
        arrow_time = arrow.get(datetime(*time_yaml[0]), time_yaml[1])
        return arrow_time

    def __len__(self):
        return len(self.pm25)

    def __getitem__(self, index):
        # return torch tensors so DataLoader's default_collate handles batches
        pm25 = torch.from_numpy(self.pm25[index])
        feature = torch.from_numpy(self.feature[index])
        time_arr = torch.from_numpy(self.time_arr[index])
        return pm25, feature, time_arr

if __name__ == '__main__':
    from graph import Graph
    graph = Graph()
    train_data = HazeData(graph, flag='Train')
    val_data = HazeData(graph, flag='Val')
    test_data = HazeData(graph, flag='Test')

    print(len(train_data))
    print(len(val_data))
    print(len(test_data))
