# PM2.5-GNN

PM2.5-GNN: A Domain Knowledge Enhanced Graph Neural Network For PM2.5 Forecasting

## Dataset

- Download dataset **KnowAir** from [Google Drive](https://drive.google.com/open?id=1R6hS5VAgjJQ_wu8i5qoLjIxY0BG7RD1L) or [Baiduyun](https://pan.baidu.com/s/18D6Etl5Lm1E4vOLVrX0ZAw) with code `t82d`.

## KnowAir-V2

🚀 Dataset Update: Announcing KnowAir-V2! 🚀

We are excited to announce a major upgrade to the original KnowAir (PM2.5-GNN) dataset with the official release of KnowAir-V2! This is a brand-new, higher-quality benchmark dataset for air quality forecasting.

Key improvements in KnowAir-V2 include:
- Longer Temporal Span: Data covers from 2016 to 2023.
- Richer Variables: Includes not only PM2.5 but also O3 and more related meteorological variables.
- Higher Data Quality: The data has undergone rigorous preprocessing and imputation, reaching an operational-level standard.

For all new research and projects, we strongly recommend using KnowAir-V2. This dataset is designed to provide a powerful benchmarking platform for more advanced spatio-temporal prediction models that integrate physical-chemical knowledge, such as PCDCNet.

How to Access and Cite
Dataset Download (KnowAir-V2):

- Wang, S., Cheng, Y., Meng, Q., Saukh, O., Zhang, J., Fan, J., Zhang, Y., Yuan, X., & Thiele, L. (2025). KnowAir-V2: A Benchmark Dataset for Air Quality Forecasting with PCDCNet [Data set]. Zenodo. https://doi.org/10.5281/zenodo.15614907

- Related Paper (PCDCNet):
Please refer to the paper: "PCDCNet: A Surrogate Model for Air Quality Forecasting with Physical-Chemical Dynamics and Constraints" (arXiv:2505.19842). https://www.arxiv.org/abs/2505.19842

## Requirements

```
Python 3.7.3
PyTorch 1.7.0
PyG: https://github.com/rusty1s/pytorch_geometric#pytorch-170
```

```bash
pip install -r requirements.txt
```

## Experiment Setup

open `config.yaml`, do the following setups.

- set data path after your server name. Like mine.

![](https://tva1.sinaimg.cn/large/0081Kckwly1gjy8kojsfmj30i202g746.jpg)

```python
filepath:
  GPU-Server:
    knowair_fp: /data/wangshuo/haze/pm25gnn/KnowAir.npy
    results_dir: /data/wangshuo/haze/pm25gnn/results

```

- Uncomment the model you want to run.

```python
#  model: MLP
#  model: LSTM
#  model: GRU
#  model: GC_LSTM
#  model: nodesFC_GRU
   model: PM25_GNN
#  model: PM25_GNN_nosub
```

- Choose the sub-dataset number in [1,2,3,4]. 1-3 are date-range subsets of **KnowAir** (184 China cities, 3-hourly). 4 is a geographically and structurally distinct second dataset - a 197-node hourly low-cost PM2.5 sensor network - added for diversity rather than for strict comparability with 1-3; see [Dataset 4: Hourly Sensor Network](#dataset-4-hourly-sensor-network) below.

```python
 dataset_num: 3
```

- Set weather variables you wish to use. Following is the default setting in the paper. You can uncomment specific variables. Variables in dataset **KnowAir** is defined in `metero_var`.

```python
  metero_use: ['2m_temperature',
               'boundary_layer_height',
               'k_index',
               'relative_humidity+950',
               'surface_pressure',
               'total_precipitation',
               'u_component_of_wind+950',
               'v_component_of_wind+950',]

```

## Run

```bash
python train.py
```

## Running on Kaggle

The repo works out of the box on a fresh clone (Kaggle, Colab, or otherwise) - `config.yaml`'s `filepath:` section only lists two specific machines by hostname, and `util.py` automatically falls back to `KnowAir.npy` / `results/` right next to the repo for everything else, so you don't need to edit `config.yaml` just to run it somewhere new.

1. In a Kaggle notebook cell:

   ```bash
   !git clone https://github.com/Grace-VN/PM25GNN.git
   %cd PM25GNN
   !pip install -r requirements.txt -q
   ```

   Kaggle's GPU notebooks already ship PyTorch, so this mainly installs `torch_geometric`, `torchdiffeq`, `arrow`, `geopy`, `MetPy`, `bresenham` - the packages that aren't preinstalled. `train.py` picks up the GPU automatically (`torch.cuda.is_available()`).

2. Get `KnowAir.npy` onto the notebook - it's ~300MB and intentionally not committed to the repo (see [Dataset](#dataset) above for download links). Either:

   - **Attach it as a Kaggle Dataset** (Add Data -> upload `KnowAir.npy`, or reuse an existing one), then point at it with env vars instead of editing any tracked file:

     ```bash
     %env KNOWAIR_FP=/kaggle/input/<your-dataset-name>/KnowAir.npy
     %env RESULTS_DIR=/kaggle/working/results
     ```

   - **Or** download it straight into the repo folder as `KnowAir.npy` (e.g. `!gdown <google-drive-id>` for the Google Drive link above) - no env vars needed, since that's exactly where the fallback path in `util.py` already looks.

3. Pick a model / `dataset_num` / `pred_len` etc. in `config.yaml` (see [Experiment Setup](#experiment-setup) above), or leave the defaults.

4. Run:

   ```bash
   !python train.py
   ```

## Dataset 4: Hourly Sensor Network

A second, structurally distinct dataset - 197 nodes, hourly, 2023-05-14 00:00 through 2023-06-02 16:00 (~19.7 days) - built from `data/data.csv` (a ~2.27M-row, 6,031-site low-cost PM2.5 sensor feed spanning Sep 2022 - Jun 2023) via `data/prepare_sensor_dataset.py`. Chosen for diversity, not strict comparability with KnowAir 1-3: most of the source's 6,031 sites are short-lived (median site reports only ~4% of the ~6,000-hour collection span), so the script picks the one window/site-set combination where coverage is actually usable - the 197 sites with >=95% hourly coverage in the final ~20 days, when a large batch of new sensors joined the network. See the script's module docstring for the full reasoning and what's approximated (a shared ~20-hour network outage is linear-interpolated, affecting ~8% of values including PM2.5 itself; PM values are raw/uncalibrated low-cost-sensor readings).

Unlike the region/graph limitations of an earlier, since-removed dataset_num 4 attempt, this one has **real** measured `wind_speed10`/`wind_direction10` (not a derived or placeholder direction) and **real** per-site elevation (not zero-filled) - see `graph.py`'s `Graph._gen_nodes` (reads `data/site_sensor.txt`'s 5th column directly) and `dataset.py`'s `family == 'sensor'` branch. Nodes are also genuinely locally-spaced (dense regional clusters, e.g. the Bay Area/LA/Seattle), which is closer to what the wind-advection graph models (PM25_GNN, AirDDE, AirPhyNet, AirDualODE, AirFormer) are actually designed for than a coarse one-node-per-state/-country set would be.

Unlike `KnowAir.npy`, the ~268MB raw source (`data/data.csv`) is too large to commit and isn't shipped - but `data/SensorAir.npy` (4.3MB) and `data/site_sensor.txt` (the derived node list) *are* committed directly, so this needs no external download at all: just set `dataset_num: 4` in `config.yaml` and run `train.py`. On Kaggle/Colab this means skipping step 2 of [Running on Kaggle](#running-on-kaggle) entirely (no `KnowAir.npy` to attach, nothing to download) - clone, `pip install -r requirements.txt`, set `dataset_num: 4`, train.

Only re-run the conversion if you want a different window/site-coverage tradeoff than the one already committed (see `prepare_sensor_dataset.py`'s docstring for how that tradeoff was chosen) - it needs `data/data.csv` present, which you'd have to get from the original source yourself:

```bash
python data/prepare_sensor_dataset.py
```

## Reference

Paper: https://dl.acm.org/doi/10.1145/3397536.3422208

```
@inproceedings{10.1145/3397536.3422208,
author = {Wang, Shuo and Li, Yanran and Zhang, Jiang and Meng, Qingye and Meng, Lingwei and Gao, Fei},
title = {PM2.5-GNN: A Domain Knowledge Enhanced Graph Neural Network For PM2.5 Forecasting},
year = {2020},
isbn = {9781450380195},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3397536.3422208},
doi = {10.1145/3397536.3422208},
abstract = {When predicting PM2.5 concentrations, it is necessary to consider complex information sources since the concentrations are influenced by various factors within a long period. In this paper, we identify a set of critical domain knowledge for PM2.5 forecasting and develop a novel graph based model, PM2.5-GNN, being capable of capturing long-term dependencies. On a real-world dataset, we validate the effectiveness of the proposed model and examine its abilities of capturing both fine-grained and long-term influences in PM2.5 process. The proposed PM2.5-GNN has also been deployed online to provide free forecasting service.},
booktitle = {Proceedings of the 28th International Conference on Advances in Geographic Information Systems},
pages = {163–166},
numpages = {4},
keywords = {air quality prediction, graph neural network, spatio-temporal prediction},
location = {Seattle, WA, USA},
series = {SIGSPATIAL '20}
}
```
