from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

CMAPSS_COLS = (
    ['engine_id', 'cycle'] +
    [f'setting{i}' for i in range(1, 4)] +
    [f's{i}' for i in range(1, 22)]
)

# Sensors with near-zero variance on FD001 — drop before modeling
CMAPSS_DROP = ['s1', 's5', 's6', 's10', 's16', 's18', 's19']


def load_cmapss(data_dir: str = 'data/cmapss', subset: str = 'FD001'):
    """Load CMAPSS train/test/RUL files for a given subset (FD001–FD004)."""
    base = Path(data_dir)
    train = pd.read_csv(base / f'train_{subset}.txt', sep=r'\s+', header=None, names=CMAPSS_COLS)
    test  = pd.read_csv(base / f'test_{subset}.txt',  sep=r'\s+', header=None, names=CMAPSS_COLS)
    rul   = pd.read_csv(base / f'RUL_{subset}.txt',   sep=r'\s+', header=None, names=['true_rul'])
    train.drop(columns=CMAPSS_DROP, inplace=True)
    test.drop(columns=CMAPSS_DROP,  inplace=True)
    return train, test, rul


def load_cwru(data_dir: str = 'data/cwru'):
    """
    Load CWRU .mat files. Returns dict: {label: 1D numpy array of vibration}.

    Expected files (0 HP load):
        Normal_0.mat, IR007_0.mat, B007_0.mat, OR007_0.mat
    """
    base = Path(data_dir)
    files = {
        'normal':     'Normal_0.mat',
        'inner_race': 'IR007_0.mat',
        'ball':       'B007_0.mat',
        'outer_race': 'OR007_0.mat',
    }
    signals = {}
    for label, fname in files.items():
        fpath = base / fname
        if not fpath.exists():
            print(f"[load_cwru] WARNING: {fname} not found — skipping {label}")
            continue
        mat = scipy.io.loadmat(str(fpath))
        # Find the drive-end time series key (ends with _DE_time)
        de_key = [k for k in mat.keys() if k.endswith('_DE_time')]
        if de_key:
            signals[label] = mat[de_key[0]].ravel().astype(np.float32)
        else:
            # Fallback: try FE (fan end) channel
            fe_key = [k for k in mat.keys() if k.endswith('_FE_time')]
            if fe_key:
                signals[label] = mat[fe_key[0]].ravel().astype(np.float32)
            else:
                print(f"[load_cwru] WARNING: no DE/FE channel found in {fname}")
    return signals
