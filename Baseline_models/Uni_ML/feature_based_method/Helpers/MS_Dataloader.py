"""
baseline correction + val train으로부터

Generating 시 바꿔주기. 데이터로더용으로는 바꾸지 않아도 됌.

2class 기준  (misc스코어 4 삭제) 
0~3 : low = 0
5~9 : high = 1
"""

import os
from tqdm import tqdm
from pathlib import Path
from scipy import stats
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from Helpers.Variables import device, RAW_DIR, COL_NAMES
import mne
import random


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    
seed_everything()

class BIODataset(Dataset):
    def __init__(self, phase, device, data_load_dir):
        super().__init__()
        self.device = device
        
        self.data = np.load(f'{data_load_dir}/{phase}.npz') # sbj/1fold/phase.npz
        self.X = self.data['data']
        self.y = self.data['label']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.X[idx]).to(self.device)
        y = torch.FloatTensor(self.y[idx]).to(self.device)
        return x, y
    

class BIODataLoader(DataLoader): 
    def __init__(self, *args, **kwargs):
        super(BIODataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn
    
def _collate_fn(batch): # 배치사이즈 지정 방법 확인
    x_batch, y_batch = [], torch.Tensor().to(device)
    xe_batch, xc_batch, xr_batch, xp_batch, xg_batch = torch.Tensor().to(device), torch.Tensor().to(device), \
                                                       torch.Tensor().to(device), torch.Tensor().to(device), \
                                                       torch.Tensor().to(device)
    for (_x, _y) in batch:
        # 1. 데이터(x)에서 EEG와 나머지 분리하기
        # 2. 데이터 shape 3차원으로 맞춰주기
        # 3. numpy -> tensor
        xe = _x[:, :-4]                        # EEG
        xc = torch.unsqueeze(_x[:, -4], 1)     # ECG
        xr = torch.unsqueeze(_x[:, -3], 1)     # Resp
        xp = torch.unsqueeze(_x[:, -2], 1)     # PPG
        xg = torch.unsqueeze(_x[:, -1], 1)     # GSR    

        # Dimension swap: (N, Seq, Ch) -> (N, Ch, Seq)
        xe = torch.permute((xe), (1, 0)).to(dtype=torch.float32)
        xc = torch.permute((xc), (1, 0)).to(dtype=torch.float32)
        xr = torch.permute((xr), (1, 0)).to(dtype=torch.float32)
        xp = torch.permute((xp), (1, 0)).to(dtype=torch.float32)
        xg = torch.permute((xg), (1, 0)).to(dtype=torch.float32)

        xe = torch.unsqueeze(xe, 0) # (28, sr*sec) -> (1, 28, sr*sec)
        xc = torch.unsqueeze(xc, 0) # (1, sr*sec) -> (1, 1, sr*sec)
        xr = torch.unsqueeze(xr, 0)
        xp = torch.unsqueeze(xp, 0)
        xg = torch.unsqueeze(xg, 0)

        xe_batch = torch.cat((xe_batch, xe), 0)
        xc_batch = torch.cat((xc_batch, xc), 0)
        xr_batch = torch.cat((xr_batch, xr), 0)
        xp_batch = torch.cat((xp_batch, xp), 0)
        xg_batch = torch.cat((xg_batch, xg), 0)

        _y = torch.unsqueeze(_y, 0)

        y_batch = torch.cat((y_batch, _y), 0) # (3, ) -> (1, 3)    

        
    x_batch = [xe_batch, xc_batch, xr_batch, xp_batch, xg_batch]
    
    return {'data': x_batch, 'label': y_batch}

