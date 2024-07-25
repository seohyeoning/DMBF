"""
🔥 **모든 데이터는 S1, S2, … 형식으로 수정**

최소한의 전처리만 수행
EEG (band-pass filtering, ICA, resampling 이미 되어있음)
1. PPMs와 통일
2. KSS 기반 레이블링 (5미만은 alert, 5이상은 drowsy)

7을 기준으로 하면, S1,4, 5(good), 8,13,14,15, 16,18,19,20 (11명 가능)

PPMs (low pass filtering, detrending은 이미 되어있음)
1. resampling (256->250Hz)
"""

import os
from pathlib import Path
from scipy.io import loadmat
import numpy as np
from sklearn.model_selection import KFold, train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import random
from scipy import stats
from .Variables import device



def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

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
    xe_batch, xg_batch, xc_batch, xr_batch = torch.Tensor().to(device), torch.Tensor().to(device), torch.Tensor().to(device), torch.Tensor().to(device)

    for (_x, _y) in batch:
        # 1. 데이터(x)에서 EEG와 나머지 분리하기
        # 2. 데이터 shape 3차원으로 맞춰주기
        # 3. numpy -> tensor
        xe = torch.unsqueeze(_x[:-3, :], 0)    # EEG (N, 64, Seq)
        xc = torch.unsqueeze(_x[-3, :], 0)     # ECG (N, 1, Seq)
        xg = torch.unsqueeze(_x[-2, :], 0)     # GSR (N, 1, Seq)
        xr = torch.unsqueeze(_x[-1, :], 0)     # Resp (N, 1, Seq)

        xe_batch = torch.cat((xe_batch, xe), 0)
        xc_batch = torch.cat((xc_batch, xc), 0)
        xg_batch = torch.cat((xg_batch, xg), 0)
        xr_batch = torch.cat((xr_batch, xr), 0)
        
        _y = torch.unsqueeze(_y, 0)
        y_batch = torch.cat((y_batch, _y), 0) # (2, ) -> (1, 2)    

        
    x_batch = [xe_batch, xc_batch, xg_batch, xr_batch]
    
    return {'data': x_batch, 'label': y_batch}
