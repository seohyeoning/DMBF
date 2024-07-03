"""
🔥 


<데이터 구조>
- x = (N, 13, Seq): Poz, Fz, Cz, C3, C4, F3, F4, P3, P4 (9), PPG, EDA, TEMP, ECG (4)


- 2 class : non distraction(0), distraction(1)
    - 4 fold cross validation (train:valid:test = 12:3:5)
        - **SPLIT** tr+val : ts = 3 : 1, tr : val = 8 : 2로 split

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
    xe_batch, xp_batch, xd_batch, xt_batch, xc_batch = torch.Tensor().to(device), torch.Tensor().to(device), torch.Tensor().to(device), torch.Tensor().to(device), torch.Tensor().to(device)

    for (_x, _y) in batch:
        # 1. 데이터(x)에서 EEG와 나머지 분리하기
        # 2. 데이터 shape 3차원으로 맞춰주기
        # 3. numpy -> tensor

        # PPG, EDA, TEMP, ECG 
        xe = torch.unsqueeze(_x[:9, :], 0)       # EEG (N, 9, Seq)
        xp = torch.unsqueeze(_x[9, :], 0)     # PPG (N, 1, Seq)
        xd = torch.unsqueeze(_x[10, :], 0)     # EDA (N, 1, Seq)
        xt = torch.unsqueeze(_x[11, :], 0)     # TEMP (N, 1, Seq)
        xc = torch.unsqueeze(_x[12, :], 0)     # ECG (N, 1, Seq)

        xe_batch = torch.cat((xe_batch, xe), 0)
        xp_batch = torch.cat((xp_batch, xp), 0)
        xd_batch = torch.cat((xd_batch, xd), 0)
        xt_batch = torch.cat((xt_batch, xt), 0)
        xc_batch = torch.cat((xc_batch, xc), 0)

        
        _y = torch.unsqueeze(_y, 0)
        y_batch = torch.cat((y_batch, _y), 0) # (2, ) -> (1, 2)    

        
    x_batch = [xe_batch, xp_batch, xd_batch, xt_batch, xc_batch]
    
    return {'data': x_batch, 'label': y_batch}
