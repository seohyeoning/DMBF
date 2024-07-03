"""
🔥 **모든 데이터는 S1, S2, … 형식으로 수정**

<matfile 내부 전처리>
1. [4,35] band-pass filtering
2. PPG, ECG de-trending based on Heart rate
3. Syncronization
4. distraction 마커 사이의 최소 거리 기반으로 segment 길이 계산
    인당 200Hz * 6.8sec = 1360 정도의 datapoint 가짐

=> non-distraction과 disctraction의 길이가 다르기 때문에 총 datapoint가 다름
=> distraction (2000, 34, N1) 
=> non-distraction (5522, 34, N2) 
=> 해결방법: distraction은 2초길이로 세그먼트 (400, 34, N1*5)\
            non-distraction은 5200으로 다운샘플링 (26초) 후, 2초 길이로 세그먼트 (400, 34, N2*13) 을 수행해서 두 데이터들을 둘 중 작은 datapoint 수로 맞춰줌

<데이터 구조>

<데이터 구조>
- x = EEG, misc = PPG_data, hrPPG, hrECG,
    - 'Fp1', 'Fp2', 'F7', 'F3'	'Fz',	'F4',	'F8',	'FC5', 'FC1', 
    'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 
    'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10’

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
    xe_batch, xp_batch, xc_batch = torch.Tensor().to(device), torch.Tensor().to(device), torch.Tensor().to(device)

    for (_x, _y) in batch:
        # 1. 데이터(x)에서 EEG와 나머지 분리하기
        # 2. 데이터 shape 3차원으로 맞춰주기
        # 3. numpy -> tensor
        xe = torch.unsqueeze(_x[:-2, :], 0)       # EEG (N, 32, Seq)
        xp = torch.unsqueeze(_x[-2, :], 0)     # PPG (N, 1, Seq)
        xc = torch.unsqueeze(_x[-1, :], 0)     # ECG (N, 1, Seq)

        xe_batch = torch.cat((xe_batch, xe), 0)
        xp_batch = torch.cat((xp_batch, xp), 0)
        xc_batch = torch.cat((xc_batch, xc), 0)
        
        _y = torch.unsqueeze(_y, 0)
        y_batch = torch.cat((y_batch, _y), 0) # (2, ) -> (1, 2)    

        
    x_batch = [xe_batch, xp_batch, xc_batch]
    
    return {'data': x_batch, 'label': y_batch}
