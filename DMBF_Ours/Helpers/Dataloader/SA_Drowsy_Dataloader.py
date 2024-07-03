"""
🔥 **모든 데이터는 S1, S2, … 형식으로 수정**

형태 (600*38*N)

<데이터 구조>
- x = EEG(0-31), PPG(32), ECG(33), hrPPG(34), hrECG(35), RT(36), KSS(37)
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
from scipy.stats import mode
from collections import Counter

from Helpers.Variables import device


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



"""
Make a EEG dataset
X: EEG data
Y: class score
"""

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
    x_batch, y_batch = torch.Tensor().to(device), torch.Tensor().to(device)
  
    for (_x, _y) in batch:
        # 1. 데이터(x)에서 EEG와 나머지 분리하기
        # 2. 데이터 shape 3차원으로 맞춰주기
        # 3. numpy -> tensor
        x = torch.unsqueeze(_x, 0)       # EEG (N, 32, Seq)
        _y = torch.unsqueeze(_y, 0)

        x_batch = torch.cat((x_batch, x), 0)
        
        _y = torch.unsqueeze(_y, 0)
        y_batch = torch.cat((y_batch, _y), 0) # (2, ) -> (1, 2)    
    
    return {'data': x_batch, 'label': y_batch}


def data_generate(args):

    # 데이터 불러오기
    mat_data = loadmat(f'/opt/workspace/Seohyeon/Journal/DATA/matfile/SA_Drowsy/{args.data_type}.mat') # dataset, unbalanced_dataset 
    
    x = mat_data["EEGsample"]
    subIdx = mat_data["subindex"]
    label = mat_data["substate"]

    subIdx.astype(int)

    for subj in range(1, args.num_sbj+1): # balance dataset 11
        # processed data 저장 경로 설정
        data_save_dir = f'/opt/workspace/Seohyeon/Journal/DATA/preprocessed/SA_Drowsy/{args.data_type}/S{subj}' 
        Path(data_save_dir).mkdir(parents=True, exist_ok=True)
        
        indexes = np.where(subIdx == subj)[0] # 해당 subj의 index 찾기
        min_index, max_index = min(indexes), max(indexes)

        sbj_features = x[min_index:max_index+1]
        sbj_label = label[min_index:max_index+1]

        drowsy = 0
        alert = 0

        for i in range (sbj_label.shape[0]):
            if sbj_label[i] == 0:
                alert += 1
            else:
                drowsy += 1

        print(f'S{subj} data distribution --> alert : {alert}, drowsy : {drowsy}')

        
        # train, validation, test set 나누기
        kf = KFold(n_splits=4, shuffle=True, random_state=args.SEED) # 4 fold
        kf_gen = kf.split(sbj_features)

        # Generate 4-fold cross validation set
        cnt = 0
        for tr_idx, ts_idx in kf_gen:
            cnt += 1
            final_datasave_dir = os.path.join(data_save_dir, f'{cnt}fold')
            Path(final_datasave_dir).mkdir(parents=True, exist_ok=True)

            # nth-fold 내부에서 train, validation, test set 나누기
            x_0, x_test = sbj_features[tr_idx], sbj_features[ts_idx]
            y_0, y_test = sbj_label[tr_idx], sbj_label[ts_idx]
            x_train, x_valid, y_train, y_valid = train_test_split(
                x_0, 
                y_0, 
                test_size = 0.2,
                random_state = 0
            ) 

            # 레이블 원핫 인코딩 (0->01, 1->10)
            onehot_y_train = to_categorical(y_train, num_classes=args.n_classes)
            onehot_y_valid = to_categorical(y_valid, num_classes=args.n_classes)
            onehot_y_test = to_categorical(y_test, num_classes=args.n_classes)       

            # 저장     
            fname_train = f"train.npz"
            fname_valid = f"valid.npz"
            fname_test = f"test.npz"

            np.savez(f"{final_datasave_dir}/{fname_train}", data=x_train, label=onehot_y_train)
            np.savez(f"{final_datasave_dir}/{fname_valid}", data=x_valid, label=onehot_y_valid)
            np.savez(f"{final_datasave_dir}/{fname_test}", data=x_test, label=onehot_y_test)
        
        print(f'Done processing S{subj}.')
        

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    y = y.astype(int)
    return np.eye(num_classes, dtype='uint8')[y]


if __name__ == "__main__":   
    import os
    import argparse
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 

    parser = argparse.ArgumentParser(description='Generating Distraction dataset')
    parser.add_argument("--SEED", default=42)
    parser.add_argument('--freq_time', default=384, help='sampling rate * time') 
    parser.add_argument('--data_type', default='unbalanced_dataset', choices=['dataset', 'unbalanced_dataset'])
    parser.add_argument('--num_sbj', default=11)

    ########## 클래스 및 채널 수 지정
    parser.add_argument('--n_channels', default=30)
    parser.add_argument('--n_classes', default=2)
    args = parser.parse_args()

    seed_everything(args.SEED)

    DATASET_DIR = '/opt/workspace/Seohyeon/Journal/DATA/preprocessed/SA_Drowsy/S1/1fold/train.npz'
    # Data Generation at first time 
    # if not os.path.exists(os.path.join(DATASET_DIR)):
    data_generate(args)