import numpy as np
import torch
import argparse

import torch.nn as nn

import pandas as pd
from pathlib import Path
import time
import datetime
import random

from Helpers.Variables import device, METRICS, DATASET_DIR, WD, FILENAME_FOLDRES, FILENAME_MODEL, FILENAME_FOLDSUM
from Helpers.SA_Drowsy_Dataloader import BIODataset, BIODataLoader
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def Experiment(args):
    ##### SAVE DIR for .pt file

    res_dir = os.path.join (WD,'res', f'{args.backbone}_{args.scheduler}/{args.modality}_{args.data_type}_{args.optimizer}_{args.lr}')
    Path(res_dir).mkdir(parents=True, exist_ok=True)
    
    res_flen = str(len(os.listdir(res_dir))) # 신규 폴더 구분용
    save_file = f'/{res_flen}/'

    RESD = res_dir + save_file
    Path(RESD).mkdir(parents=True, exist_ok=True) # 저장 경로 생성


    ################## 피험자 별 경로 설정 및 실험 시작 ##################
    ts_fold = pd.DataFrame(columns=METRICS)
    num_fold = 4 # 4

    for subj in range(1,args.num_sbj+1): 
        if subj not in include_sbj:
                continue
        for nf in range(1, num_fold+1):
            ts_total = pd.DataFrame(columns=METRICS)

            print('='*30)
            print(' '*4, '피험자{} - fold{}을 test로 학습 시작'.format(subj, nf))
            print('='*30)

            # 데이터 정보 불러오기
            if args.data_type == 'SA_Drowsy_bal':
                Dataset_directory = f'{DATASET_DIR}/SA_Drowsy/dataset'
            elif args.data_type == 'SA_Drowsy_unbal':
                Dataset_directory = f'{DATASET_DIR}/SA_Drowsy/unbalanced_dataset'

            data_load_dir = f'{Dataset_directory}/S{subj}/{nf}fold'
            print(f'Loaded data from --> {DATASET_DIR}/S{subj}/{nf}fold')

            # 결과 저장 경로 설정 2
            res_name = f'S{subj}'
            nfoldname = f'fold{nf}'

            res_dir = os.path.join(RESD, res_name, nfoldname)  # 최종 저장 경로 생성
            Path(res_dir).mkdir(parents=True, exist_ok=True) 
            print(f"Saving results to ---> {res_dir}")            
        

            tr_dataset = BIODataset('train', device, data_load_dir)
            train_loader = BIODataLoader(dataset=tr_dataset, batch_size=int(args.BATCH), \
                                                num_workers=0, shuffle=True, drop_last=True)
            
            vl_dataset = BIODataset('valid', device, data_load_dir)
            valid_loader = BIODataLoader(dataset=vl_dataset, batch_size=int(args.BATCH), \
                                            num_workers=0, shuffle=True, drop_last=True)
            ts_dataset = BIODataset('test', device, data_load_dir)
            test_loader = BIODataLoader(dataset=ts_dataset, batch_size=int(args.BATCH), \
                                            num_workers=0, shuffle=True, drop_last=True)
            
            ###### 모델 생성
            my_model = Net(args).to(device)

            MODEL_PATH = os.path.join(res_dir, FILENAME_MODEL)
            MODEL_PATH = f'{MODEL_PATH}'
            
            # 학습
            trainer = Trainer(args, my_model, MODEL_PATH, res_dir) 
            tr_history = trainer.train(train_loader, valid_loader)
            print('End of Train\n')

            # Test set 성능 평가

            ts_history = trainer.eval('test', test_loader)

            print('End of Test\n')
            

            # Save Results
            trainer.save_result(tr_history, ts_history, res_dir)

            ts_total = pd.concat([ts_total, ts_history], axis=0, ignore_index=True)

        ts_total.to_csv(os.path.join(res_dir, FILENAME_FOLDRES))
        ts_total.describe().to_csv(os.path.join(res_dir, FILENAME_FOLDSUM))

        ts_fold = pd.concat([ts_fold, ts_total], axis=0, ignore_index=True)

        print(f"Saving results to res/{res_flen}\n") 
  



if __name__ == "__main__": 

    import os
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

    start = time.time()

    parser = argparse.ArgumentParser(description='MS_mmdynamic')

    parser.add_argument("--SEED", default=42)
    parser.add_argument('--dataset', default='SA') # Don't change this
    ########################################################################
    
    ### data_type 과 model_type 선택 필수
    parser.add_argument('--data_type', default='SA_Drowsy_bal', choices=['SA_Drowsy_bal', 'SA_Drowsy_unbal'])
    parser.add_argument('--modality', default='eeg')
    parser.add_argument('--backbone', default='Transformer', choices = ['EEGNet4', 'DeepConvNet', 'ResNet8', 'ResNet18', 'EEGConformer'])
    parser.add_argument('--scheduler', default='CosineAnnealingLR', choices=['CosineAnnealingLR', 'StepLR'])
    parser.add_argument('--num_sbj', default=11)
    ### early stopping on-off
    parser.add_argument('--early_stop', default=False, choices=[True, False])

    ########## 실험 하이퍼 파라미터 설정 
    parser.add_argument('--BATCH', default=16, help='Batch Size') # original 16, set 32
    parser.add_argument('--EPOCH', default=100, help='Epoch') # original: 50, set: 10

    parser.add_argument('--optimizer', default='AdamW', help='Optimizer') 
    parser.add_argument('--lr', default=0.002, help='Adam Learning Rate') # original: 1e-4

    parser.add_argument('--step_size', default=500, help='step size for StepLR scheduler')
    parser.add_argument('--freq_time', default=384)
    parser.add_argument('--in_dim', default=[30], choices=[[28], [28,1], [28,1,1,1,1]], help='num of channel list for each modality')


    ########## 클래스 및 채널 수 지정
    parser.add_argument('--n_channels', default=30)
    parser.add_argument('--n_classes', default=2)
    
    args = parser.parse_args()

    seed_everything(args.SEED)

    
    if args.modality == 'eeg':
        args.n_channels = 30
    else :
        args.n_channels = 1

    if args.backbone == 'EEGNet4':
        from models.EEGNet4 import Net
    elif args.backbone == 'DeepConvNet':
        from models.DeepConvNet import Net
    elif args.backbone == 'Transformer':
        from models.Transformer import Net
    elif args.backbone == 'ResNet18':
        from models.ResNet1D18 import Net
    elif args.backbone == 'EEGConformer':
        from models.EEGConformer import Net

    from Helpers.trainer import Trainer

    include_sbj = [1,2,3,4,5,6,7,8,9,10,11]
    
    Experiment(args)