import os
import numpy as np
import torch
import argparse

import torch.nn as nn

import pandas as pd
from pathlib import Path
import time
import datetime
import random

from Helpers.Variables import device, DATASET_DIR, METRICS, WD, FILENAME_FOLDRES, FILENAME_MODEL, FILENAME_FOLDSUM

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")



def Experiment(args):

    ##### SAVE DIR for .pt file

    res_dir = os.path.join (WD, f'res/{args.model_type}_{args.data_type}/bs{args.BATCH}_{args.scheduler}_{args.optimizer}_{args.lr}') 

    Path(res_dir).mkdir(parents=True, exist_ok=True)
    
    res_flen = str(len(os.listdir(res_dir))) # 신규 폴더 구분용
    save_file = f'/{res_flen}/'

    RESD = res_dir + save_file
    Path(RESD).mkdir(parents=True, exist_ok=True) # 저장 경로 생성


    ################## 피험자 별 경로 설정 및 실험 시작 ##################
    ts_fold = pd.DataFrame(columns=METRICS)
    num_fold = 4

    for subj in range(12, args.num_sbj+1) : # args.num_sbj+1): 
        if subj not in include_sbj:
                continue
        for nf in range(1, num_fold+1):
            ts_total = pd.DataFrame(columns=METRICS)

            print('='*30)
            print(' '*4, '피험자{} - fold{}을 test로 학습 시작'.format(subj, nf))
            print('='*30)

            # 데이터 정보 불러오기

            Dataset_directory = f'{DATASET_DIR}/{args.data_type}/'
            if args.data_type == "MS":
                data_load_dir = f'{Dataset_directory}/S{subj}/fold{nf}'
            else:
                data_load_dir = f'{Dataset_directory}/S{subj}/{nf}fold'
            print(f'Loaded data from --> {DATASET_DIR}/S{subj}/{nf}fold')

            # 결과 저장 경로 설정 2
            res_name = f'S{subj}'
            nfoldname = f'fold{nf}'

            result_dir = os.path.join(RESD, res_name, nfoldname)  # 최종 저장 경로 생성
            Path(result_dir).mkdir(parents=True, exist_ok=True) 
            print(f"Saving results to ---> {result_dir}")            

            ###### 데이터셋 생성
            tr_dataset = BIODataset('train', device, data_load_dir)
            train_loader = BIODataLoader(dataset=tr_dataset, batch_size=int(args.BATCH),\
                                                num_workers=0, shuffle=True, drop_last=True)

            vl_dataset = BIODataset('valid', device, data_load_dir)
            valid_loader = BIODataLoader(dataset=vl_dataset, batch_size=int(args.BATCH), \
                                            num_workers=0, shuffle=True, drop_last=True)
            ts_dataset = BIODataset('test', device, data_load_dir)
            test_loader = BIODataLoader(dataset=ts_dataset, batch_size=int(args.BATCH), \
                                            num_workers=0, shuffle=True, drop_last=True)
            
            ###### 모델 생성
            my_model = Net(args).to(device)
     

            MODEL_PATH = os.path.join(result_dir, FILENAME_MODEL)
            MODEL_PATH = f'{MODEL_PATH}'
            
            # 학습
            trainer = Trainer(args, my_model, MODEL_PATH, result_dir) 
            tr_history = trainer.train(train_loader, valid_loader)
            print('End of Train\n')


            # Test set 성능 평가

            ts_history = trainer.eval('test', test_loader)

            print('End of Test\n')
            

            # Save Results
            trainer.save_result(tr_history, ts_history, result_dir)

            ts_total = pd.concat([ts_total, ts_history], axis=0, ignore_index=True)

        ts_total.to_csv(os.path.join(result_dir, FILENAME_FOLDRES))
        ts_total.describe().to_csv(os.path.join(result_dir, FILENAME_FOLDSUM))

        ts_fold = pd.concat([ts_fold, ts_total], axis=0, ignore_index=True)




if __name__ == "__main__":   
    import os
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 
    
    def seed_everything(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
 
    start = time.time()

    parser = argparse.ArgumentParser(description='ESTCNN')

    parser.add_argument("--SEED", default=42)

    ### data_type 과 model_type 선택 필수
    parser.add_argument('--data_type', default='manD', choices=['Drowsy', 'Distraction', 'Stress', 'MS'])
    parser.add_argument('--model_type', default='ESTCNN')
    parser.add_argument('--num_sbj', default=None)
    ####################################실험 하이퍼 파라미터 설정##########################################################
    parser.add_argument('--scheduler', default='CosineAnnealingLR', choices=['StepLR', 'CosineAnnealingLR'])

    ### early stopping on-off
    parser.add_argument('--early_stop', default=False, choices=[True, False])
    parser.add_argument('--random_sampler', default=False, choices=[True, False])

    ########## 실험 하이퍼 파라미터 설정 
    parser.add_argument('--BATCH', default=16, help='Batch Size') # original 16, set 32
    parser.add_argument('--EPOCH', default=100, help='Epoch') # original: 50, set: 10
    parser.add_argument('--optimizer', default='AdamW', help='Optimizer') 
    parser.add_argument('--lr', default=0.002, help='Adam Learning Rate') # original: 1e-4
    parser.add_argument('--step_size', default=500, help='step size for StepLR scheduler')

    parser.add_argument('--freq_time', default=None, help='frequency(200)*time window(3)')
    parser.add_argument('--in_dim', default=[32], help='num of channel list for each modality')


    ########## 클래스 및 채널 수 지정
    parser.add_argument('--n_channels', default=32)
    parser.add_argument('--n_classes', default=2)
    
    args = parser.parse_args()

    seed_everything(args.SEED)

    def weight_init(m):
        if isinstance(m, nn.Conv1d):
            torch.nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0.0)

    if args.data_type == 'Drowsy':
        from Helpers.Drowsy_Dataloader import BIODataset, BIODataLoader
        args.freq_time = 600
        include_sbj =[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
        args.num_sbj = 31
        args.in_dim=[32]
        args.n_channels = 32

    elif args.data_type == 'MS':
        from Helpers.MS_Dataloader import BIODataset, BIODataLoader
        include_sbj = [5,6,8,9,10,13,15,16,17,18,19,20,21]
        args.num_sbj = 23
        args.n_channels = 28
        args.in_dim=[28]
        args.freq_time = 750

    elif args.data_type == 'Distraction':
        from Helpers.Distraction_Dataloader import BIODataset, BIODataLoader
        args.freq_time = 400
        include_sbj = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        args.num_sbj = 29
        args.in_dim=[32]
        args.n_channels = 32

    elif args.data_type == 'Stress':
        from Helpers.Stress_Dataloader import BIODataset, BIODataLoader
        args.freq_time = 400
        include_sbj = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
        args.num_sbj = 27    
        args.in_dim=[32]
        args.n_channels = 32

    elif args.data_type == 'Pilot_Fatigue':
        from Helpers.Fatigue_Dataloader import BIODataset, BIODataLoader
        args.freq_time = 250*3
        include_sbj = [1,4,5,8,13,14,15,16,18,19,20]
        args.num_sbj = 20
        args.n_channels = 64
        args.in_dim = [64]

    elif args.data_type == 'manD':
        from Helpers.manD_Dataloader import BIODataset, BIODataLoader
        args.freq_time = 256*3
        include_sbj = [13,16,17,22,23,39,41,42,43,47,48,50]
        args.num_sbj =51
        args.in_dim = [9]
        args.n_channels = 9
        args.n_classes = 5

    from Models.ETCNN import Net

    from Helpers.trainer import Trainer

    Experiment(args)
