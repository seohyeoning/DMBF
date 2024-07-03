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
from Helpers.crl_utils import History 

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")



def Experiment(args):

    ##### SAVE DIR for .pt file

    if args.CRL_user == True:
        if args.postprocessor == 'ebo':
            res_dir = os.path.join (WD,'res', f'{args.data_type}_{args.mode}/{args.model_type}_{args.backbone}_{args.postprocessor}_temp{args.temp}/bs{args.BATCH}_{args.scheduler}_{args.optimizer}_{args.lr}') 
        else: # msp, mls
            res_dir = os.path.join (WD,'res',  f'{args.data_type}_{args.mode}/{args.model_type}_{args.backbone}_{args.postprocessor}/bs{args.BATCH}_{args.scheduler}_{args.optimizer}_{args.lr}') 
    
    else: # args.CRL_user == False
        if args.postprocessor == 'ebo':
            res_dir = os.path.join (WD,'res', f'woCRL/{args.data_type}_{args.mode}/{args.model_type}_{args.backbone}_{args.postprocessor}_temp{args.temp}/bs{args.BATCH}_{args.scheduler}_{args.optimizer}_{args.lr}') 
        else: # msp, mls
            res_dir = os.path.join (WD,'res',  f'woCRL/{args.data_type}_{args.mode}/{args.model_type}_{args.backbone}_{args.postprocessor}/bs{args.BATCH}_{args.scheduler}_{args.optimizer}_{args.lr}') 


    Path(res_dir).mkdir(parents=True, exist_ok=True)
    
    res_flen = str(len(os.listdir(res_dir))) # 신규 폴더 구분용
    save_file = f'/{res_flen}/'

    RESD = res_dir + save_file
    Path(RESD).mkdir(parents=True, exist_ok=True) # 저장 경로 생성


    ################## 피험자 별 경로 설정 및 실험 시작 ##################
    ts_fold = pd.DataFrame(columns=METRICS)
    num_fold = 4
    include_sbj = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29] # 2cl_5: [5,6,8,9,10,13,15,16,17,18,19,20,21] # 2cl_5 전용
    for subj in range(1,31): 
        if subj not in include_sbj:
                continue
        for nf in range(1, num_fold+1):
            ts_total = pd.DataFrame(columns=METRICS)

            print('='*30)
            print(' '*4, '피험자{} - fold{}을 test로 학습 시작'.format(subj, nf))
            print('='*30)

            # 데이터 정보 불러오기

            Dataset_directory = f'{DATASET_DIR}/{args.data_type}'
            data_load_dir = f'{Dataset_directory}/S{subj}/{nf}fold'
            print(f'Loaded data from --> {DATASET_DIR}/S{subj}/{nf}fold')

            # 결과 저장 경로 설정 2
            res_name = f'S{subj}'
            nfoldname = f'fold{nf}'

            res_dir = os.path.join(RESD, res_name, nfoldname)  # 최종 저장 경로 생성
            Path(res_dir).mkdir(parents=True, exist_ok=True) 
            print(f"Saving results to ---> {res_dir}")            

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
            
            # EEGNet4의 경우, freq_time을 알아야 함.
            freq_time = train_loader.dataset.X.shape[2]
            ###### 모델 생성
            my_model = Net(args, device).to(device)

            MODEL_PATH = os.path.join(res_dir, FILENAME_MODEL)
            MODEL_PATH = f'{MODEL_PATH}'

            if args.CRL_user == True:
                # make History Class
                correctness_history = History(len(train_loader.dataset))

                # 학습
                trainer = Trainer(args, my_model, MODEL_PATH, res_dir) 
                tr_history = trainer.train(train_loader, valid_loader, correctness_history)
            else : 
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

    parser = argparse.ArgumentParser(description='Ablation_TCPjoint')

    parser.add_argument("--SEED", default=42)

    ### data_type 과 model_type 선택 필수
    parser.add_argument('--data_type', default='Stress', choices=['Stress'])

    parser.add_argument('--model_type', default='Model_OURS', choices=['Model_OURS', 'Model_v4_woSTAM', 
                                                                       'Model_v4_woGateMech', 'Model_v4_woGateMech_STAM']) # HJU_FE_OUR

    ####################################실험 하이퍼 파라미터 설정##########################################################
    # msp, average, rank weight:1, Backbone network: EEGNet4
    parser.add_argument('--selection_loss_type', default='CE', choices=['CE', 'Focal']) # SOTA = CE
    parser.add_argument('--postprocessor', default='msp', choices=['msp', 'mls', 'ebo']) # SOTA = msp
    parser.add_argument('--fusion_type', default='average' , choices=['average', 'sum', 'concat', 'matmul']) # SOTA = average
    parser.add_argument('--temp', default=10, help='temperature scaling for ebo') # msp는 사용안함. choice: 1.5, 0.1, 10
    parser.add_argument('--CRL_user', default=True, choices=[True, False]) # default = True
                                                                           # ABLATION: woGateMech_STAM, woGate, woCRL는 False

    ###################################### FIX 
    parser.add_argument('--backbone', default='EEGNet4', choices = ['EEGNet4', 'DeepConvNet', 'ResNet8', 'ResNet18'])
    parser.add_argument('--scaling', default='sigmoid', choices=['softmax', 'sigmoid', 'none']) # SOTA기준 성능: sigmoid > none > softmax
    parser.add_argument('--rank_weight', default=1, type=float, help='Rank loss weight') # SOTA = 1
    parser.add_argument('--BATCH', default=16, help='Batch Size') # original 16, set 32
    parser.add_argument('--EPOCH', default=100, help='Epoch') # original: 100
    parser.add_argument('--optimizer', default='AdamW', help='Optimizer') 
    parser.add_argument('--lr', default=0.002, help='Learning Rate') # original: 1e-4

    parser.add_argument('--scheduler', default='CosineAnnealingLR', choices=['CosineAnnealingLR', 'StepLR', 'ReduceLROnPlateau']) # SOTA = CosineAnnealingLR
    parser.add_argument('--step_size', default=500, help='step size for StepLR scheduler')
    ###############################################################################################################################

    ########## 클래스 및 채널 수 지정
    parser.add_argument('--n_channels', default=32)
    parser.add_argument('--in_dim', default=[32,1,1], choices=[[32], [32,1], [32,1,1]])
    parser.add_argument('--n_classes', default=2)
    parser.add_argument('--freq_time', default=400, help='frequency(200)*time window(2)')
    parser.add_argument('--mode', default='multi', choices=['uni', 'multi'])
    parser.add_argument('--modal', default = 'ALL', choices=['ALL', 'EEG',  'PPG', 'ECG'])

    args = parser.parse_args()

    seed_everything(args.SEED)

    from Helpers.Stress_Dataloader import BIODataset, BIODataLoader

    if args.backbone == 'EEGNet4':
        if args.model_type == 'Model_OURS':
            from Models.Model_OURS_EGN_LG import Net

    if args.CRL_user == True:
        from Helpers.Trainer import Trainer
    else:
        from Helpers.Trainer_woCRL import Trainer


    Experiment(args)
