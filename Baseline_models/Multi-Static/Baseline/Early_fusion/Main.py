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

from Helpers.Variables import device, METRICS, DATASET_DIR, WD, FILENAME_FOLDRES, FILENAME_MODEL, FILENAME_FOLDSUM


from torch.utils.data import WeightedRandomSampler

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def Experiment(args):
    sampler_sbj_list, no_sampler_sbj_list = [], []
    ##### SAVE DIR for .pt file
    if args.modal == 'woEEG':
        res_dir = os.path.join (WD,'res', f'{args.data_type}/woEEG_{args.backbone}_{args.model_type}_{args.fusion_type}/{args.scheduler}_{args.data_type}_{args.optimizer}_{args.lr}')
    elif args.modal == 'wEEG':
        res_dir = os.path.join (WD,'res', f'{args.data_type}/wEEG_{args.backbone}_{args.model_type}_{args.fusion_type}/{args.scheduler}_{args.data_type}_{args.optimizer}_{args.lr}')
    Path(res_dir).mkdir(parents=True, exist_ok=True)
    
    res_flen = str(len(os.listdir(res_dir))) # 신규 폴더 구분용
    save_file = f'/{res_flen}/'

    RESD = res_dir + save_file
    Path(RESD).mkdir(parents=True, exist_ok=True) # 저장 경로 생성


    ################## 피험자 별 경로 설정 및 실험 시작 ##################
    ts_fold = pd.DataFrame(columns=METRICS)
    num_fold = 4

    for subj in range(1,args.sbj_num+1): # 1,24
        if subj not in include_sbj:
                continue
        for nf in range(1, num_fold+1):
            ts_total = pd.DataFrame(columns=METRICS)

            print('='*30)
            print(' '*4, '피험자{} - fold{}을 test로 학습 시작'.format(subj, nf))
            print('='*30)

            if subj < 10:
                sbj = '0' + str(subj)
            else:
                sbj = subj

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

        
            ######### Try to apply weighted random sampler############
            if args.random_sampler == True:
                tr_dataset = BIODataset('train', device, data_load_dir)
                sampler = sampler_(tr_dataset)

                # Impossible to apply sampler
                sampler_name = f"S{sbj}_fold{nf}"
                if sampler == "Do not apply sampler": 
                    print(f"\nS{sbj}/fold{nf}는 한 클래스만 존재하므로 weighted sampler 사용 안함\n")
                    train_loader = BIODataLoader(dataset=tr_dataset, batch_size=int(args.BATCH), \
                                                num_workers=0, shuffle=True, drop_last=True)
                    no_sampler_sbj_list.append(sampler_name)
                    
                # Possible to apply sampler
                else: 
                    print(f"S{sbj}/fold{nf}는 weighted sampler 사용함\n")
                    sampler_sbj_list.append(sampler_name)
                    train_loader = BIODataLoader(dataset=tr_dataset, batch_size=int(args.BATCH), \
                                    num_workers=0, shuffle=False, drop_last=True, sampler=sampler) # 샘플러 사용 시, shuffle=False
            ##############################################

            ######### Not try to apply weighted random sampler############
            if args.random_sampler == False: 
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




if __name__ == "__main__":    
    start = time.time()

    parser = argparse.ArgumentParser(description='MS_mmdynamic')

    parser.add_argument("--SEED", default=42)

    ### data_type 과 model_type 선택 필수
    parser.add_argument('--modal', default='wEEG', choices=['woEEG', 'wEEG'])
    parser.add_argument('--data_type', default='manD', choices=['Drowsy', 'Distraction', 'Pilot_Fatigue'])
    parser.add_argument('--model_type', default='early_fusion')
    parser.add_argument('--backbone', default='EEGNet4', choices = ['EEGNet4', 'DeepConvNet', 'ResNet8', 'ResNet18','EEGConformer'])
    parser.add_argument('--fusion_type', default='concat', choices=['concat']) # best backbone은 early concat으로 수행
    parser.add_argument('--scheduler', default='CosineAnnealingLR', choices=['CosineAnnealingLR', 'StepLR', 'ReduceLROnPlateau'])
    ### early stopping on-off
    parser.add_argument('--early_stop', default=False, choices=[True, False])
    parser.add_argument('--random_sampler', default=False, choices=[True, False])

    ########## 실험 하이퍼 파라미터 설정 
    parser.add_argument('--BATCH', default=16, help='Batch Size') # original 16, set 32
    parser.add_argument('--EPOCH', default=100, help='Epoch') # original: 50, set: 10
    parser.add_argument('--optimizer', default='AdamW', help='Optimizer') 
    parser.add_argument('--lr', default=0.002, help='Adam Learning Rate') # original: 1e-4
    parser.add_argument('--step_size', default=500, help='step size for StepLR scheduler')
    parser.add_argument('--freq_time', default=750, help='frequency(250)*time window(3)')
    parser.add_argument('--in_dim', default=None, choices=[[28], [28,1], [28,1,1,1,1]], help='여기서는 사용x modal로 선택')


    ########## 클래스 및 채널 수 지정
    parser.add_argument('--n_channels', default=None)
    parser.add_argument('--n_classes', default=2)
    parser.add_argument('--sbj_num', default=None)
    
    args = parser.parse_args()
    

    random.seed(args.SEED)
    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)
    torch.cuda.manual_seed_all(args.SEED)  # GPU 연산에 대한 시드 설정

    torch.backends.cudnn.deterministic = True

    # Data Generation at first time 
    if not os.path.exists(os.path.join(DATASET_DIR, f'{args.data_type}')):
       data_generate(args)

    from Helpers.trainer import Trainer
    if args.backbone == 'EEGNet4':
        from models.early_EEGNet4 import Net
    if args.backbone == 'ShallowConvNet':
        from models.early_ShallowConvNet import Net
    if args.backbone == 'DeepConvNet':
        from models.early_DeepConvNet import Net
    if args.backbone == 'ResNet8':
        from models.early_ResNet1D8 import Net
    if args.backbone == 'ResNet18':
        from models.early_ResNet1D18 import Net
    if args.backbone == 'EEGConformer':
        from models.early_EEGConformer import Net

    if args.data_type == 'Drowsy':
        from Helpers.Drowsy_Dataloader import BIODataset, BIODataLoader 
        args.freq_time = 600
        include_sbj =[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
        args.sbj_num = 31
        args.in_dim=[32,1,1]
        args.n_channels = 32

    elif args.data_type == 'Distraction':
        from Helpers.Distraction_Dataloader import BIODataset, BIODataLoader
        args.freq_time = 400
        include_sbj = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        args.sbj_num = 29
        args.in_dim=[32,1,1]
        args.n_channels = 32

    elif args.data_type == 'manD':
        from Helpers.manD_Dataloader import BIODataset, BIODataLoader
        include_sbj = [13,16,17,22,23,39,41,42,43,47,48,50]
        args.sbj_num =51
        args.in_dim = [9,1,1,1,1]
        args.n_channels = 9
        args.n_classes = 5
        args.freq_time = 256*3

    elif args.data_type == 'Stress':
        from Helpers.Stress_Dataloader import BIODataset, BIODataLoader
        args.freq_time = 400
        include_sbj = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
        args.sbj_num = 27    
        args.in_dim=[32,1,1]
        args.n_channels = 32

    elif args.data_type == 'Pilot_Fatigue':
        from Helpers.Fatigue_Dataloader import BIODataset, BIODataLoader
        args.freq_time = 250*3
        include_sbj = [1,4,5,8,13,14,15,16,18,19,20]
        args.sbj_num = 20
        args.n_channels = 64
        args.in_dim = [64,1,1,1]

    def sampleCount(dataset):
        n_classes=2
        sample_count = [0]*(n_classes)
        for num in dataset.y:
            sample_count[num[1]] += 1
        return sample_count
    
    def sampler_(dataset):
        n_classes=2
        dataset_counts = sampleCount(dataset)
        if dataset_counts[0] == 0 or dataset_counts[1] == 0:
            notice = "Do not apply sampler" # because There is a only one class
            return notice
        else:
            num_samples = sum(dataset_counts)
            labels = [tag for _,tag in dataset]
            labels = [[int(label[0].item()), int(label[1].item())] for label in labels]
            
            # 클래스 불균형을 고려하여 가중치 계산
            class_weights = [1.0 / (dataset_counts[i] * n_classes) for i in range(n_classes)]
            weights = [class_weights[label[0]] for label in labels]
            
            sampler = WeightedRandomSampler(torch.DoubleTensor(weights), num_samples)
        return sampler
        
    Experiment(args)
