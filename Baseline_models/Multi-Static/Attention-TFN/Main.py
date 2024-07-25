import argparse
import os
import pandas as pd
from pathlib import Path
import time
import datetime
import random
import numpy as np
import torch 



from Helpers.trainer import Trainer
from Helpers.Variables import device, METRICS, WD, DATASET_DIR, FILENAME_FOLDRES, FILENAME_MODEL, FILENAME_FOLDSUM


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def Experiment(args):

    ##### SAVE DIR for .pt file
    res_dir = os.path.join (WD,'res', f'{args.data_type}/bs{args.BATCH}_{args.optimizer}_{args.lr}_{args.scheduler}')
    Path(res_dir).mkdir(parents=True, exist_ok=True)

    res_flen = str(len(os.listdir(res_dir))) # 신규 폴더 구분용
    save_file = f'/{res_flen}/'

    RESD = res_dir + save_file
    Path(RESD).mkdir(parents=True, exist_ok=True) # 저장 경로 생성

    ################## 피험자 별 경로 설정 및 실험 시작 ##################
    ts_fold = pd.DataFrame(columns=METRICS)
    num_fold = 4
    for subj in range(1, sbj_num+1):
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
            print(f'Loaded data from --> {Dataset_directory}/S{subj}/{nf}fold')

            # 결과 저장 경로 설정 2
            res_name = f'S{subj}'
            nfoldname = f'fold{nf}'

            result_dir = os.path.join(RESD, res_name, nfoldname)
            Path(result_dir).mkdir(parents=True, exist_ok=True)
            print(f"Saving results to ---> {result_dir}")   
               
            tr_dataset = BIODataset('train', device, data_load_dir)
            train_loader = BIODataLoader(dataset=tr_dataset, batch_size=int(args.BATCH), \
                                            num_workers=0, shuffle=True, drop_last=True)
            vl_dataset = BIODataset('valid', device, data_load_dir)
            valid_loader = BIODataLoader(dataset=vl_dataset, batch_size=int(args.BATCH), \
                                            num_workers=0, shuffle=True, drop_last=True)
            ts_dataset = BIODataset('test', device, data_load_dir)
            test_loader = BIODataLoader(dataset=ts_dataset, batch_size=int(args.BATCH), \
                                            num_workers=0, shuffle=True, drop_last=True)
            
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
        
    print(f"Saving results to res/{res_flen}")  
        



   
if __name__ == '__main__':
    start = time.time()
    
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('--SEED', default=42)
    
    ### data 선택
    parser.add_argument('--data_type', default='manD', choices=['Drowsy', 'Stress', 'Distraction', 'Pilot_Fatigue','manD'])
    ### early stopping on-off
    parser.add_argument('--early_stop', default=False, choices=[True, False])

    parser.add_argument('--BATCH', default=16, help='Batch Size') # original 16
    parser.add_argument('--EPOCH', default=100, help='Epoch') # original: 50, set: 10
    parser.add_argument('--optimizer', default='AdamW', help='Optimizer')
    parser.add_argument('--lr', default=0.002, help='Adam Learning Rate') # original: 1e-2, set: 2e-2
    parser.add_argument('--scheduler', default='CosineAnnealingLR', help='Scheduler')
    parser.add_argument('--n_channels', default=32)                 
    parser.add_argument('--n_classes', default=2)
   
    args = parser.parse_args()

    ########## 시드 고정
    random.seed(args.SEED)
    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)
    torch.cuda.manual_seed(args.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  
    
    if args.data_type == 'Drowsy':
        from Helpers.Drowsy_Dataloader import BIODataset, BIODataLoader 
        args.freq_time = 600
        include_sbj =[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
        sbj_num = 31
        args.n_channels = 32
        from Models.Model_3modal import Net

    elif args.data_type == 'Distraction':
        from Helpers.Distraction_Dataloader import BIODataset, BIODataLoader
        args.freq_time = 400
        include_sbj = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        sbj_num = 29
        args.n_channels = 32
        from Models.Model_3modal import Net

    elif args.data_type == 'Stress':
        from Helpers.Stress_Dataloader import BIODataset, BIODataLoader
        args.freq_time = 400
        include_sbj = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
        sbj_num = 27    
        args.n_channels = 32
        from Models.Model_3modal import Net

    elif args.data_type == 'Pilot_Fatigue':
        from Helpers.Fatigue_Dataloader import BIODataset, BIODataLoader
        args.freq_time = 250*3
        include_sbj = [1,4,5,8,13,14,15,16,18,19,20]
        sbj_num = 20
        args.n_channels = 64
        from Models.Model_4modal import Net

    elif args.data_type == 'manD':
        from Helpers.manD_Dataloader import BIODataset, BIODataLoader
        args.freq_time = 256*3
        include_sbj = [13,16,17,22,23,39,41,42,43,47,48,50]
        sbj_num = 51
        args.n_channels = 9
        args.n_classes = 5

        from Models.Model_5modal import Net

    Experiment(args) # Time -> about 1 hour 50 minutes
    print('Code Time Consumption: ', str(datetime.timedelta(seconds=time.time() - start)).split('.')[0])
