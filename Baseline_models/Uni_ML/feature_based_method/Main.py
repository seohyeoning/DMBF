import numpy as np
import torch
import argparse

import torch.nn as nn

import pandas as pd
from pathlib import Path
import time
import datetime
import random

from Helpers.Variables import device, DATASET_DIR, WD, FILENAME_FOLDRES, FILENAME_MODEL, FILENAME_FOLDSUM
from sklearn.preprocessing import StandardScaler
from Helpers.FE_algorithms import DE_PSD
from mne.decoding import CSP

from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score



# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def preprocess_data(X):
    # NaN 값을 포함하는 샘플 제거
    # mask = ~np.isnan(X).any(axis=(1, 2))
    # X = X[mask]
    
    # 데이터 정규화
    scaler = StandardScaler()
    n_samples, n_channels, n_features = X.shape
    X_reshaped = X.reshape(n_samples, -1)  # 2D로 변환
    X_scaled = scaler.fit_transform(X_reshaped)
    X = X_scaled.reshape(n_samples, n_channels, n_features)  # 원래 형태로 복원
    return X


def Experiment(args, model):

    ##### SAVE DIR for .pt file
    res_dir = os.path.join (WD,'res', f'{args.data_type}/{args.model_type}_512/')
    Path(res_dir).mkdir(parents=True, exist_ok=True)
    
    res_flen = str(len(os.listdir(res_dir))) # 신규 폴더 구분용
    save_file = f'/{res_flen}/'

    RESD = res_dir + save_file
    Path(RESD).mkdir(parents=True, exist_ok=True) # 저장 경로 생성


    ################## 피험자 별 경로 설정 및 실험 시작 ##################
    METRICS = ['Accuracy', 'F1 Score', 'Balanced Accuracy']
    ts_fold = pd.DataFrame(columns=METRICS)

    num_fold = args.num_fold

    for subj in range(1,sbj_num+1): # 1,24
        if subj not in include_sbj:
                continue
        for nf in range(1, num_fold+1):
            ts_total = pd.DataFrame(columns=METRICS)

            print('='*30)
            print(' '*4, '피험자{} - fold{}을 test로 학습 시작'.format(subj, nf))
            print('='*30)


            # 데이터 정보 불러오기
            if args.data_type == 'SA_Drowsy':
                Dataset_directory = f'{DATASET_DIR}/{args.data_type}/dataset'
            else:
                Dataset_directory = f'{DATASET_DIR}/{args.data_type}'

            data_load_dir = f'{Dataset_directory}/S{subj}/{nf}fold'

            if args.data_type == 'MS':
                data_load_dir = f'{Dataset_directory}/S{subj}/fold{nf}'

            print(f'Loaded data from --> {DATASET_DIR}/S{subj}/{nf}fold')

            # 결과 저장 경로 설정 2
            res_name = f'S{subj}'
            nfoldname = f'fold{nf}'

            res_dir = os.path.join(RESD, res_name, nfoldname)  # 최종 저장 경로 생성
            Path(res_dir).mkdir(parents=True, exist_ok=True) 
            print(f"Saving results to ---> {res_dir}")            

            tr_dataset = BIODataset('train', device, data_load_dir)
            vl_dataset = BIODataset('valid', device, data_load_dir)
            ts_dataset = BIODataset('test', device, data_load_dir)
            
            # Extract EEG features from dataset
            if args.data_type == 'MS':
                X_train_raw, y_train = tr_dataset.X.transpose(0,2,1)[:,0:args.n_channels,:], tr_dataset.y.squeeze()
                X_valid_raw, y_valid = vl_dataset.X.transpose(0,2,1)[:,0:args.n_channels,:], vl_dataset.y.squeeze()
                X_test_raw, y_test = ts_dataset.X.transpose(0,2,1)[:,0:args.n_channels,:], ts_dataset.y.squeeze()  
                X_train_raw = np.concatenate((X_train_raw, X_valid_raw), axis=0)
                y_train = np.concatenate((y_train, y_valid), axis=0)
            else:
                X_train_raw, y_train = tr_dataset.X[:,0:args.n_channels,:], tr_dataset.y.squeeze()
                X_test_raw, y_test = ts_dataset.X[:,0:args.n_channels,:], ts_dataset.y.squeeze()

            if y_train.ndim > 1 and y_train.shape[1] > 1:
                y_train = np.argmax(y_train, axis=1)
            if y_test.ndim > 1 and y_test.shape[1] > 1:
                y_test = np.argmax(y_test, axis=1)

    
            if args.model_type == 'CSP_LDA':
                X_train_raw = preprocess_data(X_train_raw)
                csp = CSP(n_components=4, norm_trace=False)
                X_train = csp.fit_transform(X_train_raw, y_train)
                X_test = csp.transform(X_test_raw)
            else:
                X_train, X_test = DE_PSD(args,X_train_raw), DE_PSD(args,X_test_raw)

            X_train_flattened = X_train.reshape(X_train.shape[0], -1)
            X_test_flattened = X_test.reshape(X_test.shape[0], -1)
            ###### 모델 생성


            model.fit(X_train_flattened, y_train)
            
            MODEL_PATH = os.path.join(res_dir, FILENAME_MODEL)
            MODEL_PATH = f'{MODEL_PATH}'
            
            print('End of Train\n')

            # Test set 성능 평가
            y_pred_test = model.predict(X_test_flattened)

            test_accuracy = accuracy_score(y_test, y_pred_test)
            test_f1 = f1_score(y_test, y_pred_test, average='macro') 
            test_bacc = balanced_accuracy_score(y_test, y_pred_test)
            print("Test Accuracy:", test_accuracy)
            print("Test F1 Score:", test_f1)
            print("Test Balanced Accuracy:", test_bacc)
            print('End of Test\n')
            

            # 폴드 결과 저장
            ts_total = pd.DataFrame([[test_accuracy, test_f1, test_bacc]], columns=METRICS)
            ts_total.to_csv(os.path.join(res_dir, f'fold_{nf}_results.csv'))

            # 각 폴드 결과를 ts_fold에 추가
            ts_fold = pd.concat([ts_fold, ts_total], ignore_index=True)


    # 전체 결과 요약
    ts_fold.describe().to_csv(os.path.join(RESD, 'summary_results.csv'))



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

    parser = argparse.ArgumentParser(description='baseline')

    parser.add_argument("--SEED", default=42)

    ### data_type 과 model_type 선택 필수
    parser.add_argument('--data_type', default='manD', choices=['MS', 'Distraction', 'Drowsy', 'SA_Drowsy', 'Stress'])

    parser.add_argument('--model_type', default='PSD_GNB', choices=['PSD_KNN','PSD_SVM','PSD_GNB','CSP_LDA'])

    # parser.add_argument('--fusion_type', default='concat', choices=['average','concat', 'sum']) # default: concat

    ########## 실험 하이퍼 파라미터 설정 
    parser.add_argument('--scheduler', default='CosineAnnealingLR', choices=['StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau'])
    parser.add_argument('--step_size', default=500, help='step size for StepLR scheduler')

    parser.add_argument('--BATCH', default=16, help='Batch Size') # original 16, set 32
    parser.add_argument('--EPOCH', default=100, help='Epoch') # original: 50, set: 10
    parser.add_argument('--optimizer', default='AdamW', help='Optimizer') 
    parser.add_argument('--lr', default=0.002, help='Learning Rate') # original: 1e-4

    ########## 클래스 및 채널 수 지정
    parser.add_argument('--n_channels', default=None)
    parser.add_argument('--n_classes', default=2)
    parser.add_argument('--in_dim', default=[32], choices=[[32], [32,1], [32,1,1],None], help='num of channel list for each modality')
    parser.add_argument('--num_fold', default=4, help='number of fold')
    
    args = parser.parse_args()

    seed_everything(args.SEED)

    if args.data_type == 'Drowsy':
        from Helpers.Drowsy_Dataloader import BIODataset, BIODataLoader 
        args.window_size = 3
        args.sr = 200
        include_sbj = [1,5,6,7,8,9,10,12,13,15,16,18,20,21,24,25,26,28,29,30,31] 
        sbj_num = 31
        args.in_dim=[32]
        args.n_channels = 32

    elif args.data_type == 'Distraction':
        from Helpers.Distraction_Dataloader import BIODataset, BIODataLoader
        args.window_size = 2
        args.sr = 200
        include_sbj = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        sbj_num = 29
        args.in_dim=[32]
        args.n_channels = 32

    elif args.data_type == 'Stress':
        from Helpers.Stress_Dataloader import BIODataset, BIODataLoader
        args.window_size = 2
        args.sr = 200
        include_sbj = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
        sbj_num = 27    
        args.in_dim=[32]
        args.n_channels = 32
    
    elif args.data_type == 'MS':
        from Helpers.MS_Dataloader import BIODataset, BIODataLoader
        args.window_size = 3
        args.sr = 250
        include_sbj = [5, 6, 9, 10, 13, 15, 16, 18, 20]
        sbj_num = 20
        args.n_channels = 28
        args.in_dim = [28]

    elif args.data_type == 'SA_Drowsy':
        from Helpers.Fatigue_Dataloader import BIODataset, BIODataLoader
        args.window_size = 3
        args.sr = 128
        include_sbj = [1,2,3,4,5,6,7,8,9,10,11]
        sbj_num = 11
        args.n_channels = 30
        args.in_dim = [30]
        
    elif args.data_type == 'manD':
        from Helpers.manD_Dataloader import BIODataset, BIODataLoader
        include_sbj = [13,16,17,22,23,39,41,42,43,47,48,50]
        sbj_num =51
        args.in_dim = [9,1,1,1,1]
        args.n_channels = 9
        args.n_classes = 5
        args.sr = 256
        args.window_size = 3


    if args.model_type == 'PSD_KNN':
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier()   
    elif args.model_type == 'PSD_RF':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(random_state=42)
    elif args.model_type == 'PSD_SVM':
        from sklearn.svm import SVC
        model = SVC(random_state=42)
    elif args.model_type == 'PSD_GNB':
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()
        
    elif args.model_type == 'CSP_LDA':
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        model = LDA()
    

    from Helpers.trainer import Trainer

    Experiment(args, model)
