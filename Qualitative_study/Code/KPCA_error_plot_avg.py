import os
import numpy as np
import torch
import argparse

import pandas as pd
from pathlib import Path
import time
import random

from Helpers.Variables import device, METRICS, DATASET_DIR, WD, FILENAME_FOLDRES, FILENAME_MODEL, FILENAME_FOLDSUM
from Helpers.Helpers_2cl import data_generate, BIODataset, BIODataLoader 

import numpy as np


import matplotlib.pyplot as plt

from matplotlib.ticker import ScalarFormatter
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier

from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

from sklearn.preprocessing import StandardScaler



def calculate_bounds(mean_list, std_list):
    sub = [mean - std for mean, std in zip(mean_list, std_list)]
    add = [mean + std for mean, std in zip(mean_list, std_list)]
    return sub, add


def Inference(args):

    ### 모델 load 경로 변경
    Model_Path = "/opt/workspace/Seohyeon/NEW_PIPE/Final/res/Model_OURS_sigmoid_EEGNet4/average_bs16/CELoss_msp_CosineAnnealingLR_AdamW_0.002/0/"
   

    ################## 피험자 별 경로 설정 및 실험 시작 ##################
    num_fold = 4
    include_sbj = [5,6,8,9,10,13,15,16,17,18,19,20,21] # 2cl_5 전용 5,6,8,9,10,13,15,16,17,18,19,20,21

    new_RESD = f"/opt/workspace/Seohyeon/Journal/DMFMS/result_viz_errors/{args.classifier}/"

    layer0_errors_list = []
    layer1_errors_list = []
    layer2_errors_list = []

    for subj in range(1,24): # 1,24
        if subj not in include_sbj:
                continue
        
        collected_features_all_folds = []  # 모든 fold에 대한 특징 수집 리스트
        collected_labels_all_folds = []  # 모든 fold에 대한 레이블 수집 리스트

        for nf in range(1, num_fold+1):

            print('='*30)
            print(' '*4, '피험자{} - fold{}을 test로 학습 시작'.format(subj, nf))
            print('='*30)

            if subj < 10:
                sbj = '0' + str(subj)
            else:
                sbj = subj

            # 데이터 정보 불러오기
            Dataset_directory = f'{DATASET_DIR}/{args.data_type}'
            data_load_dir = f'{Dataset_directory}/S{sbj}/fold{nf}'
            print(f'Loaded data from --> {DATASET_DIR}/S{sbj}/fold{nf}')

            # 결과 저장 경로 설정 2
            res_name = f'S{sbj}'
            nfoldname = f'fold{nf}'


            save_dir = os.path.join(new_RESD, res_name, nfoldname)  # 최종 저장 경로 생성  
            Path(save_dir).mkdir(parents=True, exist_ok=True) 
            print(f"Saving results to ---> {save_dir}")            


            ###### 데이터셋 생성
            ts_dataset = BIODataset('test', device, data_load_dir)
            test_loader = BIODataLoader(dataset=ts_dataset, batch_size=int(args.BATCH), \
                                            num_workers=0, shuffle=True, drop_last=True)
            
            ###### 모델 생성
            my_model = Net(args, device).to(device)

            MODEL_PATH = os.path.join(Model_Path, res_name, nfoldname, FILENAME_MODEL)
            MODEL_PATH = f'{MODEL_PATH}'

            print('Start of Test\n')

            my_model.load_state_dict(torch.load(MODEL_PATH))

            # 각 특징에 대해 데이터를 수집할 리스트 초기화
            collected_features = [[] for _ in range(12)]  # 12개의 특징 리스트
            collected_labels = []
            with torch.no_grad(): # .eval함수와 torch.no_grad함수를 같이 사용하는 경향
                for datas in test_loader:
                    data, target = datas['data'], datas['label'].to(device)

                    # 모델에서 feature maps을 추출
                    feat_list, weight_feature_list, weight_concat_f, fused_f, _ = my_model.infer(data)

                    # 추출된 특징을 리스트에 추가
                    for i, feat in enumerate(feat_list + weight_feature_list + [weight_concat_f, fused_f]):
                        collected_features[i].append(feat.cpu().numpy())  # GPU 메모리에 있을 경우 CPU로 이동

                    # 레이블을 수집
                    collected_labels.append(target.cpu().numpy())

            # 레이블 데이터 연결
            collected_labels = np.concatenate(collected_labels, axis=0)

            # 각 특징별로 모든 배치 데이터를 연결                     
            for i, feats in enumerate(collected_features):
                collected_features[i] = np.concatenate(feats)
            collected_features_all_folds.append(collected_features)
            collected_labels_all_folds.append(np.argmax(collected_labels.copy(), axis=1))        

        labels = np.concatenate(collected_labels_all_folds, axis=0)

        eeg_feat, ecg_feat, rsp_feat, ppg_feat, gsr_feat, weight_concat_f, fused_f = [], [], [], [], [], [], []
        for i in range(4):
            for n in range(12):
                if n==0:
                    eeg_feat.append(collected_features_all_folds[i][n])
                elif n==1:
                    ecg_feat.append(collected_features_all_folds[i][n])
                elif n==2:
                    rsp_feat.append(collected_features_all_folds[i][n])
                elif n==3:
                    ppg_feat.append(collected_features_all_folds[i][n])
                elif n==4:
                    gsr_feat.append(collected_features_all_folds[i][n])
                elif n==10:
                    weight_concat_f.append(collected_features_all_folds[i][n])
                elif n==11:
                    fused_f.append(collected_features_all_folds[i][n])

      
        eeg_feat = np.concatenate(eeg_feat, axis=0)
        eeg_feat = eeg_feat.reshape(eeg_feat.shape[0], -1)
        ecg_feat = np.concatenate(ecg_feat, axis=0)
        ecg_feat = ecg_feat.reshape(ecg_feat.shape[0], -1)
        rsp_feat = np.concatenate(rsp_feat, axis=0)
        rsp_feat = rsp_feat.reshape(rsp_feat.shape[0], -1)
        ppg_feat = np.concatenate(ppg_feat, axis=0)
        ppg_feat = ppg_feat.reshape(ppg_feat.shape[0], -1)
        gsr_feat = np.concatenate(gsr_feat, axis=0)
        gsr_feat = gsr_feat.reshape(gsr_feat.shape[0], -1)
        weight_concat_f = np.concatenate(weight_concat_f, axis=0)
        weight_concat_f = weight_concat_f.reshape(weight_concat_f.shape[0], -1)
        fused_f = np.concatenate(fused_f, axis=0)
        fused_f = fused_f.reshape(fused_f.shape[0], -1)


        if args.Scaler == True:
            # 데이터 전처리: 표준화
            scaler = StandardScaler()
            eeg_feat = scaler.fit_transform(eeg_feat)
            ecg_feat = scaler.fit_transform(ecg_feat)
            rsp_feat = scaler.fit_transform(rsp_feat)
            ppg_feat = scaler.fit_transform(ppg_feat)
            gsr_feat = scaler.fit_transform(gsr_feat)
            weight_concat_f = scaler.fit_transform(weight_concat_f)
            fused_f = scaler.fit_transform(fused_f)


        # feature를  stack
        features = [eeg_feat, ecg_feat, rsp_feat, ppg_feat, gsr_feat, weight_concat_f, fused_f]

        """
        KPCA
        """
        features = [x.reshape(x.shape[0], -1) for x in features]  # Flatten the input data
        all_performance = [[] for _ in range(7)]  # 각 피처별 에러율 리스트를 저장할 리스트 초기화

        for i  in range(len(features)):
            # 분류기 학습을 위해 data split
            X_train, X_test, y_train, y_test = train_test_split(features[i], labels, test_size=args.test_size, random_state=42)

            eigenvector_counts = [1, 5, 10, 15, 20, 30, 50] # [5, 15, 30, 50, 100, 250]

            current_performance = []  # 현재 피처에 대한 에러율 저장할 리스트 초기화

            for n in eigenvector_counts:
                # 2단계: Kernel PCA 적용
                kpca = KernelPCA(n_components=n, kernel='rbf', eigen_solver='dense', random_state=42)
                X_train_kpca = kpca.fit_transform(X_train)
                X_test_kpca = kpca.transform(X_test)
                
                # 3단계: 분류 및 에러율 계산
                if args.classifier == "NeuralNetwork":
                    # 예: 은닉층이 2개이고 각각의 층에 뉴런이 100개인 MLP 신경망 구성
                    if n == 1 or n == 5:
                        hidden_layer_size = (n, n) # ori -> (1,2) (5,2)
                    elif n == 15: 
                        hidden_layer_size = (n, 8)
                    else: # 10, 20, 30, 50
                        hidden_layer_size = (n, n//2)

                    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_size, activation='identity', solver='adam', max_iter=1000,  learning_rate_init=0.001, learning_rate='adaptive', random_state=42)
                elif args.classifier == "LogisticRegression":
                    clf = LogisticRegression(random_state=42,max_iter=1000)
                elif args.classifier == "SVM":
                    clf = svm.SVC(random_state=42,max_iter=1000)
                clf.fit(X_train_kpca, y_train)
                y_pred = clf.predict(X_test_kpca)
                if args.metric == 'error':
                    performance = 1 - accuracy_score(y_test, y_pred)
                elif args.metric == 'bacc':
                    performance = balanced_accuracy_score(y_test, y_pred)
                elif args.metric == 'f1':
                    performance = f1_score(y_test, y_pred, average='weighted')
     
                # current_errors.append(error_rate)  # 현재 피처에 대한 에러율 추가
                current_performance.append(performance)
            all_performance[i] = current_performance  # 각 피처 별 에러율 리스트 갱신

        layer0_errors = [sum(x)/len(x) for x in zip(*all_performance[:5])]
        layer1_errors = all_performance[5]
        layer2_errors = all_performance[6]

        layer0_errors_list.append(layer0_errors)
        layer1_errors_list.append(layer1_errors)
        layer2_errors_list.append(layer2_errors)


    # layer0_errors_list 평균 계산
    round_layer0 =  [sum(x)/len(x) for x in zip(*layer0_errors_list)]
    round_layer1 =  [sum(x)/len(x) for x in zip(*layer1_errors_list)]
    round_layer2 =  [sum(x)/len(x) for x in zip(*layer2_errors_list)]

    std_layer0 = [np.std(x) for x in zip(*layer0_errors_list)]
    std_layer1 = [np.std(x) for x in zip(*layer1_errors_list)]
    std_layer2 = [np.std(x) for x in zip(*layer2_errors_list)]

    # layer0_sub, layer0_add = calculate_bounds(round_layer0, std_layer0)
    # layer1_sub, layer1_add = calculate_bounds(round_layer1, std_layer1)
    # layer2_sub, layer2_add = calculate_bounds(round_layer2, std_layer2)

    import pandas as pd

    # CSV 파일로 저장할 데이터프레임을 생성
    data = {
        'Eigenvector Count': eigenvector_counts,
        'Average Layer 0': round_layer0,
        'Average Layer 1': round_layer1,
        'Average Layer 2': round_layer2,
        'Std Layer 0': std_layer0,
        'Std Layer 1': std_layer1,
        'Std Layer 2': std_layer2
    }

    df = pd.DataFrame(data)

    # CSV 파일로 저장
    res_dir = '/opt/workspace/Seohyeon/Journal/DMFMS/result_viz_errors/'
    file_name = f'{args.classifier}_{args.metric}_by_layer.csv'  # 원하는 파일 이름으로 변경하세요.
    if args.Scaler:
        file_path = f'{res_dir}/scaled_{file_name}'
    else:
        file_path = f'{res_dir}/{file_name}'

    df.to_csv(file_path, index=False)

    print(f'Data saved to {file_path}')

"""    # 에러율을 그래프로 표시
    plt.figure(figsize=(10, 6))

    plt.plot(eigenvector_counts, round_layer0, marker='o', linestyle='-', label=f'Average of Input (Layer 0)', color='limegreen') # average error rate
    # plt.fill_between(eigenvector_counts, layer0_sub, layer0_add, color='limegreen', alpha=0.2)

    plt.plot(eigenvector_counts, round_layer1, marker='o', linestyle='-', label=f'Layer 1', color='dodgerblue')
    # plt.fill_between(eigenvector_counts, layer1_sub, layer1_add, color='dodgerblue', alpha=0.2)

    plt.plot(eigenvector_counts, round_layer2, marker='o', linestyle='-', label=f'Layer 2', color='violet')
    # plt.fill_between(eigenvector_counts, layer2_sub, layer2_add, color='violet', alpha=0.2)

    plt.title('Errors of kPCA Components by Layer')
    plt.xticks(eigenvector_counts, eigenvector_counts)
    # X축과 Y축에 대한 그리드를 점선으로 설정
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    # Y축을 실수 형식으로 설정
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.xlabel('The Number of Principal Components') # number of eigenvector
    plt.ylabel(f'{args.metric}')
    # plt.yscale('log')  # Log scale for better visibility
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    RES_DIR = f'/opt/workspace/Seohyeon/Journal/DMFMS/result_viz_errors/{args.classifier}/'
    if args.Scaler == True:
        plt.savefig(f'{RES_DIR}/scaled_avg_{args.metric}_allfold_ts{args.test_size}_ex3.png')
    else:
        plt.savefig(f'{RES_DIR}/avg_{args.metric}_allfold_ts{args.test_size}_ex3.png')
    plt.show()

"""


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

    parser = argparse.ArgumentParser(description='')

    parser.add_argument("--SEED", default=42)

    ### data_type 과 model_type 선택 필수
    parser.add_argument('--model_type', default='DMFMS_viz', choices=['DMFMS_viz'])
    parser.add_argument('--data_type', default='bl_2cl_misc5', choices=['bl_2cl_misc5'])

    parser.add_argument('--classifier', default='NeuralNetwork', choices=['LogisticRegression', 'SVM','NeuralNetwork'])
    parser.add_argument('--Scaler', default=False, choices=[True, False]) 
    parser.add_argument('--metric', default='f1', choices=['bacc', 'f1', 'error'])
    parser.add_argument('--test_size', default=0.2, help='test size') # 
    
    ###################################### FIX 
    parser.add_argument('--selection_loss_type', default='CE', choices=['CE', 'Focal']) # SOTA = CE
    parser.add_argument('--postprocessor', default='mls', choices=['msp', 'mls', 'ebo']) # SOTA = msp
    parser.add_argument('--fusion_type', default='average' , choices=['concat', 'sum', 'average', 'matmul']) # SOTA = average
    parser.add_argument('--temp', default=10, help='temperature scaling for ebo') # choice: 1.5, 0.1, 10
    parser.add_argument('--backbone', default='EEGNet4', choices = ['EEGNet4', 'DeepConvNet', 'ResNet8', 'ResNet18'])
    parser.add_argument('--CRL_user', default=True, choices=[True, False]) 
    parser.add_argument('--scaling', default='sigmoid', choices=['softmax', 'sigmoid', 'none']) # SOTA기준 성능: sigmoid > none > softmax
    parser.add_argument('--rank_weight', default=1, type=float, help='Rank loss weight') # SOTA = 1
    parser.add_argument('--BATCH', default=16, help='Batch Size') # original 16, set 32
    parser.add_argument('--EPOCH', default=100, help='Epoch') # original: 50, set: 10
    parser.add_argument('--optimizer', default='AdamW', help='Optimizer') 
    parser.add_argument('--lr', default=0.002, help='Adam Learning Rate') # original: 1e-4
    parser.add_argument('--scheduler', default='CosineAnnealingLR', choices=['CosineAnnealingLR', 'StepLR', 'ReduceLROnPlateau']) # SOTA = CosineAnnealingLR
    parser.add_argument('--step_size', default=500, help='step size for StepLR scheduler')
    parser.add_argument('--freq_time', default=750, help='frequency(250)*time window(3)')
    parser.add_argument('--in_dim', default=[28,1,1,1,1], choices=[[28], [28,1], [28,1,1,1,1]], help='여기서는 사용 안함. 효과 x')
    parser.add_argument('--n_channels', default=28)
    parser.add_argument('--n_classes', default=2)

    args = parser.parse_args()

    seed_everything(args.SEED)


    # Data Generation at first time 
    if not os.path.exists(os.path.join(DATASET_DIR, f'{args.data_type}')):
       data_generate(args)

    if args.model_type == 'DMFMS_viz':
        from Models.DMFMS_viz import Net

    Inference(args)

