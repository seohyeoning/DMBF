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
from sklearn.manifold import TSNE
import numpy as np

import matplotlib.pyplot as plt

from matplotlib.ticker import ScalarFormatter
from sklearn.linear_model import LogisticRegression
from matplotlib.ticker import FormatStrFormatter
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from Models.Model_v4_feature_map import Net
from sklearn.metrics.pairwise import rbf_kernel
from scipy.linalg import eigh


# 상위 X%에 해당하는 값을 임계값으로 결정하는 함수
def find_threshold(data, percentile):
    # 상위 percentile%에 해당하는 값을 찾습니다.
    threshold = np.percentile(data, 100 - percentile)
    # 임계값을 넘는 첫 번째 인덱스를 찾습니다.
    threshold_index = np.argmax(data < threshold)
    return threshold_index, threshold

def plot_label_contributions(data, layer_name, color, percentile=5, y_lim=None):
    threshold_index, threshold = find_threshold(data, percentile)
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(data)), data, width=0.9, color=color, label=layer_name)
    plt.axvline(x=threshold_index, color='red', linestyle='--', label='Threshold')
    plt.text(threshold_index+3, max(y_lim)*0.8, f'Top {percentile}%\nThreshold',
             horizontalalignment='left', verticalalignment='center', color='red', fontsize=10)
    
    xtick=[0, 50, 100, 150, 200, 250, 300]
    current_xticks = list(np.array(xtick)) + [threshold_index]
    plt.xticks(sorted(set(current_xticks)))  

    plt.xlabel('Component number (i)')
    plt.ylabel('Label Contributions')
    plt.title('Average Label Contributions for each kPCA Component across all subjects')
    if y_lim:
        plt.ylim(y_lim)
    plt.legend()
    plt.savefig(f'/opt/workspace/Seohyeon/Journal/DMFMS/result_viz_label_contributions/AVG_Top5_label_contribution_{layer_name}.png')

def csv_label_contributions(data, layer_name, color, percentile=5, y_lim=None):
    threshold_index, threshold = find_threshold(data, percentile)
    import pandas as pd
    eigenvector_counts = np.array(range(0,len(data)))
    rows = len(data)  # The number of rows, based on the length of 'data'

    # Create an array with 'rows' rows and 1 column, filled with 'number'
    threshold_Arr = np.zeros((rows, )) # shape: (num_samples, 1)
    threshold_Arr[threshold_index,] = threshold
    # CSV 파일로 저장할 데이터프레임을 생성
    dataframe = {
        'Eigenvector Count': eigenvector_counts,
        'Label contribution': data,
        'threshold': threshold_Arr
    }

    df = pd.DataFrame(dataframe)

    # CSV 파일로 저장
    res_dir = '/opt/workspace/Seohyeon/Journal/DMFMS/result_viz_label_contributions/'
    file_name = f'CSV_label_contribution_{layer_name}.csv' 
    if args.Scaler:
        file_path = f'{res_dir}/scaled_{file_name}'
    else:
        file_path = f'{res_dir}/{file_name}'

    df.to_csv(file_path, index=False)

    print(f'Data saved to {file_path}')

def Inference(args):

    ### 모델 load 경로 변경
    Model_Path = "/opt/workspace/Seohyeon/NEW_PIPE/Final/res/Model_OURS_sigmoid_EEGNet4/average_bs16/CELoss_msp_CosineAnnealingLR_AdamW_0.002/0/"
   

    ################## 피험자 별 경로 설정 및 실험 시작 ##################
    num_fold = 4
    include_sbj = [5,6,8,9,10,13,15,16,17,18,19,20,21] # 2cl_5 전용 5,6,8,9,10,13,15,16,17,18,19,20,21

    new_RESD = f"/opt/workspace/Seohyeon/Journal/DMFMS/result_viz_label_contributions/"
    all_label_contribution_layer1 = []  # 전체 subject 및 fold에 대한 uTY_abs_values를 저장할 리스트
    all_label_contribution_layer2 = []  # 전체 subject 및 fold에 대한 uTY_abs_values를 저장할 리스트

    for subj in range(1,24): # 1,24
        if subj not in include_sbj:
                continue
        
        fold_uTY_abs_values_layer1 = []
        fold_uTY_abs_values_layer2 = []
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
                    feat_list, weight_feature_list, weight_concat_f, fused_f, last_feat = my_model.infer(data)

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
    
            eeg_feat, ecg_feat, rsp_feat, ppg_feat, gsr_feat = collected_features[0], collected_features[1], collected_features[2], collected_features[3], collected_features[4]
            # w_eeg_feat, w_ecg_feat, w_rsp_feat, w_ppg_feat, w_gsr_feat = collected_features[5], collected_features[6], collected_features[7], collected_features[8], collected_features[9]
            weight_concat_f, fused_f = collected_features[10], collected_features[11]
            
            
            if args.Scaler == True:
                # 데이터 전처리: 표준화
                scaler = StandardScaler()
                eeg_feat = scaler.fit_transform(eeg_feat.reshape(eeg_feat.shape[0], -1))
                ecg_feat = scaler.fit_transform(ecg_feat.reshape(eeg_feat.shape[0], -1))
                rsp_feat = scaler.fit_transform(rsp_feat.reshape(eeg_feat.shape[0], -1))
                ppg_feat = scaler.fit_transform(ppg_feat.reshape(eeg_feat.shape[0], -1))
                gsr_feat = scaler.fit_transform(gsr_feat.reshape(eeg_feat.shape[0], -1))
                weight_concat_f = scaler.fit_transform(weight_concat_f.reshape(eeg_feat.shape[0], -1))
                fused_f = scaler.fit_transform(fused_f.reshape(eeg_feat.shape[0], -1))
            
            # feature를  stack
            features = [eeg_feat, ecg_feat, rsp_feat, ppg_feat, gsr_feat, weight_concat_f, fused_f]
            name_list = ['EEG', 'ECG', 'RSP', 'PPG', 'GSR', 'Layer 1', 'Layer 2']
            color_list = ['dodgerblue','violet']
            """
            KPCA
            """
            labels = np.argmax(collected_labels.copy(), axis=1)
            if labels.max() == 0:
                continue

            else :
                features = [x.reshape(x.shape[0], -1) for x in features]  # Flatten the input data

                for f in [5,6]:
                    X = features[f].reshape(features[f].shape[0], -1)  # Flatten the input data
                    kpca = KernelPCA(n_components=None, kernel='rbf', random_state=42, eigen_solver='dense')
                    X_kpca = kpca.fit_transform(X)
                    uTY_abs_values = np.abs(X_kpca.T @ labels) # label contributions
                    if f == 5:
                        fold_uTY_abs_values_layer1.append(uTY_abs_values)
                    else:   
                        fold_uTY_abs_values_layer2.append(uTY_abs_values)
                        
        # 해당 subject의 fold별 uTY_abs_values를 전체 리스트에 추가
        all_label_contribution_layer1.extend(fold_uTY_abs_values_layer1)
        all_label_contribution_layer2.extend(fold_uTY_abs_values_layer2)

    # 전체 subject의 fold별 uTY_abs_values를 평균하여 최종 평균 uTY_abs_values를 계산
    fold_min_samples_layer1 = min(len(fold) for fold in all_label_contribution_layer1)
    avg_label_cont_layer1 = np.mean(
        [fold[:fold_min_samples_layer1] for fold in all_label_contribution_layer1], axis=0)
    fold_min_samples_layer2 = min(len(fold) for fold in all_label_contribution_layer2)
    avg_label_cont_layer2 = np.mean(
        [fold[:fold_min_samples_layer2] for fold in all_label_contribution_layer2], axis=0)
    avg_label_cont_list = [avg_label_cont_layer1, avg_label_cont_layer2]

    combined_data = np.concatenate((avg_label_cont_layer1, avg_label_cont_layer2))
    y_max = combined_data.max()
    # Layer 1과 Layer 2 데이터에 대한 그래프를 그립니다.
    csv_label_contributions(avg_label_cont_layer1, 'Layer 1', 'blue', percentile=5, y_lim=(0, y_max+1))
    csv_label_contributions(avg_label_cont_layer2, 'Layer 2', 'magenta', percentile=5, y_lim=(0, y_max+1))

"""    # 평균 uTY_abs_values를 막대 그래프로 시각화
    for i in [5,6]:
        plt.figure(figsize=(10, 6))
        data = avg_label_cont_list[i-5]
        plt.bar(range(len(data)), data, label=name_list[i], color=color_list[i-5])
        plt.xlabel("Component number (i)")
        plt.ylabel("Label Contributions")
        plt.title("Average Label Contributions for each kPCA Component across all subjects")
        plt.legend()

        # threshold_index, threshold = find_threshold_index(data)
        # plt.axvline(x=threshold_index, color='r', linestyle='--')
        gradients = np.abs(np.gradient(data))
        threshold_gradient = np.percentile(gradients, 10)  # 예를 들어 하위 10% 기울기
        # 임계값 이하로 떨어지는 첫 번째 지점 찾기
        threshold_index = np.argmax(gradients <= threshold_gradient)
        plt.axvline(x=threshold_index, color='r', linestyle='--', label='Noise Bed Start')
        plt.text(threshold_index, plt.ylim()[1] * 0.8, 'Dimensionality\nof the problem', 
        horizontalalignment='center', color='red')
        plt.fill_betweenx(plt.ylim(), threshold_index, len(data), color='grey', alpha=0.3, label='Noise Bed')
        plt.tight_layout()
        plt.show()
        plt.tight_layout()
        if args.Scaler == True:
            plt.savefig(f'/opt/workspace/Seohyeon/Journal/DMFMS/result_viz_label_contributions/avg_noise_bed_{name_list[i]}scaled.png')
        else:
            plt.savefig(f'/opt/workspace/Seohyeon/Journal/DMFMS/result_viz_label_contributions/avg_noise_bed_{name_list[i]}_90per_grad.png')
        plt.show()
    
    ### 한 Figure에 subplot 추가하여 막대 그래프 시각화
    fig, axs = plt.subplots(1, 2, figsize=(20, 6), sharex=True, sharey=True)

    for i, ax in enumerate(axs):
        ax.bar(range(len(avg_label_cont_list[i])), avg_label_cont_list[i], label=name_list[i+5], color=color_list[i])
        ax.set_xlabel("Component number (i)")
        ax.set_ylabel("Label Contributions")
        ax.set_title("Average Label Contributions for each kPCA Component across all subjects")
        ax.legend()
        
    plt.tight_layout()

    # Figure 저장 및 표시
    if args.Scaler == True:
        plt.savefig('/opt/workspace/Seohyeon/Journal/DMFMS/result_viz_label_contributions/avg_noise_bed_scaled.png')
    else:
        plt.savefig('/opt/workspace/Seohyeon/Journal/DMFMS/result_viz_label_contributions/avg_noise_bed_1.png')
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
    parser.add_argument('--Scaler', default=False, choices=[True, False]) 

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

