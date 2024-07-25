import os
import numpy as np
import torch
import argparse

import pandas as pd
from pathlib import Path
import time
import random
import seaborn as sns

from Helpers.Variables import device, METRICS, DATASET_DIR, WD, FILENAME_FOLDRES, FILENAME_MODEL, FILENAME_FOLDSUM
from Helpers.Drowsy_Dataloader import BIODataset, BIODataLoader 
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
from matplotlib.colors import ListedColormap, BoundaryNorm

from Models.DMBF_viz_Drowsy import Net
from sklearn.metrics.pairwise import rbf_kernel
from scipy.linalg import eigh

def Inference(args):


     
    ### 모델 load 경로 변경
    Model_Path = "/opt/workspace/Seohyeon/Journal/DMFMS/res/EEGNet4/Drowsy_multi_ALL/Model_OURS/0"
   

    ################## 피험자 별 경로 설정 및 실험 시작 ##################
    num_fold = 1
    include_sbj = [1,5,6,7,8,9,10,12,13,15,16,18,20,21,24,25,26,28,29,30,31] # 2cl_5 전용 5,6,8,9,10,13,15,16,17,18,19,20,21

    new_RESD = f"/opt/workspace/Seohyeon/Journal/DMFMS/result_viz_heatmap_LG_Drowsiness/all_class" # alert/drowsy 

    for subj in range(1,7): # 1,24
        if subj not in include_sbj:
                continue
        
        # 결과 저장 경로 설정 2
        res_name = f'S{subj}'
        
        save_dir = os.path.join(new_RESD, res_name)  # 최종 저장 경로 생성  
        Path(save_dir).mkdir(parents=True, exist_ok=True) 
        print(f"Saving results to ---> {save_dir}")   
        
        all_features = [[] for _ in range(3)] 
        all_W_features = [[] for _ in range(3)] 
        all_labels = []
        for nf in range(1, num_fold+1):

            nfoldname = f'fold{nf}'
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
            print(f'Loaded data from --> {data_load_dir}')
         


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
            collected_W_features = [[] for _ in range(3)]  # 3개의 모달리티 대한 특징 리스트
            collected_labels = []
            collected_features = [[] for _ in range(3)]  # 3개의 모달리티 대한 특징 리스트

                        
            with torch.no_grad(): # .eval함수와 torch.no_grad함수를 같이 사용하는 경향
                for datas in test_loader:
                    data, target = datas['data'], datas['label'].to(device)

                    # 모델에서 feature maps을 추출
                    feat_list, weight_feature_list = my_model.infer(data)

                    # 추출된 특징을 리스트에 추가
                    for i, feat in enumerate(feat_list):
                        collected_features[i].append(feat.cpu().numpy())

                    for i, feat in enumerate(weight_feature_list):
                        collected_W_features[i].append(feat.cpu().numpy())

                    collected_labels.append(target.cpu().numpy())

            # 각 특징과 레이블을 전체 리스트에 추가
            for i, feats in enumerate(collected_features):
                all_features[i].append(np.concatenate(feats, axis=0))

            for i, feats in enumerate(collected_W_features):
                all_W_features[i].append(np.concatenate(feats, axis=0))

            all_labels.extend(np.concatenate(collected_labels, axis=0))

        all_features = [np.concatenate(sublist, axis=0) for sublist in all_features]
        all_W_features = [np.concatenate(sublist, axis=0) for sublist in all_W_features]

        all_labels = np.array(all_labels)
        int_class = all_labels.argmax(axis=1)
        ## 레이블 별 feature 나누기
        label_0_indices = np.where(int_class == 0)[0]
        label_1_indices = np.where(int_class == 1)[0]

        if args.feat_type == "before_weight":
            features = all_features
        elif args.feat_type == "after_weight":
            features = all_W_features

        features_label_0 = [f[label_0_indices] for f in features] # alert
        features_label_1 = [f[label_1_indices] for f in features] # drowsy

        

        n_sample_0 = features_label_0[0].shape[0]
        eeg_feature_0 = features_label_0[0].reshape(n_sample_0, -1)
        ppg_feature_0 = features_label_0[1].reshape(n_sample_0, -1)
        ecg_feature_0 = features_label_0[2].reshape(n_sample_0, -1)

        n_sample_1 = features_label_1[0].shape[0]
        eeg_feature_1 = features_label_1[0].reshape(n_sample_1, -1)
        ppg_feature_1 = features_label_1[1].reshape(n_sample_1, -1)
        ecg_feature_1 = features_label_1[2].reshape(n_sample_1, -1)


        features_0 = np.vstack([np.mean(eeg_feature_0, axis=1), np.mean(ppg_feature_0, axis=1), np.mean(ecg_feature_0, axis=1)])
        features_1 = np.vstack([np.mean(eeg_feature_1, axis=1), np.mean(ppg_feature_1, axis=1), np.mean(ecg_feature_1, axis=1)])


        # Create a figure with two subplots
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(30, 10), 
                                gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.2})

        # Define the colormap
        cmap = "YlGnBu"
        
        # Define y-tick labels
        yticklabels = ["EEG", "PPG", "ECG"]

        # Find global min and max for the color bar
        vmin = min(np.min(features_0), np.min(features_1))
        vmax = max(np.max(features_0), np.max(features_1))

        # Plot heatmap for class 0
        sns.heatmap(features_0, cmap=cmap, ax=axes[0], cbar=False, vmin=vmin, vmax=vmax)
        axes[0].set_title("Feature Heatmap for class 0 (Alert)", fontsize=30)
        axes[0].set_yticklabels(yticklabels, fontsize=30)  # Set custom y-tick labels
        axes[0].set_xticks([])

        # Plot heatmap for class 1
        sns.heatmap(features_1, cmap=cmap, ax=axes[1], cbar=False, vmin=vmin, vmax=vmax)
        axes[1].set_title("Feature Heatmap for class 1 (Drowsy)", fontsize=30)
        axes[1].set_yticklabels(yticklabels, fontsize=30)  # Set custom y-tick labels
        axes[1].set_xticks([])
    
        cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7]) 
        # Create the color bar
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
                    cax=cbar_ax)
        cbar.ax.tick_params(labelsize=30)

        # Adjust layout to make room for the colorbar
        plt.tight_layout()

        # Save the figure
        plt.savefig(f"{save_dir}/1fold_modality_Heatmap.png")

        # Display the figure
        plt.show()



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
    parser.add_argument('--data_type', default='Drowsy', choices=['Drowsy'])
    parser.add_argument('--Scaler', default=False, choices=[True, False]) 

    ###################################### FIX 
    parser.add_argument('--selection_loss_type', default='CE', choices=['CE', 'Focal']) # SOTA = CE
    parser.add_argument('--postprocessor', default='msp', choices=['msp', 'mls', 'ebo']) # SOTA = msp
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

    parser.add_argument('--freq_time', default=600, help='frequency(250)*time window(3)')
    parser.add_argument('--in_dim', default=[32,1,1])
    parser.add_argument('--n_channels', default=32)
    parser.add_argument('--n_classes', default=2)

    parser.add_argument('--feat_type', default='before_weight', choices=['before_weight', 'after_weight'])

    args = parser.parse_args()

    seed_everything(args.SEED)


    Inference(args)

