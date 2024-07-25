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
from mpl_toolkits.mplot3d import Axes3D  # 3D plot을 위해 필요
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as T
from Models.Model_v4_feature_map import Net
import umap
  
def plot_features_3d(feat_3d_list, labels, save_dir):
    label_colors = {0: 'skyblue', 1: 'navy'}  # 수정된 색상
    label_names = {0: 'alert', 1: 'drowsy'}
    label_markers = {0: 'o', 1: '^'}  # 새로 추가된 마커 설정

    for i in range(12):
        X_3d = feat_3d_list[i]
        title = ['EEG feature', 'ECG feature', 'RSP feature', 'PPG feature', 'GSR feature',
                'Weighted EEG feature', 'Weighted ECG feature', 'Weighted RSP feature',
                'Weighted PPG feature', 'Weighted GSR feature', 'Weighted Concatenated feature',
                'Fused feature'][i]
        
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # 라벨별로 데이터 포인트를 그립니다.
        for label in np.unique(labels):
            idx = np.where(labels == label)
            ax.scatter(X_3d[idx, 0], X_3d[idx, 1], X_3d[idx, 2], 
                       label=label_names[label], 
                       color=label_colors[label], 
                       marker=label_markers[label])  # marker 추가

        plt.title('UMAP of the ' + title)
        plt.legend()  # 범례 추가


        S_directory = os.path.join(save_dir, title)
        plt.savefig(S_directory)
        plt.close()

def plot_features_2d(feat_2d_list, labels, save_dir):
    label_colors = {0: 'skyblue', 1: 'navy'}  # 수정된 색상
    label_names = {0: 'alert', 1: 'drowsy'}
    label_markers = {0: 'o', 1: '^'}  # 새로 추가된 마커 설정



    titles = ['EEG (Layer 0)', 'ECG (Layer 0)', 'RSP (Layer 0)', 'PPG (Layer 0)', 'GSR (Layer 0)',
              'Weighted EEG feature', 'Weighted ECG feature', 'Weighted RSP feature',
              'Weighted PPG feature', 'Weighted GSR feature', 'Layer 1',
              'Layer 2']

    # i가 0, 1, 2, 3, 4일 때의 피쳐들을 한 figure에 subplot으로 그립니다.
    plt.figure(figsize=(20, 14))
    for i in range(5):
        plt.subplot(2, 3, i+1)
        X_2d = feat_2d_list[i]
        
        for label in np.unique(labels):
            idx = np.where(labels == label)
            plt.scatter(X_2d[idx, 0], X_2d[idx, 1], 
                        label=label_names[label], 
                        color=label_colors[label], 
                        marker=label_markers[label])
        plt.title('UMAP of the ' + titles[i])
        plt.legend()
        plt.yticks([])
        plt.xticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Input_feature.png"))
    plt.close()

    # i가 10, 11일 때의 피쳐들을 별도의 figure에 subplot으로 그립니다.
    plt.figure(figsize=(20, 7))
    for i in range(10, 12):
        plt.subplot(1, 2, i-9)
        X_2d = feat_2d_list[i]
        for label in np.unique(labels):
            idx = np.where(labels == label)
            plt.scatter(X_2d[idx, 0], X_2d[idx, 1], 
                        label=label_names[label], 
                        color=label_colors[label], 
                        marker=label_markers[label])
        plt.title('UMAP of the ' + titles[i])
        plt.legend()
        plt.yticks([])
        plt.xticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Layer1_2.png"))
    plt.close()

def Inference(args):

    ### 모델 load 경로 변경
    Model_Path = "/opt/workspace/Seohyeon/NEW_PIPE/Final/res/Model_OURS_sigmoid_EEGNet4/average_bs16/CELoss_msp_CosineAnnealingLR_AdamW_0.002/0/"
   

    ################## 피험자 별 경로 설정 및 실험 시작 ##################
    ts_fold = pd.DataFrame(columns=METRICS)
    num_fold = 4
    include_sbj = [6] # 2cl_5 전용 5,6,8,9,10,13,15,16,17,18,19,20,21


    if args.tool == 'umap':
        new_RESD = f"/opt/workspace/Seohyeon/Journal/DMFMS/result_viz_{args.tool}/neighbor{args.n_neighbors}_{args.metric}_{args.min_dist}/"
        
        for subj in range(1,24): # 1,24
            if subj not in include_sbj:
                    continue
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
                        data, target = datas['data'], datas['label'].cpu()
    
                        # 모델에서 feature maps을 추출
                        feat_list, weight_feature_list, weight_concat_f, fused_f = my_model.infer(data)

                        # 추출된 특징을 리스트에 추가
                        for i, feat in enumerate(feat_list + weight_feature_list + [weight_concat_f, fused_f]):
                            collected_features[i].append(feat.cpu().numpy())  # GPU 메모리에 있을 경우 CPU로 이동
                        # 레이블을 수집
                        collected_labels.append(target.cpu().numpy())

                # 레이블 데이터 연결
                collected_labels = np.concatenate(collected_labels, axis=0)
                collected_labels = np.argmax(collected_labels, axis=1)
                if collected_labels.max() == 0:
                    continue
                else:
                    # 각 특징별로 모든 배치 데이터를 연결                     
                    for i, feats in enumerate(collected_features):
                        collected_features[i] = np.concatenate(feats)
            
                    eeg_feat, ecg_feat, rsp_feat, ppg_feat, gsr_feat = collected_features[0], collected_features[1], collected_features[2], collected_features[3], collected_features[4]
                    w_eeg_feat, w_ecg_feat, w_rsp_feat, w_ppg_feat, w_gsr_feat = collected_features[5], collected_features[6], collected_features[7], collected_features[8], collected_features[9]
                    weight_concat_f, fused_f = collected_features[10], collected_features[11]

                    if args.tool == 'tsne':
                        # t-SNE 모델 생성, 2차원으로 축소
                        tsne = TSNE(n_components=args.dim, perplexity=args.perplexity_num)

                        n_samples = eeg_feat.shape[0]
                        eeg_feat_3d = tsne.fit_transform(eeg_feat.reshape(n_samples,-1))
                        ecg_feat_3d = tsne.fit_transform(ecg_feat.reshape(n_samples,-1))
                        rsp_feat_3d = tsne.fit_transform(rsp_feat.reshape(n_samples,-1))
                        ppg_feat_3d = tsne.fit_transform(ppg_feat.reshape(n_samples,-1))
                        gsr_feat_3d = tsne.fit_transform(gsr_feat.reshape(n_samples,-1))
                        
                        w_eeg_feat_3d = tsne.fit_transform(w_eeg_feat.reshape(n_samples,-1))
                        w_ecg_feat_3d = tsne.fit_transform(w_ecg_feat.reshape(n_samples,-1))
                        w_rsp_feat_3d = tsne.fit_transform(w_rsp_feat.reshape(n_samples,-1))
                        w_ppg_feat_3d = tsne.fit_transform(w_ppg_feat.reshape(n_samples,-1))
                        w_gsr_feat_3d = tsne.fit_transform(w_gsr_feat.reshape(n_samples,-1))

                        weight_concat_f_3d = tsne.fit_transform(weight_concat_f.reshape(n_samples,-1))
                        fused_f_3d = tsne.fit_transform(fused_f.reshape(n_samples,-1))

                        feat_3d_list = [eeg_feat_3d, ecg_feat_3d, rsp_feat_3d, ppg_feat_3d, gsr_feat_3d, w_eeg_feat_3d, w_ecg_feat_3d, w_rsp_feat_3d, w_ppg_feat_3d, w_gsr_feat_3d, weight_concat_f_3d, fused_f_3d]
                    
                    elif args.tool == 'umap':
                        reducer = umap.UMAP(n_components=args.dim, n_neighbors=args.n_neighbors, min_dist=args.min_dist, metric=args.metric)
                        n_samples = eeg_feat.shape[0]

                        eeg_feat_3d = reducer.fit_transform(eeg_feat.reshape(n_samples,-1))
                        ecg_feat_3d = reducer.fit_transform(ecg_feat.reshape(n_samples,-1))
                        rsp_feat_3d = reducer.fit_transform(rsp_feat.reshape(n_samples,-1))
                        ppg_feat_3d = reducer.fit_transform(ppg_feat.reshape(n_samples,-1))
                        gsr_feat_3d = reducer.fit_transform(gsr_feat.reshape(n_samples,-1))
                        
                        w_eeg_feat_3d = reducer.fit_transform(w_eeg_feat.reshape(n_samples,-1))
                        w_ecg_feat_3d = reducer.fit_transform(w_ecg_feat.reshape(n_samples,-1))
                        w_rsp_feat_3d = reducer.fit_transform(w_rsp_feat.reshape(n_samples,-1))
                        w_ppg_feat_3d = reducer.fit_transform(w_ppg_feat.reshape(n_samples,-1))
                        w_gsr_feat_3d = reducer.fit_transform(w_gsr_feat.reshape(n_samples,-1))

                        weight_concat_f_3d = reducer.fit_transform(weight_concat_f.reshape(n_samples,-1))
                        fused_f_3d = reducer.fit_transform(fused_f.reshape(n_samples,-1))


                        feat_3d_list = [eeg_feat_3d, ecg_feat_3d, rsp_feat_3d, ppg_feat_3d, gsr_feat_3d, w_eeg_feat_3d, w_ecg_feat_3d, w_rsp_feat_3d, w_ppg_feat_3d, w_gsr_feat_3d, weight_concat_f_3d, fused_f_3d]
                
                
                    if args.dim == 3:
                        plot_features_3d(feat_3d_list, collected_labels, save_dir)
                    elif args.dim == 2:
                        plot_features_2d(feat_3d_list, collected_labels, save_dir)




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

    parser.add_argument('--tool', default='umap', choices=['tsne', 'umap'])
    parser.add_argument('--min_dist', default=0.1, choices=[0.25, 0.1])
    parser.add_argument('--metric', default='cosine', choices=['euclidean','cosine','manhattan','correlation'])
    parser.add_argument('--dim', default=2, choices=[2,3])

    parser.add_argument('--n_neighbors', default=15, choices=[15, 35])
    parser.add_argument('--perplexity_num', default=30, type=int)



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

