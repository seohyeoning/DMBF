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

import matplotlib.pyplot as plt

import torchvision.transforms as T

from Models.Model_v4_feature_map import Net

def GEN_ConfMap_Graph(x_axis, MAP, SAMPLE_conf, target_values, save_dir):
        # Creating two y-axes
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        # ax3 = ax1.twinx()

        # Plotting MAP values on the first y-axis (left)
        ax1.plot(x_axis, MAP, label='Temporal attention score', color='b')
        ax1.set_xlabel('Sample Number', fontsize=14)
        ax1.set_ylabel('Temporal attention score', color='b', fontsize=14)
        ax1.tick_params(axis='y', labelcolor='b',  labelsize=12)
        ax1.set_yticks(np.arange(0, 1.0, 0.1))   

        # Plotting SAMPLE_conf values on the second y-axis (right)
        ax2.plot(x_axis, SAMPLE_conf.squeeze(), label='Confidence score', color='r')
        ax2.set_ylabel('Average of confidence score', color='r', fontsize=14)
        ax2.tick_params(axis='y', labelcolor='r', labelsize=12)

        # Plotting target values as bar graph on the third y-axis (right)
        # ax3.bar(x_axis, target_values, label='Target', color='g', alpha=0.5)
        # ax3.set_yticklabels([])

        plt.title('Graph of temporal attention score and confidence score', fontsize=16)

        # Combining legends from all three axes
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=13)

        # Save the figure
        plt.savefig(f'{save_dir}/Conf_Map_graph_sig.png')       

def plot_3d_surface(attention_map, name, save_dir):
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111, projection='3d')

    # X, Y 값은 attention 맵의 time points와 channels에 해당합니다.
    X = np.arange(attention_map.shape[1])
    Y = np.arange(attention_map.shape[0])
    X, Y = np.meshgrid(X, Y)
    Z = attention_map  # Z 값은 attention 가중치에 해당합니다.

    # 표면 그래프를 그립니다.
    surface = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

    # 레이블을 추가합니다.
    ax.set_xlabel('Temporal dimension')
    ax.set_ylabel('Spatial dimension')
    ax.set_zlabel('Attention Weights')
    ax.set_title(f'3D Surface Plot of {name} Attention Map')
    
    # 컬러바를 추가합니다.
    fig.colorbar(surface, shrink=0.5, aspect=5)

    # 그래프를 보여줍니다.
    plt.show()
    plt.savefig(f'{save_dir}/3d_plot_for_{name}.png')

def Inference(args):
    ##### SAVE DIR for .pt file
    if args.CRL_user == True:
        if args.postprocessor == 'ebo':
            model_dir = os.path.join (WD,'res', f'{args.model_type}_{args.scaling}_{args.backbone}/{args.fusion_type}_bs{args.BATCH}/{args.selection_loss_type}Loss_{args.postprocessor}_temp{args.temp}_{args.scheduler}_{args.optimizer}_{args.lr}') 
        else: # msp, mls
            model_dir = os.path.join (WD,'res',  f'{args.model_type}_{args.scaling}_{args.backbone}/{args.fusion_type}_bs{args.BATCH}/{args.selection_loss_type}Loss_{args.postprocessor}_{args.scheduler}_{args.optimizer}_{args.lr}') 
    
    else: # args.CRL_user == False
        if args.postprocessor == 'ebo':
            model_dir = os.path.join (WD,'res', f'woCRL/{args.model_type}_{args.scaling}_{args.backbone}/{args.fusion_type}/{args.selection_loss_type}Loss_{args.postprocessor}_temp{args.temp}_{args.scheduler}_{args.optimizer}_{args.lr}') 
        else: # msp, mls
            model_dir = os.path.join (WD,'res',  f'woCRL/{args.model_type}_{args.scaling}_{args.backbone}/{args.fusion_type}/{args.selection_loss_type}Loss_{args.postprocessor}_{args.scheduler}_{args.optimizer}_{args.lr}') 
    
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    res_flen = str(len(os.listdir(model_dir))) # 신규 폴더 구분용
    save_file = f'/{res_flen}/'

    ### 모델 load 경로 변경
    Model_Path = "/opt/workspace/Seohyeon/NEW_PIPE/Final/res_Ablation/Model_v4_sample_CRL_sigmoid_EEGNet4/average/CELoss_msp_CosineAnnealingLR_AdamW_0.002/1"
   

    ################## 피험자 별 경로 설정 및 실험 시작 ##################
    ts_fold = pd.DataFrame(columns=METRICS)
    num_fold = 4
    include_sbj = [6, 17] # 2cl_5 전용
    for subj in range(1,24): # 1,24
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
            data_load_dir = f'{Dataset_directory}/S{sbj}/fold{nf}'
            print(f'Loaded data from --> {DATASET_DIR}/S{sbj}/fold{nf}')

            # 결과 저장 경로 설정 2
            res_name = f'S{sbj}'
            nfoldname = f'fold{nf}'

            new_RESD = "/opt/workspace/Seohyeon/NEW_PIPE/Final/res_viz/"
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

            with torch.no_grad(): # .eval함수와 torch.no_grad함수를 같이 사용하는 경향
                for datas in test_loader:
                    data, target = datas['data'], datas['label'].cpu()
                    # Convert target to binary values
                    target_values = torch.argmax(target, dim=1)
                    # 모델에서 feature maps을 추출
                    tmp_map, sp_map, SAMPLE_conf = my_model.infer(data)

                    tmp_map, sp_map = tmp_map.cpu(), sp_map.cpu()
                    SAMPLE_conf = torch.sigmoid(SAMPLE_conf).unsqueeze(-1).cpu() 

                    # joint_map = tmp_map + sp_map.permute(0,2,1)
                    # plot_3d_surface(joint_map[0], name = 'Spatial-Temporal', save_dir=save_dir) # 샘플 0 의 3d surface plot
                    # plot_3d_surface(tmp_map[0], name = 'Temporal', save_dir=save_dir) # 샘플 0 의 3d surface plot
                    # plot_3d_surface(sp_map.permute(0,2,1)[0], name = 'Spatial', save_dir=save_dir) # 샘플 0 의 3d surface plot

                    temMAP = torch.mean(tmp_map, dim=(1,2))
                    # spaMAP = torch.mean(sp_map, dim=(1,2))
                    
                
                    # # Create a 1x16 grid of subplots
                    # fig2, axes2 = plt.subplots(1, 16, figsize=(28, 4),  sharex=True, sharey=True)  # Adjusted for a more elongated layout
                    # fig2.suptitle('temporal attention map', fontsize=20)

                    # for i, ax in enumerate(axes2.flat):
                    #     # Visualize each feature map
                    #     im = ax.imshow(MAP[i], cmap='viridis')
                    #     ax.axis('off')  # Turn off axis labels

                    # # Add x-axis legends
                    # for i in range(16):
                    #     axes2[i].text(0.5, -0.1, str(i), va='center', ha='center',  fontsize=15, transform=axes2[i].transAxes)

                    # # Remove spacing between subplots
                    # plt.subplots_adjust(wspace=0, hspace=0)

                    # # Position the color bar at the bottom of the figure
                    # plt.subplots_adjust(bottom=0.1)

                    # # Add a horizontal color bar at the bottom
                    # cbar_ax = fig2.add_axes([0.12, 0.1, 0.78, 0.02])  # Adjust axes to place the color bar at the bottom
                    # fig2.colorbar(im, cax=cbar_ax, orientation='horizontal')

                    # # Display the plot
                    # plt.show()

                    # plt.savefig(f'{save_dir}/3d_plot_for_joint_map.png')

                    '''
                    Plot the confidence score & spatial attention map
                    '''
                    x_axis = np.arange(16)
                    plt.figure(figsize=(20, 8))
                    GEN_ConfMap_Graph(x_axis, temMAP, SAMPLE_conf, target_values, save_dir)     


                    break

                    # # selected_feature_maps = [fmap.cpu().detach().numpy() for fmap in fused_feat]
                    # selected_feature_maps = [fmap.cpu().detach().numpy() for fmap in MAP]

                    # # Subplots 생성 (4x4 grid)
                    # fig, axes = plt.subplots(4, 4, figsize=(20, 20))

                    # for i, ax in enumerate(axes.flat):
                    #     # 각 feature map 시각화
                    #     im = ax.imshow(selected_feature_maps[i], cmap='viridis')
                    #     ax.set_title(f'Feature Map {i}')
                    #     ax.axis('off')  # 축 레이블 끄기

                    #     # 각 subplot에 대한 color bar 추가
                    #     plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                    # # Plot 표시
                    # plt.tight_layout()  # subplot 간의 간격 조정
                    # plt.show()

                    # plt.savefig(f'{save_dir}/temporal_map_per_sample.png')

                    """
                    # Recalculate global color limits for the common color bar
                    vmin = min(np.min(fm) for fm in selected_feature_maps)
                    vmax = max(np.max(fm) for fm in selected_feature_maps)
                
                    # Create a 1x16 grid of subplots
                    fig2, axes2 = plt.subplots(1, 16, figsize=(28, 4),  sharex=True, sharey=True)  # Adjusted for a more elongated layout
                    fig2.suptitle('temporal_map per Sample', fontsize=20)

                    for i, ax in enumerate(axes2.flat):
                        # Visualize each feature map
                        im = ax.imshow(selected_feature_maps[i], cmap='viridis', vmin=vmin, vmax=vmax)
                        ax.axis('off')  # Turn off axis labels

                    # Add x-axis legends
                    for i in range(16):
                        axes2[i].text(0.5, -0.1, str(i), va='center', ha='center',  fontsize=15, transform=axes2[i].transAxes)

                    # Remove spacing between subplots
                    plt.subplots_adjust(wspace=0, hspace=0)

                    # Position the color bar at the bottom of the figure
                    plt.subplots_adjust(bottom=0.1)

                    # Add a horizontal color bar at the bottom
                    cbar_ax = fig2.add_axes([0.12, 0.1, 0.78, 0.02])  # Adjust axes to place the color bar at the bottom
                    fig2.colorbar(im, cax=cbar_ax, orientation='horizontal')

                    # Display the plot
                    plt.show()

                    # Save the figure
                    horizontal_color_bar_path = os.path.join(save_dir, 'B4_STAM_relative.png')

                    # plt.savefig(horizontal_color_bar_path)

                    plt.close()  # Close the figure after saving to avoid display
  
                    horizontal_color_bar_path
                    """
                    break # Exit after the first batch
                    

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
    parser.add_argument('--data_type', default='bl_2cl_misc5', choices=['bl_2cl_misc5', 'BP_misc5'])
    parser.add_argument('--model_type', default='Model_v4_feature_map', choices=['Model_v4_sample_CRL', 'Model_v4_feature_map']) # HJU_FE_OUR

    ####################################실험 하이퍼 파라미터 설정##########################################################
    # msp, average, rank weight:1, Backbone network: EEGNet4
    parser.add_argument('--selection_loss_type', default='CE', choices=['CE', 'Focal']) # SOTA = CE
    parser.add_argument('--postprocessor', default='mls', choices=['msp', 'mls', 'ebo']) # SOTA = msp
    parser.add_argument('--fusion_type', default='average' , choices=['concat', 'sum', 'average', 'matmul']) # SOTA = average
    parser.add_argument('--temp', default=10, help='temperature scaling for ebo') # choice: 1.5, 0.1, 10

    ###################################### FIX 
    parser.add_argument('--backbone', default='EEGNet4', choices = ['EEGNet4', 'DeepConvNet', 'ResNet8', 'ResNet18'])
    parser.add_argument('--CRL_user', default=True, choices=[True, False]) 
    parser.add_argument('--scaling', default='sigmoid', choices=['softmax', 'sigmoid', 'none']) # SOTA기준 성능: sigmoid > none > softmax
    parser.add_argument('--rank_weight', default=1, type=float, help='Rank loss weight') # SOTA = 1
    parser.add_argument('--BATCH', default=16, help='Batch Size') # original 16, set 32
    parser.add_argument('--EPOCH', default=100, help='Epoch') # original: 50, set: 10
    parser.add_argument('--optimizer', default='AdamW', help='Optimizer') 
    parser.add_argument('--lr', default=0.002, help='Adam Learning Rate') # original: 1e-4
    parser.add_argument('--scheduler', default='CosineAnnealingLR', choices=['CosineAnnealingLR', 'StepLR', 'ReduceLROnPlateau']) # SOTA = CosineAnnealingLR
    ###############################################################################################################################

    parser.add_argument('--step_size', default=500, help='step size for StepLR scheduler')
    parser.add_argument('--freq_time', default=750, help='frequency(250)*time window(3)')
    parser.add_argument('--in_dim', default=[28,1,1,1,1], choices=[[28], [28,1], [28,1,1,1,1]], help='여기서는 사용 안함. 효과 x')

    ########## 클래스 및 채널 수 지정
    parser.add_argument('--n_channels', default=28)
    parser.add_argument('--n_classes', default=2)
    
    args = parser.parse_args()

    seed_everything(args.SEED)


    # Data Generation at first time 
    if not os.path.exists(os.path.join(DATASET_DIR, f'{args.data_type}')):
       data_generate(args)


    if args.backbone == 'EEGNet4':
        if args.model_type == 'Model_v4_sample_CRL':
            from Models.Model_v4_woSTAM import Net
        elif args.model_type == 'Model_v4_feature_map':    
            from Models.Model_v4_feature_map import Net


    Inference(args)