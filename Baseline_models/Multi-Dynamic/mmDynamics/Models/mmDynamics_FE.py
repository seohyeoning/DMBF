""" 
 *********** NOTICE ************  
  - classifier 뒤에 F.softmax 붙여야 할지 말지 고민. 
  - 붙이게 되면 loss가 줄지만, TCP 학습에 사용하는 TCPLogit도 softmax가 안되어있어서 학습 로스차이가 너무 크게 날 수도 있음
  - 그렇게 되면 모듈간의 학습 밸런스가 달라질 것 같음.. 
  - 우선은 원본 코드기반으로 수행하기
  - 문제점은 L1 패널티가 상대적으로 값이 매우 작음 (다른 로스들 300이면, L1정규화 놈은 0.5 정도) => sparsity 학습이 제대로 안될 것 같음

       Componets of the model
모달리티 수 자동으로 변경되는 general한 최종 모델.

Information encoder --> EEGNet4로 변경

refered by. https://github.com/TencentAILabHealthcare/mmdynamics
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from Helpers.Variables import device

def matmul(A, B, bs):
    batch_size = bs
    res_list = []
    for i in range (batch_size):
        c = torch.matmul(A[i],B[i].transpose(0,1))
        res_list.append(c)
    res = torch.stack(res_list)
    return res

def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)

class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x) 
        return x
    
class EEGNet4(nn.Module): # 4 temporal, 2 spatial per temporal
    def __init__(self, args, mod, track_running=True):
        super(EEGNet4, self).__init__()
        self.args = args
        if mod == 0: ## only EEG
            input_ch = args.n_channels
        else:        ## other
            input_ch = 1 
        self.modal_index = mod
        self.n_classes = args.n_classes
        freq = args.freq_time #################### num_seg = frequency*window size

        self.convnet = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(1, freq//2), stride=1, bias=False, padding=(0 , freq//4)),
            nn.BatchNorm2d(4, track_running_stats=track_running),
            nn.Conv2d(4, 8, kernel_size=(input_ch, 1), stride=1, groups=4),
            nn.BatchNorm2d(8, track_running_stats=track_running),
            nn.ELU(),
            # nn.AdaptiveAvgPool2d(output_size = (1,265)),
            nn.AvgPool2d(kernel_size=(1,4)),
            nn.Dropout(p=0.25),
            nn.Conv2d(8, 8, kernel_size=(1,freq//4),padding=(0,freq//4), groups=8),
            nn.Conv2d(8, 8, kernel_size=(1,1)),
            nn.BatchNorm2d(8, track_running_stats=track_running),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p=0.25),
            )
    
    def forward(self, x):
        # select the data from modality index
        if len(x.shape)==2: # ch=1인 모달리티는 unsqueeze 2번 수행 (bs, 1,1,sr*sec) 
            x = x.unsqueeze(dim=1) 
        x = x.unsqueeze(dim=1) 
        out = self.convnet(x)

        return out


class MMDynamic(nn.Module):
    def __init__(self, args, in_channel, hidden_dim, dropout): # in_channel 형태 = [28, 1] # m1의 채널수, m2의 채널수 리스트
        super().__init__()
        in_channel = [x for x in in_channel]

        self.modalities = len(in_channel) 
        self.classes = args.n_classes
        self.dropout = dropout
        self.bs = args.BATCH
        self.in_dim = args.in_dim
        self.sr = args.freq_time
        self.args = args

        self.FeatureEncoder = nn.ModuleList([EEGNet4(args, mode) for mode in range(self.modalities)])

        self.TCPConfidenceLayer = nn.ModuleList([LinearLayer(64, 1) for mode in range(self.modalities)])
        self.TCPClassifierLayer = nn.ModuleList([LinearLayer(64, self.classes) for mode in range(self.modalities)])

        self.MMClasifier = []
        linear_layers = len(self.in_dim) * [64]
        self.MMClasifier.append(LinearLayer(sum(linear_layers), self.classes))
        self.MMClasifier = nn.Sequential(*self.MMClasifier)


    def forward(self, data_list, label=None, infer=False):
        data_list = data_resize(self, data_list)
        
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        Feature, feature, feat_2d, TCPLogit, TCPConfidence = dict(), dict(), dict(), dict(), dict()

        for modal in range(self.modalities):
            Feature[modal] = self.FeatureEncoder[modal](data_list[modal]).squeeze(dim=2)

        # re-weight를 위한 alpha score 생성
        Features = torch.cat(list(Feature.values()), dim=1)  # 각 모달들의 피쳐 연결
        FeatureInfo = torch.sigmoid(Features) # 전체 모달 고려해서 모달간의 중요도 스코어 뽑음 (16, 40, 46) : 모달간의 중요도 가중치 적용
        alpha = torch.split(FeatureInfo, split_size_or_sections=8, dim=1)  # 각 피쳐에 곱해주기 위해 다시 모달 별로 나눔 (16, 8, 46) 

        for modal in range(self.modalities):
            feature[modal] = matmul(alpha[modal], Feature[modal], self.bs) # re-weighting
            feature[modal] = F.relu(feature[modal])
            feature[modal] = F.dropout(feature[modal], self.dropout, training=self.training)
            feat_2d[modal] = feature[modal].reshape(self.bs, -1)

            TCPLogit[modal] = self.TCPClassifierLayer[modal](feat_2d[modal])
            TCPConfidence[modal] = self.TCPConfidenceLayer[modal](feat_2d[modal])
            # layer norm 추가?!
            feat_2d[modal] = feat_2d[modal] * TCPConfidence[modal]

        MMfeature = torch.cat([i for i in feat_2d.values()], dim=1) # bs 기준으로 합함 (bs, 320)
        MMlogit = self.MMClasifier(MMfeature)
 
        if infer:
            return MMlogit
        
        """
        원본 코드
        """
        if self.args.loss_change == False:
            label = label.squeeze()
            MMLoss = torch.mean(criterion(MMlogit, label))
            label_indices = torch.nonzero(label)[:, 1]
            MMLoss_list = []

            for m in range(self.modalities): # 전체 모달이 합쳐져 만들어진 MMLogit + 각 모달마다의 TCP 학습을 위해 각각 계산 
                MMLoss = MMLoss+torch.mean(alpha[m]) #  FeatureInfo score의 sparsity 촉진을 위한 L1 정규화
                pred = F.softmax(TCPLogit[m], dim=1) # TCP신뢰도 학습을 위해 TCP 햇 유사하게 만드는 학습

                label = label.to(torch.int64)
                p_target = torch.gather(input=pred, dim=1, index=label_indices.view(-1, 1))  # TCP(p_target): 모달리티 별 분류기 f의 real label에 해당하는 predictive probability
                label = label.to(torch.float64)

                confidence_loss = torch.mean(F.mse_loss(TCPConfidence[m], p_target)+criterion(TCPLogit[m], label)) # 신뢰도 loss = TCP햇과 TCP 유사하도록 학습 + 만들어진 TCP로짓 기반 분류 학습 로스
                MMLoss = MMLoss+confidence_loss
                MMLoss_list.append(MMLoss.to(device))

            MMLoss_tensor = torch.stack(MMLoss_list) 


            """
            모든 loss구하기 전에 로짓에 F.softmax 수행
            """
        elif self.args.loss_change == True:

            MMLoss = torch.mean(criterion(F.softmax(MMlogit), label))
            label_indices = torch.nonzero(label)[:, 1]
            MMLoss_list = []
        
            for m in range(self.modalities): # 전체 모달이 합쳐져 만들어진 MMLogit + 각 모달마다의 TCP 학습을 위해 각각 계산 
                MMLoss = MMLoss+torch.mean(alpha[m]) #  FeatureInfo score의 sparsity 촉진을 위한 L1 정규화
                pred = F.softmax(TCPLogit[m], dim=1) # TCP신뢰도 학습을 위해 TCP 햇 유사하게 만드는 학습

                label = label.to(torch.int64)
                p_target = torch.gather(input=pred, dim=1, index=label_indices.view(-1, 1))  # TCP(p_target): 모달리티 별 분류기 f의 real label에 해당하는 predictive probability
                label = label.to(torch.float64)

                confidence_loss = torch.mean(F.mse_loss(F.softmax(TCPConfidence[m]), p_target)+criterion(TCPLogit[m], label)) # 신뢰도 loss = TCP햇과 TCP 유사하도록 학습 + 만들어진 TCP로짓 기반 분류 학습 로스
                MMLoss = MMLoss+confidence_loss
                MMLoss_list.append(MMLoss.to(device))

            MMLoss_tensor = torch.stack(MMLoss_list) 

        return MMLoss_tensor, MMlogit
    
    def infer(self, data_list):
        MMlogit = self.forward(data_list, infer=True)
        return MMlogit


def data_resize(self, data_list):
    new_data_list = []
    for i, dim in enumerate(self.in_dim):
        new_data_list.append(data_list[i])  
    data_list = new_data_list
    return data_list      


"""
모델 구조 체크용!!
NOTICE: 사용 시, mmDynamics의 forward에서 loss 관련된 부분은 다 주석처리 해야함
"""

if __name__ == "__main__":  
    import os
    import numpy as np
    import torch
    import argparse

    import torch.nn as nn

    import pandas as pd
    from pathlib import Path

    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    DATASET_DIR = '/opt/workspace/Seohyeon/NEW_PIPE/Dataset'
    METRICS = ['loss', 'acc', 'bacc', 'f1', 'preci', 'recall']

    parser = argparse.ArgumentParser(description='MS_mmdynamic')
    parser.add_argument("--SEED", default=42)
    ### data 선택
    parser.add_argument('--data_type', default='bl_2cl_misc5', choices=['bl_2cl_misc5', 'bl_3cl', 'VLinTS_bl_3cl'])
    parser.add_argument('--model_type', default='model_FE', choices=['model_FE', 'model_basic'])
    parser.add_argument('--loss_change', default=True, choices=[True, False], help='original loss --> True, changed loss by F.softmax --> False')

    ### early stopping on-off
    parser.add_argument('--early_stop', default=False, choices=[True, False])

    ########## 실험 하이퍼 파라미터 설정 
    parser.add_argument('--BATCH', default=16, help='Batch Size') # original 16, set 32
    parser.add_argument('--EPOCH', default=100, help='Epoch') # original: 50, set: 10
    parser.add_argument('--optimizer', default='AdamW', help='Optimizer') 
    parser.add_argument('--lr', default=0.002, help='Adam Learning Rate') # original: 1e-4
    parser.add_argument('--step_size', default=500, help='step size for StepLR scheduler')
    parser.add_argument('--freq_time', default=750, help='frequency(250)*time window(3)')
    parser.add_argument('--in_dim', default=[28], choices=[[28], [28,1], [28,1,1,1,1]], help='num of channel list for each modality')
    parser.add_argument('--hid_dim', default=[200], choices=[[500], [300]])

    ########## 클래스 및 채널 수 지정
    parser.add_argument('--n_channels', default=28)
    parser.add_argument('--n_classes', default=2)
    
    args = parser.parse_args()  

    ###### 모델 생성
    my_model = MMDynamic(args, in_channel=args.in_dim, hidden_dim=args.hid_dim, dropout=0.5).to(device)
    
    from torchsummary import summary
    summary(my_model, (16,28,750))

    # from torchviz import make_dot

    # import torch
    # x = torch.zeros(16, 1, 28, 750).to(device)
    # make_dot(my_model(x), params=dict(list(my_model.named_parameters())))


