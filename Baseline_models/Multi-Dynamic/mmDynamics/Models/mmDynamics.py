""" Componets of the model
모달리티 수 자동으로 변경되는 general한 최종 모델.
loss 문제있던 것도 변경함
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from Helpers.Variables import device

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

        self.FeatureInforEncoder = nn.ModuleList([LinearLayer(self.sr, self.sr) for mode in range(self.modalities)])
        self.TCPConfidenceLayer = nn.ModuleList([LinearLayer(hidden_dim[0]*in_channel[mode], 1) for mode in range(self.modalities)])
        self.TCPClassifierLayer = nn.ModuleList([LinearLayer(hidden_dim[0]*in_channel[mode], self.classes) for mode in range(self.modalities)])
        self.FeatureEncoder = nn.ModuleList([LinearLayer(self.sr, hidden_dim[0]) for mode in range(self.modalities)])

        self.MMClasifier = []
        linear_layers = [in_channel[i]*hidden_dim[0] for i in range(self.modalities)] 
        for layer in range(1, len(hidden_dim)-1):
            self.MMClasifier.append(LinearLayer(self.modalities*hidden_dim[0], hidden_dim[layer]))
            self.MMClasifier.append(nn.ReLU())
            self.MMClasifier.append(nn.Dropout(p=dropout))
        if len(self.MMClasifier):
            self.MMClasifier.append(LinearLayer(hidden_dim[-1], self.classes))
        else:
            self.MMClasifier.append(LinearLayer(sum(linear_layers), self.classes))
        self.MMClasifier = nn.Sequential(*self.MMClasifier)


    def forward(self, data_list, label=None, infer=False):
        data_list = data_resize(self, data_list)
        
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        FeatureInfo, feature, feat_2d, TCPLogit, TCPConfidence = dict(), dict(), dict(), dict(), dict()
        for modal in range(self.modalities):
            FeatureInfo[modal] = torch.sigmoid(self.FeatureInforEncoder[modal](data_list[modal]))
            feature[modal] = data_list[modal] * FeatureInfo[modal]
            feature[modal] = self.FeatureEncoder[modal](feature[modal])
            feature[modal] = F.relu(feature[modal])
            feature[modal] = F.dropout(feature[modal], self.dropout, training=self.training)
            feat_2d[modal] = feature[modal].reshape(self.bs, -1)

            TCPLogit[modal] = self.TCPClassifierLayer[modal](feat_2d[modal])
            TCPConfidence[modal] = self.TCPConfidenceLayer[modal](feat_2d[modal])
            
            feat_2d[modal] = feat_2d[modal] * TCPConfidence[modal]

        MMfeature = torch.cat([i for i in feat_2d.values()], dim=1) # bs 기준으로 합함 (bs, 6200)
        MMlogit = self.MMClasifier(MMfeature)
        
        if infer:
            return MMlogit
        label = label.squeeze()
        MMLoss = torch.mean(criterion(MMlogit, label))

        label_indices = torch.nonzero(label)[:, 1]

        MMLoss_list = []
        for m in range(self.modalities): # 전체 모달이 합쳐져 만들어진 MMLogit + 각 모달마다의 TCP 학습을 위해 각각 계산 
            MMLoss = MMLoss+torch.mean(FeatureInfo[m]) #  피쳐 정보도(시그모이드)기반 loss에 가중치 부여 
            pred = F.softmax(TCPLogit[m], dim=1) # TCP신뢰도 학습을 위해 TCP 햇 유사하게 만드는 학습

            label = label.to(torch.int64)
            p_target = torch.gather(input=pred, dim=1, index=label_indices.view(-1, 1))  # TCP와 TCP 햇 유사하게 만들기 위해 TCP 학습
            label = label.to(torch.float64)

            confidence_loss = torch.mean(F.mse_loss(TCPConfidence[m], p_target)+criterion(TCPLogit[m], label)) # 신뢰도 loss = TCP 신뢰도 학습 로스 + 만들어진 TCP로짓 기반 분류 학습 로스
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
    parser.add_argument('--model_type', default='model_basic', choices=['model_FE', 'model_basic'])
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