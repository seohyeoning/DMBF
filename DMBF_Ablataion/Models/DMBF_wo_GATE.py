import torch.nn as nn
import torch
import torch.nn.functional as F

import argparse

from .Backbone.EEGNet4 import EEGNet4
# from .Backbone.DeepConvNet import DeepConvNet
# from .Backbone.ResNet8 import ResNet8
# from .Backbone.ResNet18 import ResNet18

"""
<w/o GATE>

Confidence layer 삭제 및 Confidence score 사용 x
=> crl loss와 gate mechanism을 사용하지 않는 모델
"""

def average(out_list):
    return torch.mean(torch.stack(out_list, dim=1), dim=1)
    
def concat(out_list):
    return torch.cat(out_list, dim=1) 

def summation(out_list):
    return torch.sum(torch.stack(out_list, dim=1), dim=1)

class SpatialAttn(nn.Module):
    def __init__(self, args, channel_size, hidden_size):
        super(SpatialAttn, self).__init__()
        self.args = args
        self.fc = nn.Linear(channel_size, hidden_size)

    def forward(self, weighted_features): 
        concat_feature = weighted_features
        concat_feature = concat_feature.permute(0,2,1) # (16, 256, 5) : 채널에 대한 attention
        
        attention_weights = F.softmax(self.fc(concat_feature))


        weighted_sp = (concat_feature * attention_weights).permute(0,2,1)

        return weighted_sp
    
class TemporalAttn(nn.Module):
    def __init__(self, args, time_step_size, hidden_size):
        super(TemporalAttn, self).__init__()
        self.fc = nn.Linear(time_step_size, hidden_size)

    def forward(self, weighted_features):
        concat_feature = weighted_features
        
        attention_weights = F.relu(self.fc(concat_feature))   # ReLU to retain only positive values
        weighted_tmp = concat_feature * attention_weights

        return weighted_tmp


class Net(nn.Module):
    def __init__(self, args, device):
        super(Net, self).__init__()
        self.modalities = len(args.in_dim)

        self.args = args
        self.device = device
        self.n_classes = args.n_classes
        self.bs = args.BATCH
        self.in_dim = args.in_dim
        if args.data_type == 'Distraction' or args.data_type == 'Stress':
            self.hid_dim = [200, 600] # confidence layer -> classifier
            self.attn_dim = [25, 24] # tem -> spa
        elif args.data_type == "MS": 
            self.hid_dim = [400, 1840]
            self.attn_dim = [46, 40]
        elif args.data_type == "Drowsy":
            self.hid_dim = [296, 888]
            self.attn_dim = [37, 24]
        elif args.data_type == "manD":
            self.hid_dim = [384, 384*5]
            self.attn_dim = [48, 40]
   

        self.lossfn = torch.nn.CrossEntropyLoss() # reduction="mean" for MSLoss
        if args.backbone == 'EEGNet4':
            self.FeatureExtractor = nn.ModuleList([EEGNet4(args, m) for m in range(self.modalities)])

        self.TemporalAttn = TemporalAttn(args, time_step_size=self.attn_dim[0], hidden_size=self.attn_dim[0])
        self.SpatialAttn = SpatialAttn(args, channel_size=self.attn_dim[1], hidden_size=self.attn_dim[1])

        self.ConfidenceLayer = nn.ModuleList([nn.Linear(self.hid_dim[0], self.n_classes) for m in range(self.modalities)])
        self.MSclassifier = nn.Sequential(nn.Linear(self.hid_dim[1], self.n_classes))


    def forward(self, x, label=None, infer=False):
        data_list = data_resize(self, x)
        feat_dict = dict()

        for mod in range(self.modalities):
            feat_dict[mod] = self.FeatureExtractor[mod](data_list[mod]).squeeze(dim=2) # (16, 8, 46)

        feat_list = list(feat_dict.values())


        SAMPLE_conf =0
        """
        위에서 중요도 점수 뽑고나서, 모달리티의 상대적인 점수 주기위해 stack 후 softmax하면 그 효과가 사라지는 것이 아닌가?
        성능 확인해보고, softmax가 아닌 다른 방법으로 상대적 중요도 점수를 구해보자. 
        """


        ################
        #|     CRL    |# 
        ################

        #### CRL은 학습 시 confidence 기반 샘플 별 ranking 수행함. => confidence도 리턴해서 train에서 loss 계산 후 합산. 
        
        weight_feature = concat(feat_list).squeeze(dim=2) # (16, 40, 46) 

        tmp_x = self.TemporalAttn(weight_feature)
        sp_x = self.SpatialAttn(weight_feature)

        if self.args.fusion_type == "concat":
            fused_feat = concat([tmp_x, sp_x])


        elif self.args.fusion_type == "sum":
            fused_feat = summation([tmp_x, sp_x])


        elif self.args.fusion_type == "average":
            fused_feat = average([tmp_x, sp_x])

        elif self.args.fusion_type == "matmul":
            fused_feat = torch.matmul(tmp_x, sp_x.transpose(1,2)) # (16, 40, 40)

        fused_feat = fused_feat.reshape(self.bs, -1)

        # fused feature에 샘플 별 confidence (모달평균) 추가 적용 
        # weighted_fused_feat = fused_feat * F.softmax(SAMPLE_conf.unsqueeze(dim=1), dim=1) # (16, 3680)
        MS_logit = F.softmax(self.MSclassifier(fused_feat), dim=1)

        if infer:
            return MS_logit
        
        label = label.squeeze()
        MSLoss = torch.mean(self.lossfn(MS_logit, label)) # cross entropy loss  with "reduction=mean"
   
        return SAMPLE_conf, MSLoss, MS_logit # confidence도 리턴해서 train에서 loss 계산 후 합산.
    
    def infer(self, data_list):
        MS_logit = self.forward(data_list, infer=True)
        return MS_logit

    
def data_resize(self, data_list):
    new_data_list = []
    for i, dim in enumerate(self.in_dim):
        new_data_list.append(data_list[i])  
    data_list = new_data_list
    return data_list      


if __name__ == "__main__":    
    import time
    start = time.time()

    parser = argparse.ArgumentParser(description='Ablation_TCPjoint')

    parser.add_argument("--SEED", default=42)

    ### data_type 과 model_type 선택 필수
    parser.add_argument('--data_type', default='bl_2cl_misc5', choices=['bl_2cl_misc5'])
    parser.add_argument('--model_type', default='model_v01', choices=['model_v01', ''])
    parser.add_argument('--fusion_type', default='sum')
    parser.add_argument('--postprocessor', default='ebo', choices=['msp', 'mls', 'ebo'])
    parser.add_argument('--temp', default=1.5, help='temperature scaling for ebo')

    parser.add_argument('--backbone', default='EEGNet4', choices = ['EEGNet4'])

    ### early stopping on-off
    parser.add_argument('--early_stop', default=False, choices=[True, False])

    ########## 실험 하이퍼 파라미터 설정 
    parser.add_argument('--BATCH', default=16, help='Batch Size') # original 16, set 32
    parser.add_argument('--EPOCH', default=100, help='Epoch') # original: 50, set: 10
    parser.add_argument('--optimizer', default='AdamW', help='Optimizer') 
    parser.add_argument('--lr', default=0.002, help='Adam Learning Rate') # original: 1e-4
    parser.add_argument('--step_size', default=500, help='step size for StepLR scheduler')
    parser.add_argument('--freq_time', default=750, help='frequency(250)*time window(3)')
    parser.add_argument('--in_dim', default=[28,1,1,1,1], choices=[[28], [28,1], [28,1,1,1,1]], help='num of channel list for each modality')

    ########## 클래스 및 채널 수 지정
    parser.add_argument('--n_channels', default=28)
    parser.add_argument('--n_classes', default=2)
    
    args = parser.parse_args()

    import numpy as np
    import random

    random.seed(args.SEED)
    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)
    torch.cuda.manual_seed_all(args.SEED)  # GPU 연산에 대한 시드 설정

    device = torch.device(f'cuda:1' if torch.cuda.is_available() else 'cpu')

    model = Net(args, device).to(device)

    # 훈련 데이터 (더미 데이터 예제)
    train_data = [torch.rand(16, 28, 750).to(device),torch.rand(16, 1, 750).to(device),torch.rand(16, 1, 750).to(device),torch.rand(16, 1, 750).to(device),torch.rand(16, 1, 750).to(device)]  # 16개의 훈련 데이터 예제


    model.train()  # 모델을 훈련 모드로 설정
    outputs = model(train_data)



