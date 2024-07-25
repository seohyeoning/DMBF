import torch.nn as nn
import torch
import torch.nn.functional as F

import argparse

from .Backbone.EEGNet4 import EEGNet4
# from .Backbone.DeepConvNet import DeepConvNet
# from .Backbone.ResNet8 import ResNet8
# from .Backbone.ResNet18 import ResNet18

"""
scoring -> fusion
<Confidence Method>
MSP / EBO / MLS

<FUSION Method>
Spatial-temporal attention
=> spatial = 1 FC layer + softmax
   temporal = 1 FC layer + relu
   후 fusion (concat, sum, average)

구조 특징: select layer (in_dim, 2) 학습
손실 특징: sample selection loss, correctness loss 
최종 손실: L_CLS, L_CONF, L_CRL
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
        self.modalities = 3

        self.args = args
        self.device = device
        self.n_classes = args.n_classes
        self.bs = args.BATCH
        self.in_dim = args.in_dim


        self.lossfn = torch.nn.CrossEntropyLoss() # reduction="mean" for MSLoss
        if args.backbone == 'EEGNet4':
            self.FeatureExtractor = nn.ModuleList([EEGNet4(args, m) for m in range(self.modalities)])
        
        # Stress & Distraction
        if args.data_type == 'Drowsy':

            self.TemporalAttn = TemporalAttn(args, time_step_size=37, hidden_size=37)
            self.SpatialAttn = SpatialAttn(args, channel_size=24, hidden_size=24)
            self.MSclassifier = nn.Sequential(nn.Linear(888, self.n_classes))

            self.ConfidenceLayer = nn.ModuleList([nn.Linear(296, 2) for m in range(self.modalities)])
            


        if args.postprocessor == 'ebo':
            self.temperature = nn.Parameter(torch.ones(1, device=self.device) * self.args.temp)  # initialize T = original 1.5
          
    def forward(self, x, label=None, infer=False):
        data_list = data_resize(self, x)


        feat_dict = dict()
        logit_dict = dict()
        for mod in range(self.modalities):
            feat_dict[mod] = self.FeatureExtractor[mod](data_list[mod]) # (16, 8, 5, 25)
            feat_dict[mod] = feat_dict[mod].view(feat_dict[mod].size(0), -1, feat_dict[mod].size(3)) # (16, 8*5, 25)
            logit_dict[mod] = self.ConfidenceLayer[mod](feat_dict[mod].reshape(self.bs, -1)) # (16, 2)

        feat_list = list(feat_dict.values())
        logit_list = list(logit_dict.values())

        conf_list = []  
        for i in range (self.modalities):
            if self.args.postprocessor == 'msp':
                conf= torch.max(F.softmax(logit_list[i], dim=1), dim=1)[0].to(self.device) # (16)
            elif self.args.postprocessor == 'mls':
                conf = torch.max(logit_list[i], dim=1)[0].to(self.device)
            elif self.args.postprocessor == 'ebo':
                conf = self.temperature * torch.logsumexp(logit_list[i] / self.temperature, dim=1).to(self.device)
            conf_list.append(conf)
        conf_stack = torch.stack(conf_list, dim=1)

        conf_b4_scale = conf_stack.clone() # scaling 전 conf_stack 저장

        SAMPLE_conf = torch.mean(conf_b4_scale, dim=1) # 모달별 confidence를 평균해서 샘플 별 confidence (sample_conf)계산. (16)
        """
        위에서 중요도 점수 뽑고나서, 모달리티의 상대적인 점수 주기위해 stack 후 softmax하면 그 효과가 사라지는 것이 아닌가?
        성능 확인해보고, softmax가 아닌 다른 방법으로 상대적 중요도 점수를 구해보자. 
        """
        if self.args.scaling == 'softmax':
            confidence = F.softmax(conf_stack.squeeze(), dim=1) # 모달별 상대적 중요도 점수 [0,1] 계산하기 위해 softmax 적용
            
        elif self.args.scaling == 'sigmoid':
            confidence = torch.sigmoid(conf_stack.squeeze()) # scaling으로 sigmoid 적용

        if self.modalities == 1 :
            conf_list = [confidence]
        else:
            conf_list = [confidence[:, i] for i in range(confidence.size(1))] # 모달 별 confidence list
        conf_list = [element.unsqueeze(dim=1).unsqueeze(dim=2) for element in conf_list] # (16) -> (16,1,1)

        # feature concat 하기 전에 confidence 곱하기
        weight_feature_list = []
        for i in range(self.modalities):
            weight_feature =  feat_list[i].to(self.device) * conf_list[i] # concat된 피쳐에 confidence 적용
            weight_feature_list.append(weight_feature)

        ################
        #|     CRL    |# 
        ################

        #### CRL은 학습 시 confidence 기반 샘플 별 ranking 수행함. => confidence도 리턴해서 train에서 loss 계산 후 합산. 
        
        weight_feature = concat(weight_feature_list).squeeze(dim=2) # (16, 40, 46) 

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
        # weighted_fused_feat = fused_feat * SAMPLE_conf.unsqueeze(dim=1) # (16, 3680)
        MS_logit = F.softmax(self.MSclassifier(fused_feat), dim=1)

        if infer:
            return  feat_list, weight_feature_list
        
        ###### Selection layer 학습 loss 설정
        if self.args.selection_loss_type == 'CE':
            criterion = torch.nn.CrossEntropyLoss(reduction='none') # none: 모든 픽셀의 정답과 예측값의 cross entropy를 return
        # elif self.args.selection_loss_type == 'Focal':
        #     kwargs = {"alpha": 1, "gamma": 2.0, "reduction": 'none'} # reduction: none, mean, sum
        #     criterion = FocalLoss(**kwargs) 
        label = label.squeeze().squeeze()
        MSLoss = torch.mean(self.lossfn(MS_logit, label)) # cross entropy loss  with "reduction=mean"

        # 모달리티 별 selection layer의 loss 계산 후 합산
        conf_loss_list = []
        for m in range(self.modalities): 
            conf_loss = criterion(F.softmax(logit_list[m], dim=1), label)  # confidence score에는 적용안하고, conf_loss 계산시에만 사용.
            conf_loss_list.append(conf_loss)

        confidenceLoss = torch.mean(torch.stack(conf_loss_list))

        Loss = MSLoss + confidenceLoss


        return SAMPLE_conf, Loss, MS_logit # confidence도 리턴해서 train에서 loss 계산 후 합산.
    
    def infer(self, data_list):
        feat_list, weight_feature_list = self.forward(data_list, infer=True)
        return feat_list, weight_feature_list

    
def data_resize(self, data_list):
    new_data_list = []
    for i, dim in enumerate(self.in_dim):
        if i != 0:
            data_list[i] = data_list[i].unsqueeze(dim=1)
        new_data_list.append(data_list[i])  
            
    data_list = new_data_list
    return data_list      



