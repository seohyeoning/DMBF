import torch.nn as nn
import torch
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.modalities = 5
        self.args = args
        self.n_classes = args.n_classes
        self.in_dim = args.in_dim

        self.FeatureExtractor = EEGNet4(self.args)
        if self.args.data_type == 'Pilot_Fatigue':
            dim_list = [200, 23552]
        elif self.args.data_type == 'Drowsy': # Drowsy
            dim_list = [200, 9176]
        elif self.args.data_type == 'manD':
            dim_list = [200, 3840]
        else: # Distraction, Stress
            dim_list = [200, 6200]
        if args.modal == 'woEEG':
            hid_dim = dim_list[0]
        else : 
            hid_dim = dim_list[1]

        self.classifier = nn.Sequential(nn.Linear(hid_dim, self.n_classes))   


    def forward(self, x):
        data_list = data_resize(self, x)
        if len(data_list[-1].shape) == 2:
            if self.args.data_type == 'Pilot_Fatigue':
                data_list[1] = data_list[1].unsqueeze(1)    
                data_list[2] = data_list[2].unsqueeze(1)
                data_list[3] = data_list[3].unsqueeze(1)
            elif self.args.data_type == 'Drowsy' or self.args.data_type == 'Distraction' or self.args.data_type == 'Stress':
                data_list[1] = data_list[1].unsqueeze(1)    
                data_list[2] = data_list[2].unsqueeze(1)
            elif self.args.data_type == 'manD' or self.args.data_type =='MS':
                data_list[1] = data_list[1].unsqueeze(1)    
                data_list[2] = data_list[2].unsqueeze(1)
                data_list[3] = data_list[3].unsqueeze(1)
                data_list[4] = data_list[4].unsqueeze(1)    
    
        data = concat(data_list)
        out = self.FeatureExtractor(data)
        output = F.softmax(self.classifier(out), dim=1)

        return output

def average(out_list):
    return torch.mean(torch.stack(out_list, dim=1), dim=1)
    
def concat(out_list):
    return torch.cat(out_list, dim=1) 

def sum(out_list):
    return torch.sum(torch.stack(out_list, dim=1), dim=1) 
    
class EEGNet4(nn.Module): # 4 temporal, 2 spatial per temporal
    def __init__(self, args, input_ch=4, track_running=True): 
        super(EEGNet4, self).__init__()
        self.args = args
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
        x = x.unsqueeze(dim=1).permute(0,1,2,3)  
        out = self.convnet(x)
        out = out.view(out.size()[0], -1)

        # output = self.classifier(out)
        return out
    
def data_resize(self, data_list):
    new_data_list = []
    if self.args.modal == 'woEEG':
        for i, dim in enumerate(self.in_dim):       
            if i == 0:
                continue
            new_data_list.append(data_list[i])  
        data_list = new_data_list

    elif self.args.modal == 'wEEG':
        data_list = data_list

    return data_list       



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MS_mmdynamic')

    parser.add_argument("--SEED", default=42)

    ### data_type 과 model_type 선택 필수
    parser.add_argument('--data_type', default='bl_2cl_misc5', choices=['bl_2cl_misc5', 'bl_2cl_misc4'])
    parser.add_argument('--model_type', default='early_fusion')
    parser.add_argument('--backbone', default='EEGNet4', choices = ['EEGNet4'])
    parser.add_argument('--fusion_type', default='concat', choices=['average','concat', 'sum'])

    ### early stopping on-off
    parser.add_argument('--early_stop', default=False, choices=[True, False])
    parser.add_argument('--random_sampler', default=False, choices=[True, False])

    ########## 실험 하이퍼 파라미터 설정 
    parser.add_argument('--BATCH', default=16, help='Batch Size') # original 16, set 32
    parser.add_argument('--EPOCH', default=100, help='Epoch') # original: 50, set: 10
    parser.add_argument('--optimizer', default='AdamW', help='Optimizer') 
    parser.add_argument('--lr', default=0.002, help='Adam Learning Rate') # original: 1e-4
    parser.add_argument('--step_size', default=500, help='step size for StepLR scheduler')
    parser.add_argument('--freq_time', default=750, help='frequency(250)*time window(3)')
    parser.add_argument('--in_dim', default=[1,1,1,1], choices=[[28], [28,1], [28,1,1,1,1]], help='num of channel list for each modality')
    parser.add_argument('--hid_dim', default=[200], choices=[[500], [300]])

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

    model = Net(args)

    # 훈련 데이터 (더미 데이터 예제)
    train_data = [torch.rand(16, 28, 750),torch.rand(16, 1, 750),torch.rand(16, 1, 750),torch.rand(16, 1, 750),torch.rand(16, 1, 750)]  # 16개의 훈련 데이터 예제


    model.train()  # 모델을 훈련 모드로 설정
    outputs = model(train_data)
