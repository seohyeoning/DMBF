import torch.nn as nn
import torch
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.modalities = len(args.in_dim)
        self.args = args

        if args.modality == 'eeg':
            self.n_channels = args.n_channels
        else :
            self.n_channels = 1

        self.n_classes = args.n_classes

        self.FeatureExtractor = EEGNet4(self.args, input_ch=self.n_channels)

        if args.data_type == 'Pilot_Fatigue':
            hid_dim = 368

        elif args.data_type == "MS" : 
            hid_dim = 368

        elif args.data_type == 'SA_Drowsy_bal' or args.data_type == 'SA_Drowsy_unbal':
            hid_dim = 552
       
        elif args.data_type == 'Drowsy':
                hid_dim = 296

        elif args.data_type == "Distraction":
                hid_dim = 200

        elif args.data_type == "manD":
                hid_dim = 384


        self.classifier = nn.Sequential(nn.Linear(hid_dim, self.n_classes))   


    def forward(self, x):
        if self.args.data_type == 'Pilot_Fatigue':
            if self.args.modality == 'eeg':
                data = x[0]
            elif self.args.modality == 'ecg':
                data = x[1]
            elif self.args.modality == 'gsr':
                data = x[2]
            elif self.args.modality == 'rsp':
                data = x[3]

        elif self.args.data_type == 'MS': 
            if self.args.modality == 'eeg':
                data = x[0]
            elif self.args.modality == 'ecg':
                data = x[1]
            elif self.args.modality == 'ppg':
                data = x[2]
            elif self.args.modality == 'gsr':
                data = x[3]
            elif self.args.modality == 'rsp':
                data = x[4]
            
        elif self.args.dataset == 'SA':
            data = x

        elif self.args.dataset == 'LG':
            if self.args.modality == 'eeg':
                data = x[0]
            elif self.args.modality == 'ppg':
                data = x[1]
            elif self.args.modality == 'ecg':
                data = x[2]
                
        elif self.args.data_type == "manD" : 
            if self.args.modality == 'eeg':
                data = x[0]
            elif self.args.modality == 'ppg':
                data = x[1]
            elif self.args.modality == 'eda':
                data = x[2]
            elif self.args.modality == 'temp':
                data = x[3]
            elif self.args.modality == 'ecg':
                data = x[4]

        data = data.unsqueeze(dim=1)
        out = self.FeatureExtractor(data)
        output = F.softmax(self.classifier(out), dim=1)

        return output

    
class EEGNet4(nn.Module): # 4 temporal, 2 spatial per temporal
    def __init__(self, args, input_ch, track_running=True): 
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
        if len(x.size()) == 3:
            x = x.unsqueeze(dim=1)
        out = self.convnet(x)
        out = out.view(out.size()[0], -1)

        # output = self.classifier(out)
        return out
    


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MS_mmdynamic')

    parser.add_argument("--SEED", default=42)
    
    ### data_type 과 model_type 선택 필수
    parser.add_argument('--data_type', default='MS', choices=['Drowsy'])
    parser.add_argument('--dataset', default='None', choices=['SA', 'LG', None])
    parser.add_argument('--modality', default='eeg', choices=['eeg', 'ecg', 'rsp', 'ppg', 'gsr'])
    parser.add_argument('--backbone', default='EEGNet4', choices = ['EEGNet4'])

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
    parser.add_argument('--in_dim', default=[28], choices=[[28], [28,1], [28,1,1,1,1]], help='num of channel list for each modality')


    ########## 클래스 및 채널 수 지정
    parser.add_argument('--n_channels', default=None)
    parser.add_argument('--n_classes', default=2)
    
    args = parser.parse_args()
    
    if args.modality == 'eeg':
        args.n_channels = 28
    else :
        args.n_channels = 1
        
    import numpy as np
    import random

    random.seed(args.SEED)
    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)
    torch.cuda.manual_seed_all(args.SEED)  # GPU 연산에 대한 시드 설정

    model = Net(args).cuda()

    # 훈련 데이터 (더미 데이터 예제)
    train_data = [torch.rand(16, 28, 750),torch.rand(16, 1, 750),torch.rand(16, 1, 750),torch.rand(16, 1, 750),torch.rand(16, 1, 750)]  # 16개의 훈련 데이터 예제
    
    from torchsummary import summary

    train_data = [torch.rand(16, 28, 750),torch.rand(16, 1, 750),torch.rand(16, 1, 750),torch.rand(16, 1, 750),torch.rand(16, 1, 750)]  # 16개의 훈련 데이터 예제
    summary(model, (16, 28, 750), batch_size=16)


