import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torchsummary import summary

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.modalities = 5
        self.args = args
        self.n_classes = args.n_classes

        self.FeatureExtractor = DeepConvNet(self.args, input_ch=self.args.n_channels)
        self.Classifier = nn.Sequential(nn.Linear(258, self.n_classes))

    def forward(self, x):
        if self.args.modality == 'eeg':
            data = x[0]
        elif self.args.modality == 'ecg':
            data = x[1]
        elif self.args.modality == 'rsp':
            data = x[2]
        elif self.args.modality == 'ppg' :
            data = x[3]
        elif self.args.modality == 'gsr':
            data = x[4]  
        data = data.unsqueeze(dim=1)
        out = self.FeatureExtractor(data) # include classifier
        output = F.softmax(out, dim=1)

        return output

class DeepConvNet(nn.Module):
    def __init__(self, args, input_ch,
                 batch_norm=True, batch_norm_alpha=0.1):
        super(DeepConvNet, self).__init__()

        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.n_classes = args.n_classes
        input_time = args.freq_time
        n_ch1 = 25
        n_ch2 = 50
        n_ch3 = 100
        self.n_ch4 = 200

        if self.batch_norm:
            self.convnet = nn.Sequential(
                nn.Conv2d(1, n_ch1, kernel_size=(1, 10), stride=1),
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(input_ch, 1), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch1, momentum=self.batch_norm_alpha, affine=True, eps=1e-5, track_running_stats=True),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                # MixStyle(p=0.5, alpha=0.1, mix="random"), # MixStyle
                nn.Dropout(p=0.5),
                
                nn.Conv2d(n_ch1, n_ch2, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch2, momentum=self.batch_norm_alpha, affine=True, eps=1e-5, track_running_stats=True),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                # MixStyle(p=0.5, alpha=0.1, mix="random"), # MixStyle
                nn.Dropout(p=0.5),

                nn.Conv2d(n_ch2, n_ch3, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch3, momentum=self.batch_norm_alpha, affine=True, eps=1e-5, track_running_stats=True),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                # MixStyle(p=0.5, alpha=0.1, mix="random"), # MixStyle
                nn.Dropout(p=0.5),

                nn.Conv2d(n_ch3, self.n_ch4, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(self.n_ch4, momentum=self.batch_norm_alpha, affine=True, eps=1e-5, track_running_stats=True),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)), 
                # MixStyle(p=0.5, alpha=0.1, mix="random"), # MixStyle, maxpool 뒤에 mixstyle 추가
                )
        else:
            self.convnet = nn.Sequential(
                nn.Conv2d(1, n_ch1, kernel_size=(1, 10), stride=1,bias=False),
                nn.BatchNorm2d(n_ch1, momentum=self.batch_norm_alpha,
                               affine=True, eps=1e-5, track_running_stats=True),
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(input_ch, 1), stride=1),
                # nn.InstanceNorm2d(n_ch1),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch1, n_ch2, kernel_size=(1, 10), stride=1),
                # nn.InstanceNorm2d(n_ch2),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch2, n_ch3, kernel_size=(1, 10), stride=1),
                # nn.InstanceNorm2d(n_ch3),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch3, self.n_ch4, kernel_size=(1, 10), stride=1),
                # nn.InstanceNorm2d(self.n_ch4),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            )
        self.convnet.eval()
        out = self.convnet(torch.zeros(16, 1, input_ch, input_time))
        
        n_out_time = out.cpu().data.numpy().shape[3]
        self.final_conv_length = n_out_time

        self.n_outputs = out.size()[1]*out.size()[2]*out.size()[3]

        self.clf = nn.Sequential(nn.Linear(self.n_outputs, self.n_classes), nn.Dropout(p=0.2))  ####################### classifier 
        # DG usually doesn't have classifier
        # so, add at the end

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        # output = self.l2normalize(output)
        output=self.clf(output) 
        return output

    def get_embedding(self, x):
        return self.forward(x)

    def l2normalize(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)





if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MS_mmdynamic')

    parser.add_argument("--SEED", default=42)
    
    ### data_type 과 model_type 선택 필수
    parser.add_argument('--data_type', default='bl_2cl_misc5', choices=['bl_2cl_misc5', 'BP_misc5'])
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
    parser.add_argument('--hid_dim', default=[200], choices=[[500], [300]])

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
    summary(model, (16, 28, 750), batch_size=16)
    # model.train()  # 모델을 훈련 모드로 설정
    # outputs = model(train_data)
    


