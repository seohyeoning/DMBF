'''ResNet-1D in PyTorch.
Dong-Kyun Han 2020/09/17
dkhan@korea.ac.kr

Reference:
[1] K. He, X. Zhang, S. Ren, J. Sun
    "Deep Residual Learning for Image Recognition," arXiv:1512.03385
[2] J. Y. Cheng, H. Goh, K. Dogrusoz, O. Tuzel, and E. Azemi,
    "Subject-aware contrastive learning for biosignals,"
    arXiv preprint arXiv :2007.04871, Jun. 2020
[3] D.-K. Han, J.-H. Jeong
    "Domain Generalization for Session-Independent Brain-Computer Interface,"
    in Int. Winter Conf. Brain Computer Interface (BCI),
    Jeongseon, Republic of Korea, 2020.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.modalities = 5
        self.args = args
        self.n_classes = args.n_classes
        self.in_dim = [28,1,1,1,1]

        self.FeatureExtractor = Resnet18(self.args)


    def forward(self, x):
        data_list = data_resize(self, x)
                             
        data = concat(data_list)
        out = self.FeatureExtractor(data)
        output = F.softmax(out, dim=1)

        return output

def average(out_list):
    return torch.mean(torch.stack(out_list, dim=1), dim=1)
    
def concat(out_list):
    return torch.cat(out_list, dim=1) 

def sum(out_list):
    return torch.sum(torch.stack(out_list, dim=1), dim=1) 



class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, kernel_size, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.bn0 = norm_layer(inplanes)
        self.elu = nn.ELU(inplace=True)
        self.dropdout0 = nn.Dropout(p=0.1)

        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=kernel_size//2,bias=False)
        self.bn1 = norm_layer(planes)
        self.dropdout1 = nn.Dropout(p=0.1)

        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, stride=1, padding=kernel_size//2,bias=False)
        # self.dropdout2 = nn.Dropout(p=0.5)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out0 = self.bn0(x)
        out0 = self.elu(out0)
        out0 = self.dropdout0(out0)

        identity = out0

        out = self.conv1(out0)
        out = self.bn1(out)
        out = self.elu(out)
        out = self.dropdout1(out)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(out0)

        out += identity

        return out

class Resnet18(nn.Module):
    def __init__(self, args, input_ch=4,
                batch_norm=True, batch_norm_alpha=0.1):
        super(Resnet18, self).__init__()
        
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.num_classes = args.n_classes
        n_ch1 = 25
        n_ch2 = 50
        n_ch3 = 100
        self.n_ch4 = 200
        self.num_hidden = 1024

        self.dilation = 1
        self.groups = 1
        self.base_width = input_ch
        norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer
        self.inplanes = 32
        self.conv1 = nn.Conv1d(input_ch, 32, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=3, padding=1)
        self.elu = nn.ELU(inplace=True)

        block = BasicBlock

        layers = [2,2,2,2]
        kernel_sizes = [3, 3, 3, 3]
        self.layer1 = self._make_layer(block, 32, kernel_sizes[0], layers[0], stride=1, layer_num=1)
        self.layer2 = self._make_layer(block, 64, kernel_sizes[1], layers[1], stride=1, layer_num=2)
        self.layer3 = self._make_layer(block, 128, kernel_sizes[2],layers[2], stride=2, layer_num=3)
        self.layer4 = self._make_layer(block, 256, kernel_sizes[2], layers[2], stride=2, layer_num=4)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256 * block.expansion, self.num_classes)

    def _make_layer(self, block, planes, kernel_size, blocks, stride=1, layer_num=0, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes,planes * block.expansion, kernel_size=1, stride=stride,bias=False),
                norm_layer(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, kernel_size,stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        # See note [TorchScript super()]
        x = x.squeeze(1)

        x = self.conv1(x)
        x = self.layer1(x) # basic block 2개
        x = self.layer2(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        x = self.maxpool(x)
        x = self.layer4(x)

        x = self.elu(x)
        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
        
    def get_embedding(self, x):
        return self.forward(x)

    def l2normalize(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)

def data_resize(self, data_list):
    new_data_list = []
    for i, dim in enumerate(self.in_dim):
        if i == 0:
            continue
        new_data_list.append(data_list[i])  
    data_list = new_data_list
    return data_list       



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MS_mmdynamic')

    parser.add_argument("--SEED", default=42)

    ### data_type 과 model_type 선택 필수
    parser.add_argument('--data_type', default='bl_2cl_misc5', choices=['bl_2cl_misc5', 'bl_2cl_misc4'])
    parser.add_argument('--model_type', default='early_fusion')
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
    parser.add_argument('--in_dim', default=[28,1,1,1,1], choices=[[28], [28,1], [28,1,1,1,1]], help='num of channel list for each modality')
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
