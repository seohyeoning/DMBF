import torch.nn as nn
import torch
import torch.nn.functional as F
from base.layers import Conv2dWithConstraint, LinearWithConstraint
from base.utils import initialize_weight

torch.set_printoptions(linewidth=1000)

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.modalities = 5
        self.args = args
        self.n_classes = args.n_classes
        self.in_dim = [28,1,1,1,1]

        self.FeatureExtractor = ShallowConvNet(self.args)

        hid_dim = 368


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



class ShallowConvNet(nn.Module):
    def __init__(
            self, args,
            F1=None,
            T1=None,
            F2=None,
            P1_T=None,
            P1_S=None,
            drop_out=None,
            pool_mode=None,
            weight_init_method=None,
            last_dim=None,
    ):
        super(ShallowConvNet, self).__init__()
        n_classes = args.n_classes
        s = 1
        pooling_layer = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]
        self.net = nn.Sequential(
            Conv2dWithConstraint(1, F1, (1, T1), max_norm=2),
            Conv2dWithConstraint(F1, F2, (s, 1), bias=False, max_norm=2),
            nn.BatchNorm2d(F2),
            ActSquare(),
            pooling_layer((1, P1_T), (1, P1_S)),
            ActLog(),
            nn.Dropout(drop_out),
            nn.Flatten(),
            LinearWithConstraint(last_dim, n_classes, max_norm=0.5)
        )

        initialize_weight(self, weight_init_method)

    def forward(self, x):
        x = x.unsqueeze(dim=2)
        out = self.net(x)
        return out
    
class ActSquare(nn.Module):
    def __init__(self):
        super(ActSquare, self).__init__()
        pass

    def forward(self, x):
        return torch.square(x)


class ActLog(nn.Module):
    def __init__(self, eps=1e-06):
        super(ActLog, self).__init__()
        self.eps = eps

    def forward(self, x):
        return torch.log(torch.clamp(x, min=self.eps))
    
def data_resize(self, data_list):
    new_data_list = []
    for i, dim in enumerate(self.in_dim):
        new_data_list.append(data_list[i+1])  
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
