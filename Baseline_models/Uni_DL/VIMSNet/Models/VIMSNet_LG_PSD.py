import torch.nn as nn
import torch
import torch.nn.functional as F
from Helpers.FE_algorithms import DE_PSD
from Helpers.Variables import device

"""
논문에서 사용한 데이터: Absolute band power (ABP)
논문에서 사용한 데이터 쉐입: (bs, 5, 4) # 5개 대역, eeg 4채널

내 전처리 된 데이터에 맞게 레이어 dimension만 수정

refer. https://github.com/threedteam/VIMSNet
"""

class VIMSNet(nn.Module):
    def __init__(self, args):
        super(VIMSNet, self).__init__()
        self.args = args
        self.bs = 16
        self.Block1 = nn.Sequential(
            nn.Conv1d(
                in_channels=5, # original = 1
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
                dilation=1
            ),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.ReLU(),
        )

        self.Block2 = nn.Sequential(
            nn.Conv1d(5, 32, 6, 2, 2, 1), # original = nn.Conv1d(1, 8, 6, 2, 2, 1)
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.ReLU(),
        )

        self.Block3 = nn.Sequential(
            nn.Conv1d(5, 32, 2, 1, 1, 1), # original = nn.Conv1d(1, 8, 2, 1, 1, 1)
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.squeeze = nn.AdaptiveMaxPool1d(1)
        
        if args.data_type == "Distraction" or args.data_type == "Stress" or args.data_type == "Drowsy":
            hid_dim = [48, 1536, 672]
        elif args.data_type == "MS":
            hid_dim = [42, 1344, 652]
        elif args.data_type == 'manD':
            hid_dim = [13, 416, 557]

        self.excitation = nn.Sequential(
            nn.Linear(hid_dim[0], 5), # original = nn.Linear(10,5)
            nn.ReLU(inplace=True),
            nn.Linear(5, hid_dim[0]), # original = nn.Linear(5,10)
            nn.Sigmoid(),
        )
        self.dense_1 = nn.Sequential(
            nn.Linear(hid_dim[1], 512), # original: nn.Linear(48*10, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        self.dense_2 = nn.Sequential(
            nn.Linear(hid_dim[2], 128),  # original: nn.Linear(532, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3))

        self.out = nn.Linear(128, self.args.n_classes)
        nn.init.constant(self.out.bias, 0.1)
        torch.nn.init.kaiming_normal_(self.out.weight)

    def forward(self, x):
        x = x[0]
        x = DE_PSD(self.args, x.cpu())
        x = x.transpose(0,2,1)
        x = torch.from_numpy(x).float().to(device)
        x_1 = self.Block1(x)
        x_2 = self.Block2(x)
        x_3 = self.Block3(x)

        x_cat = torch.cat([x_1, x_2, x_3], 2)

        a = x_cat.permute(0, 2, 1)
        squeeze = self.squeeze(a)
        squeeze = squeeze.view(self.bs, -1) # original: view(-1, 10)
        excitation = self.excitation(squeeze)
        excitation = excitation.view(self.bs,1,-1) # original:  excitation.view(-1, 1, 10)

        scale = torch.mul(x_cat, excitation, out=None)

        input_dense = scale.view(scale.size(0), -1)

        input_dense = self.dense_1(input_dense)

        x_input_dense = x.reshape(x.size(0), -1)

        input_dense = torch.cat([input_dense, x_input_dense], 1)

        input_dense = self.dense_2(input_dense)

        out = self.out(input_dense)

        return out
    