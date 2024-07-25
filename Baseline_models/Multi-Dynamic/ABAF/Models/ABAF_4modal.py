import torch
import torch.nn as nn
import torch.nn.functional as F
import os
"""
AP_dataTrain, AP_label_Train, AP_dataTest, AP_labelTest, -> M1_
PVP_ -> M2
PC_ -> M3

"""
class ABAFNet(nn.Module):
    def __init__(self, args):
        super(ABAFNet, self).__init__()
        # 3DConv -> 1DConv
        self.args = args
        num_channel = args.num_channel

        self.bs = args.BATCH
        if args.data_type == "Pilot_Fatigue":
            hidden = 11904 # freq*sec = 750 (3sec)

        self.conv1_1 = nn.Conv1d(num_channel, 32, kernel_size=3)
        self.conv1_2 = nn.Conv1d(32, 64, kernel_size=3)
        self.fc1_1 = nn.Linear(hidden, 500)
        self.fc1_2 = nn.Linear(500, 50)            

        self.conv2_1 = nn.Conv1d(1, 32, kernel_size=3) # 다른 모달리티들은 채널이 1이라서 따로 지정 x
        self.conv2_2 = nn.Conv1d(32, 64, kernel_size=3)
        self.fc2_1 = nn.Linear(hidden, 500)
        self.fc2_2 = nn.Linear(500, 50)

        self.conv3_1 = nn.Conv1d(1, 32, kernel_size=3)
        self.conv3_2 = nn.Conv1d(32, 64, kernel_size=3)
        self.fc3_1 = nn.Linear(hidden, 500)
        self.fc3_2 = nn.Linear(500, 50)

        self.conv4_1 = nn.Conv1d(1, 32, kernel_size=3)
        self.conv4_2 = nn.Conv1d(32, 64, kernel_size=3)
        self.fc4_1 = nn.Linear(hidden, 500)
        self.fc4_2 = nn.Linear(500, 50)

        self.fc_M1= nn.Linear(50, self.args.n_classes)
        self.fc_M2 = nn.Linear(50, self.args.n_classes)
        self.fc_M3 = nn.Linear(50, self.args.n_classes)
        self.fc_M4 = nn.Linear(50, self.args.n_classes)

        self.fc_fusion1 = nn.Linear(200, 50) # MLP
        self.fc_fusion2 = nn.Linear(50, self.args.n_classes) 
        self.relu = nn.ReLU()
    

    def AAFFM(self, inputs):
        (x1, x2, x3, x4) = (inputs[0], inputs[1].reshape(self.bs, 1, -1), inputs[2].reshape(self.bs, 1, -1), inputs[2].reshape(self.bs, 1, -1)) # xe, xc, xg, xr
        x1 = self.relu(self.conv1_1(x1))
        x1 = F.max_pool1d(x1, 2)
        x1 = F.relu(self.conv1_2(x1))
        x1 = F.max_pool1d(x1, 2)
        x1 = x1.view(self.bs, -1) # batch_size=self.bs
        x1 = F.relu(self.fc1_1(x1))
        x1 = F.dropout(x1, p=0.5)
        fc_out_12 = F.relu(self.fc1_2(x1))            

        x2 = F.relu(self.conv2_1(x2))
        x2 = F.max_pool1d(x2, 2)
        x2 = F.relu(self.conv2_2(x2))
        x2 = F.max_pool1d(x2, 2)
        x2 = x2.view(self.bs, -1)
        x2 = F.relu(self.fc2_1(x2))
        x2 = F.dropout(x2, p=0.5)
        fc_out_22 = F.relu(self.fc2_2(x2))

        x3 = F.relu(self.conv3_1(x3))
        x3 = F.max_pool1d(x3, 2)
        x3 = F.relu(self.conv3_2(x3))
        x3 = F.max_pool1d(x3, 2)
        x3 = x3.view(self.bs, -1)
        x3 = F.relu(self.fc3_1(x3))
        x3 = F.dropout(x3, p=0.5)
        fc_out_32 = F.relu(self.fc3_2(x3))

        # calculate for x4
        x4 = F.relu(self.conv4_1(x4))
        x4 = F.max_pool1d(x4, 2)
        x4 = F.relu(self.conv4_2(x4))
        x4 = F.max_pool1d(x4, 2)
        x4 = x4.view(self.bs, -1)
        x4 = F.relu(self.fc4_1(x4))
        x4 = F.dropout(x4, p=0.5)
        fc_out_42 = F.relu(self.fc4_2(x4))
        

        mid1 = self.fc_M1(fc_out_12)
        mid2 = self.fc_M2(fc_out_22)
        mid3 = self.fc_M3(fc_out_32)
        mid4 = self.fc_M4(fc_out_42)

        concat1 = torch.cat([fc_out_12, fc_out_22, fc_out_32, fc_out_42], dim=1) 
        return mid1, mid2, mid3, mid4, fc_out_12, fc_out_22, fc_out_32, fc_out_42, concat1

    def ADSFM(self, inputs):
        *_, fc_out_12, fc_out_22, fc_out_32, fc_out_42, concat1 = self.AAFFM(inputs)

        f11 = self.fc_fusion1(concat1)
        f21 = self.fc_fusion1(concat1)
        f31 = self.fc_fusion1(concat1)
        f41 = self.fc_fusion1(concat1)

        alpha1 = torch.sigmoid(f11)
        alpha2 = torch.sigmoid(f21)
        alpha3 = torch.sigmoid(f31)
        alpha4 = torch.sigmoid(f41)

        f1 = torch.mul(alpha1, fc_out_12) # feature reweighting
        f2 = torch.mul(alpha2, fc_out_22)
        f3 = torch.mul(alpha3, fc_out_32) 
        f4 = torch.mul(alpha4, fc_out_42)         

        f15 = torch.sum(f1, dim=1)
        f25 = torch.sum(f2, dim=1)
        f35 = torch.sum(f3, dim=1)
        f45 = torch.sum(f4, dim=1)

        f5 = torch.stack([f15, f25, f35, f45], dim=1)
        beta = F.softmax(f5, dim=1)            

        f14 = beta[:, 0]
        f24 = beta[:, 1]
        f34 = beta[:, 2]
        f44 = beta[:, 3]
        feature_cat = torch.cat([f1, f2, f3, f4], dim=1) 

        return f14, f24, f34, f44, feature_cat

    def forward(self, inputs):

        mid1, mid2, mid3, mid4, fc_out_12, fc_out_22, fc_out_32, fc_out_42, concat1 = self.AAFFM(inputs)
        f14, f24, f34, f44, feature_cat = self.ADSFM(inputs)

        fc_out_f1 = F.relu(self.fc_fusion1(feature_cat))
        mid = self.fc_fusion2(fc_out_f1) 
        # mid = torch.reshape(mid, (x1.shape[0], mid.shape[0] // x1.shape[0], -1))
        prediction = F.softmax(mid, dim=1)

        return f14, f24, f34, f44, mid1, mid2, mid3, mid4, prediction
