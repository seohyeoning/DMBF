"""
EEG Conformer 

Convolutional Transformer for EEG decoding

Couple CNN and Transformer in a concise manner with amazing results

https://github.com/eeyhsong/EEG-Conformer/tree/main
"""
# remember to change paths

import os

import math


import torch.nn as nn
import torch.nn.functional as F
import torch

import torch
import torch.nn.functional as F


from torch import nn
from torch import Tensor

from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
# from common_spatial_pattern import csp

import matplotlib.pyplot as plt

# Convolution module
# use conv to capture local features, instead of postion embedding.
class PatchEmbedding(nn.Module):
    def __init__(self, args, emb_size=40):
        # self.patch_size = patch_size
        super().__init__()
        self.args = args
        
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (32, 1), (1, 1)), # original:  nn.Conv2d(40, 40, (22, 1), (1, 1))
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )


    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        
        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.fc = nn.Sequential(
            nn.Linear(1760, 256), 
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return out # original: x, out => tocken과 output


class EEGConformer(nn.Sequential):
    def __init__(self, args, emb_size=40, depth=6, n_classes=2, **kwargs):
        super().__init__(

            PatchEmbedding(args, emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )
        
class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.n_classes = args.n_classes
        self.in_dim = [28,1,1,1,1]

        self.FeatureExtractor = EEGConformer(self.args)


    def forward(self, x):
        # data_list = data_resize(self, x)
                             
        data = concat(x)
        out = self.FeatureExtractor(data)
        output = F.softmax(out, dim=1)

        return output

def average(out_list):
    return torch.mean(torch.stack(out_list, dim=1), dim=1)
    
def concat(out_list):
    return torch.cat(out_list, dim=1) 

def sum(out_list):
    return torch.sum(torch.stack(out_list, dim=1), dim=1) 


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
    parser.add_argument('--in_dim', default=[28,1,1,1,1], choices=[[28], [28,1], [28,1,1,1,1]], help='num of channel list for each modality')
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

    model = Net(args)

    # 훈련 데이터 (더미 데이터 예제)
    train_data = [torch.rand(16, 28, 750),torch.rand(16, 1, 750),torch.rand(16, 1, 750),torch.rand(16, 1, 750),torch.rand(16, 1, 750)]  # 16개의 훈련 데이터 예제


    model.train()  # 모델을 훈련 모드로 설정
    outputs = model(train_data)