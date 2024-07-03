"""
Transformer for EEG classification

The core idea is slicing, which means to split the signal along the time dimension. Slice is just like the patch in Vision Transformer.

https://github.com/eeyhsong/EEG-Transformer

수정사항은 아래 참조
https://github.com/sylyoung/DeepTransferEEG
"""


import os
import numpy as np
import math
import random
import time
import scipy.io

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchsummary import summary

import torch
import torch.nn.functional as F

from torch import nn
from torch import Tensor

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

import matplotlib.pyplot as plt

from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True


class Conv2dWithConstraint(nn.Conv2d):
    """
    가중치에 제약을 추가한 conv layer
    가중치에 L2 Norm이 max_norm을 초과하지 않도록 제약 가함
    오버피팅 방지,  일반화 성능 높임
    """
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)
    

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size, sampling_rate, num_channels):
        super().__init__()
        kernel_size = int(sampling_rate * 0.16)
        pooling_kernel_size = int(sampling_rate * 0.3)
        pooling_stride_size = int(sampling_rate * 0.06)
        
        self.temporal_conv = Conv2dWithConstraint(1, emb_size, kernel_size=[1, kernel_size], padding='same', max_norm=2.)
        self.spatial_conv = Conv2dWithConstraint(emb_size, emb_size, kernel_size=[num_channels, 1], padding='valid', max_norm=2.)
        self.avg_pool = nn.AvgPool2d(kernel_size=[1, pooling_kernel_size], stride=[1, pooling_stride_size])
        self.bn = nn.BatchNorm2d(emb_size)
        self.rearrange = Rearrange('b e (h) (w) -> b (h w) e')
        
        
    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = torch.square(x) # 각 요소에 대해 제곱 (특정 피쳐 중요도 증가)
        x = self.avg_pool(x)
        x = torch.log(torch.clamp(x, min=1e-06))
        x = self.bn(x)
        x = self.rearrange(x)
        
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout=0.5):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size) # (emb_size, emb_size) # 12

    def forward(self, x, return_attention=False):
        queries = rearrange(self.queries(x), 'b n (h d) -> b h n d', h=self.num_heads)
        keys = rearrange(self.keys(x), 'b n (h d) -> b h n d', h=self.num_heads)
        values = rearrange(self.values(x), 'b n (h d) -> b h n d', h=self.num_heads)
        energy = torch.einsum('b h q d, b h k d -> b h q k', queries, keys)
        
        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        output = torch.einsum('b h a l, b h l v -> b h a v', att, values)
        output = rearrange(output, 'b h n d -> b n (h d)')
        output = self.projection(output)
        if return_attention:
            return output, att
        else:
            return output



class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        if len(x.shape) == 5:
            x = x.squeeze(dim=1)
        elif len(x.shape) == 6:
            x = x.squeeze(dim=2)
            x = x.squeeze(dim=1)
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
                 num_heads=1,
                 drop_p=0.5,
                 forward_expansion=2, # ori: 4
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
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

    def forward(self, x): 
        out = self.clshead(x)
        return out # x, out


class channel_attention(nn.Module):
    def __init__(self, n_channels, sequence_num=1000, inter=30):
        super(channel_attention, self).__init__()
        self.sequence_num = sequence_num
        self.n_channels = n_channels
        self.inter = inter
        self.extract_sequence = int(self.sequence_num / self.inter)  # You could choose to do that for less computation

        self.query = nn.Sequential(
            nn.Linear(n_channels, n_channels),
            nn.LayerNorm(n_channels),  # also may introduce improvement to a certain extent
            nn.Dropout(0.3)
        )
        self.key = nn.Sequential(
            nn.Linear(n_channels, n_channels),
            # nn.LeakyReLU(),
            nn.LayerNorm(n_channels),
            nn.Dropout(0.3)
        )

        # self.value = self.key
        self.projection = nn.Sequential(
            nn.Linear(n_channels, n_channels),
            # nn.LeakyReLU(),
            nn.LayerNorm(n_channels),
            nn.Dropout(0.3),
        )

        self.drop_out = nn.Dropout(0)
        self.pooling = nn.AvgPool2d(kernel_size=(1, self.inter), stride=(1, self.inter))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(dim=1)
        elif len(x.shape)  == 6:
            x = x.squeeze(dim=2)
            x = x.squeeze(dim=1)
        temp = rearrange(x, 'b o c s->b o s c')
        temp_query = rearrange(self.query(temp), 'b o s c -> b o c s')
        temp_key = rearrange(self.key(temp), 'b o s c -> b o c s')

        channel_query = self.pooling(temp_query)
        channel_key = self.pooling(temp_key)

        scaling = self.extract_sequence ** (1 / 2)

        channel_atten = torch.einsum('b o c s, b o m s -> b o c m', channel_query, channel_key) / scaling

        channel_atten_score = F.softmax(channel_atten, dim=-1)
        channel_atten_score = self.drop_out(channel_atten_score)

        out = torch.einsum('b o c s, b o c m -> b o c s', x, channel_atten_score)
        '''
        projections after or before multiplying with attention score are almost the same.
        '''
        out = rearrange(out, 'b o c s -> b o s c')
        out = self.projection(out)
        out = rearrange(out, 'b o s c -> b o c s')
        return out


class Transformer(nn.Module):
    def __init__(self, args, mod, emb_size=12, depth=1, is_gap: bool=False):
        super(Transformer, self).__init__()
        self.args = args
        self.mod = mod
        if self.mod == 0: ## only EEG
            input_ch = args.n_channels
        else:        ## other
            input_ch = 1 

        self.n_channels = input_ch

        self.sampling_rate = args.freq_time
        self.args = args
        self.n_classes = args.n_classes
  
        temporal_dim = args.freq_time
        
        self.ch_att = ResidualAdd(nn.Sequential(
                            nn.LayerNorm(temporal_dim),
                            channel_attention(self.n_channels),
                            nn.Dropout(0.5)
                        ))
        
        # self.patch_embedding = PatchEmbedding(emb_size, num_channels)
        self.patch_embedding = PatchEmbedding(emb_size, self.sampling_rate, self.n_channels)
        self.encoder = TransformerEncoder(depth, emb_size)
        self.clf_head = ClassificationHead(emb_size=12, n_classes=self.n_classes)
        
    
    def forward(self, x):
        if self.mod != 0:
            x = x.unsqueeze(dim=1) 
        x = x.unsqueeze(dim=1)

        x = self.ch_att(x)
        x = self.patch_embedding(x)
        x = self.encoder(x)
        # x = self.clf_head(x)

        return x


def average(out_list):
    return torch.mean(torch.stack(out_list, dim=1), dim=1)
    
def concat(out_list):
    return torch.cat(out_list, dim=1) 

def sum(out_list):
    return torch.sum(torch.stack(out_list, dim=1), dim=1) 



def data_resize(self, data_list):
    new_data_list = []
    for i, dim in enumerate(self.in_dim):
        new_data_list.append(data_list[i])  
    data_list = new_data_list
    return data_list      