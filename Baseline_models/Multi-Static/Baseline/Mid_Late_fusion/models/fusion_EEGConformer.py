import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor

import math

from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
# from common_spatial_pattern import csp


class PatchEmbedding(nn.Module):
    def __init__(self, args, mod, emb_size=40):
        # self.patch_size = patch_size
        super().__init__()
        self.args = args

        self.mod = mod
        if self.mod == 0: ## only EEG
            input_ch = args.n_channels
        else:        ## other
            input_ch = 1 

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (input_ch, 1), (1, 1)), # original:  nn.Conv2d(40, 40, (22, 1), (1, 1))
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
        if self.mod != 0:
            x = x.unsqueeze(dim=1) 
        x = x.unsqueeze(dim=1)
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
    def __init__(self, emb_size, n_classes, args):
        super().__init__()
        if args.data_type == 'Drowsy':
            hid_dim = 1360
        elif args.data_type == 'Distraction' or args.data_type == 'Stress':
            hid_dim = 840
        elif args.data_type == 'Pilot_Fatigue':
            hid_dim = 1760 
        elif args.data_type == 'manD':
            hid_dim = 1800

        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.fc = nn.Sequential(
            nn.Linear(hid_dim, 256), 
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3))
            # nn.Linear(32, n_classes) # classifier  삭제
        

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return out # original: x, out => tocken과 output


class EEGConformer(nn.Sequential):
    def __init__(self, args, mod, emb_size=40, depth=6, n_classes=2, **kwargs):
        super().__init__(

            PatchEmbedding(args, mod, emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes, args)
        )

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.modalities = len(args.in_dim)
        self.args = args
        self.n_classes = args.n_classes
        self.in_dim = args.in_dim
        self.FeatureExtractor = nn.ModuleList([EEGConformer(args, m) for m in range(self.modalities)])

        hid_dim = 32

        self.classifier = nn.Sequential(nn.Linear(hid_dim, self.n_classes))  

        if args.model_type == 'feature_based' and args.fusion_type == 'concat':
            self.classifier = nn.Sequential(nn.Linear(hid_dim*self.modalities, self.n_classes))   

        if args.model_type == 'logit_based' and args.fusion_type == 'concat' :
            self.fc_layer = nn.Sequential(nn.Linear(self.n_classes*self.modalities, self.n_classes)) # for logit-based concat fusion
            
 

    def forward(self, x):
        data_list = data_resize(self, x)
        feat_dict = dict()
        logit_dict = dict()

        if self.args.model_type == 'feature_based':
            for mod in range(self.modalities):
                feat_dict[mod] = self.FeatureExtractor[mod](data_list[mod])
            feat_list = list(feat_dict.values())

            if self.args.fusion_type == 'average':
                out = average(feat_list)
            elif self.args.fusion_type == 'concat':
                out = concat(feat_list)
            elif self.args.fusion_type == 'sum':
                out = sum(feat_list)
            output = F.softmax(self.classifier(out))

        
        elif self.args.model_type == 'logit_based':
            for mod in range(self.modalities):
                feat_dict[mod] = self.FeatureExtractor[mod](data_list[mod])
                logit_dict[mod] = self.classifier(feat_dict[mod])
            logit_list = list(logit_dict.values())

            if self.args.fusion_type == 'average':
                output = F.softmax(average(logit_list))
            elif self.args.fusion_type == 'concat':
                output = concat(logit_list)
                output = F.softmax(self.fc_layer(output))
            elif self.args.fusion_type == 'sum':
                output = F.softmax(sum(logit_list))

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
        new_data_list.append(data_list[i])  
    data_list = new_data_list
    return data_list      