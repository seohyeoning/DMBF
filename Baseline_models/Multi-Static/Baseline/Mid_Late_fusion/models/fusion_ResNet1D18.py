import torch.nn as nn
import torch
import torch.nn.functional as F
from Helpers.Variables import device

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
    def __init__(self, args, mod,
                batch_norm=True, batch_norm_alpha=0.1):
        super(Resnet18, self).__init__()

        self.args = args
        self.mod = mod
        if self.mod == 0:
            input_ch = args.n_channels
        else:
            input_ch = 1 

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
        # self.fc = nn.Linear(256 * block.expansion, self.num_classes)

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
        if self.mod != 0:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.layer1(x) # basic block 2개
        x = self.layer2(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        x = self.maxpool(x)
        x = self.layer4(x)

        x = self.elu(x)
        x = self.avgpool(x)

        # x = torch.flatten(x, 1)
        # x = self.fc(x) # classifier

        return x
        
class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.modalities = len(args.in_dim)
        self.args = args
        self.n_classes = args.n_classes
        self.in_dim = args.in_dim
        self.FeatureExtractor = nn.ModuleList([Resnet18(args, m) for m in range(self.modalities)])

        if args.data_type == "MS" or args.data_type == 'manD':
            hid_dim = 256
        elif args.data_type == "Distraction" or args.data_type == "Stress":
            hid_dim = 256
        elif args.data_type == "Drowsy":
            hid_dim = 256
        elif args.data_type == "Pilot_Fatigue":
            hid_dim = 256

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

            out = torch.flatten(out, 1)
            output = F.softmax(self.classifier(out))

        
        elif self.args.model_type == 'logit_based':
            for mod in range(self.modalities):
                feat_dict[mod] = torch.flatten(self.FeatureExtractor[mod](data_list[mod]),1)
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