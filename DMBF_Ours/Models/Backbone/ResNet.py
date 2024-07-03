import torch
import torch.nn as nn
"""
ResNet-1D in PyTorch.
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
"""
DROPOUT_RATIO = 0.1 # 0.1
        
class conv_task(nn.Module):
    
    def __init__(self, in_planes, planes, kernel_size=3, stride=1, padding=None, bias=False, momentum=0.1):
        super(conv_task, self).__init__()
        
        if padding is None:
            padding = kernel_size//2
        
        self.conv = nn.Conv1d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm1d(planes, momentum=momentum)
    
    def forward(self, x):
        
        y = self.conv(x)
        y = self.bn(y)

        return y

# No projection: identity shortcut
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, kernel_size=3, stride=1, downsample=None, momentum=0.1):
        super(BasicBlock, self).__init__()
        
        self.inplanes = in_planes
        self.planes = planes
        
        self.conv1 = conv_task(in_planes, planes, kernel_size, stride, bias=False, momentum=momentum)
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout(p=DROPOUT_RATIO)
        
        self.conv2 = conv_task(planes, planes, kernel_size, 1, bias=False, momentum=momentum)
        self.elu2 = nn.ELU()
        self.dropout2 = nn.Dropout(p=DROPOUT_RATIO)
        
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        y = self.conv1(x)
        y = self.elu1(y)
        y = self.dropout1(y)
        
        y = self.conv2(y)
        
        if self.downsample:
            identity = self.downsample(x)
                
        y += identity
        
        y = self.elu2(y)
        y = self.dropout2(y)
        return y
    
class Resnet(nn.Module):
    def __init__(self, args, m, momentum=0.1):
        super(Resnet, self).__init__()
        self.n_classes = args.n_classes
        if m == 0: ## only EEG
            self.input_ch = args.n_channels
        else:        ## other
            self.input_ch = 1
        self.m=m
        
        self.momentum=momentum
        self.inplanes = 32
        self.n_outputs = 256
        
        self.elu = nn.ELU()
        
    def set_pre_layers(self):
        raise NotImplementedError
    
    def custom_forward(self, x):
        raise NotImplementedError
    
    def forward(self, x):
        output = self.custom_forward(x)
        return output
        
    def _make_layer(self, block, planes, kernel_size, blocks, stride=2):
        downsample = None
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes,planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion, momentum=self.momentum))

        layers = []
        layers.append(block(self.inplanes, planes, kernel_size, stride, downsample=downsample, momentum=self.momentum))
        
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size, 1, momentum=self.momentum))
        return nn.Sequential(*layers)
    

class Resnet8(Resnet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.set_pre_layers()
        block = BasicBlock

        layers = [1,1,1]
        kernel_sizes = [11, 9, 7]
        self.layer1 = self._make_layer(block, 32, kernel_sizes[0], layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, kernel_sizes[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, kernel_sizes[2],layers[2], stride=2)
        
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=3, padding=1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
    
    def set_pre_layers(self):
        self.pre_layers_conv = conv_task(self.input_ch, 32, kernel_size=13, stride=2, bias=False)
        self.dropdout = nn.Dropout(p=DROPOUT_RATIO)

    def custom_forward(self, x):
        if self.m == 0:
            x= x.squeeze(1)

        x = self.pre_layers_conv(x)
        x = self.elu(x)
        x = self.dropdout(x)
        
        x = self.layer1(x) # basic block 2개
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        

        x = self.elu(x)
        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        return x



class Resnet18(Resnet):
    def __init__(self, *args, **kwargs):
        super(Resnet18, self).__init__(*args, **kwargs)
                
        self.set_pre_layers()
            
        block = BasicBlock
        nblocks = [2,2,2,2]
        
        kernel_sizes = [3,3,3,3]
        self.layer1 = self._make_layer(block, 32, kernel_sizes[0], nblocks[0], stride=2)
        self.layer2 = self._make_layer(block, 64, kernel_sizes[1], nblocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, kernel_sizes[2], nblocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, kernel_sizes[3], nblocks[3], stride=2)
        
        self.maxpool = nn.MaxPool1d(kernel_size=4, stride=4, padding=0)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
    def set_pre_layers(self):
        self.pre_layers_conv = conv_task(self.input_ch, 32, kernel_size=7, stride=1)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=DROPOUT_RATIO)
        
    def custom_forward(self, x):
        if self.m == 0:
            x = x.squeeze(1)
        
        x = self.pre_layers_conv(x)
        x = self.elu(x)
        x = self.dropout(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        x = self.maxpool(x)
        x = self.layer4(x)
        
        x = self.elu(x)
        x = self.avgpool(x)
        
        x = torch.flatten(x, 1)
        
        return x
    

