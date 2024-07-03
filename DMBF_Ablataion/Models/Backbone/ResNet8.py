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
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, kernel_size, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, track_running=True):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.bn0 = norm_layer(inplanes, track_running_stats=track_running)
        self.elu = nn.ELU(inplace=True)
        self.dropdout0 = nn.Dropout(p=0.1)

        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=kernel_size//2,bias=False)
        self.bn1 = norm_layer(planes, track_running_stats=track_running)
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

class Resnet8(nn.Module):
    def __init__(self, args, m,
                batch_norm=True, batch_norm_alpha=0.1): 
#    def __init__(self, args, input_ch=32 , batch_norm=True, batch_norm_alpha=0.1):
        super(Resnet8, self).__init__()

        self.track_running=True
        if m == 0: ## only EEG
            input_ch = args.n_channels
        else:        ## other
            input_ch = 1
        self.m = m
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
        self.conv1 = nn.Conv1d(input_ch, 32, kernel_size=13, stride=2, padding=3,
                               bias=False)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=3, padding=1)
        self.elu = nn.ELU(inplace=True)

        block = BasicBlock

        layers = [1,1,1]
        kernel_sizes = [11, 9, 7]
        self.layer1 = self._make_layer(block, 32, kernel_sizes[0], layers[0], stride=1, layer_num=1)
        self.layer2 = self._make_layer(block, 128, kernel_sizes[1], layers[1], stride=1, layer_num=2)
        self.layer3 = self._make_layer(block, 256, kernel_sizes[2],layers[2], stride=2, layer_num=3)

        self.avgpool = nn.AdaptiveAvgPool1d(1)


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
                norm_layer(planes * block.expansion, track_running_stats=self.track_running),)

        layers = []
        layers.append(block(self.inplanes, planes, kernel_size,stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, self.track_running))
        
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, track_running=self.track_running))

        return nn.Sequential(*layers)

    def forward(self, x):
        # See note [TorchScript super()]
        if self.m == 0:
            x= x.squeeze(1)

        x = self.conv1(x)
        x = self.layer1(x) # basic block 2개
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.maxpool(x)
        x = self.layer3(x)

        x = self.elu(x)
        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        return x
        