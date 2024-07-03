import torch
import torch.nn as nn

class EEGNet4(nn.Module): # 4 temporal, 2 spatial per temporal
    def __init__(self, args, mod, track_running=True): ### use only EEG
        super(EEGNet4, self).__init__()
        self.args = args
        self.mod = mod
        if self.mod == 0: ## only EEG
            input_ch = args.n_channels  
        else:        ## other
            input_ch = 1 
        self.modal_index = mod
        self.n_classes = args.n_classes
        freq = args.freq_time #################### num_seg = frequency*window size

        self.convnet = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(1, freq//2), stride=1, bias=False, padding=(0 , freq//4)),
            nn.BatchNorm2d(4, track_running_stats=track_running),
            nn.Conv2d(4, 8, kernel_size=(input_ch, 1), stride=1, groups=4),
            nn.BatchNorm2d(8, track_running_stats=track_running),
            nn.ELU(),
            # nn.AdaptiveAvgPool2d(output_size = (1,265)),
            nn.AvgPool2d(kernel_size=(1,4)),
            nn.Dropout(p=0.25),
            nn.Conv2d(8, 8, kernel_size=(1,freq//4),padding=(0,freq//4), groups=8),
            nn.Conv2d(8, 8, kernel_size=(1,1)),
            nn.BatchNorm2d(8, track_running_stats=track_running),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p=0.25),
            )
    
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(dim=1)
        x = x.unsqueeze(dim=1)
        # x = x.unsqueeze(dim=1).permute(0,1,2,3)  
        out = self.convnet(x)

        return out