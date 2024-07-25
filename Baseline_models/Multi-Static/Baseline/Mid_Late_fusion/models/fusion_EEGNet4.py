import torch.nn as nn
import torch
import torch.nn.functional as F
from Helpers.Variables import device

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.modalities = len(args.in_dim)
        self.args = args
        self.n_classes = args.n_classes
        self.in_dim = args.in_dim
        self.FeatureExtractor = nn.ModuleList([EEGNet4(args, m) for m in range(self.modalities)])

        if args.data_type == 'Drowsy':
            hid_dim = 296
        
        elif args.data_type == 'Pilot_Fatigue':
            hid_dim = 368

        elif args.data_type == 'Stress' or args.data_type == 'Distraction':
            hid_dim = 200
        
        elif args.data_type == 'manD':
            hid_dim = 384                       

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

            out = out.view(out.size(0), -1)
            output = F.softmax(self.classifier(out))

        
        elif self.args.model_type == 'logit_based':
            for mod in range(self.modalities):
                feat_dict[mod] = self.FeatureExtractor[mod](data_list[mod])
                logit_dict[mod] = self.classifier(feat_dict[mod].view(feat_dict[mod].size(0), -1))
                
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
    
class EEGNet4(nn.Module): # 4 temporal, 2 spatial per temporal
    def __init__(self, args, mod, track_running=True): ### use only EEG
        super(EEGNet4, self).__init__()
        self.args = args
        self.mod = mod
        if self.mod == 0: ## only EEG
            input_ch = args.n_channels
        else:        ## other
            input_ch = 1 
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
        if self.mod != 0:
            x = x.unsqueeze(dim=1) 
        x = x.unsqueeze(dim=1)
        out = self.convnet(x)
        out = out.squeeze()
        # output = self.classifier(out)
        return out
    
def data_resize(self, data_list):
    new_data_list = []
    for i, dim in enumerate(self.in_dim):
        new_data_list.append(data_list[i])  
    data_list = new_data_list
    return data_list      