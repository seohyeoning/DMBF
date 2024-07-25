import torch.nn as nn
import torch
import torch.nn.functional as F
from Helpers.Variables import device


class DeepConvNet(nn.Module):
    def __init__(self, args, mod, batch_norm=True, batch_norm_alpha=0.1):
        super(DeepConvNet, self).__init__()

        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.n_classes = args.n_classes
        input_time = args.freq_time
        self.mod = mod
        if self.mod == 0: ## only EEG
            input_ch = args.n_channels
        else:        ## other
            input_ch = 1 
        n_ch1 = 25
        n_ch2 = 50
        n_ch3 = 100
        self.n_ch4 = 200

        if self.batch_norm:
            self.convnet = nn.Sequential(
                nn.Conv2d(1, n_ch1, kernel_size=(1, 10), stride=1),
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(input_ch, 1), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch1, momentum=self.batch_norm_alpha, affine=True, eps=1e-5, track_running_stats=True),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),
                
                nn.Conv2d(n_ch1, n_ch2, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch2, momentum=self.batch_norm_alpha, affine=True, eps=1e-5, track_running_stats=True),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),

                nn.Conv2d(n_ch2, n_ch3, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch3, momentum=self.batch_norm_alpha, affine=True, eps=1e-5, track_running_stats=True),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),

                nn.Conv2d(n_ch3, self.n_ch4, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(self.n_ch4, momentum=self.batch_norm_alpha, affine=True, eps=1e-5, track_running_stats=True),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)), 
                )
        else:
            self.convnet = nn.Sequential(
                nn.Conv2d(1, n_ch1, kernel_size=(1, 10), stride=1,bias=False),
                nn.BatchNorm2d(n_ch1, momentum=self.batch_norm_alpha,
                               affine=True, eps=1e-5, track_running_stats=True),
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(input_ch, 1), stride=1),
                # nn.InstanceNorm2d(n_ch1),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch1, n_ch2, kernel_size=(1, 10), stride=1),
                # nn.InstanceNorm2d(n_ch2),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch2, n_ch3, kernel_size=(1, 10), stride=1),
                # nn.InstanceNorm2d(n_ch3),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch3, self.n_ch4, kernel_size=(1, 10), stride=1),
                # nn.InstanceNorm2d(self.n_ch4),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            )
        self.convnet.eval()
        out = self.convnet(torch.zeros(16, 1, input_ch, input_time))
        
        n_out_time = out.cpu().data.numpy().shape[3]
        self.final_conv_length = n_out_time

        self.n_outputs = out.size()[1]*out.size()[2]*out.size()[3]

        # self.clf = nn.Sequential(nn.Linear(self.n_outputs, self.n_classes), nn.Dropout(p=0.2))  ####################### classifier 
        # DG usually doesn't have classifier
        # so, add at the end

    def forward(self, x):
        if self.mod != 0: 
            x = x.unsqueeze(dim=1)
        x = x.unsqueeze(dim=1)
        output = self.convnet(x)
        # output = output.view(output.size()[0], -1)
        # output = self.l2normalize(output)
        # output=self.clf(output) 
        return output



class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.modalities = len(args.in_dim)
        self.args = args
        self.n_classes = args.n_classes
        self.in_dim = args.in_dim
        self.FeatureExtractor = nn.ModuleList([DeepConvNet(args, m) for m in range(self.modalities)])
        if args.data_type == "Distraction" or args.data_type == "Stress" :
            hid_dim = 3200  
        elif args.data_type == "Drowsy":
            hid_dim = 5800
        elif args.data_type == "Pilot_Fatigue":
            hid_dim = 7600
        elif args.data_type == 'manD':
            hid_dim = 7800

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

            output = out.view(out.size()[0], -1)
            output = F.softmax(self.classifier(output))

        
        elif self.args.model_type == 'logit_based':
            for mod in range(self.modalities):
                feat_dict[mod] = self.FeatureExtractor[mod](data_list[mod]).view(out.size()[0], -1)
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