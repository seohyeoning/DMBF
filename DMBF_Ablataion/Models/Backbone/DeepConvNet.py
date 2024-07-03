import torch
import torch.nn as nn

class DeepConvNet(nn.Module):
    def __init__(self, args, input_ch=4,
                 batch_norm=True, batch_norm_alpha=0.1):
        super(DeepConvNet, self).__init__()

        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.n_classes = args.n_classes
        input_time = args.freq_time
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
                # MixStyle(p=0.5, alpha=0.1, mix="random"), # MixStyle
                nn.Dropout(p=0.5),
                
                nn.Conv2d(n_ch1, n_ch2, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch2, momentum=self.batch_norm_alpha, affine=True, eps=1e-5, track_running_stats=True),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                # MixStyle(p=0.5, alpha=0.1, mix="random"), # MixStyle
                nn.Dropout(p=0.5),

                nn.Conv2d(n_ch2, n_ch3, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch3, momentum=self.batch_norm_alpha, affine=True, eps=1e-5, track_running_stats=True),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                # MixStyle(p=0.5, alpha=0.1, mix="random"), # MixStyle
                nn.Dropout(p=0.5),

                nn.Conv2d(n_ch3, self.n_ch4, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(self.n_ch4, momentum=self.batch_norm_alpha, affine=True, eps=1e-5, track_running_stats=True),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)), 
                # MixStyle(p=0.5, alpha=0.1, mix="random"), # MixStyle, maxpool 뒤에 mixstyle 추가
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


    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)

        return output



