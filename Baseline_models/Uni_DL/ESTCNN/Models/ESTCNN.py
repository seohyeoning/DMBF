import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class CoreBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CoreBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3))
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3))
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=2)

        # Initialize weights
        self.apply(self.init_weights)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        return x

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.n_channels = args.n_channels
        self.n_classes = args.n_classes

        self.core_block1 = CoreBlock(1, 16)
        self.core_block2 = CoreBlock(16, 32)
        self.core_block3 = CoreBlock(32, 64)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1)) # Automatically adapt to the required output size
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(64, 50) # After adaptive pooling, the size will be (batch_size, 64, 1, 1)
        self.dense2 = nn.Linear(50, self.n_classes)

        # Initialize weights for the dense layers
        self.dense1.apply(self.init_weights)
        self.dense2.apply(self.init_weights)

    def forward(self, x):
        if self.args.data_type == "SA_Drowsy_bal" or self.args.data_type == "SA_Drowsy_unbal":
            x = x.unsqueeze(dim=1)
        else:
            x = x[0].unsqueeze(dim=1)
        x = self.core_block1(x)
        x = self.core_block2(x)
        x = self.core_block3(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        return F.log_softmax(x, dim=1)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)

# Assuming `input_data` is a torch.Tensor of shape (batch_size, 1, 30, 100) representing the EEG signals
# model = ESTCNN()
# input_data = torch.randn(batch_size, 1, 30, 100)
# output = model(input_data)
