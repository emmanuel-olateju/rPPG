import torch
import torch.nn as nn
import torch.nn.functional as F

class rPPG_PSD_MLP(nn.Module):
    def __init__(self):
        super(rPPG_PSD_MLP, self).__init__()
        self.drop1 = nn.Dropout(0.1)
        self.btc1 = nn.BatchNorm1d(32)
        self.fc1 = nn.Linear(2 * 16, 1000)  # First fully connected layer
        self.btc_conv1 = nn.BatchNorm1d(1000)
        self.conv1 = nn.Conv1d(1,1,kernel_size=101)
        self.fc2 = nn.Linear(1 * 900, 16)      # Second fully connected layer

    def forward(self, x):
        x = x.view(-1, 2 * 16)  # Flatten the input tensor
        x = self.drop1(x)
        x = self.btc1(x)
        x = F.leaky_relu(self.fc1(x))
        x = self.btc_conv1(x)
        x = x.view(x.size(0),1,x.size(-1))
        x = F.leaky_relu(self.conv1(x))
        x = x.view(x.size(0),x.size(-1))
        x = self.fc2(x)
        
        
        return x