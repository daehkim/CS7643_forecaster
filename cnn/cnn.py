#!/apps/anaconda3/bin/python

import torch.nn as nn
import torch

from torch.utils.data import Dataset, DataLoader

#dataset loader
class VolDataset(Dataset):
    def __init__(self,feature,target):
        self.feature = feature
        self.target = target
    
    def __len__(self):
        return len(self.feature)
    
    def __getitem__(self,idx):
        item = self.feature[idx]
        label = self.target[idx]
        
        return item,label

# Here we define our model as a class
class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, fc1_input_dim, fc1_output_dim, fc2_input_dim, fc2_output_dim,
                 device, kernel_size=1, stride=1, padding=0, model='CNN'):
        super(CNN, self).__init__()

        # Initialize the variables
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.fc1_input_dim = fc1_input_dim
        self.fc1_output_dim = fc1_output_dim
        self.fc2_input_dim = fc2_input_dim
        self.fc2_output_dim = fc2_output_dim 
        self.model = model

        assert self.kernel_size == 1            #time series not images
        assert self.stride == 1                 #time series, dont want to skip obs
        assert self.padding == 0                #same as above

        #(in, out, kernel size)
        self.conv1d = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size) 
        self.fc1 = nn.Linear(self.fc1_input_dim, self.fc1_output_dim)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(self.fc2_input_dim, self.fc2_output_dim)

        self.device = device

    def forward(self, x):
        x = self.conv1d(x)
        x = x.view(-1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
