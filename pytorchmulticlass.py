import torch
import torch.nn as nn
import torch.nn.functional as F

class PytorchMultiClass(nn.Module):
    def __init__(self, num_features):
        super(PytorchMultiClass, self).__init__()
        
        self.layer_1 = nn.Linear(num_features, 8192) # 2**15
        self.layer_2 = nn.Linear(8192, 4096)
        self.layer_3 = nn.Linear(4096, 2048)
        self.layer_4 = nn.Linear(2048, 1024)
        self.layer_5 = nn.Linear(1024, 512)
        self.layer_6 = nn.Linear(512, 256)
        self.layer_7 = nn.Linear(256, 128) 
        self.layer_out = nn.Linear(128, 104)
        self.softmax = nn.Softmax(dim=1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(8192)
        self.batchnorm2 = nn.BatchNorm1d(4096)
        self.batchnorm3 = nn.BatchNorm1d(2048)
        self.batchnorm4 = nn.BatchNorm1d(1024)
        self.batchnorm5 = nn.BatchNorm1d(512)
        self.batchnorm6 = nn.BatchNorm1d(256)
        self.batchnorm7 = nn.BatchNorm1d(128)


    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        
        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_4(x)
        x = self.batchnorm4(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_5(x)
        x = self.batchnorm5(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_6(x)
        x = self.batchnorm6(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_7(x)
        x = self.batchnorm7(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_out(x)
        return x # nn.CrossEntropyLoss does log_softmax() for us so we can simply return x