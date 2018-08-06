import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


def reorg(x):
    stride = 2
    assert(x.data.dim() == 4)
    B = x.data.size(0)
    C = x.data.size(1)
    H = x.data.size(2)
    W = x.data.size(3)
    assert(H % stride == 0)
    assert(W % stride == 0)
    ws = stride
    hs = stride
    x = x.view(B, C, int(H/hs), hs, int(W/ws), ws).transpose(3,4).contiguous()
    x = x.view(B, C, int(H/hs*W/ws), hs*ws).transpose(2,3).contiguous()
    x = x.view(B, C, hs*ws, int(H/hs), int(W/ws)).transpose(1,2).contiguous()
    x = x.view(B, hs*ws*C, int(H/hs), int(W/ws))
    return x



class ComplexYOLO(nn.Module):
    def __init__(self):
        super(ComplexYOLO, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=3,out_channels=24,kernel_size=3,stride=1,padding=1)
        self.bn_1   = nn.BatchNorm2d(num_features=24)
        self.pool_1 = nn.MaxPool2d(2)

        self.conv_2 = nn.Conv2d(in_channels=24,out_channels=48,kernel_size=3,stride=1,padding=1)
        self.bn_2   = nn.BatchNorm2d(num_features=48)
        self.pool_2 = nn.MaxPool2d(2)

        self.conv_3 = nn.Conv2d(in_channels=48,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.bn_3   = nn.BatchNorm2d(num_features=64)
        self.conv_4 = nn.Conv2d(in_channels=64,out_channels=32,kernel_size=1,stride=1,padding=0)
        self.bn_4   = nn.BatchNorm2d(num_features=32)
        self.conv_5 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.bn_5   = nn.BatchNorm2d(num_features=64)
        self.pool_3 = nn.MaxPool2d(2)

        self.conv_6 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.bn_6   = nn.BatchNorm2d(num_features=128)
        self.conv_7 = nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.bn_7   = nn.BatchNorm2d(num_features=64)
        self.conv_8 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.bn_8   = nn.BatchNorm2d(num_features=128)
        self.pool_4 = nn.MaxPool2d(2)

        self.conv_9  = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.bn_9    = nn.BatchNorm2d(num_features=256)
        self.conv_10 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=1,stride=1,padding=0)
        self.bn_10   = nn.BatchNorm2d(num_features=256)
        self.conv_11 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.bn_11   = nn.BatchNorm2d(num_features=512)
        self.pool_5  = nn.MaxPool2d(2)

        self.conv_12 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.bn_12   = nn.BatchNorm2d(num_features=512)
        self.conv_13 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=1,stride=1,padding=0)
        self.bn_13   = nn.BatchNorm2d(num_features=512)
        self.conv_14 = nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,stride=1,padding=1)
        self.bn_14   = nn.BatchNorm2d(num_features=1024)
        self.conv_15 = nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,stride=1,padding=1)
        self.bn_15   = nn.BatchNorm2d(num_features=1024)
        self.conv_16 = nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,stride=1,padding=1)
        self.bn_16   = nn.BatchNorm2d(num_features=1024)

        self.conv_17 = nn.Conv2d(in_channels=2048,out_channels=1024,kernel_size=3,stride=1,padding=1)
        self.bn_17   = nn.BatchNorm2d(num_features=1024)
        self.conv_18 = nn.Conv2d(in_channels=1024,out_channels=75,kernel_size=1,stride=1,padding=0)

        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.relu(self.bn_1(self.conv_1(x)))
        x = self.pool_1(x)
        
        x = self.relu(self.bn_2(self.conv_2(x)))
        x = self.pool_2(x)

        x = self.relu(self.bn_3(self.conv_3(x)))
        x = self.relu(self.bn_4(self.conv_4(x)))
        x = self.relu(self.bn_5(self.conv_5(x)))
        x = self.pool_3(x)

        x = self.relu(self.bn_6(self.conv_6(x)))
        x = self.relu(self.bn_7(self.conv_7(x)))
        x = self.relu(self.bn_8(self.conv_8(x)))
        x = self.pool_4(x)
        
        x = self.relu(self.bn_9(self.conv_9(x)))
        route_1 = x            # 12 layer
        reorg_result = reorg(route_1)
        
        x = self.relu(self.bn_10(self.conv_10(x)))
        x = self.relu(self.bn_11(self.conv_11(x)))
        x = self.pool_5(x)

        x = self.relu(self.bn_12(self.conv_12(x)))
        x = self.relu(self.bn_13(self.conv_13(x)))
        x = self.relu(self.bn_14(self.conv_14(x)))
        x = self.relu(self.bn_15(self.conv_15(x)))
        x = self.relu(self.bn_16(self.conv_16(x)))

        x = torch.cat((reorg_result,x),1)
        x = self.relu(self.bn_17(self.conv_17(x)))
        x = self.conv_18(x)

        return x
