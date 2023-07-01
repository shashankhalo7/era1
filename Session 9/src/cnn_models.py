import torch.nn as nn
import torch.nn.functional as F



class CustomDilatedNet(nn.Module):
    def __init__(self, dropout_value=0):
        super(CustomDilatedNet, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False), #input 3x32x32 output 64x32x32 RF 3X3
            nn.ReLU(),
            nn.BatchNorm2d(64), 
            nn.Dropout(dropout_value),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),  #input 64x32x32 Output 128x32x32 RF 5X5
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value), 
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), padding=1, bias=False), #input 128x32x32 Output 128x34x34 RF 5X5
        )
        self.pool1 = nn.MaxPool2d(2, 2) #input 128x34x34 Output 128x17x17 RF 6X6
        # Block with 1 Dilation
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False, dilation=2), #input 128x17x17 Output 128x15x15 RF 10X10
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False), #input 128x15x15 Output 128x15x15 RF 18X18
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value)
        )
        self.pool2 = nn.MaxPool2d(2, 2) #input 128x15x15 Output 128x7x7 RF 22X22
        # Block with Depthwise separable Convolution
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1,groups=128), #input 128x7x7 Output 128x7x7 RF 38X38
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=1,stride=1,padding=0,groups=1), #input 128x7x7 Output 128x7x7 RF 38*38
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=2, bias=False), #input 128x7x7 Output 128x9x9 RF 54X54
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value)
        )
        self.pool3 = nn.MaxPool2d(2, 2) #input 128x9X9 Output 128x4x4 RF 62X62
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=2, bias=False), #input 128x4x4 Output 128x6x6 RF 94X94
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=2, bias=False), #input 128x6x6 Output 128x8x8 RF  126X126 
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value)
        )
        self.gap = nn.AvgPool2d(kernel_size=8) #input 128x8x8 Output 128x1x1 RF  238X238
        self.output = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), padding=0, bias=False), #input 128x1x1 Output 64x1X1 RF 238X238
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=(1, 1), padding=0, bias=False), #input 64x1x1 Output 10x1x1 RF 238X238
        )

    def forward(self, x):
        x = self.convblock1(x)
        x = self.pool1(x)
        x = self.convblock2(x)
        x = self.pool2(x)
        x = self.convblock3(x)
        x = self.pool3(x)
        x = self.convblock4(x)
        x = self.gap(x)
        x = self.output(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
