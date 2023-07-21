import torch
import torch.nn as nn
import torch.nn.functional as F



import torch.nn.functional as F
dropout_value = 0.05
class CustomResnet(nn.Module):
    def __init__(self):
        super(CustomResnet, self).__init__()
        self.prep_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False), #input 3x32x32 output 64x32x32 RF 3X3
            nn.ReLU(),
            nn.BatchNorm2d(64),
            #nn.Dropout(dropout_value),
        )

        # Conv Block 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False), #input 128x17x17 Output 128x15x15 RF 10X10
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            #nn.Dropout(dropout_value),
        )

        # Res Block 1
        self.res_block1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False), #input 3x32x32 output 64x32x32 RF 3X3
            nn.ReLU(),
            nn.BatchNorm2d(128),
            #nn.Dropout(dropout_value),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False), #input 3x32x32 output 64x32x32 RF 3X3
            nn.ReLU(),
            nn.BatchNorm2d(128),
            #nn.Dropout(dropout_value),
        )
        
        #Conv Block 2
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False), #input 128x17x17 Output 128x15x15 RF 10X10
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            #nn.Dropout(dropout_value),
        )

        # Conv Block 3
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, bias=False), #input 128x17x17 Output 128x15x15 RF 10X10
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            #nn.Dropout(dropout_value),
        )

        # Res Block 3
        self.res_block3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, bias=False), #input 3x32x32 output 64x32x32 RF 3X3
            nn.ReLU(),
            nn.BatchNorm2d(512),
            #nn.Dropout(dropout_value),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, bias=False), #input 3x32x32 output 64x32x32 RF 3X3
            nn.ReLU(),
            nn.BatchNorm2d(512),
            #nn.Dropout(dropout_value),
        )

        self.mp = nn.MaxPool2d(4,2) #input 128x8x8 Output 128x1x1 RF  238X238
        self.output = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=10, kernel_size=(1, 1), padding=0, bias=False), #input 128x1x1 Output 64x1X1 RF 238X238
        )

    def forward(self, x):
        x = self.prep_layer(x)
        x = self.convblock1(x)
        r1 = self.res_block1(x)
        x = x + r1
        x = self.convblock2(x)
        x = self.convblock3(x)
        r3 = self.res_block3(x)
        x = x + r3
        x = self.mp(x)
        x = self.output(x)
        x = x.view(-1, 10)
        return x
