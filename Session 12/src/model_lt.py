import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from __future__ import print_function
import torch.optim as optim
from torchvision import datasets, transforms
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from torchsummary import summary
from tqdm import tqdm
from torch_lr_finder import LRFinder
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


dropout_value = 0.05

class CustomResnet(pl.LightningModule):
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
        
        self.criterion = nn.CrossEntropyLoss(reduction='sum')

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
    
    def configure_optimizer(self):
        optimizer = optim.Adam(self.parameters(), lr=8e-04,weight_decay = 1e-4)
        scheduler = OneCycleLR(optimizer, max_lr=8e-02, steps_per_epoch=len(train_loader), epochs=20,div_factor=100,pct_start = 5/20)
        return [optimizer],[scheduler]
    
    def training_step(self, batch, batch_idx):
        data, target = batch
        y_pred = self(data)
        train_loss = self.criterion(y_pred, target)
        self.log("train_loss", train_loss)
        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct = pred.eq(target.view_as(pred)).sum().item()
        processed = len(data)
        train_accuracy = 100*float(correct)/float(processed)
        self.log("train_accuracy", train_accuracy)
        return loss
    
    def validation_step(self,batch,batch_idx):
        data, target = batch
        output = self(data)
        test_loss += self.criterion(output, target).item()  # sum up batch loss
        self.log("test_loss", test_loss)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct = pred.eq(target.view_as(pred)).sum().item()
        processed = len(data)
        test_accuracy = 100*float(correct)/float(processed)
        self.log("test_accuracy", test_accuracy)
        return test_loss
    
    def predict_step(self, batch, batch_idx):
        data = batch
        output = self(data)
        pred = output.argmax(dim=1, keepdim=True)
        return pred
    
        
        
        
        
        
        
        
        
    
    