
dropout_value = 0.1
group_size = 2

class Net(nn.Module):
    def __init__(self,norm = 'bn'):
        super(Net, self).__init__()
        # Input Block size = 32
        
        if  norm == 'bn':
            self.n1 = nn.BatchNorm2d(8)
        elif norm == 'gn':
            self.n1 = nn.GroupNorm(group_size,8)
        elif norm == 'ln':
            self.n1 = nn.GroupNorm(1,8)
        
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            self.n1,
            nn.Dropout(dropout_value)
        ) # output_size = 32

        # CONVOLUTION BLOCK 1
        
        if  norm == 'bn':
            self.n2 = nn.BatchNorm2d(8)
        elif norm == 'gn':
            self.n2 = nn.GroupNorm(group_size,8)
        elif norm == 'ln':
            self.n2 = nn.GroupNorm(1,8)
            
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            self.n2,
            nn.Dropout(dropout_value)
        ) # output_size = 32


        if  norm == 'bn':
            self.n3 = nn.BatchNorm2d(8)
        elif norm == 'gn':
            self.n3 = nn.GroupNorm(group_size,8)
        elif norm == 'ln':
            self.n3 = nn.GroupNorm(1,8)
            
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            self.n3,
            nn.Dropout(dropout_value)
        ) # output_size = 32

        # TRANSITION BLOCK 1
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 32
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 16

        # CONVOLUTION BLOCK 2
        
        if  norm == 'bn':
            self.n4 = nn.BatchNorm2d(16)
        elif norm == 'gn':
            self.n4 = nn.GroupNorm(group_size,16)
        elif norm == 'ln':
            self.n4 = nn.GroupNorm(1,16)
            
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            self.n4,
            nn.Dropout(dropout_value)
        ) # output_size = 16

        if  norm == 'bn':
            self.n5 = nn.BatchNorm2d(16)
        elif norm == 'gn':
            self.n5 = nn.GroupNorm(group_size,16)
        elif norm == 'ln':
            self.n5 = nn.GroupNorm(1,16)
            
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            self.n5,
            nn.Dropout(dropout_value)
        ) # output_size = 16

        if  norm == 'bn':
            self.n6 = nn.BatchNorm2d(16)
        elif norm == 'gn':
            self.n6 = nn.GroupNorm(group_size,16)
        elif norm == 'ln':
            self.n6 = nn.GroupNorm(1,16)
            
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            self.n6,
            nn.Dropout(dropout_value)
        ) # output_size = 16

         # TRANSITION BLOCK 2
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 16
        self.pool2 = nn.MaxPool2d(2, 2) # output_size = 8

          # CONVOLUTION BLOCK 3
          
        if  norm == 'bn':
            self.n7 = nn.BatchNorm2d(32)
        elif norm == 'gn':
            self.n7 = nn.GroupNorm(group_size,32)
        elif norm == 'ln':
            self.n7 = nn.GroupNorm(1,32)
            
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            self.n7,
            nn.Dropout(dropout_value)
        ) # output_size = 8
        
        if  norm == 'bn':
            self.n8 = nn.BatchNorm2d(32)
        elif norm == 'gn':
            self.n8 = nn.GroupNorm(group_size,32)
        elif norm == 'ln':
            self.n8 = nn.GroupNorm(1,32)
            
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            self.n8,
            nn.Dropout(dropout_value)
        ) # output_size = 8
        
        if  norm == 'bn':
            self.n9 = nn.BatchNorm2d(32)
        elif norm == 'gn':
            self.n9 = nn.GroupNorm(group_size,32)
        elif norm == 'ln':
            self.n9 = nn.GroupNorm(1,32)
            
        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            self.n9,
            nn.Dropout(dropout_value)
        ) # output_size = 8

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=8)
        ) # output_size = 1

        self.flat = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        )


        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x_temp = self.convblock1(x)
        x = self.convblock2(x_temp)
        x = self.convblock3(x) + x_temp
        x = self.convblock4(x)
        x = self.pool1(x)
        x_temp = self.convblock5(x)
        x = self.convblock6(x_temp)
        x = self.convblock7(x) + x_temp
        x = self.convblock8(x)
        x = self.pool2(x)
        x_temp = self.convblock9(x)
        x = self.convblock10(x_temp)
        x = self.convblock11(x) + x_temp
        x = self.gap(x)
        x = self.flat(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    
    
    
    
class Net_session7(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value)
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 24

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 10
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 8
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 6
        """self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 6
        """
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # output_size = 1

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        )


        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        #x = self.convblock7(x)
        x = self.gap(x)
        x = self.convblock8(x)
        x = self.convblock9(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
