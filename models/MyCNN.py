import torch 
import torch.nn as nn

class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.batch_norm1 = nn.BatchNorm2d(3)
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(32, 128, kernel_size=3, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.batch_norm4 = nn.BatchNorm2d(128)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.batch_norm5 = nn.BatchNorm2d(256)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        
    def forward(self, x):
        x = self.batch_norm1(x)

        x = self.conv1(x)
        x = self.batch_norm2(x)
        x = nn.ReLU()(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.batch_norm3(x)
        x = nn.ReLU()(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.batch_norm4(x)
        x = nn.ReLU()(x)
        x = self.maxpool3(x)

        x = self.conv4(x)
        x = self.batch_norm5(x)
        x = nn.ReLU()(x)
        x = self.maxpool4(x)

        return x