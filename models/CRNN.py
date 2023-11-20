import sys 
sys.path.append("../")

import torch
import torch.nn as nn

from settings import settings 
from models.MyCNN import MyCNN

class CRNN(nn.Module):
    def __init__(self, img_h, img_w, hidden_size, num_classes):
        super(CRNN, self).__init__()
        # feature extraction
        self.cnn = MyCNN().to(settings.DEVICE)
        b, fc, fh, fw = self.cnn(torch.randn(1, 3, img_h, img_w).to(settings.DEVICE)).shape
        # recurrent layers
        self.rnn1 = nn.LSTM(fh * fc, hidden_size, 1, bidirectional=True)
        self.rnn2 = nn.LSTM(hidden_size * 2 , hidden_size, 1, bidirectional=True)
        # hidden to label
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = self.cnn(x) # (batch, fc, fh, fw)
        b, fc, fh, fw = x.shape
        x = x.view(b, fc * fh, fw) # (batch, fc * fh, fw)
        x = x.permute(2, 0, 1) # (fw, batch, fc * fh)
        x, _ = self.rnn1(x) # (fw, batch, hidden * 2)
        x, _ = self.rnn2(x) # (fw, batch, hidden * 2)
        x = self.fc(x) # (fw, batch, num_classes)
        return x