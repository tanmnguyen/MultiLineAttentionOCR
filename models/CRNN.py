import sys 
sys.path.append("../")

import torch
import torch.nn as nn

from settings import settings 
from .MyCNN import MyCNN

class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()
        # feature extraction
        self.cnn = MyCNN().to(settings.DEVICE)
        b, fc, fh, fw = self.cnn(torch.randn(1, 3, settings.IMG_H, settings.IMG_W).to(settings.DEVICE)).shape
        # recurrent layers
        self.rnn1 = nn.LSTM(fh * fc, 64, 1, bidirectional=True)
        self.rnn2 = nn.LSTM(64 * 2 , 64, 1, bidirectional=True)
        # hidden to label
        self.fc = nn.Linear(64 * 2, len(settings.CHAR2IDX))

    def forward(self, x):
        x = self.cnn(x) # (batch, fc, fh, fw)
        b, fc, fh, fw = x.shape
        x = x.view(b, fc * fh, fw) # (batch, fc * fh, fw)
        x = x.permute(2, 0, 1) # (fw, batch, fc * fh)
        x, _ = self.rnn1(x) # (fw, batch, hidden * 2)
        x, _ = self.rnn2(x) # (fw, batch, hidden * 2)
        x = self.fc(x) # (fw, batch, num_classes)
        return x