import sys 
sys.path.append("../")
sys.path.append("../../")

import torch
import torch.nn as nn 

from settings import settings 
from models.MyIncept import MyIncept 
from models.AttentionDecoderOCR import PositionEmbedding

class Encoder(nn.Module):
    def __init__(self, img_w, img_h, hidden_size):
        super().__init__()
        self.cnn = MyIncept().to(settings.DEVICE)

        # get the shape of feature output
        b, fc, fh, fw = self.cnn(torch.randn(1, 3, img_h, img_w).to(settings.DEVICE)).shape

        # vertical position embedding
        self.emb_i = PositionEmbedding(fh)

        # horizontal position embedding
        self.emb_j = PositionEmbedding(fw)

        # feature embedding
        self.emb_f = nn.Linear(fh + fw + fc, hidden_size)

    def forward(self, img):
        # feature extraction 
        x = self.cnn(img) 
        b, fc, fh, fw = x.shape

        # positional grid
        i_grid, j_grid = torch.meshgrid(
            torch.arange(fh), 
            torch.arange(fw), 
            indexing='ij'
        )

        # build embedding for each i and j positions
        i_emb = self.emb_i(i_grid.to(settings.DEVICE)) # (fh, fh)
        j_emb = self.emb_j(j_grid.to(settings.DEVICE)) # (fw, fw)

        # build positional embedding 
        p_emb = torch.cat([i_emb, j_emb], dim=-1).expand(b, -1, -1, -1) # (b, fh, fw, fh + fw)

        # combine features with positional embedding (move the channel dimension to the last dimension)
        x = torch.cat([x.permute(0, 2, 3, 1), p_emb], dim=-1) # (b, fh, fw, fc + fh + fw)
        
        # flatten pixels into sequence of features
        x = x.view(b, -1, fc + fh + fw) # (b, fh * fw, fc + fh + fw)
       
        # build embedded features with image features and positional embedding
        x = self.emb_f(x) # (b, fh * fw, hidden)

        return x 
