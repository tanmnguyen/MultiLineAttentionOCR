import torch 
import torch.nn as nn

from settings import settings 

from .TransformerLayers.Encoder import Encoder
from .TransformerLayers.Decoder import Decoder

class TransformerOCR(nn.Module):
    def __init__(self, img_w, img_h, num_classes, n_head, d_model, block_size, num_heads, dropout, max_len):
        super().__init__()
        self.max_len = max_len 

        self.encoder = Encoder(img_w, img_h, d_model).to(settings.DEVICE)
        self.decoder = Decoder(num_classes, d_model, block_size, n_head, num_heads, dropout).to(settings.DEVICE)
            
    def forward(self, img, txt=None, teacher_forcing_ratio=0):
        enc_emb = self.encoder(img)

        # max_len = txt.shape[1] - 1 if txt is not None else self.max_len

        logits = self.decoder(txt[:, :-1], enc_emb)
        logits = logits.log_softmax(-1)
        return logits
    
        # # use teacher forcing
        # if torch.rand(1).item() < teacher_forcing_ratio:
            
        
        # # no teacher forcing
        # else:
        #     pred = []
        #     txt = torch.tensor([settings.CHAR2IDX[settings.SOS]]).expand(enc_emb.shape[0], 1).to(settings.DEVICE)
        #     for _ in range(max_len):
        #         logits = self.decoder(txt, enc_emb)
        #         # retrieve the last token
        #         logits = logits[:, -1].unsqueeze(1)
        #         # append to pred
        #         pred.append(logits)
        #         # append to txt 
        #         txt = torch.cat([txt, logits.argmax(-1)], dim=1)
        #     pred = torch.cat(pred, dim=1)
        #     pred = pred.log_softmax(-1)
        #     return pred
        

        
