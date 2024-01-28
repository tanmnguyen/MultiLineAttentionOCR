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

        max_len = txt.shape[1] - 1 if txt is not None else self.max_len

        # use teacher forcing
        if torch.rand(1).item() < teacher_forcing_ratio:
            logits = self.decoder(txt[:, :-1], enc_emb)
            logits = logits.log_softmax(-1)
            return logits
        
        # no teacher forcing
        else:
            pred_list = []
            sos_token = torch.tensor([settings.CHAR2IDX[settings.SOS]]).expand(enc_emb.shape[0], 1).to(settings.DEVICE)

            for _ in range(max_len):
                logits = self.decoder(sos_token, enc_emb)
                last_token_logits = logits[:, -1].unsqueeze(1)
                
                pred_list.append(last_token_logits)
                sos_token = torch.cat([sos_token, last_token_logits.argmax(-1)], dim=1)

            pred_tensor = torch.cat(pred_list, dim=1).log_softmax(-1)
            return pred_tensor
        

        
