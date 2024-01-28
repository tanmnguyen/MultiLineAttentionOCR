import torch 
import torch.nn as nn

from settings import settings 
from .DecoderBlock import DecoderBlock

# GPT-based decoder 
class Decoder(nn.Module):
    def __init__(self, num_classes, d_model, block_size, n_head, num_heads, dropout):
        super().__init__()

        self.token_embedding_table = nn.Embedding(num_classes, d_model)
        self.position_embedding_table = nn.Embedding(block_size, d_model)

        # Decoder blocks
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, n_head, num_heads, block_size, dropout) for _ in range(3)
        ])

        # Final layer norm
        self.ln = nn.LayerNorm(d_model)
        self.ff = nn.Linear(d_model, num_classes)

    def forward(self, txt, enc_emb):
        B, T = txt.shape

        # (B,T,d_model)
        tok_emb = self.token_embedding_table(txt)

        # (T,d_model)
        pos = torch.arange(T).unsqueeze(0).expand(B, -1).to(settings.DEVICE)
        pos_emb = self.position_embedding_table(pos)

        # Add positional encodings to encodings
        # (B,T,d_model)
        x = tok_emb + pos_emb

        # Mix up the token representations over and over via the blocks
        # (B,T,d_model)
        for dec_block in self.blocks:
            x = dec_block(x, enc_emb)

        # Apply layer norm
        # (B,T,d_model)
        x = self.ln(x)

        # Apply the final linear map, to get to dimension vocab_size
        # (B,T,vocab_size)
        logits = self.ff(x)

        return logits 
