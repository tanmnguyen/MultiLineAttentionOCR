import torch
import torch.nn as nn

from .Head import Head
class MultiHeadAttention(nn.Module):
    """
    Multiple heads of self-attention in parallel
    You can have just sequential code below
    """

    def __init__(self, num_heads, d_head, d_model, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(d_head, d_model, block_size, dropout) for _ in range(num_heads)])
        # This is to project back to the dimension of d_model. In this case, it is just a learned linear map
        self.W_O = nn.Linear(d_head * num_heads, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_emb=None):
        # Concatenate the different representations per head along the last dimension
        out = torch.cat([h(x, enc_emb) for h in self.heads], dim=-1)
        # Project the concatenation and apply dropout; this is the W_O in "Attention is all you need"
        out = self.W_O(out)
        out = self.dropout(out)

        return out