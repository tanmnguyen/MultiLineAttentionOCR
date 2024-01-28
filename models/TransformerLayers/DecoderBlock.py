import torch.nn as nn

from .FeedForward import FeedFoward
from .MultiheadAttention import MultiHeadAttention

class DecoderBlock(nn.Module):
    """
    Transformer decoder block: communication followed by computation
    These are stacked on top of each other one after another
    """

    def __init__(self, d_model, n_head, num_heads, block_size, dropout):
        super().__init__()
        # Each head gets a smaller dimensional representation of the data
        # Assume each head gets a representation of dimension d_head and d_model is divisible by n_head
        d_head = d_model // n_head

        self.masked_multihead_attention = MultiHeadAttention(num_heads, d_head, d_model, block_size, dropout)
        self.cross_attention = MultiHeadAttention(num_heads, d_head, d_model, block_size, dropout)
        self.feed_forward = FeedFoward(d_model, dropout)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_emb=None):
        """
        In the "Attention is all you need" paper, we had
        x = self.ln1(x + self.sa(x))
        x = self.ln2(x + self.ffwd(x))
        See Figure 1 here, and mimic that: https://arxiv.org/pdf/2002.04745.pdf
        """

        x = self.ln1(x + self.masked_multihead_attention(x))
        x = self.ln2(x + self.cross_attention(x, enc_emb=enc_emb))
        x = self.ln3(x + self.feed_forward(x))

        return x