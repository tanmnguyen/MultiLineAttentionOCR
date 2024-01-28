import torch 
import torch.nn as nn 
from settings import settings 

class Head(nn.Module):
    """
    This class represents one head of self-attention
    Note that since this is a Decoder, this is masked-self-attention
    There is no Encoder, so there is no cross-self-attention
    """

    def __init__(self, d_head, d_model, block_size, dropout):
        super().__init__()
        self.d_head = d_head
        # Map each key, query, or value in to a d_head dimensional model.
        # Each should be matrices from d_model to d_head
        self.W_K = nn.Linear(d_model, d_head, bias=False)
        self.W_Q = nn.Linear(d_model, d_head, bias=False)
        self.W_V = nn.Linear(d_model, d_head, bias=False)
        self.d_head = d_head
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_emb=None):
        # (B, T, d_model)
        # B = batch_size, T = block_size in the below
        B,T,d = x.shape
        # Get the key and query representations from the embedding x
        # (B,T,d_head)
        k = self.W_K(x if enc_emb is None else enc_emb)
        # (B,T,d_head)
        q = self.W_Q(x)
        # (B,T,d_head)
        v = self.W_V(x if enc_emb is None else enc_emb)

        # Compute attention scores, and get the new representations for this head

        # (B, T, d_head) @ (B, d_head, T) = (B, T, T)
        # Multiply q by k and divide by the appropriate constant
        scores = q.bmm(k.transpose(1,2)) / torch.sqrt(torch.tensor(k.shape[-1]))

        # (B, T, T)
        # Apply a mask to scores, making all scores above the diagonal -inf
        # Do not apply mask for cross attention
        if enc_emb is None:
            mask = torch.tril(torch.ones(scores.shape[1], scores.shape[2])).to(settings.DEVICE)
            scores = scores * mask + torch.where(mask == 1, torch.tensor(0), torch.tensor(float('-inf')))

        # (B, T, T)
        # Apply softmax to the final dimension of scores
        a = nn.Softmax(dim=-1)(scores)

        # Apply dropout
        a = self.dropout(a)
        # Perform the weighted aggregation of the values
        # Using a and v, get the new representations
        # (B, T, T) @ (B, T, d_head) -> (B, T, d_head)
        out = a.bmm(v)
        # For each token, return the weighted sum of the values
        return out