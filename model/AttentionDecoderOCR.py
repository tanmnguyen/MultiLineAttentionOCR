import sys 
sys.path.append("../")

import torch 
import torchvision.models as models

from torch import nn 
import torch.functional as F
from constants import IMG_H, IMG_W, SOS, EOS, PAD, CHAR2IDX, MAX_LEN

from .attention_ocr import MyIncept

class PositionEmbedding(nn.Module):
    def __init__(self, num_positions: int):
        super(PositionEmbedding, self).__init__()
        self.emb = nn.Embedding(num_positions, num_positions)
        self.init_weight(num_positions)

    def init_weight(self, num_positions: int):
        self.emb.weight.data = torch.eye(num_positions)
        self.emb.weight.requires_grad = False

    def forward(self, input_):
        return self.emb(input_)

class Attention(nn.Module):
    def __init__(self, hidden: int):
        super(Attention, self).__init__()
        self.fc = nn.Linear(hidden * 2, hidden)
        self.v = nn.Parameter(torch.rand(hidden), requires_grad=True)
        self.init_weight() 

    def init_weight(self):
        # TODO init weight for the self.v parameter
        pass 

    def forward(self, x, hidden):
        # compute attention score 
        # TODO: select the right hidden layer when there are > 1 layers 
        attn = self.score(
            x, # (batch, timestep, hidden)
            hidden.expand(x.shape[1], -1, -1).transpose(0,1) # (batch, timestep, hidden)
        )
        # apply softmax 
        attn = torch.softmax(attn, dim=-1) # (batch, timestep, 1)
        return attn 

    def score(self, x, hidden):
        sim = torch.tanh(self.fc(torch.cat([x, hidden], dim=-1))) # (batch, timestep, hidden)
        v = self.v.expand(x.shape[0], -1).unsqueeze(1) # (batch, 1, hidden)
        attn = torch.bmm(v, sim.permute(0, 2, 1)) # (batch, 1, timestep)
        attn = attn.permute(0, 2, 1) # (batch, timestep, 1)
        return attn

class Decoder(nn.Module):
    def __init__(self, hidden: int, rnn_layers: int = 1):
        super(Decoder, self).__init__()
        self.max_len = MAX_LEN 
        self.rnn_layers = rnn_layers

        self.emb = nn.Embedding(len(CHAR2IDX), hidden)
        self.attention = Attention(hidden)
        self.rnn = nn.GRU(hidden * 2, hidden, 1, batch_first=True)
        self.fc = nn.Linear(hidden, len(CHAR2IDX))
        
    def forward_step(self, x, s, hidden):
        # compute attention matrix using encoder outputs "X" and hidden state "hidden"
        attn = self.attention(x, hidden) # (batch, timestep, 1)

        # compute context vector using attention matrix and encoder outputs "X"
        # (Activate which input pixels' hidden states to focus on)
        context = x.transpose(1, 2).bmm(attn).transpose(1, 2) # (batch, 1, hidden)

        # compute previous character embedding
        emb = self.emb(s) # (batch, 1, hidden)

        # compute rnn input 
        rnn_input = torch.cat([context, emb], dim=-1) # (batch, 1, hidden * 2)

        # pass to rnn layer 
        output, hidden = self.rnn(rnn_input, hidden) # output (batch, 1, hidden), hidden (layers, batch, hidden)

        # compute output 
        output = self.fc(output) # (batch, 1, len(CHAR2IDX))

        # return output and hidden state
        return output, hidden 

    def forward(self, x, y = None, teacher_forcing_ratio : float = 0.5):
        max_len = y.shape[1] - 1 if y is not None else self.max_len

        # initial hidden state
        hidden = torch.zeros(self.rnn_layers, x.shape[0], x.shape[-1]) # (layers, batch, hidden)

        decoder_outputs = [] 
        # use teacher forcing 
        if torch.rand(1).item() <= teacher_forcing_ratio:
            for i in range(max_len):
                # compute next output using encoder outputs, last predicted character, hidden state
                output, hidden = self.forward_step(x, y[:, i].unsqueeze(1), hidden) 
                # save output 
                decoder_outputs.append(output.squeeze(1)) # each output (batch, len(CHAR2IDX))
        else:
            inputs = torch.tensor([CHAR2IDX[SOS]]).expand(x.shape[0], 1) # (batch, 1)
            for i in range(max_len):
                # compute next output using encoder outputs, last predicted character, hidden state
                output, hidden = self.forward_step(x, inputs, hidden)
                # save output
                decoder_outputs.append(output.squeeze(1)) # each output (batch, len(CHAR2IDX))
                # update inputs
                inputs = output.argmax(dim=-1) # (batch, 1)
        
        decoder_outputs = torch.stack(decoder_outputs, dim=1) # (batch, max_len, len(CHAR2IDX))
        return decoder_outputs

class AttentionDecoderOCR(nn.Module):
    def __init__(self, hidden = 256):
        super(AttentionDecoderOCR, self).__init__()
        # image feature extractor
        # self.cnn = models.alexnet().features
        self.cnn = MyIncept()
        # get the shape of feature output
        b, fc, fh, fw = self.cnn(torch.randn(1, 3, IMG_H, IMG_W)).shape
        # vertical position embedding
        self.emb_i = PositionEmbedding(fh)
        # horizontal position embedding
        self.emb_j = PositionEmbedding(fw)
        # feature embedding
        self.emb_f = nn.Linear(fh + fw + fc, hidden)
        # decoder 
        self.decoder = Decoder(hidden)

    def forward(self, x, y = None, teacher_forcing_ratio: float = 0.5):
        # feature extraction
        x = self.cnn(x) 
        b, fc, fh, fw = x.shape

        # positional grid
        i_grid, j_grid = torch.meshgrid(
            torch.arange(fh), 
            torch.arange(fw), 
            indexing='ij'
        )
       
        # build embedding for each i and j positions
        i_emb = self.emb_i(i_grid) # (fh, fh)
        j_emb = self.emb_j(j_grid) # (fw, fw)

        # build positional embedding 
        p_emb = torch.cat([i_emb, j_emb], dim=-1).expand(b, -1, -1, -1) # (b, fh, fw, fh + fw)

        # combine features with positional embedding (move the channel dimension to the last dimension)
        x = torch.cat([x.permute(0, 2, 3, 1), p_emb], dim=-1) # (b, fh, fw, fc + fh + fw)
        
        # flatten pixels into sequence of features
        x = x.view(b, -1, fc + fh + fw) # (b, fh * fw, fc + fh + fw)
       
        # build embedded features with image features and positional embedding
        x = self.emb_f(x) # (b, fh * fw, hidden)

        # decoder 
        x = self.decoder(x, y, teacher_forcing_ratio=teacher_forcing_ratio)

        # apply log softmax
        x = nn.LogSoftmax(dim=-1)(x)

        return x