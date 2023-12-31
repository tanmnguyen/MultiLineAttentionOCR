import sys 
sys.path.append("../")  

import math
import torch
import torch.nn as nn

from settings import settings 
from models.MyIncept import MyIncept

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size), requires_grad=True)
        self.init_weight()
    
    def init_weight(self):
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(1)
        h = hidden.expand(timestep, -1, -1).transpose(0, 1)
        attn_energies = self.score(h, encoder_outputs)
        return attn_energies.softmax(2)

    def score(self, hidden, encoder_outputs):
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)
        v = self.v.expand(encoder_outputs.size(0), -1).unsqueeze(1)
        energy = torch.bmm(v, energy)
        return energy

class Decoder(nn.Module):
    def __init__(self, max_len, hidden_size, num_classes):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size 
        self.max_len = max_len

        self.emb = nn.Embedding(num_classes, hidden_size)
        self.attention = Attention(hidden_size)
        self.rnn = nn.GRU(hidden_size * 2, hidden_size, 1)
        self.out = nn.Linear(hidden_size, num_classes)

    def forward_step(self, input_, last_hidden, encoder_outputs):
        emb = self.emb(input_.transpose(0, 1)) # (1, batch, hidden)
        attn = self.attention(last_hidden, encoder_outputs)  # (batch, 1, total pixels)
        context = attn.bmm(encoder_outputs).transpose(0, 1)  # (1, batch, hidden)
        rnn_input = torch.cat((emb, context), dim=2) # (1, batch, hidden * 2)
        outputs, hidden = self.rnn(rnn_input, last_hidden) # (1, batch, hidden), (layers, batch, hidden)
        outputs = self.out(outputs.contiguous().squeeze(0)).log_softmax(1) # (batch, vocab_size)
        return outputs, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size).to(settings.DEVICE)
    
    def forward(self, x, y = None, teacher_forcing_ratio = 0):
        max_len = y.shape[1] - 1 if y is not None else self.max_len
        
        # initial hidden state
        hidden = self.init_hidden(x.shape[0]) # (1, batch, hidden)

        # improve performance by flattenning the input
        self.rnn.flatten_parameters()

        outputs = []
        # use teacher forcing
        if torch.rand(1).item() < teacher_forcing_ratio:
            for di in range(max_len):
                # compute next output using encoder outputs, last predicted character, hidden state
                output, hidden = self.forward_step(y[:, di].unsqueeze(1), hidden, x)
                # save output
                outputs.append(output.squeeze(1))
        else:
            decoder_input = torch.tensor([settings.CHAR2IDX[settings.SOS]]).expand(x.shape[0], 1).to(settings.DEVICE)
            for di in range(max_len):
                # compute next output using encoder outputs, last predicted character, hidden state
                output, hidden = self.forward_step(decoder_input, hidden, x)
                # save output 
                outputs.append(output.squeeze(1))
                # update input 
                decoder_input = output.argmax(dim=-1).unsqueeze(1)

        outputs = torch.stack(outputs).permute(1, 0, 2)
        return outputs

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

class AttentionDecoderOCR(nn.Module):
    def __init__(self, img_w, img_h, max_len, hidden_size, num_classes):
        super(AttentionDecoderOCR, self).__init__()
        # self.cnn = MyCNN().to(settings.DEVICE)
        self.cnn = MyIncept().to(settings.DEVICE)
        # get the shape of feature output
        b, fc, fh, fw = self.cnn(torch.randn(1, 3, img_h, img_w).to(settings.DEVICE)).shape
        # vertical position embedding
        self.emb_i = PositionEmbedding(fh)
        # horizontal position embedding
        self.emb_j = PositionEmbedding(fw)
        # feature embedding
        self.emb_f = nn.Linear(fh + fw + fc, hidden_size)
        # decoder 
        self.decoder = Decoder(max_len, hidden_size, num_classes)
            
    def forward(self, x, y=None, teacher_forcing_ratio=0):
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

        decoder_outputs = self.decoder(x=x, y=y, teacher_forcing_ratio=teacher_forcing_ratio) # (batch, seq_len, num_classes)

        return decoder_outputs