import torch 
from typing import OrderedDict 
from torchtext.vocab import vocab

SOS = "<SOS>"
EOS = "<EOS>" 
PAD = "<PAD>"
UNK = "<UNK>"
IMG_H = 224 
IMG_W = 224
BATCH_SIZE = 32
MAX_LEN = 10
DEVICE = torch.device("mps")

CHAR2IDX = vocab(
    OrderedDict([(token, 1) for token in "0123456789ABCDEFGHKLMNPRSTUVXYZ"]), 
    specials=[PAD, UNK, SOS, EOS]
)
CHAR2IDX.set_default_index(CHAR2IDX[UNK])