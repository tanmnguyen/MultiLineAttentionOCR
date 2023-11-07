import torch 

PAD = '[PAD]'
SOS = '[SOS]'
EOS = '[EOS]'
IMG_W = 224
IMG_H = 50

lets = [
    PAD, SOS, EOS, 
    '0', '1', '2', '3', 
    '4', '5', '6', '7', 
    '8', '9', 'A', 'B', 
    'C', 'D', 'E', 'F', 
    'G', 'H', 'K', 'L', 
    'M', 'N', 'P', 'R', 
    'S', 'T', 'U', 'V', 
    'X', 'Y', 'Z'
]

CHAR2IDX = {c : i for i, c in enumerate(lets)}
IDX2CHAR = {i : c for i, c in enumerate(lets)}

BATCH_SIZE = 32
MAX_LEN = 10

DEVICE = torch.device("cpu")
