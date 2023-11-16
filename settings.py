import os 
import torch 
import datetime

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

DEVICE = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")

# create default save directory for each run
save_directory = f"train-outputs/train-{datetime.datetime.now().strftime('%Y.%m.%d. %H.%M.%S')}"
os.makedirs(save_directory, exist_ok=True)

image_directory = os.path.join(save_directory, "wrong-predictions")

