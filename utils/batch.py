import sys 
sys.path.append("../")

import cv2
import torch

from settings import settings 

def img_processing(img):
    img = cv2.resize(img, (settings.IMG_W, settings.IMG_H))
    img = torch.from_numpy(img).permute(2, 0, 1)
    return img

def lbl_processing(lbl, seq_len):
    lbl = [settings.SOS] + [c for c in lbl] + [settings.EOS]
    lbl = torch.tensor([settings.CHAR2IDX[c] for c in lbl], dtype=torch.int32)
    lbl = torch.nn.functional.pad(lbl, (0, seq_len - len(lbl)), value=settings.CHAR2IDX[settings.PAD])
    return lbl

def collate_fn(batch):
    seq_len = max([len(lbl) for (_, lbl, _) in batch]) + 2 # include settings.SOS and settings.EOS 
    img_path  = [path for (_, _, path) in batch]
    img_batch = torch.stack([img_processing(img) for (img, _, _) in batch]).to(torch.float32)
    lbl_batch = torch.stack([lbl_processing(lbl.upper(), seq_len) for (_, lbl, _) in batch]).to(torch.int32)

    return img_batch.to(settings.DEVICE), lbl_batch.to(settings.DEVICE), img_path