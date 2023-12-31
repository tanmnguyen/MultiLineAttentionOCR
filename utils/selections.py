import sys 
sys.path.append("../")

import torch 
from models.CRNN import CRNN
from models.AttentionDecoderOCR import AttentionDecoderOCR

from settings import settings 
from utils.metrics import CTC_loss, CTC_decode, autoregressive_loss, autoregressive_decode

def get_model(arch: str):
    # Convolutional Recurrent Neural Network
    if arch.upper() == "CRNN":
        return CRNN(
            img_h=settings.IMG_H,
            img_w=settings.IMG_W,
            hidden_size=64, 
            num_classes=len(settings.CHAR2IDX)
        ).to(settings.DEVICE), CTC_loss, CTC_decode
    
    # Attention Decoder Neural Network
    if arch.upper() == "ATTDEC":
        return AttentionDecoderOCR(
            img_h=settings.IMG_H,
            img_w=settings.IMG_W,
            max_len=settings.MAX_LEN, 
            hidden_size=512, 
            num_classes=len(settings.CHAR2IDX)
        ).to(settings.DEVICE), autoregressive_loss, autoregressive_decode
    
    raise ValueError(f"Invalid model architecture: {arch}")
