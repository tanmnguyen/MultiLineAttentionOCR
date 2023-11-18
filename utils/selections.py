import sys 
sys.path.append("../")

from models.CRNN import CRNN
from models.AttentionDecoderOCR import AttentionDecoderOCR

from settings import settings 
from utils.metrics import CTC_loss, CTC_decode, autoregressive_loss, autoregressive_decode

def get_model(arch: str):
    # Convolutional Recurrent Neural Network
    if arch.upper() == "CRNN":
        return CRNN().to(settings.DEVICE), CTC_loss, CTC_decode
    
    # Attention Decoder Neural Network
    if arch.upper() == "ATTDEC":
        return AttentionDecoderOCR().to(settings.DEVICE), autoregressive_loss, autoregressive_decode
    
    raise ValueError(f"Invalid model architecture: {arch}")
