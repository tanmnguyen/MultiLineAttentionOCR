import sys 
sys.path.append("../")

from models.CRNN import CRNN
from models.AttentionDecoderOCR import AttentionDecoderOCR

from settings import DEVICE
from utils.metrics import CTC_loss, CTC_accuracy_fn, autoregressive_loss, autoregressive_accuracy_fn

def get_model(arch: str):
    # Convolutional Recurrent Neural Network
    if arch.upper() == "CRNN":
        return CRNN().to(DEVICE), CTC_loss, CTC_accuracy_fn
    
    # Attention Decoder Neural Network
    if arch.upper() == "ATTDEC":
        return AttentionDecoderOCR().to(DEVICE), autoregressive_loss, autoregressive_accuracy_fn
    
    raise ValueError(f"Invalid model architecture: {arch}")
