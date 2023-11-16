import os 
import torch

from tqdm import tqdm
from settings import DEVICE
from models.CRNN import CRNN
from utils.io import log, save_image_prediction
from utils.metrics import accuracy_fn, to_string

def train(model, train_dataloader, optimizer, loss_fn, decode_fn, epoch, num_epochs):
    teacher_forcing_ratio = 0.8 if epoch <= 10 else 0.5

    epoch_loss, ch_acc, sq_acc = 0.0, 0.0, 0.0
    for x_train, y_train, _ in tqdm(train_dataloader, leave=False):
        optimizer.zero_grad()

        y_pred = model(x_train.to(DEVICE)) if isinstance(model, CRNN) else \
                 model(x_train.to(DEVICE), y_train.to(DEVICE), teacher_forcing_ratio=teacher_forcing_ratio)
            
        loss = loss_fn(y_pred, y_train)

        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            c_acc, s_acc = accuracy_fn(y_pred, y_train, decode_fn)
            epoch_loss += loss.item()
            ch_acc += c_acc
            sq_acc += s_acc

    # normalize metrics 
    epoch_loss /= len(train_dataloader)
    ch_acc /= len(train_dataloader)
    sq_acc /= len(train_dataloader)
    
    log([
        f"[Train] Epoch {epoch} / {num_epochs} | " + \
        f"Loss {epoch_loss:.4f} | " + \
        f"Char Accuracy {ch_acc:.4f} | " + \
        f"Sequence Accuracy {sq_acc:.4f}" 
    ])

    return {
        "epoch_loss" : epoch_loss,
        "character_acc" : ch_acc,
        "sequence_acc" : sq_acc,
    } 

def valid(model, valid_dataloader, loss_fn, decode_fn, epoch, num_epochs):
    epoch_loss, ch_acc, sq_acc = 0.0, 0.0, 0.0
    for x_valid, y_valid, x_pth in tqdm(valid_dataloader, leave=False):
        # for the purpose of testing, we will provide y train to the attention decoder model to ensure the 
        # sequence length is correct. 
        y_pred = model(x_valid.to(DEVICE)) if isinstance(model, CRNN) else \
                 model(x_valid.to(DEVICE), y_valid.to(DEVICE), teacher_forcing_ratio=0) 
        
        loss = loss_fn(y_pred, y_valid) 

        with torch.no_grad():
            c_acc, s_acc = accuracy_fn(y_pred, y_valid, decode_fn)
            epoch_loss += loss.item()
            ch_acc += c_acc
            sq_acc += s_acc

    # normalize metrics 
    epoch_loss /= len(valid_dataloader)
    ch_acc /= len(valid_dataloader)
    sq_acc /= len(valid_dataloader)
    
    log([
        f"[Valid] Epoch {epoch} / {num_epochs} | " + \
        f"Loss {epoch_loss:.4f} | " + \
        f"Char Accuracy {ch_acc:.4f} | " + \
        f"Sequence Accuracy {sq_acc:.4f}" 
    ])

    return {
        "epoch_loss" : epoch_loss,
        "character_acc" : ch_acc,
        "sequence_acc" : sq_acc,
    }

def save_wrong_predictions(model, dataloader, decode_fn):
    for x_valid, y_valid, x_pth in tqdm(dataloader, leave=False):
        y_pred = model(x_valid.to(DEVICE)) if isinstance(model, CRNN) else \
                 model(x_valid.to(DEVICE), y_valid.to(DEVICE), teacher_forcing_ratio=0) 

        with torch.no_grad():
            y_pred = decode_fn(y_pred)
            for i in range(y_pred.shape[0]):
                y_pred_str = to_string(y_pred[i])
                y_true_str = to_string(y_valid[i,1:])

                if y_pred_str != y_true_str:
                    save_image_prediction(y_pred_str, y_true_str, x_valid[i], x_pth[i])