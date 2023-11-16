import os 
import torch

from tqdm import tqdm
from utils.io import log
from models.CRNN import CRNN
from settings import DEVICE, image_directory

def train(model, train_dataloader, optimizer, loss_fn, accuracy_fn, epoch, num_epochs):
    teacher_forcing_ratio = 0.8 if epoch < 10 else 0.5 if epoch < 30 else 0.3

    epoch_loss, ch_acc, sq_acc = 0.0, 0.0, 0.0
    for x_train, y_train, _ in tqdm(train_dataloader, leave=False):
        optimizer.zero_grad()

        y_pred = model(x_train.to(DEVICE)) if isinstance(model, CRNN) else \
                 model(x_train.to(DEVICE), y_train.to(DEVICE), teacher_forcing_ratio=teacher_forcing_ratio)
            
        loss = loss_fn(y_pred, y_train)

        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            c_acc, s_acc = accuracy_fn(y_pred, y_train)
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

def valid(model, valid_dataloader, loss_fn, accuracy_fn, epoch, num_epochs):
    epoch_loss, ch_acc, sq_acc = 0.0, 0.0, 0.0
    for x_valid, y_valid, x_pth in tqdm(valid_dataloader, leave=False):
        # for the purpose of testing, we will provide y train to the attention decoder model to ensure the 
        # sequence length is correct. 
        y_pred = model(x_valid.to(DEVICE)) if isinstance(model, CRNN) else \
                 model(x_valid.to(DEVICE), y_valid.to(DEVICE), teacher_forcing_ratio=0) 
        
        loss = loss_fn(y_pred, y_valid) 

        with torch.no_grad():
            if epoch >= 10:
                c_acc, s_acc = accuracy_fn(y_pred, y_valid, x_valid, x_pth, f"{image_directory}-Epoch-{epoch}")
            else:
                c_acc, s_acc = accuracy_fn(y_pred, y_valid)
                
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