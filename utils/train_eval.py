import torch
import numpy as np
from tqdm import tqdm
from constants import DEVICE

def train(model, train_dataloader, optimizer, loss_fn, accuracy_fn, epoch, num_epochs):
    epoch_loss, ch_acc, sq_acc = 0.0, 0.0, 0.0
    for x_train, y_train in tqdm(train_dataloader, leave=False):
        optimizer.zero_grad()
        y_pred = model(x_train.to(DEVICE))
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
    
    print(f"[Train] Epoch {epoch} / {num_epochs} | "
        f"Loss {epoch_loss:.4f} | "
        f"Char Accuracy {ch_acc:.4f} | "
        f"Sequence Accuracy {sq_acc:.4f}")

    return {
        "epoch_loss" : epoch_loss,
        "character_acc" : ch_acc,
        "sequence_acc" : sq_acc,
    }

def valid(model, valid_dataloader, loss_fn, accuracy_fn, epoch, num_epochs):
    epoch_loss, ch_acc, sq_acc = 0.0, 0.0, 0.0
    for x_train, y_train in tqdm(valid_dataloader, leave=False):
        y_pred = model(x_train.to(DEVICE))
        loss = loss_fn(y_pred, y_train)

        with torch.no_grad():
            c_acc, s_acc = accuracy_fn(y_pred, y_train)
            epoch_loss += loss.item()
            ch_acc += c_acc
            sq_acc += s_acc

    # normalize metrics 
    epoch_loss /= len(valid_dataloader)
    ch_acc /= len(valid_dataloader)
    sq_acc /= len(valid_dataloader)
    
    print(f"[valid] Epoch {epoch} / {num_epochs} | "
        f"Loss {epoch_loss:.4f} | "
        f"Char Accuracy {ch_acc:.4f} | "
        f"Sequence Accuracy {sq_acc:.4f}")

    return {
        "epoch_loss" : epoch_loss,
        "character_acc" : ch_acc,
        "sequence_acc" : sq_acc,
    }
