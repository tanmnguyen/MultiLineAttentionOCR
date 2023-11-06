import torch 
import argparse 
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from dataset.LicensePlateDataset import LicensePlateDataset

from constants import BATCH_SIZE
from utils.batch import collate_fn
from utils.metrics import accuracy
from model.attention_ocr import OCR
from constants import DEVICE

# torch.autograd.set_detect_anomaly(True)
def train(model : nn.Module, loss_fn, optimizer, train_dataloader : DataLoader, epoch : int, EPOCHS : int):
    epoch_loss, epoch_chr_acc, epoch_seq_acc = 0.0, 0.0, 0.0
    for x_train, y_train in tqdm(train_dataloader):
        x_train = x_train.to(DEVICE)
        y_train = y_train.to(DEVICE)

        optimizer.zero_grad()
        y_pred = model(x_train, y_train, teacher_forcing_ratio=1) # (batch, timestep, len(CHAR2IDX))

        loss = 0
        # auto-regressive loss
        for i in range(y_pred.shape[1]):
            # compute loss with y_train with label shift i + 1 to ignore the starting SOS token
            # as we feed the SOS token first to generate the output 
            loss += loss_fn(
                y_pred[:, i, :],                    # (batch, len(CHAR2IDX))
                y_train[:, i + 1].to(torch.long)    # (batch) 
            )

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            epoch_loss += loss.item() / y_pred.shape[1]

            chr_acc, seq_acc = accuracy(y_pred, y_train)
            epoch_chr_acc += chr_acc
            epoch_seq_acc += seq_acc

    print(  
        f"[Train] Epoch {epoch+1}/{EPOCHS} | "
        f"Loss: {epoch_loss / len(train_dataloader):.4f} | "
        f"Char Acc: {epoch_chr_acc / len(train_dataloader):.4f} | "
        f"Seq Acc: {epoch_seq_acc / len(train_dataloader):.4f}")
    
    return {
        "loss" : epoch_loss / len(train_dataloader),
        "char_acc" : epoch_chr_acc / len(train_dataloader),
        "seq_acc" : epoch_seq_acc / len(train_dataloader)
    }

def valid(model: nn.Module, loss_fn, valid_dataloader : DataLoader, epoch : int, EPOCHS : int):
    epoch_loss, epoch_chr_acc, epoch_seq_acc = 0.0, 0.0, 0.0
    for x_valid, y_valid in tqdm(valid_dataloader):
        x_valid = x_valid.to(DEVICE)
        y_valid = y_valid.to(DEVICE)

        y_pred = model(x_valid, y_valid, teacher_forcing_ratio=1) # (batch, timestep, len(CHAR2IDX))

        loss = 0
        # auto-regressive loss
        for i in range(y_pred.shape[1]):
            # compute loss with y_train with label shift i + 1 to ignore the starting SOS token
            # as we feed the SOS token first to generate the output 
            loss += loss_fn(
                y_pred[:, i, :],                    # (batch, len(CHAR2IDX))
                y_valid[:, i + 1].to(torch.long)    # (batch) 
            )

        with torch.no_grad():
            epoch_loss += loss.item() / y_pred.shape[1]

            chr_acc, seq_acc = accuracy(y_pred, y_valid)
            epoch_chr_acc += chr_acc
            epoch_seq_acc += seq_acc

    print(  
        f"[Valid] Epoch {epoch+1}/{EPOCHS} | "
        f"Loss: {epoch_loss / len(valid_dataloader):.4f} | "
        f"Char Acc: {epoch_chr_acc / len(valid_dataloader):.4f} | "
        f"Seq Acc: {epoch_seq_acc / len(valid_dataloader):.4f} | <------------------")
    
    return {
        "loss" : epoch_loss / len(valid_dataloader),
        "char_acc" : epoch_chr_acc / len(valid_dataloader),
        "seq_acc" : epoch_seq_acc / len(valid_dataloader)
    }
    
def main(args):
    train_dataset, valid_dataset = random_split(LicensePlateDataset(args.data), [0.9, 0.1])

    train_dataloader = DataLoader(
        train_dataset, 
        collate_fn=collate_fn,
        batch_size=BATCH_SIZE, 
        shuffle=True
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        collate_fn=collate_fn,
        batch_size=BATCH_SIZE,
        shuffle=False
    )


    print("-"*100)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Valid dataset size: {len(valid_dataset)}")
    print("-"*100)

    EPOCHS = args.epochs
    model = OCR().to(DEVICE)

    print("-"*100)
    print(model)
    print("-"*100)
    
    # count parameters 
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    loss_fn = nn.NLLLoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.8)

    best = 0.0
    train_history, valid_history = [], []
    for epoch in range(EPOCHS):
        train_history.append(train(model, loss_fn, optimizer, train_dataloader, epoch, EPOCHS))
        if (epoch + 1) % 5 == 0:
            valid_history.append(valid(model, loss_fn, valid_dataloader, epoch, EPOCHS))

        if train_history[-1]["seq_acc"] > best:
            best = train_history[-1]["seq_acc"]
            print("Saving model...")
            torch.save(model.state_dict(), "model.pt")

        # print learning rate 
        print(f"Learning rate: {scheduler.get_last_lr()[0]}")
        scheduler.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-data',
                        '--data',
                        required=True,
                        help="Path to training data .zip file")
    
    parser.add_argument('-epochs',
                        '--epochs',
                        type=int,
                        default=50,
                        required=False,
                        help="Training epochs")

    args = parser.parse_args()
    main(args)