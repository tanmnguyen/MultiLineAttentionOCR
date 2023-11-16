import os 
import torch
import argparse 
import torch.optim as optim

from utils.io import log
from utils.batch import collate_fn
from utils.selections import get_model
from utils.epochFns import train, valid
from torch.utils.data import random_split, DataLoader

from settings import BATCH_SIZE, save_directory
from dataset.LicensePlateDataset import LicensePlateDataset

def main(args):
    model, loss_fn, accuracy_fn = get_model(args.arch)
    dataset = LicensePlateDataset(args.data)
    train_ds, valid_ds = random_split(dataset, [0.9, 0.1])

    train_dataloader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        collate_fn=collate_fn, 
        shuffle=True
    )

    valid_dataloader = DataLoader(
        valid_ds, 
        batch_size=BATCH_SIZE, 
        collate_fn=collate_fn, 
        shuffle=False
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5, verbose=False)

    log([model, 
        "-" * 100, 
        f"Number of parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}",
        f"Total dataset = {len(dataset)} | Train dataset = {len(train_ds)} | Valid dataset = {len(valid_ds)}",
        "-" * 100
    ])

    best_sequence_acc = -1
    train_history, valid_history = [], []
    for epoch in range(1, args.epochs+1):
        train_history.append(train(model, train_dataloader, optimizer, loss_fn, accuracy_fn, epoch, args.epochs))
        valid_history.append(valid(model, valid_dataloader, loss_fn, accuracy_fn, epoch, args.epochs))

        if valid_history[-1]['sequence_acc'] > best_sequence_acc:
            best_sequence_acc = valid_history[-1]['sequence_acc']
            torch.save(model.state_dict(), os.path.join(save_directory, f"{args.arch}-model.pt"))

        lr_scheduler.step()
        log([f"Learning Rate {lr_scheduler.get_last_lr()[0]}", ''])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-data',
                        '--data',
                        required=True,
                        help="Path to training data .zip file")
    
    parser.add_argument('-arch',
                        '--arch',
                        type=str,
                        required=True,
                        help="Model Archiecture: CRNN, ATTDEC")
    
    parser.add_argument('-epochs',
                        '--epochs',
                        type=int,
                        default=20,
                        required=False,
                        help="Training epochs")

    args = parser.parse_args()
    main(args)