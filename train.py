import os 
import torch
import argparse 
import torch.optim as optim

from utils.io import log
from utils.batch import collate_fn
from utils.selections import get_model
from torch.utils.data import random_split, DataLoader
from utils.epochFns import train, valid, save_wrong_predictions

from settings import BATCH_SIZE, save_directory
from dataset.OCRZippedDataset import OCRZippedDataset

def main(args):
    model, loss_fn, decode_fn = get_model(args.arch)
    dataset = OCRZippedDataset(args.data)
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

    best_sequence_acc, train_history, valid_history = -1, [], []
    try:
        for epoch in range(1, args.epochs+1):
            train_history.append(train(model, train_dataloader, optimizer, loss_fn, decode_fn, epoch, args.epochs))
            valid_history.append(valid(model, valid_dataloader, loss_fn, decode_fn, epoch, args.epochs))

            if valid_history[-1]['sequence_acc'] > best_sequence_acc:
                best_sequence_acc = valid_history[-1]['sequence_acc']
                torch.save(model.state_dict(), os.path.join(save_directory, f"{args.arch}-model.pt"))

            lr_scheduler.step()
            log([f"Learning Rate {lr_scheduler.get_last_lr()[0]}", ''])
    except KeyboardInterrupt:
        pass 

    # load best model 
    model.load_state_dict(torch.load(os.path.join(save_directory, f"{args.arch}-model.pt")))
    save_wrong_predictions(model, valid_dataloader, decode_fn)

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
                        default=50,
                        required=False,
                        help="Training epochs")

    args = parser.parse_args()
    main(args)