import argparse 
import torch.optim as optim

from models.CRNN import CRNN
from utils.batch import collate_fn
from utils.train_eval import train, valid
from utils.metrics import CTC_loss, CTC_accuracy_fn
from torch.utils.data import random_split, DataLoader

from constants import DEVICE
from dataset.LicensePlateDataset import LicensePlateDataset

def main(args):
    model = CRNN().to(DEVICE)
    dataset = LicensePlateDataset(args.data)
    train_ds, valid_ds = random_split(dataset, [0.8, 0.2])

    train_dataloader = DataLoader(train_ds, batch_size=16, collate_fn=collate_fn, shuffle=True)
    valid_dataloader = DataLoader(valid_ds, batch_size=16, collate_fn=collate_fn, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=5)

    print(model)
    print("-" * 100)
    print(f"Number of parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Total dataset = {len(dataset)} | Train dataset = {len(train_ds)} | Valid dataset = {len(valid_ds)}")
    print("-" * 100)

    train_history, valid_history = [], []
    for epoch in range(1, args.epochs+1):
        train_history.append(
            train(model, train_dataloader, optimizer, CTC_loss, CTC_accuracy_fn, epoch, args.epochs)
        )

        valid_history.append(
            valid(model, valid_dataloader, CTC_loss, CTC_accuracy_fn, epoch, args.epochs)
        )

        lr_scheduler.step(train_history[-1]["epoch_loss"])

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