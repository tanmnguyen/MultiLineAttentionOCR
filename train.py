import os 
import torch
import argparse 
import torch.optim as optim

from utils.batch import collate_fn
from utils.selections import get_model
from utils.io import log, plot_learning_curve

from torch.utils.data import random_split, DataLoader
from utils.epochFns import train, valid, save_wrong_predictions

from settings import settings 
from dataset.OCRZippedDataset import OCRZippedDataset

def main(args):
    settings.load_configuration(args.cfg)
   
    model, loss_fn, decode_fn = get_model(settings.ARCH)

    dataset = OCRZippedDataset(settings.DATA)
    train_ds, valid_ds = random_split(dataset, [0.9, 0.1])

    train_dataloader = DataLoader(
        train_ds, 
        batch_size=settings.BATCH_SIZE, 
        collate_fn=collate_fn, 
        shuffle=True
    )

    valid_dataloader = DataLoader(
        valid_ds, 
        batch_size=settings.BATCH_SIZE, 
        collate_fn=collate_fn, 
        shuffle=False
    )

    optimizer = optim.Adam(model.parameters(), lr=settings.LR, weight_decay=1e-3)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.75, verbose=False)

    log([model, 
        "-" * 100, 
        f"Number of parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}",
        f"Total dataset = {len(dataset)} | Train dataset = {len(train_ds)} | Valid dataset = {len(valid_ds)}",
        "-" * 100
    ])

    best_sequence_acc, train_history, valid_history = -1, [], []
    try:
        for epoch in range(1, settings.EPOCHS + 1):
            train_history.append(train(model, train_dataloader, optimizer, loss_fn, decode_fn, epoch, settings.EPOCHS))
            valid_history.append(valid(model, valid_dataloader, loss_fn, decode_fn, epoch, settings.EPOCHS))

            if valid_history[-1]['sequence_acc'] > best_sequence_acc:
                best_sequence_acc = valid_history[-1]['sequence_acc']
                torch.save(model.state_dict(), os.path.join(settings.SAVE_DIR, f"{settings.ARCH}-model.pt"))

            lr_scheduler.step()
            log([f"Learning Rate {lr_scheduler.get_last_lr()[0]}", ''])
    except KeyboardInterrupt:
        pass 

    print("Best Sequence Accuracy:", best_sequence_acc)

    # plot learning curve
    plot_learning_curve(train_history, "train")
    plot_learning_curve(valid_history, "valid")

    # load best model 
    model.load_state_dict(torch.load(os.path.join(settings.SAVE_DIR, f"{settings.ARCH}-model.pt")))
    save_wrong_predictions(model, valid_dataloader, decode_fn)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-cfg',
                        '--cfg',
                        required=True,
                        help="Path to configuration .cfg file")

    args = parser.parse_args()
    main(args)