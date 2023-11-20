import sys 
sys.path.append("../")

import os 
import numpy as np
from settings import settings 
import matplotlib.pyplot as plt

def plot_learning_curve(history, title: str):
    epoch_loss = [item["epoch_loss"] for item in history]
    character_acc = [item["character_acc"] for item in history]
    sequence_acc = [item["sequence_acc"] for item in history]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epoch_loss, label="Epoch Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title} - Epoch Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(character_acc, label="Character Accuracy")
    plt.plot(sequence_acc, label="Sequence Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title} - Accuracy")
    plt.legend()
    plt.ylim(0, 1)  # Set the y-axis limits to ensure values between 0 and 1

    plt.tight_layout()
    plt.savefig(os.path.join(settings.SAVE_DIR, f"{title}.png"))
    plt.close()
    

def save_image_prediction(y_pred_str, y_true_str, img, img_path):
    os.makedirs(settings.IMG_DIR, exist_ok=True)
    plt.imshow(np.uint8(img.permute(1,2,0).cpu().numpy()))
    plt.title(f"Pred: {y_pred_str}\nTrue: {y_true_str}")
    plt.savefig(os.path.join(settings.IMG_DIR, os.path.basename(img_path)))
    plt.close()

def log(contents, verbose=True):
    with open(f"{settings.SAVE_DIR}/log.txt", "a") as f:
        for c in contents:
            if verbose:
                print(c)
            f.write(f"{c}\n")