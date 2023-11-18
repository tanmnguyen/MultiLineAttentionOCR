import sys 
sys.path.append("../")

import os 
import numpy as np
import matplotlib.pyplot as plt
from settings import settings 

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