import sys 
sys.path.append("../")

import os 
import numpy as np
import matplotlib.pyplot as plt
from settings import save_directory, image_directory

def save_image_prediction(y_pred_str, y_true_str, img, img_path):
    os.makedirs(image_directory, exist_ok=True)
    plt.imshow(np.uint8(img.permute(1,2,0).cpu().numpy()))
    plt.title(f"Pred: {y_pred_str}\nTrue: {y_true_str}")
    plt.savefig(os.path.join(image_directory, os.path.basename(img_path)))
    plt.close()

def log(contents, verbose=True):
    with open(f"{save_directory}/log.txt", "a") as f:
        for c in contents:
            if verbose:
                print(c)
            f.write(f"{c}\n")