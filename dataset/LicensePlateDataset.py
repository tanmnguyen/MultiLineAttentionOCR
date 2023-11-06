import os 
import cv2 
import zipfile
import numpy as np

from torch.utils.data import Dataset

class LicensePlateDataset(Dataset):
    def __init__(self, data_path: str):
        self.zipfile = zipfile.ZipFile(data_path, mode="r")

        img_paths, lbls = [], []

        has_file = {}
        for file in self.zipfile.namelist():
            has_file[file] = True
        
        for file in has_file:
            # check image format
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                lbl_path = file.split(".")[0] + ".txt"
                # check label format
                if lbl_path in has_file:
                    img_paths.append(file)
                    # read content of lbl file
                    with self.zipfile.open(lbl_path) as lbl_file:
                        lbls.append(lbl_file.read().decode("utf-8"))

        self.img_paths = img_paths
        self.lbls      = lbls

    def __len__(self):
        assert len(self.img_paths) == len(self.lbls)
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path, lbl = self.img_paths[idx], self.lbls[idx]
        with self.zipfile.open(img_path) as img_file:
            img = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img, lbl
           