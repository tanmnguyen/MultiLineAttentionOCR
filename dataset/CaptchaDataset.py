import os 
import cv2 
from torch.utils.data import Dataset

class CAPTCHADataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.img_files = os.listdir(data_dir)
        self.img_files.sort()

        # find invalid files
        inv = []
        for filename in self.img_files:
            if not filename.endswith(".png") and not filename.endswith(".jpg") and not filename.endswith(".jpeg"):
                inv.append(filename)

        # remove invalid files
        for inv_file in inv:
            self.img_files.remove(inv_file)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path= os.path.join(self.data_dir, self.img_files[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lbl = self.img_files[idx].split('.')[0]
        return img, lbl