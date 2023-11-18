import os
import torch
import datetime
import configparser

class Settings:
    def __init__(self):
        self.PAD = '[PAD]'
        self.SOS = '[SOS]'
        self.EOS = '[EOS]'
    
        self.DEVICE = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
        self.SAVE_DIR = f"train-outputs/train-{datetime.datetime.now().strftime('%Y.%m.%d. %H.%M.%S')}"
        self.IMG_DIR = os.path.join(self.SAVE_DIR, "wrong-predictions")

        os.makedirs(self.SAVE_DIR, exist_ok=True)

    def load_configuration(self, cfg_file):
        config = configparser.ConfigParser()
        config.read(cfg_file)

        # Access the configuration values
        self.ARCH = config['DEFAULT']['ARCH']
        self.EPOCHS = config.getint('DEFAULT', 'EPOCHS')
        self.BATCH_SIZE = config.getint('DEFAULT', 'BATCH_SIZE')
        self.LR = config.getfloat('DEFAULT', 'LR')
        self.IMG_W = config.getint('DEFAULT', 'IMG_W')
        self.IMG_H = config.getint('DEFAULT', 'IMG_H')
        self.LETTERS = config['DEFAULT']['LETTERS']
        self.MAX_LEN = config.getint('DEFAULT', 'MAX_LEN')
        self.DATA = config['DEFAULT']['DATA']

        self.LETTERS = [self.PAD, self.SOS, self.EOS] + list(self.LETTERS)
        self.CHAR2IDX = {c : i for i, c in enumerate(self.LETTERS)}
        self.IDX2CHAR = {i : c for i, c in enumerate(self.LETTERS)}

settings = Settings()


