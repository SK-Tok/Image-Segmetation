# import torch module
import torch.utils.data as data_utils

# import python module
import numpy as np
from PIL import Image
from pathlib import Path

class NPSegDataset(data_utils.Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_paths = list(Path(self.img_dir).iterdir())
        self.label_paths = list(Path(self.label_dir).iterdir())
        self.transform = transform
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label_path = self.label_paths[idx]
        img = np.load(img_path)
        label = np.load(label_path)
        if len(label.shape) == 2:
            label = label[np.newaxis, :, :]
        if self.transform:
            img, label = self.transform([img,label])
        return img, label
    
    def __len__(self):
        return len(self.img_paths)
    
class ImageSegDataset(data_utils.Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_paths = list(Path(self.img_dir).iterdir())
        self.label_paths = list(Path(self.label_dir).iterdir())
        self.transform = transform
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label_path = self.label_paths[idx]
        img = Image.open(img_path)
        label = Image.open(label_path)
        
        if self.transform:
            img, label = self.transform([img,label])
            
            return img,label
    
    def __len__(self):
        return len(self.img_paths)
