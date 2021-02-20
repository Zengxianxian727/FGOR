
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch

def default_loader(path):
    return Image.open(path).convert('RGB')

class csv_Dataset(Dataset):
    def __init__(self, label_list, transform=None, target_transform=None, loader=default_loader):
        imgs = []
        for index, row in label_list.iterrows():
            imgs.append((row['img_path'], row['label']))
        self.imgs = imgs
        #print(len(self.imgs))
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        filename, label = self.imgs[index]
        img = self.loader(filename)
        if self.transform is not None:
            img = self.transform(img)
        return img, label-1

    def __len__(self):
        return len(self.imgs)

