# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 09:48:42 2018

@author: dell
"""
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd

class planetDataset(Dataset):
    """
    img_list: list of path to imgs
    labels: (n,2) pandas dataframe, 1st col is path to img, 2nd col is 0/1
    """
    def __init__(self, labels, subset=False, transform=None):
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        fullname = self.labels.iloc[idx,0]
        image = Image.open(fullname)
        label = self.labels.iloc[idx,1].astype(np.float32)
        if self.transform:
            image = self.transform(image)
        return [image, label]


# TODO - get the labels.csv ready

data_dir = '../input/dog-breed-identification/labels.csv'
labels = pd.read_csv(data_dir)

train = labels.sample(frac=0.8)
valid = labels[~labels['id'].isin(train['id'])]



normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

ds_trans = transforms.Compose([transforms.Scale(224),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               normalize])
    
train_ds = planetDataset(train, transform=ds_trans)
valid_ds = planetDataset(valid, transform=ds_trans)

train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4)
valid_dl = DataLoader(valid_ds, batch_size=4, shuffle=True, num_workers=4)
