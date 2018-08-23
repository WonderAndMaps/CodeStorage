# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 09:48:42 2018

@author: yang

Borrow lots of the codes from 
https://www.kaggle.com/pvlima/use-pretrained-pytorch-models
"""

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import time
from torch.autograd import Variable
import torch
from torch.optim import lr_scheduler


#=======================================Data loader part
class planetDataset(Dataset):
    """
    img_list: list of path to imgs
    labels: (n,3) pandas dataframe, 2nd col is path to img, 3rd col is 0/1
    """
    def __init__(self, labels, subset=False, transform=None):
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        fullname = self.labels.iloc[idx,1]
        image = Image.open(fullname)
        image = image.convert("RGB")
        label = self.labels.iloc[idx,2].astype(np.float32)
        if self.transform:
            image = self.transform(image)
        return [image, label]


# TODO - get the labels.csv ready

data_dir = 'F:/sky_obs'
labels = pd.read_csv(data_dir+'/sky_obs_labels.csv')

train = labels.sample(frac=0.8)
valid = labels[~labels['img_path'].isin(train['img_path'])]



normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
ds_trans = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), normalize])
    
train_ds = planetDataset(train, transform=ds_trans)
valid_ds = planetDataset(valid, transform=ds_trans)

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=32, shuffle=True)

dloaders = {'train':train_dl, 'valid':valid_dl}

'''
def imshow(axis, inp):
    """Denormalize and show"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, None)
    axis.imshow(inp)
    
    
img, label = next(iter(train_dl))
print(img.size(), label.size())
fig = plt.figure(1, figsize=(16, 4))
grid = ImageGrid(fig, 111, nrows_ncols=(1, 4), axes_pad=0.05)    
for i in range(img.size()[0]):
    ax = grid[i]
    imshow(ax, img[i])
'''

#==============================model part

def train_model(dataloders, model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    use_gpu = torch.cuda.is_available()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    dataset_sizes = {'train': len(dataloders['train'].dataset), 
                     'valid': len(dataloders['valid'].dataset)}

    for epoch in range(num_epochs):
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train(True)
                print('Training')
            else:
                model.train(False)
                print('Validating')

            running_loss = 0.0
            running_corrects = 0

            count = 0
            for inputs, labels in dataloders[phase]:
                count += 1
                if count%50 ==0:
                    print('Done',round(count/len(dataloders[phase])*100,2),'%')
                    print('Accuracy',running_corrects.type(torch.FloatTensor)/inputs.size(0)/count)
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                
                labels = labels.type(torch.LongTensor)
                
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()  # Does the update

                #running_loss += loss.data[0]
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                train_epoch_loss = running_loss / dataset_sizes[phase]
                train_epoch_acc = running_corrects.type(torch.FloatTensor) / dataset_sizes[phase]
            else:
                valid_epoch_loss = running_loss / dataset_sizes[phase]
                valid_epoch_acc = running_corrects.type(torch.FloatTensor) / dataset_sizes[phase]
                
            if phase == 'valid' and valid_epoch_acc > best_acc:
                best_acc = valid_epoch_acc
                best_model_wts = model.state_dict()

        print('Epoch [{}/{}] train loss: {:.4f} acc: {:.4f} ' 
              'valid loss: {:.4f} acc: {:.4f}'.format(
                epoch, num_epochs - 1,
                train_epoch_loss, train_epoch_acc, 
                valid_epoch_loss, valid_epoch_acc))
            
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Time spent:',time.time()-since,'s')

    model.load_state_dict(best_model_wts)
    return model



resnet = models.resnet50(pretrained=True)
# freeze all model parameters
for param in resnet.parameters():
    param.requires_grad = False

# new final layer with 2 classes
num_ftrs = resnet.fc.in_features
resnet.fc = torch.nn.Linear(num_ftrs, 2)

use_gpu = torch.cuda.is_available()
if use_gpu:
    resnet = resnet.cuda()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet.fc.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.01)


model = train_model(dloaders, resnet, criterion, optimizer, exp_lr_scheduler, num_epochs=2)
