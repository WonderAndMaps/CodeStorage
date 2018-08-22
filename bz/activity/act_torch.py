# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 18:47:04 2018

@author: dell
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from PIL import Image
import torch
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import torch.optim as optim
import torch.nn as nn
import cv
import copy


INPUT_SIZE = 224
NUM_CLASSES = 2

# TODO - change here
#data_dir = '../input/dog-breed-identification/'
#labels = pd.read_csv(join(data_dir, 'labels.csv'))




def loadRgbImage(idxs,img_list,width,height):
    imgs = np.zeros([len(idxs),3,width,height])
    for i in range(len(idxs)):
        imgs[i,:,:,:] = cv.resize(cv.imread(img_list[i]),(width,height)).transpose()
    return imgs.astype(np.float32)


x_np = loadRgbImage([0,100,1000],img_list,INPUT_SIZE,INPUT_SIZE)
x_tensor = torch.from_numpy(x_np)
x = Variable(x_tensor,requires_grad=False)




model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False


fc_features = model.fc.in_features  
model.fc = nn.Linear(fc_features, 2)
optimizer = optim.SGD(model.fc.parameters(), lr=1e-4, momentum=0.9)
criterion = nn.CrossEntropyLoss()


model(x)


'''


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    use_gpu = torch.cuda.is_available()


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

'''