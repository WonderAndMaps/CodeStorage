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

np.random.seed(666)

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

train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=16, shuffle=True)

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
                print('Start training!')
            else:
                model.train(False)
                print('Start validating')

            running_loss = 0.0
            running_corrects = 0
            running_false_neg = 0

            count = 0
            num_of_samples = 0
            num_of_pos = 0
            for inputs, labels in dataloders[phase]:
                count += 1
                if count%50 ==0:
                    print('Done',count,'/',len(dataloders[phase]))
                    print('Running acc',running_corrects.type(torch.FloatTensor).numpy()/num_of_samples)
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                outputs = model(inputs)
                labels = labels.type(torch.cuda.LongTensor)

                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()  # Does the update

                #running_loss += loss.data[0]
                num_of_samples += inputs.size(0)
                num_of_pos += torch.sum(labels.data==1).type(torch.FloatTensor)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_false_neg += torch.sum(preds[labels.data==1] == 0)
            
            if phase == 'train':
                train_epoch_loss = running_loss / dataset_sizes[phase]
                train_epoch_acc = running_corrects.type(torch.FloatTensor) / dataset_sizes[phase]
                train_epoch_false_neg = running_false_neg.type(torch.FloatTensor) / num_of_pos
            else:
                valid_epoch_loss = running_loss / dataset_sizes[phase]
                valid_epoch_acc = running_corrects.type(torch.FloatTensor) / dataset_sizes[phase]
                valid_epoch_false_neg = running_false_neg.type(torch.FloatTensor) / num_of_pos
                
            if phase == 'valid' and valid_epoch_acc > best_acc:
                best_acc = valid_epoch_acc
                type_II = valid_epoch_false_neg
                best_model_wts = model.state_dict()

        print('Epoch [{}/{}] \n train loss: {:.4f} acc: {:.4f} type II: {:.4f} \n' 
              'valid loss: {:.4f} acc: {:.4f} type II: {:.4f}'.format(
                epoch, num_epochs - 1,
                train_epoch_loss, train_epoch_acc, train_epoch_false_neg,
                valid_epoch_loss, valid_epoch_acc, valid_epoch_false_neg))
            
    print('Best val Acc: {:4f} and its type II: {:4f}'.format(best_acc,type_II))
    print('Time spent:',time.time()-since,'s')

    model.load_state_dict(best_model_wts)
    return model

def false_neg(dataloders, model):
    since = time.time()
    use_gpu = torch.cuda.is_available()
    dataset_sizes = {'train': len(dataloders['train'].dataset), 
                     'valid': len(dataloders['valid'].dataset)}

    running_corrects = 0
    running_false_neg = 0
    running_false_pos = 0

    count = 0
    num_of_samples = 0
    num_of_pos = 0
    for inputs, labels in dataloders['valid']:
        count += 1
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)


        outputs = model(inputs)
        labels = labels.type(torch.cuda.LongTensor)

        _, preds = torch.max(outputs.data, 1)


        num_of_samples += inputs.size(0)
        num_of_pos += torch.sum(labels.data==1).type(torch.FloatTensor)
        running_corrects += torch.sum(preds == labels.data)
        running_false_neg += torch.sum(preds[labels.data==1] == 0)
        running_false_pos += torch.sum(preds[labels.data==0] == 1)
        
        #print([name for name,p,l in zip(fullname,preds.tolist(),labels.tolist()) if p==0 and l==1])
            
    valid_epoch_acc = running_corrects.type(torch.FloatTensor) / dataset_sizes['valid']
    valid_epoch_false_neg = running_false_neg.type(torch.FloatTensor) / num_of_pos
    valid_epoch_false_pos = running_false_pos.type(torch.FloatTensor) / (num_of_samples-num_of_pos)                                
            
    print('Val Acc: {:4f}'.format(valid_epoch_acc))
    print('Type II: {:4f}'.format(valid_epoch_false_neg))
    print('Type I: {:4f}'.format(valid_epoch_false_pos))
    print('Time spent:',time.time()-since,'s')
    return None


#===============================================================training models

#==================================================resnet50
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
optimizer = torch.optim.SGD(resnet.fc.parameters(), lr=1e-3, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


model1 = train_model(dloaders, resnet, criterion, optimizer, exp_lr_scheduler, num_epochs=20)

# Save model
torch.save(model1, data_dir+'/resnet50_180822.pkl')
#0.789702


# Load model
#the_model = torch.load(PATH)
model1 = torch.load(data_dir+'/resnet50_180822.pkl')
false_neg(dloaders,model1)


#==================================================densenet161
densenet = models.densenet161(pretrained=True)
# freeze all model parameters
for param in densenet.parameters():
    param.requires_grad = False

# new final layer with 2 classes
num_ftrs = densenet.classifier.in_features
densenet.classifier = torch.nn.Linear(num_ftrs, 2)

use_gpu = torch.cuda.is_available()
if use_gpu:
    densenet = densenet.cuda()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(densenet.classifier.parameters(), lr=1e-3, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


model2 = train_model(dloaders, densenet, criterion, optimizer, exp_lr_scheduler, num_epochs=20)

# Save model
torch.save(model2, data_dir+'/densenet161_180823.pkl')
#0.788258

model2 = torch.load(data_dir+'/densenet161_180823.pkl')
false_neg(dloaders,model2)



#=======================================ResNet-34
resnet = models.resnet34(pretrained=True)
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
optimizer = torch.optim.SGD(resnet.fc.parameters(), lr=1e-3, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


model3 = train_model(dloaders, resnet, criterion, optimizer, exp_lr_scheduler, num_epochs=20)

# Save model
torch.save(model3, data_dir+'/resnet34_180824.pkl')
#0.785611

model3 = torch.load(data_dir+'/resnet34_180824.pkl')
false_neg(dloaders,model3)

#======================================Inception 3*299*299
# see another file


#==================================================resnet152
resnet = models.resnet152(pretrained=True)
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
optimizer = torch.optim.SGD(resnet.fc.parameters(), lr=1e-3, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


model5 = train_model(dloaders, resnet, criterion, optimizer, exp_lr_scheduler, num_epochs=20)

# Save model
torch.save(model5, data_dir+'/resnet152_180825.pkl')
#0.789702

model5 = torch.load(data_dir+'/resnet152_180825.pkl')
false_neg(dloaders,model5)

#=======================================vgg19bn
vgg = models.vgg19_bn(pretrained=True)

for param in vgg.parameters():
    param.requires_grad = False

# new final layer with 2 classes
num_ftrs = vgg.classifier[6].in_features
vgg.classifier[6] = torch.nn.Linear(num_ftrs, 2)

use_gpu = torch.cuda.is_available()
if use_gpu:
    vgg = vgg.cuda()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(vgg.classifier[6].parameters(), lr=1e-3, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


model6 = train_model(dloaders, vgg, criterion, optimizer, exp_lr_scheduler, num_epochs=20)

# Save model
torch.save(model6, data_dir+'/vgg19bn_180825.pkl')
#0.779115


model6 = torch.load(data_dir+'/vgg19bn_180825.pkl')
false_neg(dloaders,model6)

#==================================================densenet201
densenet = models.densenet201(pretrained=True)
# freeze all model parameters
for param in densenet.parameters():
    param.requires_grad = False

# new final layer with 2 classes
num_ftrs = densenet.classifier.in_features
densenet.classifier = torch.nn.Linear(num_ftrs, 2)

use_gpu = torch.cuda.is_available()
if use_gpu:
    densenet = densenet.cuda()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(densenet.classifier.parameters(), lr=1e-3, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


model7 = train_model(dloaders, densenet, criterion, optimizer, exp_lr_scheduler, num_epochs=20)

# Save model
torch.save(model7, data_dir+'/densenet201_180825.pkl')
#0.791627


model7 = torch.load(data_dir+'/densenet201_180825.pkl')
false_neg(dloaders,model7)
