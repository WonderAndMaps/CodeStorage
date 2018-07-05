# -*- coding: utf-8 -*-
"""
Created on Mon May 12 10:02:31 2018

@author: fuyang
"""
# In[1]:

from __future__ import division
import time
import _pickle as cPickle

from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from torch.autograd import Variable
from sklearn.metrics import average_precision_score as sk_average_precision_score
from scipy.stats import rankdata
from multiprocessing.dummy import Pool as ThreadPool
import gc
import xarray as xr
import os

from PIL import Image

import torch
import torch.nn as nn

from math import ceil, floor

# In[2]:

# # Outline:
# This tutorial contains the following sections:
# - Visualizing objects in an image
# - Region proposals with selective search
# - Projecting bounding boxes to feature space
# - Extracting features for a bounding box
# 
# #### Feature extraction:
# Given a bounding box and an image, we extract features 
# corresponding to the bounding box in a two-step procedure.
# The first step is to "project" the bounding box onto the feature space.
# The second step to use the Featurizer module to extract features corresponding to this projected bounding box.

# selective search tutorial: https://www.learnopencv.com/selective-search-for-object-detection-cpp-python/



# In[2]:
    
def average_precision_score(y_true,y_pred,threshold=0):
    pred = []
    actual = []
    for (s, l) in zip(y_pred, y_true):
        if s>threshold or l==1.0:
            pred += [s]
            actual += [l]
    ap = sk_average_precision_score(actual, pred)
    return ap
    
    
def iou(rect1, rect2): # rect = [x, y, w, h]
    x1, y1, w1, h1 = rect1
    X1, Y1 = x1+w1, y1 + h1
    x2, y2, w2, h2 = rect2
    X2, Y2 = x2+w2, y2 + h2
    a1 = (X1 - x1 + 1) * (Y1 - y1 + 1)
    a2 = (X2 - x2 + 1) * (Y2 - y2 + 1)
    x_int = max(x1, x2) 
    X_int = min(X1, X2) 
    y_int = max(y1, y2) 
    Y_int = min(Y1, Y2) 
    a_int = (X_int - x_int + 1) * (Y_int - y_int + 1) * 1.0 
    if x_int > X_int or y_int > Y_int:
        a_int = 0.0 
    return a_int / (a1 + a2 - a_int)  


# nearest neighbor in 1-based indexing
def _nnb_1(x):                                                                                                                               
    x1 = int(floor((x + 8) / 16.0))
    x1 = max(1, min(x1, 13))
    return x1


def project_onto_feature_space(rect, image_dims):
    # project bounding box onto conv net
    # @param rect: (x, y, w, h)
    # @param image_dims: (imgx, imgy), the size of the image
    # output bbox: (x, y, x'+1, y'+1) where the box is x:x', y:y'

    # For conv 5, center of receptive field of i is i_0 = 16 i for 1-based indexing
    imgx, imgy = image_dims
    x, y, w, h = rect
    # scale to 224 x 224, standard input size.
    x1, y1 = ceil((x + w) * 224 / imgx), ceil((y + h) * 224 / imgy)
    x, y = floor(x * 224 / imgx), floor(y * 224 / imgy)
    px = _nnb_1(x + 1) - 1 # inclusive
    py = _nnb_1(y + 1) - 1 # inclusive
    px1 = _nnb_1(x1 + 1) # exclusive
    py1 = _nnb_1(y1 + 1) # exclusive

    return [px, py, px1, py1]


class Featurizer:
    dim = 11776 # for small features
    def __init__(self):
        # pyramidal pooling of sizes 1, 3, 6
        self.pool1 = nn.AdaptiveMaxPool2d(1)
        self.pool3 = nn.AdaptiveMaxPool2d(3)                                                                                                 
        self.pool6 = nn.AdaptiveMaxPool2d(6)
        self.lst = [self.pool1, self.pool3, self.pool6]
        
    def featurize(self, projected_bbox, image_features):
        # projected_bbox: bbox projected onto final layer
        # image_features: C x W x H tensor : output of conv net
        full_image_features = torch.from_numpy(image_features)
        x, y, x1, y1 = projected_bbox
        crop = full_image_features[:, x:x1, y:y1] 
#         return torch.cat([self.pool1(crop).view(-1), self.pool3(crop).view(-1),  
#                           self.pool6(crop).view(-1)], dim=0) # returns torch Variable
        return torch.cat([self.pool1(crop).view(-1), self.pool3(crop).view(-1),  
                          self.pool6(crop).view(-1)], dim=0).data.numpy() # returns numpy array


# In[18]:

# read features: 
    
class loader_detection:
    
    def __init__(self,dataDir='D:/uw/stat548/proj/data',dataType='train2014',load_feat_now=True):
        self.dataDir = dataDir
        self.dataType = dataType
        self.annFile = '{}/annotations/instances_{}.json'.format(dataDir,dataType)
        self.coco = COCO(self.annFile)
        
        if load_feat_now:
            [self.img_ids, self.feats] = cPickle.load(open('{}/features_small/{}.p'.format(self.dataDir, self.dataType),'rb'),encoding='bytes')
        else:
            self.feats=None
        
        [self.img_ids, self.bboxes] = cPickle.load(open('{}/bboxes/{}_bboxes.p'.format(self.dataDir, self.dataType),'rb'),encoding='bytes')
        self.featurizer = Featurizer()
        
        cats = self.coco.loadCats(self.coco.getCatIds()) # categories
        #cat_names = list(set([cat['name'] for cat in cats]))
        #self.cat_order = {_: i for (_,i) in zip(cat_names,range(80))}
        #self.cat_order_to_name = {i: _ for (_,i) in zip(cat_names,range(80))}
        self.cat_id_to_name = {cat['id']: cat['name'] for cat in cats}
        
        #use these in the future
        self.cat_order = {x: i for (x,i) in zip(self.cat_id_to_name,range(80))}
        self.cat_order_to_name = {i: x for (x,i) in zip(self.cat_id_to_name,range(80))}
        
        self.box_img_idx = []
        self.box_id_for_img = []
    
    def data(self,sample_id,sample_box_id=[],get_feat=True,get_lab=True,feats=None):
        '''
        This function returns boxes features and labels of some boxes in some images.
        -----------
        sample_id: list of shape n_sample. Each element is an integer id of an image.
        sample_box_id: list of shape n_sample. Each element is a list of patch id of an image. Does not matter when get_feat=False.
        feats: array (n,d). Will pass to self.feats
        '''
        t1 = time.time()
        sample_img_ids = list(np.array(self.img_ids)[sample_id])
        num_of_bbox = [len(x) if x is not None else 0 for x in self.bboxes]
        self.feats = feats
        
        if sample_box_id == []:
            sample_box_id = [[x for x in range(num_of_bbox[j])] for j in sample_id]
                
        k = 0   #counting the number of img visited
        count_feat = 0   #counting the number of bbox visited
        count_lab = 0
        
        if get_lab is True:
            bbox_labels = np.array([[0.0]*80]*sum([len(_) for _ in sample_box_id])).astype(np.float32)
        
        if get_feat is True:
            bbox_feats = np.array([[0.0]*11776]*sum([len(_) for _ in sample_box_id])).astype(np.float32)
    
        for img_id in sample_img_ids:
            idx = self.img_ids.index(img_id)
            img = self.coco.loadImgs([img_id])[0]
            img_pil = Image.open('%s/%s/%s'%(self.dataDir, self.dataType, img['file_name']))
            
            #features
            if get_feat is True and self.bboxes[sample_id[k]] is not None:
                img_feats = self.feats[idx]
                for i in range(len(self.bboxes[sample_id[k]])): 
                    #extract features for each box
                    if i in sample_box_id[k]:
                        projected_bbox = project_onto_feature_space(self.bboxes[sample_id[k]][i], img_pil.size)
                        bbox_feats[count_feat,:] = self.featurizer.featurize(projected_bbox, img_feats)
                        count_feat += 1
            
            #labels 
            if get_lab is True and self.bboxes[sample_id[k]] is not None:
                annIds = self.coco.getAnnIds(imgIds=img['id'], iscrowd=None)
                anns = self.coco.loadAnns(annIds)

                true_bboxes = []
                true_bboxes_cat = []
                L = len(self.bboxes[sample_id[k]])
                self.box_img_idx = np.concatenate((self.box_img_idx,[sample_id[k]]*L))
                self.box_id_for_img = np.concatenate((self.box_id_for_img,[j for j in range(L)]))
                bbox_labels_temp = [[0]*L for j in range(80)]
                for ann in anns:
                    true_bboxes += [ann['bbox']]
                    true_bboxes_cat += [ann['category_id']]
                    true_rect = true_bboxes[-1]
                    ious = np.asarray([iou(true_rect, r) for r in self.bboxes[sample_id[k]]])
                    bbox_labels_temp[self.cat_order[true_bboxes_cat[-1]]] += (ious>=0.5)*1.0
                
                bbox_labels[count_lab:count_lab+L,:] = np.array(bbox_labels_temp).transpose().astype(np.float32)
                count_lab += L
                
            k += 1
        
        print('time to load features/labels =', time.time() - t1, 'sec')
                
        animal_and_vehicle = [2,3,4,5,6,7,8,9,16,17,18,19,20,21,22,23,24,25]
        if get_lab is True:
            wanted = [self.cat_order[i] for i in animal_and_vehicle]
            bbox_labels = bbox_labels[:,wanted]
        
        if get_lab is True and get_feat is True:
            return bbox_feats, bbox_labels
        elif get_lab is True:
            return bbox_labels
        elif get_feat is True:
            return bbox_feats
        
        
        
# In[18.8]:
# classifier

class mlp_hvy_ball:
    
    def __init__(self, reg=10, hidden=10, learn_rate=1e-3, gamma=0.9, batch_size=32, tol_err=-1, max_round=100):
        self.h = np.int(hidden)
        self.learn_rate = learn_rate
        self.tol_err = tol_err
        self.max_round = max_round
        self.w1 = None
        self.w2 = None
        self.train_score=[]
        self.val_score=[]
        self.train_obj=[]
        self.val_obj=[]
        self.batch_size = batch_size
        self.runtime = []
        self.gamma = gamma
        self.v1 = None
        self.v2 = None
        self.reg = reg

    
    def fit(self, X_train, y_train, X_val, y_val, loss_func='l2'):
        n,d = X_train.shape
        n,k = y_train.shape
        
        if self.w1 is None:
            self.w1 = np.random.randn(d,self.h).astype(np.float32)/100
            self.w2 = np.random.randn(self.h,k).astype(np.float32)/100
            self.v1 = np.zeros([d,self.h]).astype(np.float32)
            self.v2 = np.zeros([self.h,k]).astype(np.float32)
            
    
        X_train_torch = Variable(torch.from_numpy(X_train), requires_grad=False)
        y_train_torch = Variable(torch.from_numpy(y_train), requires_grad=False)
        train_err = self.tol_err + 1
        
        print('Start training MLP!')
        
        time_start=time.time()
        count = 0
        i = 0
        while(train_err > self.tol_err):
            i += 1
            shuffle = np.random.choice([_1 for _1 in range(n)],size=n,replace=False)
            split_point = [0+self.batch_size*_2 for _2 in range(np.ceil(n/self.batch_size).astype(np.int))]+[n]
            
            for j in range(len(split_point)-1):
                count += 1
                
                #do gradient
                w1_torch = Variable(torch.from_numpy(self.w1), requires_grad=True)
                w2_torch = Variable(torch.from_numpy(self.w2), requires_grad=True)
                X_b = X_train_torch[shuffle[split_point[j]:split_point[j+1]],:].contiguous().view([-1,d])
                y_b = y_train_torch[shuffle[split_point[j]:split_point[j+1]],:].contiguous().view([-1,k])
                
                y_hat = torch.clamp(X_b.mm(w1_torch),min=0).mm(w2_torch)
                
                if loss_func is 'l2':
                    loss = (y_b-y_hat)**2/2
                
                if loss_func is 'log':
                    loss = y_b*torch.log(1+torch.exp(-y_hat)) + (1-y_b)*torch.log(1+torch.exp(y_hat))
                
                if loss_func is 'hinge':
                    loss = torch.clamp(1-(2*y_b-1)*y_hat,min=0)**2/4
                                              
                Loss = torch.sum(loss)/self.batch_size + self.reg*(torch.norm(w1_torch)**2+torch.norm(w2_torch)**2)/2
                Loss.backward()
                self.v1 = self.gamma*self.v1 + w1_torch.grad.data.numpy()
                self.v2 = self.gamma*self.v2 + w2_torch.grad.data.numpy()
                self.w1 -= self.learn_rate * self.v1
                self.w2 -= self.learn_rate * self.v2

                #reset gradient
                if w1_torch.grad is not None: w1_torch.grad.data.zero_()
                if w2_torch.grad is not None: w2_torch.grad.data.zero_()

                if count%np.floor(n/2/self.batch_size) == 0:
                    #sample = np.random.choice([_1 for _1 in range(n)],size=np.int(n/1),replace=False)
                    sample = np.random.choice([_1 for _1 in range(n)],size=np.int(n/5),replace=False)
                    temp1 = self.predict(X_train[sample,:])
                    temp2 = self.predict(X_val)
                    temp4 = self.reg*(np.linalg.norm(self.w1)**2+np.linalg.norm(self.w2)**2)/2
                    self.train_score.append(average_precision_score(y_train[sample,:],temp1))
                    self.val_score.append(average_precision_score(y_val,temp2))
                                        
                    if loss_func is 'l2':
                        self.train_obj.append(((temp1 - y_train[sample,:])**2).sum()/sample.shape[0]/2+temp4)
                        self.val_obj.append(((temp2 - y_val)**2).sum()/X_val.shape[0]/2+temp4)
                    
                    if loss_func is 'log':
                        self.train_obj.append((y_train[sample,:]*np.log(1+np.exp(-temp1)) + (1-y_train[sample,:])*np.log(1+np.exp(temp1))).sum()/sample.shape[0]+temp4)
                        self.val_obj.append((y_val*np.log(1+np.exp(-temp2)) + (1-y_val)*np.log(1+np.exp(temp2))).sum()/X_val.shape[0] + temp4)
    
                    if loss_func is 'hinge':
                        self.train_obj.append((np.clip(1-(2*y_train[sample,:]-1)*temp1,0,None)**2/4).sum()/sample.shape[0]+temp4)
                        self.val_obj.append((np.clip(1-(2*y_val-1)*temp2,0,None)**2/4).sum()/X_val.shape[0]+temp4)
                    
                    print('Step',count,'\t training loss:',self.train_obj[-1],'\t at hidden',self.h)
                    print('Step',count,'\t training score:',self.train_score[-1],'\t at hidden',self.h)
                    print('Step',count,'\t validate score:',self.val_score[-1],'\t at hidden',self.h)
                    train_err = 1-self.train_score[-1]
                    self.runtime.append(time.time()-time_start)

        
            if (i>self.max_round): 
                print('Reached max_iter, did not converge.')
                break
    
        print('End training!')
        
    def predict(self, X_test):
        return np.clip(X_test.dot(self.w1),0,None).dot(self.w2)


# In[18.88]:

# get positive/negative sample features for each cat
def get_pos_feat(cat,pos,loader,feats):
    '''
    pos: array of shape (n,2). Idx of samples in y_all
    cat: int. The cat to deal with
    loader: data_sample_loader class.
    '''
    idx = pos[pos[:,1]==cat,0]
    sample_id_temp = loader.box_img_idx.astype(int)[idx]
    sample_box_id = []
    temp = loader.box_id_for_img.astype(int)[idx]
    sample_id = list(set(sample_id_temp))
    sample_id.sort()
    for j in sample_id:
        sample_box_id.append(list(temp[sample_id_temp==j]))
    
    return loader.data(sample_id,sample_box_id,get_lab=False,feats=feats)

#classifier with hard neg sampling for one category
def one_cat_cl(cat,y_dir,pos_train,pos_val,
               loader_train,loader_val,train_feat,val_feat,
               max_round=10,reg=10):
    
    y_train_all = pd.read_table(y_dir+'/train_det_lab.txt',sep=',',dtype=np.float32,usecols=[cat]).values    
    y_val_all = pd.read_table(y_dir+'/val_det_lab.txt',sep=',',dtype=np.float32,usecols=[cat]).values 
    
    #get positive examples for each cat
    X_train_pos = get_pos_feat(cat,pos_train,loader_train,feats=train_feat)
    X_val_pos = get_pos_feat(cat,pos_val,loader_val,feats=val_feat)
    
    n_train = X_train_pos.shape[0]
    n_val = X_val_pos.shape[0]
    
    if n_train>25000:
        sample = np.random.choice([_1 for _1 in range(n_train)],size=25000,replace=False)
        X_train_pos = X_train_pos[sample,:]
        
    if n_val>5000:
        sample = np.random.choice([_1 for _1 in range(n_val)],size=5000,replace=False)
        X_val_pos = X_val_pos[sample,:]
        
    neg_train = np.argwhere(y_train_all[:,0]==0)
    neg_train = np.concatenate((neg_train,np.array([cat]*neg_train.shape[0]).reshape([-1,1])),axis=1)
    neg_val = np.argwhere(y_val_all[:,0]==0)
    neg_val = np.concatenate((neg_val,np.array([cat]*neg_val.shape[0]).reshape([-1,1])),axis=1)

    n_train = X_train_pos.shape[0]
    n_val = X_val_pos.shape[0]
    
    #first uniformly randomly draw 2n neg examples
    sample_idx_train = random.sample([i for i in range(neg_train.shape[0])],k=2*n_train)
    sample_idx_val = random.sample([i for i in range(neg_val.shape[0])],k=2*n_val)
    X_train_neg = get_pos_feat(cat,neg_train[sample_idx_train,:],loader_train,feats=train_feat)
    X_val_neg = get_pos_feat(cat,neg_val[sample_idx_val,:],loader_val,feats=val_feat)
    
    y_train = np.array([1]*n_train+[0]*2*n_train).reshape([-1,1]).astype(np.float32)
    y_val = np.array([1]*n_val+[0]*2*n_val).reshape([-1,1]).astype(np.float32)
    
    #build a model
    model_for_cat = []
    
    for j in range(max_round):
        print('Start\t',j,'-th round of category\t',cat,'!')
        
        model = mlp_hvy_ball(reg=reg, learn_rate=8e-5, gamma=0.9, batch_size=32, max_round=4)
        model.fit(np.concatenate((X_train_pos,X_train_neg)),y_train,np.concatenate((X_val_pos,X_val_neg)),y_val,loss_func='hinge')
        model_for_cat.append(model)
        
        ##free some memory
        #del X_train_neg, X_val_neg
        #gc.collect()
        
        if j != max_round-1:
            #get hard n neg examples
            if 5*n_train < 30000:
                sample_idx_train = random.sample([i for i in range(neg_train.shape[0])],k=5*n_train)
                sample_idx_val = random.sample([i for i in range(neg_val.shape[0])],k=5*n_val)
            else:
                sample_idx_train = random.sample([i for i in range(neg_train.shape[0])],k=int(1.5*n_train))
                sample_idx_val = random.sample([i for i in range(neg_val.shape[0])],k=int(1.5*n_val))
        
            X_train_neg = get_pos_feat(cat,neg_train[sample_idx_train,:],loader_train,feats=train_feat)
            X_train_neg = X_train_neg[np.argwhere(rankdata(-model.predict(X_train_neg),method='ordinal')<=n_train)[:,0],:]

            X_val_neg = get_pos_feat(cat,neg_val[sample_idx_val,:],loader_val,feats=val_feat)
            X_val_neg = X_val_neg[np.argwhere(rankdata(-model.predict(X_val_neg),method='ordinal')<=n_val)[:,0],:]
        
            #get random n neg examples
            sample_idx_train = random.sample([i for i in range(neg_train.shape[0])],k=n_train)
            sample_idx_val = random.sample([i for i in range(neg_val.shape[0])],k=n_val)
            X_train_neg = np.concatenate((X_train_neg,get_pos_feat(cat,neg_train[sample_idx_train,:],loader_train,train_feat)))
            X_val_neg = np.concatenate((X_val_neg,get_pos_feat(cat,neg_val[sample_idx_val,:],loader_val,val_feat)))
    
    with open('D:/uw/stat548/proj/models/cat_'+str(cat)+'_models.p', 'wb') as f:
        cPickle.dump(model_for_cat, f)
    
    #score = np.array([sum(model_for_cat[-1].train_score)/len(model_for_cat[-1].train_score),sum(model_for_cat[-1].val_score)/len(model_for_cat[-1].val_score)])
    #np.savetxt('D:/uw/stat548/proj/cv/cat_'+str(cat)+'_models_'+str(max_round)+'_lambd_'+str(reg)+'.txt',score)
    
    return model_for_cat

# In[18.888]:
dataDir='D:/uw/stat548/proj/data'
#dataType='train2014' 
#dataType='val2014'
#dataType='test2014'


#===========================preprocessing and save

"""
#================feat

dataType='train2014' # uncomment to access the train set
#dataType='val2014' # uncomment to access the validation set
#dataType='test2014' # uncomment to access the train set

#Load extracted features
t1 = time.time()
with open(os.path.join(dataDir, 'features_small', '{}.p'.format(dataType)),'rb') as f:
    [img_list, feats] = cPickle.load(f,encoding='bytes')

print('time to load features =', time.time() - t1, 'sec')
print('num images =', len(img_list))
print('shape of features =', feats.shape)

feats = xr.DataArray(feats)
feats.to_netcdf(dataDir+'/train_feat.nc')

del feats
gc.collect()
#=============lab

#train
loader_train = loader_detection(dataDir,'train2014',load_feat_now=False)

all_id = [i for i in range(10000)]  #train 1000; val 2000; test 2000
y_all = loader_train.data(all_id,get_feat=False)
y_all[y_all>1] = 1
y_all = pd.DataFrame(y_all,dtype=np.float32)

y_all.to_csv(dataDir+'/train_det_lab.txt',index=False)
y_all = None
gc.collect()

with open(dataDir+'/loader_det_train.p', 'wb') as f:
    cPickle.dump(loader_train, f)
    

#val
loader_val = loader_detection(dataDir,'val2014')

all_id = [i for i in range(2000)]
y_all = loader_val.data(all_id,get_feat=False)
y_all[y_all>1] = 1.0
y_all = pd.DataFrame(y_all,dtype=np.float32)

y_all.to_csv(dataDir+'/val_det_lab.txt',index=False)
y_all = None                   
gc.collect()

with open(dataDir+'/loader_det_val.p', 'wb') as f:
    cPickle.dump(loader_val, f)
    

#test
loader_test = loader_detection(dataDir,'test2014')
all_id = [i for i in range(2000)]
y_all = loader_test.data(all_id,get_feat=False)
y_all[y_all>1] = 1
y_all = pd.DataFrame(y_all,dtype=np.float32)

y_all.to_csv(dataDir+'/test_det_lab.txt',index=False)
y_all = None                   
gc.collect()

with open(dataDir+'/loader_det_test.p', 'wb') as f:
    cPickle.dump(loader_test, f)                 
"""

#============load data and ...
img_train_feat = xr.open_dataarray(dataDir+'/train_feat.nc').values
img_val_feat = xr.open_dataarray(dataDir+'/val_feat.nc').values

y_train_all = pd.read_table(dataDir+'/train_det_lab.txt',sep=',',dtype=np.float32).values
y_val_all = pd.read_table(dataDir+'/val_det_lab.txt',sep=',',dtype=np.float32).values

                         
pos_train = np.argwhere(y_train_all>0)
pos_val = np.argwhere(y_val_all>0)
cats = [i for i in range(18)]

del y_train_all,y_val_all
gc.collect()
                         
                         
                         
with open(dataDir+'/loader_det_train.p', 'rb') as f:
    loader_train = cPickle.load(f)
with open(dataDir+'/loader_det_val.p', 'rb') as f:
    loader_val = cPickle.load(f)

loader_train.dataDir = dataDir 
loader_val.dataDir = dataDir




time_start=time.time()
pool = ThreadPool(2)
results = pool.map(lambda cat: one_cat_cl(cat,dataDir,pos_train,pos_val,
                                          loader_train,loader_val,img_train_feat,img_val_feat,
                                          max_round=5,reg=5), [0,7])
pool.close()
pool.join()
print('Time spent: ', time.time()-time_start, 's')


'''
model_for_cat=one_cat_cl(5,dataDir,pos_train,pos_val,loader_train,loader_val,img_train_feat,img_val_feat,max_round=20,reg=5)
sum(model_for_cat[-1].val_score)/len(model_for_cat[-1].val_score)
'''
# In[30624700]:
#=============================plots
#models with no retrain
modelDir = 'D:/uw/stat548/proj/models'
models_noretrain = []

cat_names = [loader_train.cat_id_to_name[i] for i in [2,3,4,5,6,7,8,9,16,17,18,19,20,21,22,23,24,25]]

for i in range(18):
    with open(modelDir+'/cat_'+str(i)+'_models.p', 'rb') as f:
        model = cPickle.load(f)
    
    models_noretrain.append([model])
    
    plt.figure(12,figsize=(12,2))
    plt.subplots_adjust(wspace=0.5)
    plt.subplot(221)
    l1,=plt.plot([(_+1)*1/2 for _ in range(len(model[-1].train_obj))], model[-1].train_obj, linewidth=1, color='r')
    l2,=plt.plot([(_+1)*1/2 for _ in range(len(model[-1].train_obj))], model[-1].val_obj, linewidth=1, color='g')
    plt.xlabel('Pass')
    plt.ylabel('Object')
    plt.legend(handles = [l1, l2], labels = ['train','val'], loc = 'best')

    plt.subplot(222)
    l1,=plt.plot([(_+1)*1/2 for _ in range(len(model[-1].train_obj))], model[-1].train_score, linewidth=1, color='r')
    l2,=plt.plot([(_+1)*1/2 for _ in range(len(model[-1].train_obj))], model[-1].val_score, linewidth=1, color='g')
    plt.xlabel('Pass')
    plt.ylabel('Score')
    plt.legend(handles = [l1, l2], labels = ['train','val'], loc = 'best')

    plt.suptitle(cat_names[i],fontsize=20);
    plt.savefig('/Users/fuyang/Desktop/plots/cat_'+str(i)+'.png')
    plt.close()
    
    
del loader_val,pos_train,pos_val,img_train_feat,img_val_feat
gc.collect()

# In[30624770]:
#=========================test
img_test_feat = xr.open_dataarray(dataDir+'/test_feat.nc').values


with open(dataDir+'/loader_det_test.p', 'rb') as f:
    loader_test = cPickle.load(f)

loader_test.dataDir = dataDir 

y_test_all = pd.read_table(dataDir+'/test_det_lab.txt',sep=',',dtype=np.float32).values


y_hat_all_bagging = np.array([0.0]*(y_test_all.shape[0]*y_test_all.shape[1])).reshape(y_test_all.shape).astype(np.float32)

start = 0
for j in range(2000):
    X_test = loader_test.data([j],get_lab=False,feats=img_test_feat)
    
    if loader_test.bboxes[j] is None:
        continue
    
    L = X_test.shape[0]
    for i in range(18):       
        y_hat_bagging = np.array([0.0]*L).reshape([-1,1])
        for model in models_noretrain[i][0]:
            y_hat_bagging += model.predict(X_test)

        y_hat_all_bagging[start:start+L,i] = y_hat_bagging.astype(np.float32).reshape([-1,])/len(models_noretrain[i][0])
    
    start += L
    print('Finish predicting',j+1,'of 2000!')
    
del y_hat_bagging
gc.collect()

ap_bagging = []
for i in range(18):
    ap_bagging += [average_precision_score(y_test_all[:,i],y_hat_all_bagging[:,i])]




count = 0.0
nonnan = 0
for x in ap_bagging:
    if ~np.isnan(x):
        count += x
        nonnan += 1
count/nonnan