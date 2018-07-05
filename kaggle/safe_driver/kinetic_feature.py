# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 16:30:23 2017

@author: dell
"""
#Four types of features(individual, car, calc)
#Calculate kinetic energy (similar to Shannon entropy) of each type for every sample 
#https://alexandrudaia.quora.com/Kinetic-things-by-Daia-Alexandru

import pandas as pd
import numpy as np
import os

os.chdir('C:\\Users\\dell\\Desktop\\Safe driver')
train = pd.read_csv("train.csv")
y = train['target']
train = train.drop(['id', 'target'], axis=1)
test = pd.read_csv("test.csv")
test = test.drop(['id'], axis=1)


def kinetic(row):
    probs=np.unique(row,return_counts=True)[1]/len(row)
    kinetic=np.sum(probs**2)
    return kinetic    

##################################kinetic feature for train####################
first_kin_names=[col for  col in train.columns  if '_ind_' in col]
subset_ind=train[first_kin_names]
kinetic_1=[]
for row in range(subset_ind.shape[0]):
    row=subset_ind.iloc[row]
    k=kinetic(row)
    kinetic_1.append(k)
    
second_kin_names= [col for  col in train.columns  if '_car_' in col and col.endswith('cat')]
subset_ind=train[second_kin_names]
kinetic_2=[]
for row in range(subset_ind.shape[0]):
    row=subset_ind.iloc[row]
    k=kinetic(row)
    kinetic_2.append(k)
    
third_kin_names= [col for  col in train.columns  if '_calc_' in col and  not col.endswith('bin')]
subset_ind=train[second_kin_names]
kinetic_3=[]
for row in range(subset_ind.shape[0]):
    row=subset_ind.iloc[row]
    k=kinetic(row)
    kinetic_3.append(k)
    
fd_kin_names= [col for  col in train.columns  if '_calc_' in col and  col.endswith('bin')]
subset_ind=train[fd_kin_names]
kinetic_4=[]
for row in range(subset_ind.shape[0]):
    row=subset_ind.iloc[row]
    k=kinetic(row)
    kinetic_4.append(k)

kinetic_feature = pd.DataFrame(np.array([kinetic_1, kinetic_2, kinetic_3, kinetic_4])).transpose()
kinetic_feature.to_csv('kinetic_feature_train.csv', index=False, float_format='%.5f')

##################################kinetic feature for test####################
first_kin_names=[col for  col in test.columns  if '_ind_' in col]
subset_ind=test[first_kin_names]

kinetic_1=[]
for row in range(subset_ind.shape[0]):
    row=subset_ind.iloc[row]
    k=kinetic(row)
    kinetic_1.append(k)
    
second_kin_names= [col for  col in test.columns  if '_car_' in col and col.endswith('cat')]
subset_ind=test[second_kin_names]
kinetic_2=[]
for row in range(subset_ind.shape[0]):
    row=subset_ind.iloc[row]
    k=kinetic(row)
    kinetic_2.append(k)
    
third_kin_names= [col for  col in test.columns  if '_calc_' in col and  not col.endswith('bin')]
subset_ind=test[second_kin_names]
kinetic_3=[]
for row in range(subset_ind.shape[0]):
    row=subset_ind.iloc[row]
    k=kinetic(row)
    kinetic_3.append(k)
    
fd_kin_names= [col for  col in test.columns  if '_calc_' in col and  col.endswith('bin')]
subset_ind=test[fd_kin_names]
kinetic_4=[]
for row in range(subset_ind.shape[0]):
    row=subset_ind.iloc[row]
    k=kinetic(row)
    kinetic_4.append(k)

kinetic_feature = pd.DataFrame(np.array([kinetic_1, kinetic_2, kinetic_3, kinetic_4])).transpose()
kinetic_feature.to_csv('kinetic_feature_test.csv', index=False, float_format='%.5f')

