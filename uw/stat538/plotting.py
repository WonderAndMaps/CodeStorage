# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 15:58:47 2018

@author: dell
"""

import pandas as pd
import matplotlib.pyplot as plt

mlpResultL0 = pd.read_csv('C:\\Users\\dell\\Desktop\\paper review\\mlpResultL0.txt',index_col=0).values.tolist()
mlpResultL1 = pd.read_csv('C:\\Users\\dell\\Desktop\\paper review\\mlpResultL1.txt',index_col=0).values.tolist()
mlpResultL2 = pd.read_csv('C:\\Users\\dell\\Desktop\\paper review\\mlpResultL2.txt',index_col=0).values.tolist()
cnnResultL1 = pd.read_csv('C:\\Users\\dell\\Desktop\\paper review\\cnnResultL1.txt',index_col=0).values.tolist()
cnnResultL2 = pd.read_csv('C:\\Users\\dell\\Desktop\\paper review\\cnnResultL2.txt',index_col=0).values.tolist()

l1,=plt.plot([i for i in range(11)], mlpResultL0, linewidth=1, color=[.8,.8,.8])
l2,=plt.plot([i for i in range(11)], mlpResultL1, linewidth=1, color=[.5,.5,.5])
l3,=plt.plot([i for i in range(11)], mlpResultL2, linewidth=1, color=[.2,.2,.2])
plt.xlabel('Number of principal components')
plt.ylabel('MLP Misclassification rate')
plt.legend(handles = [l1, l2, l3], labels = ['layer 0','layer 1','layer 2'], loc = 'best')

l1,=plt.plot([i for i in range(4)], mlpResultL0[10]+mlpResultL1[10]+mlpResultL2[10]+[0.025], linewidth=1, color='b', marker='o')
l2,=plt.plot([i for i in range(4)], mlpResultL0[10]+cnnResultL1[0]+cnnResultL2[0]+[0.025], linewidth=1, color='r', marker='^')
plt.xlabel('layer')
plt.ylabel('Misclassification rate (d=10)')
plt.xticks([0, 1, 2, 3], ['0', '1', '2', 'out'])
plt.legend(handles = [l1, l2], labels = ['MLP','CNN'], loc = 'best')
