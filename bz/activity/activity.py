# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 11:27:04 2018

@author: yang
"""

import fnmatch
import glob
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import datetime


def recursiveSearchFiles(dirPath, partFileInfo): 
    fileList = []
    pathList = glob.glob(os.path.join('/', dirPath, '*'))
    for mPath in pathList:
        if fnmatch.fnmatch(mPath, partFileInfo):
            fileList.append(mPath)
        elif os.path.isdir(mPath):
            fileList += recursiveSearchFiles(mPath, partFileInfo)
        else:
            pass
    return fileList



def loadGrayImage(idxs,img_list,width,height):
    imgs = np.zeros([len(idxs),width,height])
    for i in range(len(idxs)):
        # Read as gray image
        imgs[i,:,:] = cv.resize(cv.imread(img_list[idxs[i]],0),(width,height))
    return imgs


def hist(img):
	plt.subplot(121)
	plt.imshow(img,'gray')
	plt.xticks([])
	plt.yticks([])
	plt.title("Original")
	plt.subplot(122)
	plt.hist(img.ravel(),256,[0,256])
	plt.show()


def getPadNumber(img_path):
    return os.path.basename(os.path.abspath(os.path.join(img_path,"..")))

def getWeek(img_path):
    timestr = os.path.basename(os.path.abspath(os.path.join(img_path,"../../.."))) # week of
    return datetime.datetime.strptime(timestr,'%Y%m%d').strftime('%Y-%m-%d')

def getGoodImg(img_list):
    pad_num = [[] for i in range(len(img_list))]
    week_of = [[] for i in range(len(img_list))]
    good_img_list = [[] for i in range(len(img_list))]
    for i in range(len(img_list)):
        try:
            img = loadGrayImage([i],img_list,64,64)
        except:
            print('Something wrong here:',i)
            continue
        
        if (img>200).mean()>0.8 or (img<50).mean()>0.4:
            #os.remove(img_list[i])
            pass
        else:
            good_img_list[i] = img_list[i]
            pad_num[i] = getPadNumber(img_list[i])
            week_of[i] = getWeek(img_list[i])
            
        if i%100==0:
            print('Done iter',i)
            #gc.collect()

    return good_img_list,pad_num,week_of


dirPath = 'F:/Dropbox (Bazean)/sky_obs'
img_list = recursiveSearchFiles(dirPath, '*.tif')



#imgs = loadGrayImage([i for i in range(30,40)],img_list,64,64)
#plotPreviews(imgs,[i for i in range(10,20)])

#for img in imgs:
#    hist(img)
#    print('Over 200: ',(img>200).mean())
#    print('Less 50: ',(img<50).mean())
    
    
# Too bright: 80% >200; Too dark: 40% <50...looks reasonable
# Obtain week and pad number for each image so that its label can be found
img_list_good,pad_num,week_of = getGoodImg(img_list)
