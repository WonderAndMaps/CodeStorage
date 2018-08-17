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



def plotPreviews(data, dates, cols = 4, figsize=(15,15), denom=255.):
    """
    Utility to plot small "true color" previews.
    """
    width = data[-1].shape[1]
    height = data[-1].shape[0]
    
    rows = data.shape[0] // cols + (1 if data.shape[0] % cols else 0)
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)
    for index, ax in enumerate(axs.flatten()):
        if index < data.shape[0]:
            caption = str(index)+': '+str(dates[index])
            #ax.set_axis_off()
            ax.imshow(data[index] / denom, vmin=0.0, vmax=1.0)
            ax.text(0, -2, caption, fontsize=12, color='g')
        else:
            ax.set_axis_off()



def loadGrayImage(idxs,img_list,width,height):
    imgs = np.zeros([len(idxs),width,height])
    for i in range(len(idxs)):
        # Read as gray image
        imgs[i,:,:] = cv.resize(cv.imread(img_list[idxs[i]],0),(width,height))
    return imgs




def hist(img):
	#hist = cv.calcHist([img],[0],None,[256],[0,256])
	plt.subplot(121)
	plt.imshow(img,'gray')
	plt.xticks([])
	plt.yticks([])
	plt.title("Original")
	plt.subplot(122)
	plt.hist(img.ravel(),256,[0,256])
	plt.show()




dirPath = 'F:/Dropbox (Bazean)/20180813/imgs'
img_list = recursiveSearchFiles(dirPath, '*.tif')

imgs = loadGrayImage([i for i in range(30,40)],img_list,64,64)
#plotPreviews(imgs,[i for i in range(10,20)])

for img in imgs:
    hist(img)
    print('Over 200: ',(img>200).mean())
    print('Less 50: ',(img<50).mean())
    
    
# Too bright: 80% >200; Too dark: 40% <50...looks reasonable


# Obtain week and pad number for each image so that its label can be found




