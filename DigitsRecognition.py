# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 12:22:34 2017

@author: Veeramani Natarajan
@Competiion: Analytics Vidhya - Digit Recognition

"""
# Loading required libraries

# To save trained objects
from sklearn.externals import joblib
# from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as np
from scipy import misc
import glob

# Reading all train images
features=[]
for image_path in glob.glob("D:/temp/img/*.png"):
    image = misc.imread(image_path)
    features.append(image)
    print (image.shape)
    print (image.dtype)
    
    