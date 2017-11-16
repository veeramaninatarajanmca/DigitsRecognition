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
import os
import pandas as pd

os.chdir("C:\\Users\\user\\Dropbox\\temp\\Hackathons\\AV\\DigitsRecognition")


# Reading all train images & data
features=[]
imgpath="C:\\Users\\user\\Dropbox\\temp\\Hackathons\\AV\\DigitsRecognition\\data\\train\\images\\train\\*.png"
for image_path in glob.glob(imgpath):
    image = misc.imread(image_path,flatten=True)
    features.append(image)
    print (image.shape)
    print (image.dtype)


trpath="C:\\Users\\user\\Dropbox\\temp\\Hackathons\\AV\\DigitsRecognition\\data\\train\\train.csv"
df=pd.read_csv(trpath)
labels=df['label']


list_hog_fd = []
for feature in features:
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')


# Creating SVM Linear classifier objects
clf = LinearSVC()
clf.fit(hog_features, labels)


# Reading test images
#############################################################
t1=[]
for image_path in glob.glob("D:\\temp\\img\\*.png"):
    image = misc.imread(image_path,flatten=True)
    t1.append(image)
    print (image.shape)
    print (image.dtype)
    

test_fd= []
for feat in t1:
    fd = hog(feat.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    test_fd.append(fd)
tf = np.array(test_fd, 'float64')



mypred=clf.predict(tf)
mypred