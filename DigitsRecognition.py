# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 12:22:34 2017

@author: Veeramani Natarajan
@Competiion: Analytics Vidhya - Digit Recognition

"""
# Loading required libraries

# To save trained objects
from sklearn.externals import joblib
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as np
from scipy import misc
import glob
import os
import pandas as pd
import random
from sklearn.metrics import accuracy_score

# Reading training labels
trpath="C:\\Users\\user\\Dropbox\\temp\\Hackathons\\AV\\DigitsRecognition\\data\\train\\train.csv"
df=pd.read_csv(trpath)
labels=df['label']
fname=df['filename']

# Reading all train images & data
features=[]
imgpath="C:\\Users\\user\\Dropbox\\temp\\Hackathons\\AV\\DigitsRecognition\\data\\train\\images\\train\\"
for fn in fname:
    print(imgpath+fn)
    image = misc.imread(imgpath+fn,flatten=True)
    features.append(image)
    print (image.shape)
    print (image.dtype)


list_hog_fd = []
for feature in features:
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(8,8), cells_per_block=(2, 2), visualise=False)
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')
traindf=pd.DataFrame(hog_features)


# Split train and test 80-20
test_list=random.sample(range(0,48999),9800)
wholelist=list(range(49000))
train_list=set(wholelist)-set(test_list)
train_list=list(train_list)
tr_labels=labels.iloc[train_list]
te_labels=labels.iloc[test_list]
tr_images=traindf.iloc[train_list]
te_images=traindf.iloc[test_list]


# Creating SVM Linear classifier objects
clf = LinearSVC()
clf.fit(tr_images, tr_labels)

# internal testing - 20% of the data
test20_pred=clf.predict(te_images)
test20_pred=pd.Series(test20_pred,dtype="category")
test20_act=pd.Series(te_labels,dtype="category")
accuracy_score(test20_act,test20_pred,normalize=True)



# Reading test images
#############################################################################################################
t_timage=[]
test_fname=pd.read_csv("C:\\Users\\user\\Dropbox\\temp\\Hackathons\\AV\\DigitsRecognition\\data\\Test.csv","r")
test_fname=test_fname['filename']

testpath="C:\\Users\\user\\Dropbox\\temp\\Hackathons\\AV\\DigitsRecognition\\data\\Train\\Images\\test\\"
for fnam in test_fname:
#    print(testpath+fnam)
    image = misc.imread(testpath+fnam,flatten=True)
    t_timage.append(image)
#    print (image.shape)
#    print (image.dtype)
    

test_fd= []
for feat in t_timage:
    fd = hog(feat.reshape((28, 28)), orientations=9, pixels_per_cell=(8,8), cells_per_block=(2, 2), visualise=False)
    test_fd.append(fd)
tf = np.array(test_fd, 'float64')



mypred=clf.predict(tf)

mypred=(pd.Series(mypred))


myresults=pd.concat([test_fname,mypred],axis=1)
myresults.columns=['filename', 'label']
myresults.to_csv("C:\\Users\\user\\Dropbox\\temp\\Hackathons\\AV\\DigitsRecognition\\output\\myresults4.csv",index=False)


