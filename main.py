#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 18:42:53 2020

@author: tjards
"""

#%% IMPORT packages

import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
#import PIL
import imageio
from PIL import Image
from scipy import ndimage
import dnnModule as dnn
import pickle


#%% Set parameters for the datasets
# --------------------------------
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
np.random.seed(1) 


#%% LOAD and PREP the data
# ------------------------
path_train = 'datasets/train_catvnoncat.h5'
path_test = 'datasets/test_catvnoncat.h5'

 

#training set
train_dataset = h5py.File(path_train, "r")
train_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
train_y = np.array(train_dataset["train_set_y"][:]) # your train set labels
train_y = train_y.reshape((1, train_y.shape[0]))

#test set
test_dataset = h5py.File(path_test, "r")
test_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
test_y = np.array(test_dataset["test_set_y"][:]) # your test set labels
test_y = test_y.reshape((1, test_y.shape[0]))

classes = np.array(test_dataset["list_classes"][:]) # the list of classes

#show a sample
#index = 15
#plt.imshow(train_x_orig[index])
#print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")

#extract key vars
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

#flatten the inputs into nice vectors
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# normalize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

#%% DEFINE the DNN hyperparameters 
n_x = train_x.shape[0]
n_y = train_y.shape[0]
layers_dims = [n_x, 20, 7, 5, n_y] #  4-layer model
learning_rate=0.0075
num_iterations = 2500

#%% RUN A DNN
parameters = dnn.train(train_x, train_y, layers_dims, learning_rate, num_iterations)

#%% PREDICT
print('Prediction on the training set')
pred_train = dnn.predict(train_x, train_y, parameters)
print('Prediction on the test set')
pred_test = dnn.predict(test_x, test_y, parameters)
#show me the mislabed stuff
dnn.print_mislabeled_images(classes, test_x, test_y, pred_test)

#%% TEST a single sample
my_image = "russian_cat.jpg" # place the file in "/images" and name here 
my_label_y = [1] # the true class of your image (1, 0)

fname = "images/" + my_image
#image = np.array(ndimage.imread(fname, flatten=False))

image_arr = np.array(imageio.imread(fname))
image = Image.fromarray(image_arr)
image = image.resize(size=(num_px,num_px))
image = np.array(image) 
my_image=image.reshape((num_px*num_px*3,1))
#my_imagine = my_image.reshape((num_px*num_px*3,1))
#my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
my_image = my_image/255.
my_predicted_image = dnn.predict(my_image, my_label_y, parameters)
#
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.figure()
plt.imshow(image_arr)
print ("y = " + str(np.squeeze(my_predicted_image)) + ", your model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")

#%% Save the parameters file
file_params = open("network_params.pkl","wb")
pickle.dump(parameters,file_params)
file_params.close()





