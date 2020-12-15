#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 18:42:53 2020

@author: tjards
"""

#%% IMPORT packages
#import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
#import scipy
import imageio
from PIL import Image
#from scipy import ndimage
import dnnModule as dnn
import pickle

#%% PREPARE the data
# ------------------------

# define paths
path_train = 'datasets/train_catvnoncat.h5'
path_test = 'datasets/test_catvnoncat.h5'
path_my_image = "images/russian_cat.jpg"      # manually enter a picture of new cat

# training set
train_dataset = h5py.File(path_train, "r")
train_x_orig = np.array(train_dataset["train_set_x"][:])              # features
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # flatten (the "-1" makes reshape flatten the remaining dimensions)
train_x = train_x_flatten/255.                                        # normalize
train_y = np.array(train_dataset["train_set_y"][:])                   # labels
train_y = train_y.reshape((1, train_y.shape[0]))

# test set
test_dataset = h5py.File(path_test, "r")
test_x_orig = np.array(test_dataset["test_set_x"][:])                 # features
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T      # flatten
test_x = test_x_flatten/255.                                          # normalize
test_y = np.array(test_dataset["test_set_y"][:])                      # labels
test_y = test_y.reshape((1, test_y.shape[0]))

# pull out key parameters
classes = np.array(test_dataset["list_classes"][:])     # list of classes
num_px = train_x_orig.shape[1]                          # image size
m_train = train_x_orig.shape[0]                         # number of training samples
m_test = test_x_orig.shape[0]                           # number of test samples
my_label_y = [1]                                        # label of my image (1, 0)


#%% SET hyperparameters
# -----------------------
n_x = train_x.shape[0]              # number of input features
n_y = train_y.shape[0]              # number of outputs
layers_dims = [n_x, 20, 7, 5, n_y]  # model size [input, ..., hidden nodes, ... ,output]
learning_rate=0.0075                # learning rate (< 1.0)
num_iterations = 2500               # number of iterations
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
np.random.seed(1) 
fcost = 'mse' #'x-entropy'                 # x-entropy or mse

#%% Training a DNN on the training set
# ------------------------------------
parameters = dnn.train(train_x, train_y, layers_dims, learning_rate, num_iterations, print_cost=True,fcost=fcost)

# save the parameters to file
file_params = open("network_params.pkl","wb")
pickle.dump(parameters,file_params)
file_params.close()


#%% PREDICT on the test set
# -------------------------
print('Prediction on the training set: ')
pred_train = dnn.predict(train_x, train_y, parameters)
print('Prediction on the test set: ')
pred_test = dnn.predict(test_x, test_y, parameters)
dnn.print_mislabeled_images(classes, test_x, test_y, pred_test) # show me the mis-predicted stuff


#%% identify a single image
# ------------------------

# prepare the image
image_arr = np.array(imageio.imread(path_my_image))
image = Image.fromarray(image_arr)
image = image.resize(size=(num_px,num_px))
image = np.array(image) 
my_image=image.reshape((num_px*num_px*3,1))
my_image = my_image/255.

# predict
print('Prediction on manual input: ')
my_predicted_image = dnn.predict(my_image, my_label_y, parameters)

# plot
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.figure()
plt.imshow(image_arr)
print ("Model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")







