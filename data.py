import numpy as np
import pandas as pd
from skimage import io
import skimage.external.tifffile as tiff
import csv
import os
from scipy.misc import imread # another option to read the images
import math

NORMAL_image = 'VIS_registered.tif'
IRR_image = 'IRR_rescaled.tif'

IMAGE_SIZE = 64


def load_train_data(data_path="", validation_size=200):
    original_img = load_from_csv('./original_train/data.csv')
    irr_img = load_from_csv('./irr_train/data.csv')

    temp = [] # save temporally image information
    # converting the data in a numpy array
    for img_name in original_img:
        image_path = os.path.join('./original_train/' + img_name)
        img =  imread(image_path, flatten = False) # flattern is to a gray-scale layers
        img = img.astype('float32')
        temp.append(img)

    x_values= np.stack(temp) # (1836, 64, 64)
    #print(x_values.shape)
    # tiff.imshow(temp[0])
    # io.show()

    temp = []  # save temporally image for the irr information
    # converting the data in a numpy array
    for img_name in irr_img:
        image_path = os.path.join('./irr_train/' + img_name)
        img = imread(image_path, flatten=True)
        img = img.astype('float32')
        for r in range(64):
            for c in range(64):
                img[r][c] = math.ceil(img[r][c]) # to have a discrete amount of values
        temp.append(img)

    y_values = np.stack(temp)  # (1836, 64, 64)
   

    x_train = x_values[:1500] #336 values
    x_val = x_values[1500:1800]
    y_train = y_values[:1500]
    y_val = y_values[1500:1800]

    # reshaping data in a not flat format
    x_train = x_train.reshape(len(x_train), IMAGE_SIZE, IMAGE_SIZE, 3)
    x_val = x_val.reshape(len(x_val), IMAGE_SIZE, IMAGE_SIZE, 3)
    y_train = y_train.reshape(len(y_train), IMAGE_SIZE, IMAGE_SIZE, 1) # play with the 1 to one relation 
    y_val = y_val.reshape(len(y_val), IMAGE_SIZE, IMAGE_SIZE, 1) # play with the 1 to one relation


    return x_train, x_val, y_train, y_val # returning the values with the name of the data



def load_from_csv(path):
    with open(path , 'rt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\n')
        data = []
        for row in spamreader:
            data.append(row[0])
        return data


def load_test_data():
    original_img = load_from_csv('./original_train/data.csv')
    irr_img = load_from_csv('./irr_train/data.csv')

    temp = [] # save temporally image information
    # converting the data in a numpy array
    for img_name in original_img:
        image_path = os.path.join('./original_train/' + img_name)
        img =  imread(image_path, flatten = False) # flattern is to a gray-scale layers
        img = img.astype('float32')
        temp.append(img)

    x_values= np.stack(temp) 

    temp = []  # save temporally image for the irr information
    # converting the data in a numpy array
    for img_name in irr_img:
        image_path = os.path.join('./irr_train/' + img_name)
        img = imread(image_path, flatten=True)
        img = img.astype('float32')
        for r in range(64):
            for c in range(64):
                img[r][c] = math.ceil(img[r][c]) # to have a discrete amount of values
        temp.append(img)

    y_values = np.stack(temp)  # (1836, 64, 64)
   

    x_test = x_values[1800:1836] #336 values
    y_test = y_values[1800:1836]

    x_test = x_test.reshape(len(x_test), IMAGE_SIZE, IMAGE_SIZE, 3)
    y_test = y_test.reshape(len(y_test), IMAGE_SIZE, IMAGE_SIZE, 1)
    
    
    return x_test, y_test
