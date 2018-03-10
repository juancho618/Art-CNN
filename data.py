import numpy as np
import pandas as pd
import math


vis_image = './data/VIS_crop_registered'
irr_image = './data/IRR_crop_registered'


IMAGE_SIZE = 64

#load images
original_img = np.load('{0}.npy'.format(vis_image))
irr_img = np.load('{0}.npy'.format(irr_image))

def load_train_data(validation_size=200):    

    x_train = np.array(list(map(lambda x: x['data'], original_img[:1836]))) #336 values
    x_val = np.array(list(map(lambda x: x['data'], original_img[:1836])))
    y_train = np.array(list(map(lambda x: x['data'], irr_img[:1836])))
    y_val = np.array(list(map(lambda x: x['data'], irr_img[:1836]))) 

    return x_train, x_val, y_train, y_val 

def load_test_data(): 
    # Evaluation images and Labels
    x_test = np.array(list(map(lambda x: x['data'], original_img[:10])))
    y_test = np.array(list(map(lambda x: x['data'], irr_img[:10])))      
    
    return x_test, y_test
