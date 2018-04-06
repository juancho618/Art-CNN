import numpy as np
import pandas as pd
import math
import random

vis_image = './data/VIS_crop_registered'
irr_image = './data/IRR_crop_registered_gray'

#Face Image
vis_face = './data/00-17-VIS-HI-AT-face1'
ir_face = './data/00-17-IR-HI-AT-face1'

#Robe Image
vis_robe = './data/00-17-VIS-HI-AT-robe1'
ir_robe = './data/00-17-IR-HI-AT-robe1'

IMAGE_SIZE = 64

training_proportion = 0.9
#load images
#Old Arch
original_img =  np.load('{0}.npy'.format(vis_image))
irr_img= np.load('{0}.npy'.format(irr_image))

original_face = np.load('{0}.npy'.format(vis_face))
irr_face = np.load('{0}.npy'.format(ir_face))

original_robe = np.load('{0}.npy'.format(vis_robe))
irr_robe = np.load('{0}.npy'.format(ir_robe))
def load_train_data(validation_size=200):    
    
    # Training and validation images for face 
    x_train_face = np.array(list(map(lambda x: x['data'], original_face[:int((len(original_face)*training_proportion))]))) 
    print(x_train_face.shape)
    x_val_face = np.array(list(map(lambda x: x['data'], original_face[(int(len(original_face)*training_proportion)):])))
    # Training and validation images for robe 
    x_train_robe = np.array(list(map(lambda x: x['data'], original_robe[:(int(len(original_robe)*training_proportion))]))) 
    print(x_train_robe.shape)
    x_val_robe = np.array(list(map(lambda x: x['data'], original_robe[(int(len(original_robe)*training_proportion)):])))

    x_total_train =  np.concatenate((x_train_face, x_train_robe),axis=0)
    x_total_val =  np.concatenate((x_val_face, x_val_robe),axis=0)
    print(x_total_train.shape)

     # Training and validation images for face 
    y_train_face = np.array(list(map(lambda x: x['data'], irr_face[:(int(len(irr_face)*training_proportion))]))) 
    y_val_face = np.array(list(map(lambda x: x['data'], irr_face[(int(len(irr_face)*training_proportion)):])))
    # Training and validation images for robe 
    y_train_robe = np.array(list(map(lambda x: x['data'], irr_robe[:(int(len(irr_robe)*training_proportion))]))) 
    y_val_robe = np.array(list(map(lambda x: x['data'], irr_robe[(int(len(irr_robe)*training_proportion)):])))

    y_total_train =  np.concatenate((y_train_face, y_train_robe),axis=0)
    y_total_val =  np.concatenate((y_val_face, y_val_robe),axis=0)

    #Shuffle
    x_total_train, y_total_train = shuffle(x_total_train,y_total_train)
    print('data_x', x_total_train.shape)
    print('data_y', y_total_train.shape)

    avg_color_array = getAverage(x_total_train) 
    return x_total_train, x_total_val, y_total_train, y_total_val, avg_color_array 

def shuffle(a,b):
    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)
    return np.asarray(a), np.asarray(b)

def load_test_data(): 
    # Evaluation images and Labels
    x_test = np.array(list(map(lambda x: x['data'],original_robe[:]))) #original_img
    y_test = np.array(list(map(lambda x: x['data'], irr_robe[:]))) #irr_img      
    
    return x_test, y_test
# Get the color channel average for normalization
def getAverage (data):
    avg_array = []
    for image in data:
          local_avg_per_row = np.average(image,axis=0)
          local_avg_per_ch = np.average(local_avg_per_row,axis=0)
          avg_array.append(local_avg_per_ch)
    avg_array = np.average(np.array(avg_array), axis=0)
    print('AVG Color Channels', avg_array)

    return avg_array
