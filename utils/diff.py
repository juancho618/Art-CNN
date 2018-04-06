import numpy as np
from PIL import Image
from PIL import ImageChops


def getImage(file, index):

    data = np.load('{0}.npy'.format(file))
    current =np.squeeze( data[index])
    #print('shape', np.squeeze(data).shape)
    original = np.load('{0}.npy'.format('../data/00-17-IR-HI-AT-robe1')) #'../data/IRR_crop_registered_gray'
    # original = np.load('{0}.npy'.format('../data/VIS_crop_registered'))
    # print(data[index]['data'].dtype)
    # im_original = Image.fromarray(np.uint8(original[index]['data']))
    im2 = Image.fromarray(np.uint8(original[index]['data']))
    
    im = Image.fromarray(np.uint8(current))
   
    np3 = np.uint8(original[index]['data']) - np.uint8(current)
    im3 = Image.fromarray(np3)
    diff = ImageChops.difference(im2, im)
    # im.show()    
    # im2.show()
    # im3.show()
    # diff.show()
    
    return im3

    # im2.show()
    # im_original.show()
# getImage('../data/VIS_crop_registered',0)
#getImage('../noPooling/results/results', 7)