import numpy as np
from PIL import Image

def getImage(file, index):

    data = np.load('{0}.npy'.format(file))
    print('shape', data[index]['data'].shape)
    # gt = np.load('{0}.npy'.format('../data/IRR_crop_registered'))
    #original = np.load('{0}.npy'.format('../data/VIS_crop_registered'))
    print(data[index]['data'].dtype)
    #im_original = Image.fromarray(np.uint8(original[index]['data']))
    # im2 = Image.fromarray(np.uint8(gt[index]['data']))
    # print(data.shape)
    # im = Image.fromarray(np.uint8(data[index]['data']))
    im = Image.fromarray(data[index]['data'])
   
    im.show()  
    #im_original.show()  
    
    #return im

    # im2.show()
    # im_original.show()
# getImage('../data/VIS_crop_registered',0)
getImage('../data/00-17-VIS-HI-AT-robe1',8)