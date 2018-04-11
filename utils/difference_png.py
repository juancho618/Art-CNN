import numpy as np
from PIL import Image
from PIL import ImageChops


def getImage(file, index):

    #data = np.load('{0}.npy'.format(file))
    data = np.asarray(Image.open('{0}_{1}.png'.format(file, index)))
    # original = np.load('{0}.npy'.format('../data/IRR_crop_registered_gray')) #'../data/IRR_crop_registered_gray' # for npy values
    original = np.asarray(Image.open('{0}_{1}.png'.format('../data/images/denoised_ir_arc_x2/denoised_ir_arc_x2', index)))
    # im2 = Image.fromarray(np.uint8(original[index]['data']))
    print('shape original', original.shape)
    im2 = original
    
    im = Image.fromarray(np.uint8(data))
    np3 = (np.uint8(original) - np.uint8(data))#['data']
    # np3 = np.uint8(original[index]) - np.uint8(data)#['data']
    im3 = Image.fromarray(np3)
    
    return im3

    # im2.show()
    # im_original.show()
# getImage('../data/VIS_crop_registered',0)
#getImage('../noPooling/results/results', 7)