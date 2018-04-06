import numpy as np
from scipy import misc
from PIL import Image

#Avoid size limitations when reading the image!
Image.MAX_IMAGE_PIXELS = None

#Ti extract  apart of a bigger image
def extractPath(name):
    numpy_data = misc.imread('{0}.tif'.format(name))#Image.open('{0}.tif'.format(name))
    
    
    im = Image.fromarray(numpy_data)   
    im.show()  


extractPath('E:/data_thesis/00-17-VIS-HI-AT-crop')