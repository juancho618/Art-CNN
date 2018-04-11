import numpy as np
from scipy import misc
from PIL import Image
import imageio
import os

#Avoid size limitations when reading the image!
Image.MAX_IMAGE_PIXELS = None

csv_path = '../../../data_thesis/00-17-VIS-HI-AT'
csv_path = os.path.join(os.path.dirname(__file__), csv_path)
#Ti extract  apart of a bigger image
def extractPath(name):
    numpy_data = imageio.imread('{0}.tif'.format(name))#Image.open('{0}.tif'.format(name))
    extraction = np.array(numpy_data[1000:5000,2000:7000,:])
    print(numpy_data.shape)
    print(extraction.shape)
    # im = Image.fromarray(extraction)   
    # im.show()  
    misc.imsave('ex.jpg', extraction)


extractPath(csv_path)