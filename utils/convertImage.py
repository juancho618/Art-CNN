from PIL import Image
import numpy as np
from scipy import misc
import cv2
import imgHist as histogram

Image.MAX_IMAGE_PIXELS = None
#TOOD: change numpy for pandas    
def convertImage(name):
    images_size = 128#64
    numpy_data = misc.imread('{0}.tif'.format(name))
    from_uint16_to_uint18 = ((numpy_data / np.amax(numpy_data)) * 255).astype(np.uint8)
    #source_image = Image.open('{0}.tif'.format(name))
    #outputImg8U = cv2.convertScaleAbs(face, alpha=(255.0/65535.0)) #Convert to uint8
    
    
    im_array = np.array(from_uint16_to_uint18)
    images_row = int(im_array.shape[0]/images_size)
    images_col = int(im_array.shape[1]/images_size)

    data_array = []

    for x in range(images_row):
        for y in range(images_col):
            data_array.append({'name': 'IR_{0}_{1}'.format(x,y),
                                'data': np.array(im_array[(x*images_size):((x*images_size) + images_size),(y*images_size):((y*images_size) + images_size)])
                            })
    data_array = np.array(data_array)
    np.save('../data/00-17-VIS-HI-AT-robe1_128', data_array)#name
    histogram.save_histogram('00-17-VIS-HI-AT-robe1', 'Robe Color Histogram', 'color') #to save color histogram
convertImage('../data/00-17-VIS-HI-AT-robe1')
