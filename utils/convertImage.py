from PIL import Image
import numpy as np

#TOOD: change numpy for pandas    
def convertImage(name):
    images_size = 64
    source_image = Image.open('{0}.tif'.format(name)).convert('L')
    im_array = np.array(source_image)
    images_row = int(im_array.shape[0]/images_size)
    images_col = int(im_array.shape[1]/images_size)

    data_array = []

    for x in range(images_row):
        for y in range(images_col):
            data_array.append({'name': 'original_{0}_{1}'.format(x,y),
                                'data': np.array(im_array[(x*images_size):((x*images_size) + images_size),(y*images_size):((y*images_size) + images_size)])
                            })

    data_array = np.array(data_array)
    np.save(name+'_gray', data_array)
#convertImage('../IRR_rescaled')
