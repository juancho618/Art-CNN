import numpy as np
from PIL import Image
import os
import diff as diff
import difference_png as diff_png
import higlight as higlight
import scipy.misc


def npy_to_png(filename, name):
    #data = np.load('{0}.npy'.format(filename))
    print('starting...')
    if not os.path.exists('../data/images/{0}'.format(name)):
        os.makedirs('../data/images/{0}'.format(name))
    for i in range(1836):#len(data)
        print(i)
        image_array = higlight.getImage(filename,i)
        #img = diff.getImage(filename,i) #To get image difference
        #img = diff_png.getImage(filename,i) #To get differences in images with png
        #img = Image.fromarray(np.uint8(data[i]))#np.reshape#['data']
        scipy.misc.imsave('../data/images/{0}/{0}_{1}.png'.format(name,i), image_array)
        #img.save('../data/images/{0}/{0}_{1}.png'.format(name,i), format="PNG")
    print('done!')
#npy_to_png('../robe_face/results/results_robe','robe_face_robe_infered')#00-17-VIS-HI-AT-face1
npy_to_png('../data/images/acrh_diff_highlight_x2/acrh_diff_highlight_x2','acrh_diff_highlight_x2_both')
#npy_to_png('../robe_face/results/results_robe', 'robe_numpy_diff')