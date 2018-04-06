import numpy as np
from PIL import Image
import os
import diff as diff
import difference_png as diff_png
import scipy.misc

def invert_black_to_white(filename, name):
    print('starting...')
    if not os.path.exists('../data/images/{0}'.format(name)):
        os.makedirs('../data/images/{0}'.format(name))
    for i in range(15376):#len(data)
        print(i)
        data = np.asarray(Image.open('{0}_{1}.png'.format(filename, i)))
        img = np.invert(data)
        scipy.misc.imsave('../data/images/{0}/{0}_{1}.png'.format(name,i), img)
    print('done!')
#npy_to_png('../robe_face/results/results_robe','robe_face_robe_infered')#00-17-VIS-HI-AT-face1
invert_black_to_white('../data/images/robe_face_infered_face_diff/robe_face_infered_face_diff','face_numpy_diff_inverted')
#npy_to_png('../robe_face/results/results_robe', 'robe_numpy_diff_inverted')