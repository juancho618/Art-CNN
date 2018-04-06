import numpy as np
from PIL import Image
import os
import diff as diff
import difference_png as diff_png

def npy_to_png(filename, name):
    data = np.load('{0}.npy'.format(filename))
    print('starting...')
    if not os.path.exists('../data/images/{0}'.format(name)):
        os.makedirs('../data/images/{0}'.format(name))
    for i in range(len(data)):#len(data)
        print(i)
        img = diff.getImage(filename,i) #To get image difference
        #img = diff_png.getImage(filename,i) #To get differences in images with png
        #img = Image.fromarray(np.uint8(data[i]))#np.reshape#['data']
        img.save('../data/images/{0}/{0}_{1}.png'.format(name,i), format="PNG")
    print('done!')
#npy_to_png('../robe_face/results/results_robe','robe_face_robe_infered')#00-17-VIS-HI-AT-face1
#npy_to_png('../data/images/denoised_infered_arc/denoised_infered_arc','acrh_diff_denoised_both')
npy_to_png('../robe_face/results/results_robe', 'robe_numpy_diff')