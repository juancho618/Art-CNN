import skimage.external.tifffile as tiff
import numpy as np
from skimage import io
import matplotlib
import matplotlib.pyplot as plt

import scipy.misc


data = np.load('./11up/results/file.npy')

print(data.shape)
uInt = data[1].astype(int)
# scipy.misc.imsave('outfile.jpg', uInt)
# plt.imsave('test.png', data, cmap = plt.cm.gray)
print(uInt, data[1])
tiff.imshow(data[1], cmap='gray')
io.show()
io.imsave('test.tif',data[1])

