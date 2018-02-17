import skimage.external.tifffile as tiff
import numpy as np
from skimage import io

data = np.load('./1up/results/file.npy')

print(data.shape)

tiff.imshow(data[24], cmap='gray')
io.show()

