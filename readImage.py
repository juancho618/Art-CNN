import skimage.external.tifffile as tiff
import numpy as np
from skimage import io

data = np.load('./11up/results/file.npy')

print(data.shape)

tiff.imshow(data[1], cmap='gray')
io.show()

