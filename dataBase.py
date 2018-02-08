import h5py
import numpy as numpy
from skimage import io
import skimage.external.tifffile as tiff

# f = h5py.File("testFIle.hdf5", 'w')
# dset = f.create_dataset('testdset', (10.), dtype='f' )

# h5f = h5py.File('data.h5', 'w')
# h5f.create_dataset('data' , data=[1,213,3])
# h5f.close()

#read
h5f = h5py.File('dataVIS.h5', 'r')
# myNumpyArray= h5f['data'][:]
print(h5f[0].shape)
# tiff.imshow(h5f[0])
# io.show()

h5f.close()