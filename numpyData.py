import numpy as np
data = np.load('vgg16.npy', encoding='latin1').item()

print('hola', data['fc6'][0].shape)