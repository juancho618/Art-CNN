import tensorflow as tf
import numpy as np
from skimage import io
import skimage.external.tifffile as tiff

class Model(object):
    def __init__(self, batch_size=20, learning_rate=1e-4):
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self.layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
        )
        self.weights = [16, 32, 64, 128]

    
    def inference(self, images, keep_prob):
        net = {}
        for i, name in enumerate(layers):
            kind = name[:4]
            if kind == 'conv':
                kernel = self._create_weights([2, 2, 3, self.weights[i]])
                bias = self._create_bias(self.weights[i])
                conv = self._create_conv2d(images, kernel) 
                current = tf.add.bias(conv, bias)                
            elif kind == 'relu':
                current = tf.nn.relu(current, name = name)
                # add debug option
            elif kind == 'pool':
                current = self._create_max_pool_2x2(current)
            net[name] = current
        return net

    

    def _create_conv2d(self, x, W):
        return tf.nn.conv2d(input=x,
                            filter=W,
                            strides = [1, 1, 1, 1],
                            padding = 'SAME')
    
    def _create_max_pool_2x2(self, input):
        return tf.nn.max_pool(value = input,
                              ksize = [1, 2, 2, 1],
                              strides = [1, 2, 2, 1],
                              padding = 'SAME')
        
    def _create_weights(self, shape):
        return tf.Variable(tf.truncated_normal(shape = shape, stddev = 0.1, dtype = tf.float32))

    def _create_bias(self, shape):
        return tf.Variable(tf.constant(1., shape = shape, dtype = tf.float32))
            
