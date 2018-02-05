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
        self.weights = [64,0, 64, 0,0,128, 0, 128,0, 0,512, 0,512 ,0, 512,0,512, 0, 0, 512,0,512,0,512,0,512,0]

    def vgg_net(self, images):
        net = {}
        for i, name in enumerate(self.layers):
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


    def inference(self, images, keep_prob):
      image_net = self.vgg_net(images)
      conv_final_layer = image_net["conv5_3"]
      pool5 = self._create_max_pool_2x2(conv_final_layer)

      w6 = self._create_weights([2,2,512,4096])
      b6 = self._create_bias(4096)
      conv6 = self._create_conv2d(pool5,w6)
      result = tf.add.bias(conv6, b6)
      relu6 = tf.nn.relu(result, name="relu6")
      relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)


      w7 = self._create_weights([1,1,4096,4096])
      b7 = self._create_bias(4096)
      conv7 = self._create_conv2d(relu_dropout6,w7)
      result = tf.add.bias(conv7, b7)
      relu7 = tf.nn.relu(result, name="relu7")
      relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

      # Now it comes the deconvolution to the original size
      deconv_shape1 = image_net["pool4"].get_shape()
      w_t1 = self._create_weights([4,4, deconv_shape1[3].value, 1])
      b_t1 = self._create_bias([deconv_shape1[3].value])
      conv_t1 = tf.nn.conv2d_transpose(relu_dropout7,w_t1,b_t1, output_shape=tf.shape(image_net["pool4"]))
      fuse_1 = tf.add(conv_t1, image_net["pool4"], name = "fuse_1")

      deconv_shape2 = image_net["pool3"].get_shape()
      w_t2 = self._create_weights([4,4, deconv_shape2[3].value, 1])
      b_t2 = self._create_bias([deconv_shape2[3].value])
      conv_t2 = tf.nn.conv2d_transpose(fuse_1,w_t2,b_t2, output_shape=tf.shape(image_net["pool3"]))
      fuse_2 = tf.add(conv_t2, image_net["pool3"], name = "fuse_2") 

      shape = [64, 64, 1]
      deconv_shape3 = tf.stack(shape[0], shape[1], shape[2])
      w_t3 = self._create_weights([64,64, deconv_shape2[3].value])
      b_t3 = self._create_bias([1])
      conv_t3 = tf.nn.conv2d_transpose(fuse_2,w_t3,b_t3, output_shape=tf.shape(deconv_shape3, stride=8))
      
      return conv_t3

    def _create_conv2d(self, x, W):
        return tf.nn.conv2d(input=x,
                            filter=W,
                            strides = [1, 1, 1, 1],
                            padding = 'SAME')
    
    def _create_max_pool_2x2(self, input, name):
        return tf.nn.max_pool(value = input,
                              ksize = [1, 2, 2, 1],
                              strides = [1, 2, 2, 1],
                              padding = 'SAME')
        
    def _create_weights(self, shape):
        return tf.Variable(tf.truncated_normal(shape = shape, stddev = 0.1, dtype = tf.float32))

    def _create_bias(self, shape):
        return tf.Variable(tf.constant(1., shape = shape, dtype = tf.float32))
            
