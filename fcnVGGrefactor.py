import tensorflow as tf
import numpy as np
from skimage import io
import skimage.external.tifffile as tiff

#https://github.com/MarvinTeichmann/tensorflow-fcn/blob/master/fcn32_vgg.py

class Model(object):
    def __init__(self, batch_size=20, learning_rate=1e-4):
        self._batch_size = batch_size
        self._learning_rate = learning_rate
       
    def inference(self, images, keep_prob):
        with tf.name_scope('Processing'):
            self.conv1_1 = self._conv_layer(images, 'conv1_1')
            self.conv1_2 = self._conv_layer(images, 'conv1_2')
            self.pool1 = self._max_pool(self.conv1_2, 'pool1', False)

            self.conv2_1 = self._conv_layer(images, 'conv2_1')
            self.conv2_2 = self._conv_layer(images, 'conv2_2')
            self.pool2 = self._max_pool(self.conv2_2, 'pool2', False)

            self.conv3_1 = self._conv_layer(self.pool2, "conv3_1")
            self.conv3_2 = self._conv_layer(self.conv3_1, "conv3_2")
            self.conv3_3 = self._conv_layer(self.conv3_2, "conv3_3")
            self.pool3 = self._max_pool(self.conv3_3, 'pool3', False)

            self.conv4_1 = self._conv_layer(self.pool3, "conv4_1")
            self.conv4_2 = self._conv_layer(self.conv4_1, "conv4_2")
            self.conv4_3 = self._conv_layer(self.conv4_2, "conv4_3")
            self.pool4 = self._max_pool(self.conv4_3, 'pool4', False)

            self.conv5_1 = self._conv_layer(self.pool4, "conv5_1")
            self.conv5_2 = self._conv_layer(self.conv5_1, "conv5_2")
            self.conv5_3 = self._conv_layer(self.conv5_2, "conv5_3")
            self.pool5 = self._max_pool(self.conv5_3, 'pool5', False)


    def _conv_layer(self, input_img, name):
        with tf.variable_scope(name) as scope:
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(input_img, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            # Add summary to Tensorboard
            _activation_summary(relu)
            return relu
            
    
    def _max_pool(self, bottom, name, debug):
        pool = tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)

        if debug:
            pool = tf.Print(pool, [tf.shape(pool)],
                            message='Shape of %s' % name,
                            summarize=4, first_n=1)
        return pool

    def _activation_summary(x):
        """Helper to create summaries for activations.
        Creates a summary that provides a histogram of activations.
        Creates a summary that measure the sparsity of activations.
        Args:
        x: Tensor
        Returns:
        nothing
        """
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        tensor_name = x.op.name
        # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
        tf.summary.histogram(tensor_name + '/activations', x)
        tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
