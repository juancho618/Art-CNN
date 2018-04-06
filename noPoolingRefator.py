import tensorflow as tf
import numpy as np


class Model(object):
    def __init__(self, batch_size=18, learning_rate=1e-4, num_labels= 1):
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._filters = {
            'conv1': [3,3,3,64],
            'conv2': [3,3,64,64],
            'conv3': [3,3,64,64],
           'conv4': [3,3,64,1]
        }

    def inference(self, images, keep_prob, train, avg_ch_array = []):
        
        # Normalize images
        red, green, blue = tf.split(images, 3, 3)
        images = tf.concat([
            blue - avg_ch_array[0],
            green - avg_ch_array[1],
            red - avg_ch_array[2]], axis=3)

        self.conv1 = self._conv_layer(images, "conv1")
        self.conv2 = self._conv_layer(self.conv1, "conv2")
        self.conv3 = self._conv_layer(self.conv2, "conv3")
        self.conv4 = self._conv_layer(self.conv3, "conv4")
        if train:
            self.conv4 = tf.nn.dropout(self.conv4, keep_prob)
      
        return tf.squeeze(self.conv4) #x,64,64
            
    
    def _conv_layer(self, bottom, name):
        with tf.variable_scope(name) as scope:
            filt = self._create_weights(self._filters[name])
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self._create_bias([self._filters[name][3]])
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            # Add summary to Tensorboard
            self._activation_summary(relu)
            return relu

    def train(self, loss, global_step):
        tf.summary.scalar('learning_rate', self._learning_rate)
        train_op = tf.train.AdamOptimizer(self._learning_rate).minimize(loss, global_step=global_step)
        return train_op

    def loss(self, inference, labels):
        with tf.variable_scope('loss') as scope:
            print('inference', inference.shape)
            print('labels', labels.shape)
            cost = tf.reduce_mean(tf.squared_difference(inference, labels))
            tf.summary.scalar('cost', cost)

        return cost

    def accuracy(self, logits, labels):
        with tf.variable_scope('accuracy') as scope:
            accuracy = tf.reduce_mean(tf.squared_difference(logits, labels), dtype=tf.float32,
                                      name=scope.name)
            tf.summary.scalar('accuracy', accuracy)
        return accuracy

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

    def _activation_summary(self, x):
        tensor_name = x.op.name
        tf.summary.histogram(tensor_name + '/activations', x)
        tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

    def save_infered(self):
        print('infered data', len(self._infered[0]))
