import tensorflow as tf
import numpy as np
from skimage import io
import skimage.external.tifffile as tiff

class Model(object):
    def __init__(self, batch_size=20, learning_rate=1e-4, num_labels= 256):
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._num_labels = num_labels
       

    def inference(self, images, keep_prob):
        with tf.variable_scope('conv1') as scope:
            kernel = self._create_weights([2, 2, 3, 16])
            conv = self._create_conv2d(images, kernel) 
            bias = self._create_bias([16])
            preactivation = tf.nn.bias_add(conv, bias)
            conv1 = tf.nn.relu(preactivation, name = scope.name)
            self._activation_summary(conv1)     

        # First pool
        h_pool1 = self._create_max_pool_2x2(conv)

        with tf.variable_scope('conv2') as scope:
            kernel = self._create_weights([2, 2, 16, 32])
            conv = self._create_conv2d(h_pool1, kernel) 
            bias = self._create_bias([32])
            preactivation = tf.nn.bias_add(conv, bias)
            conv2 = tf.nn.relu(preactivation, name = scope.name)
            self._activation_summary(conv2)

        # Second pool
        h_pool2 = self._create_max_pool_2x2(conv2)   

        with tf.variable_scope('local1') as scope:
            reshape = tf.reshape(h_pool2, [-1, 16 * 16 * 32]) # the -1 refers to 1-D
            W_fc1 = self._create_weights([16 * 16 * 32, 512])
            b_fc1 = self._create_bias([512])
            local1 = tf.nn.relu(tf.matmul(reshape, W_fc1) + b_fc1, name=scope.name) # (?, 1024)
            some = tf.reshape(local1,[-1, 16,16,1])
            resize = tf.image.resize_bicubic(some[:20], [64,64]) # upscaling part TODO: select 20 latersu
        


        
        # print('some', resize.shape)
        # print('local 1 shape', local1.shape)
        
        # with tf.variable_scope('local2_linear') as scope:
        #     W_fc2 = self._create_weights([1024, self._num_labels])
        #     b_fc2 = self._create_bias([self._num_labels])
        #     local1_drop = tf.nn.dropout(local1, keep_prob)
        #     local2 = tf.nn.bias_add(tf.matmul(local1_drop, W_fc2), b_fc2, name=scope.name) # (?, 256)
        #     self._activation_summary(local2)
            
        # print('local 2 shape', local2.shape)
        return resize #local2
            
    

    def train(self, loss, global_step):
        tf.summary.scalar('learning_rate', self._learning_rate)
        train_op = tf.train.AdamOptimizer(self._learning_rate).minimize(loss, global_step=global_step)
        return train_op

    def loss(self, logits, labels):
        with tf.variable_scope('loss') as scope:
            print('logits', logits.shape)
            print('labels', labels.shape)
            #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            #cost = tf.reduce_mean(cross_entropy, name=scope.name)
            cost = tf.reduce_mean(tf.squared_difference(logits, labels))
            tf.summary.scalar('cost', cost)

        return cost

    def accuracy(self, logits, labels):
        with tf.variable_scope('accuracy') as scope:
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits), tf.argmax(labels)), dtype=tf.float32),
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
