import tensorflow as tf
import numpy as np
from skimage import io
import skimage.external.tifffile as tiff

class Model(object):
    def __init__(self, batch_size=20, learning_rate=1e-4):
        self._batch_size = batch_size
        self._learning_rate = learning_rate
       


    def inference(self, images, keep_prob):
        with tf.variable_scope('conv1_1') as scope:
            kernel = self._create_weights([2, 2, 3, 64])
            conv = self._create_conv2d(images, kernel) 
            bias = self._create_bias([64])
            preactivation = tf.nn.bias_add(conv, bias)
            conv1_1 = tf.nn.relu(preactivation, name = scope.name)
            self._activation_summary(conv1_1)  
        
        with tf.variable_scope('conv1_2') as scope:
            kernel = self._create_weights([2, 2, 64, 64])
            conv = self._create_conv2d(images, kernel) 
            bias = self._create_bias([64])
            preactivation = tf.nn.bias_add(conv, bias)
            conv1_2= tf.nn.relu(preactivation, name = scope.name)
            self._activation_summary(conv1_2)  

        h_pool1 = self._create_max_pool_2x2(conv1_2)

        with tf.variable_scope('conv2_1') as scope:
            kernel = self._create_weights([2, 2, 64, 128])
            conv = self._create_conv2d(images, kernel) 
            bias = self._create_bias([128])
            preactivation = tf.nn.bias_add(conv, bias)
            conv2_1 = tf.nn.relu(preactivation, name = scope.name)
            self._activation_summary(conv2_1) 
            
        with tf.variable_scope('conv2_2') as scope:
            kernel = self._create_weights([2, 2, 128, 128])
            conv = self._create_conv2d(images, kernel) 
            bias = self._create_bias([128])
            preactivation = tf.nn.bias_add(conv, bias)
            conv2_2 = tf.nn.relu(preactivation, name = scope.name)
            self._activation_summary(conv2_2)  

        h_pool2 = self._create_max_pool_2x2(conv2_2)
        
        with tf.variable_scope('conv3_1') as scope:
            kernel = self._create_weights([2, 2, 128, 256])
            conv = self._create_conv2d(images, kernel) 
            bias = self._create_bias([256])
            preactivation = tf.nn.bias_add(conv, bias)
            conv3_1 = tf.nn.relu(preactivation, name = scope.name)
            self._activation_summary(conv3_1)  
        
        with tf.variable_scope('conv3_2') as scope:
            kernel = self._create_weights([2, 2, 256, 256])
            conv = self._create_conv2d(images, kernel) 
            bias = self._create_bias([256])
            preactivation = tf.nn.bias_add(conv, bias)
            conv3_2 = tf.nn.relu(preactivation, name = scope.name)
            self._activation_summary(conv3_2)  

        with tf.variable_scope('conv3_3') as scope:
            kernel = self._create_weights([2, 2, 256, 256])
            conv = self._create_conv2d(images, kernel) 
            bias = self._create_bias([256])
            preactivation = tf.nn.bias_add(conv, bias)
            conv3_3 = tf.nn.relu(preactivation, name = scope.name)
            self._activation_summary(conv3_3)

        with tf.variable_scope('conv3_4') as scope:
            kernel = self._create_weights([2, 2, 256, 256])
            conv = self._create_conv2d(images, kernel) 
            bias = self._create_bias([256])
            preactivation = tf.nn.bias_add(conv, bias)
            conv3_4 = tf.nn.relu(preactivation, name = scope.name)
            self._activation_summary(conv3_4)    

        h_pool3 = self._create_max_pool_2x2(conv3_4) 

        with tf.variable_scope('conv4_1') as scope:
            kernel = self._create_weights([2, 2, 256, 512])
            conv = self._create_conv2d(images, kernel) 
            bias = self._create_bias([512])
            preactivation = tf.nn.bias_add(conv, bias)
            conv3_1 = tf.nn.relu(preactivation, name = scope.name)
            self._activation_summary(conv3_1)  
        
        with tf.variable_scope('conv4_2') as scope:
            kernel = self._create_weights([2, 2, 512, 512])
            conv = self._create_conv2d(images, kernel) 
            bias = self._create_bias([512])
            preactivation = tf.nn.bias_add(conv, bias)
            conv4_2 = tf.nn.relu(preactivation, name = scope.name)
            self._activation_summary(conv4_2)  

        with tf.variable_scope('conv4_3') as scope:
            kernel = self._create_weights([2, 2, 512, 512])
            conv = self._create_conv2d(images, kernel) 
            bias = self._create_bias([512])
            preactivation = tf.nn.bias_add(conv, bias)
            conv4_3 = tf.nn.relu(preactivation, name = scope.name)
            self._activation_summary(conv4_3)

        with tf.variable_scope('conv4_4') as scope:
            kernel = self._create_weights([2, 2, 512, 512])
            conv = self._create_conv2d(images, kernel) 
            bias = self._create_bias([512])
            preactivation = tf.nn.bias_add(conv, bias)
            conv4_4 = tf.nn.relu(preactivation, name = scope.name)
            self._activation_summary(conv4_4)   
        
        h_pool4 = self._create_max_pool_2x2(conv4_4) 

        with tf.variable_scope('conv4_1') as scope:
            kernel = self._create_weights([2, 2, 512, 512])
            conv = self._create_conv2d(images, kernel) 
            bias = self._create_bias([512])
            preactivation = tf.nn.bias_add(conv, bias)
            conv4_1 = tf.nn.relu(preactivation, name = scope.name)
            self._activation_summary(conv3_1)  
        
        with tf.variable_scope('conv4_2') as scope:
            kernel = self._create_weights([2, 2, 512, 512])
            conv = self._create_conv2d(images, kernel) 
            bias = self._create_bias([512])
            preactivation = tf.nn.bias_add(conv, bias)
            conv4_2 = tf.nn.relu(preactivation, name = scope.name)
            self._activation_summary(conv4_2)  

        with tf.variable_scope('conv4_3') as scope:
            kernel = self._create_weights([2, 2, 512, 512])
            conv = self._create_conv2d(images, kernel) 
            bias = self._create_bias([512])
            preactivation = tf.nn.bias_add(conv, bias)
            conv4_3 = tf.nn.relu(preactivation, name = scope.name)
            self._activation_summary(conv4_3)

        with tf.variable_scope('conv4_4') as scope:
            kernel = self._create_weights([2, 2, 512, 512])
            conv = self._create_conv2d(images, kernel) 
            bias = self._create_bias([512])
            preactivation = tf.nn.bias_add(conv, bias)
            conv4_4 = tf.nn.relu(preactivation, name = scope.name)
            self._activation_summary(conv4_4)   
        
        h_pool4 = self._create_max_pool_2x2(conv4_4) 

        with tf.variable_scope('conv5_1') as scope:
            kernel = self._create_weights([2, 2, 512, 512])
            conv = self._create_conv2d(images, kernel) 
            bias = self._create_bias([512])
            preactivation = tf.nn.bias_add(conv, bias)
            conv5_1 = tf.nn.relu(preactivation, name = scope.name)
            self._activation_summary(conv5_1)  
        
        with tf.variable_scope('conv5_2') as scope:
            kernel = self._create_weights([2, 2, 512, 512])
            conv = self._create_conv2d(images, kernel) 
            bias = self._create_bias([512])
            preactivation = tf.nn.bias_add(conv, bias)
            conv5_2 = tf.nn.relu(preactivation, name = scope.name)
            self._activation_summary(conv5_2)  

        with tf.variable_scope('conv5_3') as scope:
            kernel = self._create_weights([2, 2, 512, 512])
            conv = self._create_conv2d(images, kernel) 
            bias = self._create_bias([512])
            preactivation = tf.nn.bias_add(conv, bias)
            conv5_3 = tf.nn.relu(preactivation, name = scope.name)
            self._activation_summary(conv5_3)

        h_pool5 = self._create_max_pool_2x2(conv5_3) 


        with tf.variable_scope('local1') as scope:
            w6 = self._create_weights([2,2,512,4096])
            b6 = self._create_bias(4096)
            conv6 = self._create_conv2d(h_pool5,w6)
            result = tf.add.bias(conv6, b6)
            relu6 = tf.nn.relu(result, name="relu6")
            relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        with tf.variable_scope('local2') as scope:
            w7 = self._create_weights([1,1,4096,4096])
            b7 = self._create_bias(4096)
            conv7 = self._create_conv2d(relu_dropout6,w7)
            result = tf.add.bias(conv7, b7)
            relu7 = tf.nn.relu(result, name="relu7")
            relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)
      
        with tf.variable_scope('deconv_1') as scope:
            # Now it comes the deconvolution to the original size
            deconv_shape1 = h_pool4.get_shape()
            w_t1 = self._create_weights([4,4, deconv_shape1[3].value, 1])
            b_t1 = self._create_bias([deconv_shape1[3].value])
            conv_t1 = tf.nn.conv2d_transpose(relu_dropout7,w_t1,b_t1, outpu_shape=tf.shape(h_pool4))
            fuse_1 = tf.add(conv_t1, image_net["pool4"], name = "fuse_1")

        with tf.variable_scope('deconv_2') as scope:
            deconv_shape2 = h_pool3.get_shape()
            w_t2 = self._create_weights([4,4, deconv_shape2[3].value, 1])
            b_t2 = self._create_bias([deconv_shape2[3].value])
            conv_t2 = tf.nn.conv2d_transpose(fuse_1,w_t2,b_t2, outpu_shape=tf.shape(h_pool3))
            fuse_2 = tf.add(conv_t2, image_net["pool3"], name = "fuse_2") 

        with tf.variable_scope('deconv_3') as scope:
            shape = [64, 64, 1]
            deconv_shape3 = tf.stack(shape[0], shape[1], shape[2])
            w_t3 = self._create_weights([64,64, deconv_shape2[3].value])
            b_t2 = self._create_bias([1])
            conv_t3 = tf.nn.conv2d_transpose(fuse_2, w_t3, b_t3, outpu_shape=tf.shape(deconv_shape3, stride=8))
      
        return conv_t3

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
