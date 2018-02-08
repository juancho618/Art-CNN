import tensorflow as tf
import numpy as np
from skimage import io
import logging
from math import ceil
import skimage.external.tifffile as tiff

#link about deconvolution terms: https://www.quora.com/What-is-the-difference-between-Deconvolution-Upsampling-Unpooling-and-Convolutional-Sparse-Coding
#deconvolution video: https://www.youtube.com/watch?v=8DiqJj5tPlA
#good guide for transpose deconvolution: http://cv-tricks.com/image-segmentation/transpose-convolution-in-tensorflow/
class Model(object):
    def __init__(self, batch_size=20, learning_rate=1e-4):
        self._batch_size = batch_size
        self._learning_rate = learning_rate
       
  

    def inference(self, images, keep_prob):
        random_init_fc8= False
        train = True
        num_classes = 1
        debug = False
        with tf.variable_scope('conv1_1') as scope:
            kernel = self._create_weights([2, 2, 3, 64])
            conv = self._create_conv2d(images, kernel) 
            bias = self._create_bias([64])
            preactivation = tf.nn.bias_add(conv, bias)
            conv1_1 = tf.nn.relu(preactivation, name = scope.name)
            self._activation_summary(conv1_1)  
        
        with tf.variable_scope('conv1_2') as scope:
            kernel = self._create_weights([2, 2, 64, 64])
            conv = self._create_conv2d(conv1_1, kernel) 
            bias = self._create_bias([64])
            preactivation = tf.nn.bias_add(conv, bias)
            conv1_2= tf.nn.relu(preactivation, name = scope.name)
            self._activation_summary(conv1_2)  

        h_pool1 = self._create_max_pool_2x2(conv1_2)

        with tf.variable_scope('conv2_1') as scope:
            kernel = self._create_weights([2, 2, 64, 128])
            conv = self._create_conv2d(h_pool1, kernel) 
            bias = self._create_bias([128])
            preactivation = tf.nn.bias_add(conv, bias)
            conv2_1 = tf.nn.relu(preactivation, name = scope.name)
            self._activation_summary(conv2_1) 
            
        with tf.variable_scope('conv2_2') as scope:
            kernel = self._create_weights([2, 2, 128, 128])
            conv = self._create_conv2d(conv2_1, kernel) 
            bias = self._create_bias([128])
            preactivation = tf.nn.bias_add(conv, bias)
            conv2_2 = tf.nn.relu(preactivation, name = scope.name)
            self._activation_summary(conv2_2)  

        h_pool2 = self._create_max_pool_2x2(conv2_2)
        
        with tf.variable_scope('conv3_1') as scope:
            kernel = self._create_weights([2, 2, 128, 256])
            conv = self._create_conv2d(h_pool2, kernel) 
            bias = self._create_bias([256])
            preactivation = tf.nn.bias_add(conv, bias)
            conv3_1 = tf.nn.relu(preactivation, name = scope.name)
            self._activation_summary(conv3_1)  
        
        with tf.variable_scope('conv3_2') as scope:
            kernel = self._create_weights([2, 2, 256, 256])
            conv = self._create_conv2d(conv3_1, kernel) 
            bias = self._create_bias([256])
            preactivation = tf.nn.bias_add(conv, bias)
            conv3_2 = tf.nn.relu(preactivation, name = scope.name)
            self._activation_summary(conv3_2)  

        with tf.variable_scope('conv3_3') as scope:
            kernel = self._create_weights([2, 2, 256, 256])
            conv = self._create_conv2d(conv3_2, kernel) 
            bias = self._create_bias([256])
            preactivation = tf.nn.bias_add(conv, bias)
            conv3_3 = tf.nn.relu(preactivation, name = scope.name)
            self._activation_summary(conv3_3)

        with tf.variable_scope('conv3_4') as scope:
            kernel = self._create_weights([2, 2, 256, 256])
            conv = self._create_conv2d(conv3_3, kernel) 
            bias = self._create_bias([256])
            preactivation = tf.nn.bias_add(conv, bias)
            conv3_4 = tf.nn.relu(preactivation, name = scope.name)
            self._activation_summary(conv3_4)    

        h_pool3 = self._create_max_pool_2x2(conv3_4) 

        with tf.variable_scope('conv4_1') as scope:
            kernel = self._create_weights([2, 2, 256, 512])
            conv = self._create_conv2d(h_pool3, kernel) 
            bias = self._create_bias([512])
            preactivation = tf.nn.bias_add(conv, bias)
            conv3_1 = tf.nn.relu(preactivation, name = scope.name)
            self._activation_summary(conv3_1)  
        
        with tf.variable_scope('conv4_2') as scope:
            kernel = self._create_weights([2, 2, 512, 512])
            conv = self._create_conv2d(conv3_1, kernel) 
            bias = self._create_bias([512])
            preactivation = tf.nn.bias_add(conv, bias)
            conv4_2 = tf.nn.relu(preactivation, name = scope.name)
            self._activation_summary(conv4_2)  

        with tf.variable_scope('conv4_3') as scope:
            kernel = self._create_weights([2, 2, 512, 512])
            conv = self._create_conv2d(conv4_2, kernel) 
            bias = self._create_bias([512])
            preactivation = tf.nn.bias_add(conv, bias)
            conv4_3 = tf.nn.relu(preactivation, name = scope.name)
            self._activation_summary(conv4_3)

        with tf.variable_scope('conv4_4') as scope:
            kernel = self._create_weights([2, 2, 512, 512])
            conv = self._create_conv2d(conv4_3, kernel) 
            bias = self._create_bias([512])
            preactivation = tf.nn.bias_add(conv, bias)
            conv4_4 = tf.nn.relu(preactivation, name = scope.name)
            self._activation_summary(conv4_4)   
        
        h_pool4 = self._create_max_pool_2x2(conv4_4) 

        with tf.variable_scope('conv5_1') as scope:
            kernel = self._create_weights([2, 2, 512, 512])
            conv = self._create_conv2d(h_pool4, kernel) 
            bias = self._create_bias([512])
            preactivation = tf.nn.bias_add(conv, bias)
            conv5_1 = tf.nn.relu(preactivation, name = scope.name)
            self._activation_summary(conv5_1)  
        
        with tf.variable_scope('conv5_2') as scope:
            kernel = self._create_weights([2, 2, 512, 512])
            conv = self._create_conv2d(conv5_1, kernel) 
            bias = self._create_bias([512])
            preactivation = tf.nn.bias_add(conv, bias)
            conv5_2 = tf.nn.relu(preactivation, name = scope.name)
            self._activation_summary(conv5_2)  

        with tf.variable_scope('conv5_3') as scope:
            kernel = self._create_weights([2, 2, 512, 512])
            conv = self._create_conv2d(conv5_2, kernel) 
            bias = self._create_bias([512])
            preactivation = tf.nn.bias_add(conv, bias)
            conv5_3 = tf.nn.relu(preactivation, name = scope.name)
            self._activation_summary(conv5_3)

        h_pool5 = self._create_max_pool_2x2(conv5_3) 


        with tf.variable_scope('local1') as scope:
            w6 = self._create_weights([2,2,512,4096])
            b6 = self._create_bias([4096])
            conv6 = self._create_conv2d(h_pool5,w6)
            result = tf.nn.bias_add(conv6, b6)
            relu6 = tf.nn.relu(result, name="relu6")
            relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)
            print('first drop', relu_dropout6.get_shape())

            print('conv 6', conv6.get_shape())

        with tf.variable_scope('local2') as scope:
            w7 = self._create_weights([2,2,4096,4096])
            b7 = self._create_bias([4096])
            conv7 = self._create_conv2d(relu_dropout6,w7)
            result = tf.nn.bias_add(conv7, b7)
            relu7 = tf.nn.relu(result, name="relu7")
            relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)
            print('second drop', relu_dropout7.get_shape())

            print('conv 7', conv7.get_shape())

        # if random_init_fc8:
        #     self.score_fr = self._score_layer(relu_dropout7, "score_fr",
        #                                       num_classes)
        # else:
        #     self.score_fr = self._fc_layer(relu_dropout7, "score_fr",
        #                                    num_classes=num_classes,
        #                                    relu=False)
        
        self.upscore = self._upscore_layer(relu_dropout7, shape=None,
                                           num_classes=num_classes,
                                           debug=debug,
                                           name='up', ksize=64, stride=32)
     


        # with tf.variable_scope('deconv_1') as scope:
        #     # Now it comes the deconvolution to the original size
            
        #     #Different approach
        #     n_channels = 4096
        #     upscale_factor = 6
        #     kernel_size = 2*upscale_factor - upscale_factor%2
        #     stride =  upscale_factor
        #     strides = [1, stride, stride, 1]
        #     in_shape = tf.shape(relu_dropout7)
            
        #     h = ((in_shape[1] - 1) * stride) + 1
        #     w = ((in_shape[2] - 1) * stride) + 1
        #     new_shape = [in_shape[0], h, w, 1]
        #     output_shape = tf.stack(new_shape)

        #     filter_shape = [kernel_size, kernel_size, n_channels, n_channels]
        #     weights = self.get_bilinear_filter(filter_shape,upscale_factor)
        #     deconv = tf.nn.conv2d_transpose(relu_dropout7, weights, output_shape,
        #                                     strides=strides, padding='SAME')
        #     print('deconvolution shape', tf.shape(output_shape))



        # #other deconv layers
        # with tf.variable_scope('deconv_2') as scope:

        #     n_channels = 512
        #     upscale_factor = 2
        #     kernel_size = 2*upscale_factor - upscale_factor%2
        #     stride =  upscale_factor
        #     strides = [1, stride, stride, 1]
        #     in_shape = tf.shape(relu_dropout7)
            
        #     h = ((in_shape[1] - 1) * stride) + 1
        #     w = ((in_shape[2] - 1) * stride) + 1
        #     new_shape = [in_shape[0], h, w, n_channels]
        #     output_shape = tf.stack(new_shape)

        #     filter_shape = [kernel_size, kernel_size, n_channels, n_channels]
        #     weights = self.get_bilinear_filter(filter_shape,upscale_factor)
        #     deconv2 = tf.nn.conv2d_transpose(deconv, weights, output_shape,
        #                                     strides=strides, padding='SAME')
        #     print('deconvolution shape', tf.shape(deconv2))
            
        # with tf.variable_scope('deconv_3') as scope:

        #     n_channels = 256
        #     upscale_factor = 2
        #     kernel_size = 2*upscale_factor - upscale_factor%2
        #     stride =  upscale_factor
        #     strides = [1, stride, stride, 1]
        #     in_shape = tf.shape(relu_dropout7)
            
        #     h = ((in_shape[1] - 1) * stride) + 1
        #     w = ((in_shape[2] - 1) * stride) + 1
        #     new_shape = [in_shape[0], h, w, n_channels]
        #     output_shape = tf.stack(new_shape)

        #     filter_shape = [kernel_size, kernel_size, n_channels, n_channels]
        #     weights = self.get_bilinear_filter(filter_shape,upscale_factor)
        #     deconv3 = tf.nn.conv2d_transpose(deconv2, weights, output_shape,
        #                                     strides=strides, padding='SAME')
        #     print('deconvolution shape', tf.shape(deconv3))

        #     with tf.variable_scope('deconv_4') as scope:

        #         n_channels = 128
        #         upscale_factor = 2
        #         kernel_size = 2*upscale_factor - upscale_factor%2
        #         stride =  upscale_factor
        #         strides = [1, stride, stride, 1]
        #         in_shape = tf.shape(relu_dropout7)
                
        #         h = ((in_shape[1] - 1) * stride) + 1
        #         w = ((in_shape[2] - 1) * stride) + 1
        #         new_shape = [in_shape[0], h, w, n_channels]
        #         output_shape = tf.stack(new_shape)

        #         filter_shape = [kernel_size, kernel_size, n_channels, n_channels]
        #         weights = self.get_bilinear_filter(filter_shape,upscale_factor)
        #         deconv4 = tf.nn.conv2d_transpose(deconv3, weights, output_shape,
        #                                         strides=strides, padding='SAME')
        #         print('deconvolution shape', tf.shape(deconv4))

        #     with tf.variable_scope('deconv_5') as scope:

        #         n_channels = 64
        #         upscale_factor = 2
        #         kernel_size = 2*upscale_factor - upscale_factor%2
        #         stride =  upscale_factor
        #         strides = [1, stride, stride, 1]
        #         in_shape = tf.shape(relu_dropout7)
                
        #         h = ((in_shape[1] - 1) * stride) + 1
        #         w = ((in_shape[2] - 1) * stride) + 1
        #         new_shape = [in_shape[0], h, w, n_channels]
        #         output_shape = tf.stack(new_shape)

        #         filter_shape = [kernel_size, kernel_size, n_channels, n_channels]
        #         weights = self.get_bilinear_filter(filter_shape,upscale_factor)
        #         deconv5 = tf.nn.conv2d_transpose(deconv4, weights, output_shape,
        #                                         strides=strides, padding='SAME')
        #         print('deconvolution shape', tf.shape(deconv5))
        
      
        return self.upscore

    def _fc_layer(self, bottom, name, num_classes=None,
                  relu=True, debug=False):
        with tf.variable_scope(name) as scope:
            shape = bottom.get_shape().as_list()

            if name == 'fc6':
                filt = self.get_fc_weight_reshape(name, [4, 4, 512, 4096])
            elif name == 'score_fr':
                name = 'fc8'  # Name of score_fr layer in VGG Model
                filt = self.get_fc_weight_reshape(bottom, [1, 1, 4096, 1000],
                                                  num_classes=num_classes)
            else:
                filt = self.get_fc_weight_reshape(name, [1, 1, 4096, 4096])
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            conv_biases = self.get_bias(name, num_classes=num_classes)
            bias = tf.nn.bias_add(conv, conv_biases)

            if relu:
                bias = tf.nn.relu(bias)
            _activation_summary(bias)

            if debug:
                bias = tf.Print(bias, [tf.shape(bias)],
                                message='Shape of %s' % name,
                                summarize=4, first_n=1)
            return bias

    def get_fc_weight_reshape(self, name, shape, num_classes=None):
        print('Layer name: %s' % name)
        print('Layer shape: %s' % shape)
        
        weights = name
        if num_classes is not None:
            weights = self._summary_reshape(weights, shape,
                                            num_new=num_classes)
        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        return tf.get_variable(name="weights", initializer=init, shape=shape)

    def _upscore_layer(self, bottom, shape,
                       num_classes, name, debug,
                       ksize=4, stride=2):
        strides = [1, stride, stride, 1]
        with tf.variable_scope(name):
            in_features = bottom.get_shape()[3].value

            if shape is None:
                # Compute shape out of Bottom
                in_shape = tf.shape(bottom)

                h = ((in_shape[1] - 1) * stride) + 1
                w = ((in_shape[2] - 1) * stride) + 1
                new_shape = [in_shape[0], h, w, num_classes]
            else:
                new_shape = [shape[0], shape[1], shape[2], num_classes]
            output_shape = tf.stack(new_shape)

            logging.debug("Layer: %s, Fan-in: %d" % (name, in_features))
            f_shape = [ksize, ksize, num_classes, in_features]

            # create
            num_input = ksize * ksize * in_features / stride
            stddev = (2 / num_input)**0.5

            weights = self.get_deconv_filter(f_shape)
            deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                            strides=strides, padding='SAME')

            if debug:
                deconv = tf.Print(deconv, [tf.shape(deconv)],
                                  message='Shape of %s' % name,
                                  summarize=4, first_n=1)

        _activation_summary(deconv)
        return deconv
    def get_deconv_filter(self, f_shape):
        width = f_shape[0]
        height = f_shape[1]
        f = ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([f_shape[0], f_shape[1]])
        for x in range(width):
            for y in range(height):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(f_shape)
        for i in range(f_shape[2]):
            weights[:, :, i, i] = bilinear

        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        return tf.get_variable(name="up_filter", initializer=init,
                               shape=weights.shape)

    def get_bilinear_filter(self, filter_shape, upscale_factor):
            ##filter_shape is [width, height, num_in_channels, num_out_channels]
        kernel_size = filter_shape[1]
        ### Centre location of the filter for which value is calculated
        if kernel_size % 2 == 1:
            centre_location = upscale_factor - 1
        else:
            centre_location = upscale_factor - 0.5
 
        bilinear = np.zeros([filter_shape[0], filter_shape[1]])
        for x in range(filter_shape[0]):
            for y in range(filter_shape[1]):
                ##Interpolation Calculation
                value = (1 - abs((x - centre_location)/ upscale_factor)) * (1 - abs((y - centre_location)/ upscale_factor))
                bilinear[x, y] = value
        weights = np.zeros(filter_shape)
        for i in range(filter_shape[2]):
            weights[:, :, i, i] = bilinear
        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
 
        bilinear_weights = tf.get_variable(name="decon_bilinear_filter", initializer=init,
                               shape=weights.shape)
        return bilinear_weights

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