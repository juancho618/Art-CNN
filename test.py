import tensorflow as tf
import data as data
from noPooling import Model
# from fcnVGGhourglass import Model
from PIL import Image
import utils as utils
import numpy as np

FLAGS = tf.app.flags.FLAGS


def evaluate():
    model = Model()
    img_tf = tf.placeholder(tf.float32)
    with tf.Graph().as_default():
        images, labels = data.load_test_data()
        x = tf.placeholder(shape=[None, data.IMAGE_SIZE, data.IMAGE_SIZE, 3], dtype=tf.float32, name='x') 
        y = tf.placeholder(shape=[None, data.IMAGE_SIZE,  data.IMAGE_SIZE], dtype=tf.float32, name='y') 
        
        infered = model.inference(x, keep_prob=1.0, train=False)
        # loss = model.loss(logits= infered, labels= y)
        init = tf.global_variables_initializer()
        #accuracy = model.accuracy(logits, y)
        img_tf = infered
        saver = tf.train.Saver()
        results = []
        with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, FLAGS.checkpoint_file_path)

            # total_accuracy = sess.run([accuracy])
            # print('Test Error: {}'.format(total_accuracy))
            #For every image
            for i in range(len(images)):
                offset = (i * FLAGS.batch_size) % (len(images) - FLAGS.batch_size)
                batch_x, batch_y = images[offset:(offset + FLAGS.batch_size), :], labels[
                                                                                  offset:(offset + FLAGS.batch_size), :]
                infered_image  = sess.run(infered, feed_dict={x: batch_x})
                results.append(infered_image)
           
            # result_images = []
            # img = Image.fromarray(np.uint8(images[525]))
            # img_r = Image.fromarray(np.uint8(im[525]))

            # img.show()
            # img_r.show()
            # for i in range(len(images)):
            #     npImage = np.array(im[i])
            #     result_images.append(npImage)
            #     # utils.save_img('./hourglass/results', npImage)
            #     img = Image.fromarray(np.uint8(images[i]))
            #     img_r = Image.fromarray(np.uint8(im[i]))
    
            #     img.show()
            #     img_r.show()
            np.save('./noPooling/results/results.npy', np.array(results))
    


def main(argv=None):
    evaluate()


if __name__ == '__main__':
    tf.app.flags.DEFINE_integer('batch_size', 1, 'size of training batches')
    tf.app.flags.DEFINE_string('checkpoint_file_path', 'noPooling/checkpoints/model.ckpt-75-102', 'path to checkpoint file')
    tf.app.flags.DEFINE_string('test_data', 'original_train/', 'path to test data')

    tf.app.run()

    # interesting solution https://stackoverflow.com/questions/21865637/image-processing-tiff-images-in-matlab-in-grayscale