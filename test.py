import tensorflow as tf
import data as data
from noPoolingRefator import Model
# from fcnVGGhourglass import Model
from PIL import Image
import utils as utils
import numpy as np

FLAGS = tf.app.flags.FLAGS

avg_channel_color = [85.03, 40.26, 25.25]

def evaluate():
    model = Model()
    img_tf = tf.placeholder(tf.float32)
    with tf.Graph().as_default():
        images, labels = data.load_test_data()
        print('size', images.shape)
        x = tf.placeholder(shape=[None, data.IMAGE_SIZE, data.IMAGE_SIZE, 3], dtype=tf.float32, name='x') 
        y = tf.placeholder(shape=[None, data.IMAGE_SIZE,  data.IMAGE_SIZE], dtype=tf.float32, name='y') 
        
        infered = model.inference(x, keep_prob=1.0, train=False, avg_ch_array=avg_channel_color)
        loss = model.loss(inference= infered, labels= y)
        init = tf.global_variables_initializer()
        #accuracy = model.accuracy(logits, y)
        img_tf = infered
        saver = tf.train.Saver()
        results = []
        loss_array =[]
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
                cur_loss = sess.run(loss, feed_dict={x: batch_x, y: batch_y})
                print(i, np.round(cur_loss, decimals=1))
                loss_array.append(cur_loss)
            np.save('./robe_face/results/results_robe.npy', np.array(results))
            np.save('./robe_face/results/loss_results_robe.npy', np.array(loss_array))
    


def main(argv=None):
    evaluate()


if __name__ == '__main__':
    tf.app.flags.DEFINE_integer('batch_size', 1, 'size of training batches')
    tf.app.flags.DEFINE_string('checkpoint_file_path', 'robe_face/checkpoints/model.ckpt-1201-1201', 'path to checkpoint file')
    tf.app.flags.DEFINE_string('test_data', 'robe_face/', 'path to test data')

    tf.app.run()

    # interesting solution https://stackoverflow.com/questions/21865637/image-processing-tiff-images-in-matlab-in-grayscale