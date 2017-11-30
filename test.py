import tensorflow as tf

import data as data
from model import Model
from skimage import io
import skimage.external.tifffile as tiff

FLAGS = tf.app.flags.FLAGS


def evaluate():
    img_tf = tf.placeholder(tf.float32)
    with tf.Graph().as_default():
        images, labels = data.load_test_data()
        model = Model()

        logits = model.inference(images, keep_prob=1.0)
        accuracy = model.accuracy(logits, labels)
        img_tf = logits
        saver = tf.train.Saver()

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver.restore(sess, FLAGS.checkpoint_file_path)

            total_accuracy = sess.run([accuracy])
            print('Test accuracy: {}'.format(total_accuracy))
            im =sess.run(img_tf)
            tiff.imshow(images[6])
            tiff.imshow(labels[6],cmap='gray')
            tiff.imshow(im[6], cmap='gray')
            io.show()
    


def main(argv=None):
    evaluate()


if __name__ == '__main__':
    tf.app.flags.DEFINE_string('checkpoint_file_path', 'checkpoints/model.ckpt-75-75', 'path to checkpoint file')
    tf.app.flags.DEFINE_string('test_data', 'original_train/', 'path to test data')

    tf.app.run()

    # interesting solution https://stackoverflow.com/questions/21865637/image-processing-tiff-images-in-matlab-in-grayscale