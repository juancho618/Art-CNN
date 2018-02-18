import tensorflow as tf
import matplotlib.pyplot as plt
import data as data
#from model2 import Model
from VGG11up import Model
from skimage import io
import skimage.external.tifffile as tiff
import scipy.misc
from PIL import Image
import utils as utils
import numpy as np

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
            print('Test Error: {}'.format(total_accuracy))
            im =sess.run(img_tf)
            result_images = []
            for i in range(len(images)):
                npImage = np.array(im[i])
                result_images.append(npImage)
                # utils.save_img('./hourglass/results', npImage)
                tiff.imshow(images[i])
                tiff.imshow(labels[i],cmap='gray')
                # print(npImage.shape)
                # tiff.imshow(npImage)
                # tiff.imsave('./hourglass/results/temp'+str(i)+'.tiff', tiff.imshow(im[33], cmap='gray'))
                # # imag = Image.fromarray(im[i])
                # # imag.save("your_file"+i+".tiff")
                tiff.imshow(im[i], cmap='gray')
                # #scipy.misc.imsave('./hourglass/results/temp'+str(i)+'.jpg', im[i])
                io.show()
            # np.save('./11up/results/file.npy', np.array(result_images))
    


def main(argv=None):
    evaluate()


if __name__ == '__main__':
    tf.app.flags.DEFINE_string('checkpoint_file_path', '11up/checkpoints/model.ckpt-75-75', 'path to checkpoint file')
    tf.app.flags.DEFINE_string('test_data', 'original_train/', 'path to test data')

    tf.app.run()

    # interesting solution https://stackoverflow.com/questions/21865637/image-processing-tiff-images-in-matlab-in-grayscale