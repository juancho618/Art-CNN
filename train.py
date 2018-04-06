import tensorflow as tf
import data as data
import time

from noPoolingRefator import Model
# from fcnVGGhourglass import Model


FLAGS = tf.app.flags.FLAGS

#https://pythonprogramming.net/rnn-tensorflow-python-machine-learning-tutorial/
#https://github.com/MarvinTeichmann/tensorflow-fcn   #example from git

def train():
    model = Model()
    

    with tf.Graph().as_default():
        # Load the images drom the .npy file       
        images, val_images, labels, val_labels, color_channel_mean = data.load_train_data()
        
        x = tf.placeholder(shape=[None, data.IMAGE_SIZE, data.IMAGE_SIZE, 3], dtype=tf.float32, name='x') 
        y = tf.placeholder(shape=[None, data.IMAGE_SIZE,  data.IMAGE_SIZE], dtype=tf.float32, name='y') 
        keep_prob = tf.placeholder(tf.float32, name='dropout_prob')
        global_step = tf.contrib.framework.get_or_create_global_step()

        images, val_images, labels, val_labels, color_channel_mean = data.load_train_data()
        infered_images = model.inference(x, keep_prob=keep_prob, train=True, avg_ch_array = color_channel_mean) 
        loss = model.loss(inference=infered_images, labels=y) # calculate loss pixel by pixel
        #accuracy =  model.accuracy(infered_images, y) # Not needed
        summary_op = tf.summary.merge_all()
        train_op = model.train(loss, global_step = global_step)
        
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)
            sess.run(init)
            for i in range(FLAGS.num_iter): #len(images)/FLAGS.batch_size
                offset = (i * FLAGS.batch_size) % (len(images) - FLAGS.batch_size)
                batch_x, batch_y = images[offset:(offset + FLAGS.batch_size), :], labels[
                                                                                  offset:(offset + FLAGS.batch_size), :]
                
                _, cur_loss, summary = sess.run([train_op, loss, summary_op],
                                                feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
                
                writer.add_summary(summary, i)
                print(i, cur_loss)
                # if i % 5 == 0:
                #     validation_accuracy = accuracy.eval(feed_dict={x: val_images, y: val_labels, keep_prob: 1.0})
                #     print('Iter {} Error: {}'.format(i, validation_accuracy))

                if i == FLAGS.num_iter - 1:
                    saver.save(sess, FLAGS.checkpoint_file_path, global_step)
                


def main(argv=None):
    train()


if __name__ == '__main__':
    tf.app.flags.DEFINE_integer('batch_size', 18, 'size of training batches')
    tf.app.flags.DEFINE_integer('num_iter', 1201, 'number of training iterations')#102
    tf.app.flags.DEFINE_string('checkpoint_file_path', 'robe_face/checkpoints/model.ckpt-1201', 'path to checkpoint file')
    tf.app.flags.DEFINE_string('summary_dir', 'robe_face/graphs', 'path to directory for storing summaries')

    tf.app.run()