import tensorflow as tf
from DataSet import DataSet
from trained_vgg16 import vgg16
from tools import do_upsample, calculate_loss, calculate_accuracy
from Tools import conver2onehot
import numpy as np

class fcn:
    def __init__(self, sess):
        self.sess = sess
        self.CATEGORY_NUM = 151
        self.IMAGE_SIZE = [224, 224]
        self.IMAGE_CHANNEL = 3
        data_dir = '/home/give/Documents/dataset/ADEChallengeData2016'
        self.dataset = DataSet(data_dir)
        self.learning_rate = 1e-5
        self.itertator_number = int(1e+5)
        self.BATCH_SIZE = 80
        self.imgs = tf.placeholder(
            tf.float32,
            shape=[
                self.BATCH_SIZE,
                self.IMAGE_SIZE[0],
                self.IMAGE_SIZE[0],
                self.IMAGE_CHANNEL
            ]
        )
        self.y_ = tf.placeholder(
            tf.float32,
            shape=[
                self.BATCH_SIZE,
                self.IMAGE_SIZE[0],
                self.IMAGE_SIZE[1],
                self.CATEGORY_NUM
            ]
        )
        self.vgg = vgg16(self.imgs, self.sess, skip_layers=['fc6', 'fc7', 'fc8'])
        self.inference()


    def inference(self):
        # def do_upsample(name, input_tensor, output_channel, output_size, ksize, stride=[1, 1, 1, 1], is_pretrain=True):
        self.upsample32s = do_upsample(
            'upconv1_1',
            self.vgg.convs_output,
            self.CATEGORY_NUM,
            self.IMAGE_SIZE,
            [3, 3],
            stride=[1, 32, 32, 1]
        )

    def start_train(self):
        y = self.upsample32s
        loss = calculate_loss(logits=y, labels=self.y_, arg_index=3)
        tf.summary.scalar(
            'loss',
            loss
        )
        train_op = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate
        ).minimize(loss)

        accuracy_tensor = calculate_accuracy(logits=y, labels=self.y_, arg_index=3)
        merged = tf.summary.merge_all()
        self.sess.run(tf.global_variables_initializer())
        for i in range(self.itertator_number):
            train_image, train_annotation, flag = self.dataset.next_batch(self.BATCH_SIZE)
            train_annotation = conver2onehot(np.array(train_annotation))
            feed_dict = {
                self.imgs: train_image,
                self.y_: train_annotation
            }
            _, loss_value, accuracy_value = self.sess.run(
                [train_op, loss, accuracy_tensor],
                feed_dict=feed_dict
            )
            if (i % 20) == 0:
                print 'step is %d, loss value is %g, accuracy is %g' %(i, loss_value, accuracy_value)

if __name__ == '__main__':
    with tf.Session() as sess:
        train_obj = fcn(sess)
        train_obj.start_train()
