# -*- coding: utf-8 -*-
import tensorflow as tf
from trained_vgg16 import vgg16
from DataSet import DataSet
import numpy as np
import gc
from tools import calculate_loss, calculate_accuracy, save_weights

class train:
    def __init__(self):
        data_dir = '/home/give/Documents/dataset/ADEChallengeData2016'
        dataset = DataSet(data_dir)
        print np.shape(dataset.train_image)
        self.dataset = dataset
        self.BATCH_SIZE = 128
        self.sess = tf.Session()
        imgs = tf.placeholder(
            tf.float32,
            shape=[
                None,
                224,
                224,
                3
            ]
        )
        self.vgg = vgg16(imgs, '/home/give/PycharmProjects/FaceDetection/fine_tuning_vgg16/vgg16.npy', self.sess)
        


    def start_train(self):
        y_ = tf.placeholder(
            tf.float32,
            [
                None,
                2
            ]
        )
        y = self.vgg.y
        loss = calculate_loss(logits=y, labels=y_)
        tf.summary.scalar(
            'loss',
            loss
        )
        train_op = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate
        ).minimize(loss)
        # 计算准确率
        accuracy_tensor = calculate_accuracy(logits=y, labels=y_)
        merged = tf.summary.merge_all()
        max_accuracy = 0.0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            log_path = './log/train'
            val_log_path = './log/val'
            writer = tf.summary.FileWriter(log_path, tf.get_default_graph())
            val_writer = tf.summary.FileWriter(val_log_path, tf.get_default_graph())
            for i in range(self.iterator_number):
                train_images, labels, scores = self.dataset.next_batch(self.BATCH_SZIE, self.BATCH_DISTRIBUTION)
                feed_dict = {
                    self.vgg.imgs: train_images,
                    y_: labels
                }
                _, loss_value, accuracy_value, summary, y_value = sess.run(
                    [train_op, loss, accuracy_tensor, merged, y],
                    feed_dict=feed_dict
                )
                writer.add_summary(summary, i)
                if (i % 40) == 0 and i != 0:
                    val_images, labels, scores = self.val_dataset.next_batch(self.BATCH_SZIE, self.BATCH_DISTRIBUTION)
                    feed_dict = {
                        self.vgg.imgs: val_images,
                        y_: labels
                    }
                    val_loss, val_accuracy, summary = sess.run(
                        [loss, accuracy_tensor, merged],
                        feed_dict=feed_dict
                    )
                    if val_accuracy > 0.9:
                        print 'will save, accuracy is %g' % val_accuracy
                        save_weights(
                            '/home/give/PycharmProjects/FaceDetection/fine_tuning_vgg16/vgg16_trained.npy',
                            self.vgg.layers_name
                        )
                    max_accuracy = max(max_accuracy, val_accuracy)
                    val_writer.add_summary(summary, i)
                    print '-'*15, 'val loss is %g, val accuracy is %g' % (val_loss, val_accuracy), '-'*15
                if (i % 20) == 0:

                    print 'predict the number of positive number is ', np.sum(np.argmax(y_value, 1))
                    print 'loss value is %g accuracy is %g' \
                          % (loss_value, accuracy_value)
                del train_images, labels, scores
                gc.collect()
        writer.close()
        val_writer.close()

if __name__ == '__main__':
    my_train = train()
    my_train.start_train()