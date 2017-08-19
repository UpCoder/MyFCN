import tensorflow as tf
from DataSet import DataSet
from trained_vgg16 import vgg16
from tools import do_upsample, calculate_loss, calculate_accuracy, do_conv, save_weights, load_with_skip, calucltate_loss_dispate_bg
from Tools import conver2onehot
import numpy as np

class fcn:
    def __init__(self, sess, weights=None):
        self.sess = sess
        self.CATEGORY_NUM = 151
        self.IMAGE_SIZE = [224, 224]
        self.IMAGE_CHANNEL = 3
        data_dir = '/home/give/Documents/dataset/ADEChallengeData2016'
        self.model_save_path = '/home/give/PycharmProjects/MyFCN/pretrain_vgg16/model/'
        self.dataset = DataSet(data_dir)
        self.learning_rate = 1e-3
        self.itertator_number = int(1e+5)
        self.learning_rate_decay = 0.9
        self.BATCH_SIZE = 20
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
        tf.summary.image(
            "input_image",
            self.imgs,
            max_outputs=2
        )
        self.layers_name = [
            'conv6_1',
            'conv6_2',
            'conv6_3',
            'upconv1_1',
            'pooling4_score',
            'upconv2_1',
            'pooling3_score',
            'upconv3_1',
        ]
        tf.summary.image("ground_truth", tf.cast(tf.expand_dims(tf.argmax(self.y_, dimension=3), dim=3), tf.uint8), max_outputs=2)
        if weights is None:
            vgg_weights_path = './vgg16.npy'
        else:
            vgg_weights_path = None
        self.vgg = vgg16(self.imgs, weights=vgg_weights_path, sess=self.sess, skip_layers=['fc6', 'fc7', 'fc8'])
        self.inference()
        if weights is not None and sess is not None:
            load_with_skip(weights, sess, [])


    def inference(self):
        # def do_upsample(name, input_tensor, output_channel, output_size, ksize, stride=[1, 1, 1, 1], is_pretrain=True):

        conv6_1 = do_conv('conv6_1', self.vgg.convs_output, 4096, ksize=[7, 7], dropout=True)
        conv6_2 = do_conv('conv6_2', conv6_1, 4096, ksize=[1, 1], dropout=True)
        conv6_3 = do_conv('conv6_3', conv6_2, 151, ksize=[1, 1])

        tf.summary.image('downimage', tf.cast(tf.expand_dims(tf.argmax(conv6_3, dimension=3), dim=3), tf.uint8), max_outputs=2)

        pooling4_shape = self.vgg.pooling4.get_shape().as_list()
        pooling5x2 = do_upsample(
            'upconv1_1',
            conv6_3,
            self.CATEGORY_NUM,
            [pooling4_shape[1], pooling4_shape[2]],
            [4, 4],
            stride=[1, 2, 2, 1]
        )
        score4 = do_conv('pooling4_score', self.vgg.pooling4, self.CATEGORY_NUM, ksize=[1, 1])
        pooling4_sum = tf.add(score4, pooling5x2)

        pooling3_shape = self.vgg.pooling3.get_shape().as_list()
        pooling4x2 = do_upsample(
            'upconv2_1',
            pooling4_sum,
            self.CATEGORY_NUM,
            [pooling3_shape[1], pooling3_shape[2]],
            [4, 4],
            stride=[1, 2, 2, 1]
        )
        score3 = do_conv('pooling3_score', self.vgg.pooling3, self.CATEGORY_NUM, ksize=[1, 1])
        pooling3_sum = tf.add(score3, pooling4x2)

        self.upsample_8s = do_upsample(
            'upconv3_1',
            pooling3_sum,
            self.CATEGORY_NUM,
            self.IMAGE_SIZE,
            [16, 16],
            stride=[1, 8, 8, 1]
        )

    def start_train(self, load_model=False):
        y = self.upsample_8s
        global_step = tf.Variable(0, trainable=False)
        tf.summary.image("pred_annotation", tf.cast(tf.expand_dims(tf.argmax(y, dimension=3), dim=3), tf.uint8), max_outputs=2)
        loss = calculate_loss(logits=y, labels=self.y_, arg_index=3)
        learning_rate = tf.train.exponential_decay(
            self.learning_rate,
            global_step,
            len(self.dataset.train_image) / self.BATCH_SIZE,
            self.learning_rate_decay,
            staircase=False
        )
        train_op = tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate,
        ).minimize(loss, global_step=global_step)

        accuracy_tensor = calculate_accuracy(logits=y, labels=self.y_, arg_index=3)
        merged = tf.summary.merge_all()
        self.sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter('./log', sess.graph)
        saver = tf.train.Saver()
        if load_model:
            saver.restore(sess, self.model_save_path)
        for i in range(self.itertator_number):
            train_image, train_annotation, flag = self.dataset.next_batch(self.BATCH_SIZE)
            train_annotation = conver2onehot(np.array(train_annotation))
            feed_dict = {
                self.imgs: train_image,
                self.y_: train_annotation
            }
            _, loss_value, accuracy_value, summary, step, learning_rate_value = self.sess.run(
                [train_op,  loss, accuracy_tensor, merged, global_step, learning_rate],
                feed_dict=feed_dict
            )
            if (i % 1000) == 0 and i != 0:
                saver.save(sess, self.model_save_path)
                save_layers = []
                save_layers.extend(self.vgg.layers_name)
                save_layers.extend(self.layers_name)
                save_weights('./trained_vgg16.npy', save_layers)
            if (i % 20) == 0:
                summary_writer.add_summary(summary, i)
                print 'step is %d, loss value is %g, accuracy is %g, learning_rate_value is %g' %(i, loss_value, accuracy_value, learning_rate_value)
        summary_writer.close()
if __name__ == '__main__':
    with tf.Session() as sess:
        # trainvggweights = './trained_vgg16.npy'
        trainvggweights = None
        train_obj = fcn(sess, trainvggweights)
        train_obj.start_train(load_model=False)
