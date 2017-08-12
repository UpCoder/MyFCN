# -*- coding: utf-8 -*-
# 上采样之后得到的得分，然后通过卷积来得到最后的预测结果
import tensorflow as tf
from DataSet import DataSet
import numpy as np
IMAGE_SIZE = 224
IMAGE_CHANNAL = 3

CONV1_1_SIZE = 3
CONV1_1_DEEP = 96

CONV2_1_SIZE = 3
CONV2_1_DEEP = 256

CONV3_1_SIZE = 3
CONV3_1_DEEP = 384
CONV3_2_SIZE = 3
CONV3_2_DEEP = 384
CONV3_3_SIZE = 3
CONV3_3_DEEP = 256

CONV4_1_SIZE = 3
CONV4_1_DEEP = 4096
CONV4_2_SIZE = 3
CONV4_2_DEEP = 4096
CONV4_3_SIZE = 3
CONV4_3_DEEP = 150

UPSAMPLE1_1_SIZE = 5
UPSAMPLE1_1_DEEP = 150


FC1_SIZE = 1
FC1_DEEP = 1
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "50", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "'/home/give/Documents/dataset/ADEChallengeData2016'", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")
MAX_ITERATION = int(1e5 + 1)
def inference(image, keep_prob):
    with tf.variable_scope('inference'):
        with tf.variable_scope('conv1_1'):
            print image.shape
            weight = tf.get_variable(
                'weight',
                shape=[
                    CONV1_1_SIZE,
                    CONV1_1_SIZE,
                    IMAGE_CHANNAL,
                    CONV1_1_DEEP
                ],
                initializer=tf.truncated_normal_initializer(stddev=0.02)
            )
            bias = tf.get_variable(
                name='bias',
                shape=[
                    CONV1_1_DEEP
                ],
                initializer=tf.constant_initializer(value=0.0)
            )
            conv11 = tf.nn.conv2d(
                image,
                weight,
                strides=[1, 1, 1, 1],
                padding='SAME'
            )
            layer11 = tf.nn.bias_add(conv11, bias)
            layer11 = tf.nn.relu(layer11)
            print layer11.shape
        with tf.variable_scope('pooling1'):
            pooling1 = tf.nn.max_pool(
                layer11,
                strides=[1, 2, 2, 1],
                padding='SAME',
                ksize=[1, 2, 2, 1]
            )
            print pooling1.shape
        with tf.variable_scope('conv2_1'):
            weight = tf.get_variable(
                'weight',
                shape=[
                    CONV2_1_SIZE,
                    CONV2_1_SIZE,
                    CONV1_1_DEEP,
                    CONV2_1_DEEP
                ],
                initializer=tf.truncated_normal_initializer(stddev=0.02)
            )
            bias = tf.get_variable(
                'bias',
                shape=[
                    CONV2_1_DEEP
                ],
                initializer=tf.constant_initializer(value=0.0)
            )
            conv21 = tf.nn.conv2d(
                pooling1,
                weight,
                strides=[1, 1, 1, 1],
                padding='SAME'
            )
            layer21 = tf.nn.bias_add(conv21, bias)
            layer21 = tf.nn.relu(layer21)
            print layer21.shape
        with tf.variable_scope('pooling2'):
            pooling2 = tf.nn.max_pool(
                layer21,
                strides=[1, 2, 2, 1],
                padding='SAME',
                ksize=[1, 2, 2, 1]
            )
            print pooling2.shape
        with tf.variable_scope('conv3_1'):
            weight = tf.get_variable(
                'weight',
                shape=[
                    CONV3_1_SIZE,
                    CONV3_1_SIZE,
                    CONV2_1_DEEP,
                    CONV3_1_DEEP
                ],
                initializer=tf.truncated_normal_initializer(stddev=0.02)
            )
            bias = tf.get_variable(
                'bias',
                shape=[
                    CONV3_1_DEEP
                ],
                initializer=tf.constant_initializer(value=0.0)
            )
            conv31 = tf.nn.conv2d(
                pooling2,
                weight,
                strides=[1, 1, 1, 1],
                padding='SAME'
            )
            layer31 = tf.nn.bias_add(conv31, bias)
            layer31 = tf.nn.relu(layer31)
            print layer31.shape
        with tf.variable_scope('conv3_2'):
            weight = tf.get_variable(
                'weight',
                shape=[
                    CONV3_2_SIZE,
                    CONV3_2_SIZE,
                    CONV3_1_DEEP,
                    CONV3_2_DEEP
                ],
                initializer=tf.truncated_normal_initializer(stddev=0.02)
            )
            bias = tf.get_variable(
                'bias',
                shape=[
                    CONV3_2_DEEP
                ],
                initializer=tf.constant_initializer(value=0.0)
            )
            conv32 = tf.nn.conv2d(
                layer31,
                weight,
                strides=[1, 1, 1, 1],
                padding='SAME'
            )
            layer32 = tf.nn.bias_add(conv32, bias)
            layer32 = tf.nn.relu(layer32)
            print layer32.shape
        with tf.variable_scope('conv3_3'):
            weight = tf.get_variable(
                'weight',
                shape=[
                    CONV3_3_SIZE,
                    CONV3_3_SIZE,
                    CONV3_2_DEEP,
                    CONV3_3_DEEP
                ],
                initializer=tf.truncated_normal_initializer(stddev=0.02)
            )
            bias = tf.get_variable(
                'bias',
                shape=[
                    CONV3_3_DEEP
                ],
                initializer=tf.constant_initializer(value=0.0)
            )
            conv33 = tf.nn.conv2d(
                layer32,
                weight,
                strides=[1, 1, 1, 1],
                padding='SAME'
            )
            layer33 = tf.nn.bias_add(conv33, bias)
            layer33 = tf.nn.relu(layer33)
            print layer33.shape
        with tf.variable_scope('pooling3'):
            pooling3 = tf.nn.max_pool(
                layer33,
                strides=[1, 2, 2, 1],
                padding='SAME',
                ksize=[1, 2, 2, 1]
            )
            print pooling3.shape
        with tf.variable_scope('conv4_1'):
            weight = tf.get_variable(
                'weight',
                shape=[
                    CONV4_1_SIZE,
                    CONV4_1_SIZE,
                    CONV3_3_DEEP,
                    CONV4_1_DEEP
                ],
                initializer=tf.truncated_normal_initializer(stddev=0.02)
            )
            bias = tf.get_variable(
                'bias',
                shape=[
                    CONV4_1_DEEP
                ],
                initializer=tf.constant_initializer(value=0.0)
            )
            conv41 = tf.nn.conv2d(
                pooling3,
                weight,
                strides=[1, 1, 1, 1],
                padding='SAME'
            )
            layer41 = tf.nn.bias_add(conv41, bias)
            layer41 = tf.nn.relu(layer41)
            print layer41.shape
        with tf.variable_scope('conv4_2'):
            weight = tf.get_variable(
                'weight',
                shape=[
                    CONV4_2_SIZE,
                    CONV4_2_SIZE,
                    CONV4_1_DEEP,
                    CONV4_2_DEEP
                ],
                initializer=tf.truncated_normal_initializer(stddev=0.02)
            )
            bias = tf.get_variable(
                'bias',
                shape=[
                    CONV4_2_DEEP
                ],
                initializer=tf.constant_initializer(value=0.0)
            )
            conv42 = tf.nn.conv2d(
                layer41,
                weight,
                strides=[1, 1, 1, 1],
                padding='SAME'
            )
            layer42 = tf.nn.bias_add(conv42, bias)
            layer42 = tf.nn.relu(layer42)
            print layer42.shape
        with tf.variable_scope('conv4-3'):
            weight = tf.get_variable(
                'weight_conv4_3',
                shape=[
                    CONV4_3_SIZE,
                    CONV4_3_SIZE,
                    CONV4_2_DEEP,
                    CONV4_3_DEEP
                ],
                initializer=tf.truncated_normal_initializer(stddev=0.02)
            )
            bias = tf.get_variable(
                'bias_conv4_3',
                shape=[
                    CONV4_3_DEEP
                ],
                initializer=tf.constant_initializer(value=0.0)
            )
            conv43 = tf.nn.conv2d(
                layer42,
                weight,
                strides=[1, 1, 1, 1],
                padding='SAME'
            )
            layer43 = tf.nn.bias_add(conv43, bias)
            layer43 = tf.nn.relu(layer43)
            print layer43.shape
        with tf.variable_scope('upsampe1'):
            weight = tf.get_variable(
                'weight',
                shape=[
                    UPSAMPLE1_1_SIZE,
                    UPSAMPLE1_1_SIZE,
                    UPSAMPLE1_1_DEEP,
                    CONV4_3_DEEP
                ],
                initializer=tf.truncated_normal_initializer(stddev=0.02)
            )
            upsample1 = tf.nn.conv2d_transpose(
                layer43,
                weight,
                output_shape=[FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, UPSAMPLE1_1_DEEP],
                strides=[1, 8, 8, 1]
            )
            print 'upsample1 shape is ', upsample1.shape
        # argmax_upsample = tf.argmax(upsample1, 3)
        # argmax_result = tf.reshape(
        #     argmax_upsample,
        #     [
        #         FLAGS.batch_size,
        #         IMAGE_SIZE,
        #         IMAGE_SIZE,
        #         1
        #     ]
        # )
        # argmax_result = tf.cast(argmax_result, tf.float32)
        # print 'argmax_result shape is ', argmax_result.shape
        # print argmax_result
        # return argmax_result
        with tf.variable_scope('fc1'):
            weight = tf.get_variable(
                'weight',
                shape=[
                    FC1_SIZE,
                    FC1_SIZE,
                    UPSAMPLE1_1_DEEP,
                    FC1_DEEP
                ],
                initializer=tf.truncated_normal_initializer(stddev=0.02)
            )
            bias = tf.get_variable(
                'bias',
                shape=[
                    FC1_DEEP
                ],
                initializer=tf.constant_initializer(value=0.0)
            )
            fc1 = tf.nn.conv2d(
                upsample1,
                weight,
                strides=[1, 1, 1, 1],
                padding='SAME'
            )
            layer51 = tf.nn.bias_add(fc1, bias)
            layer51 = tf.nn.relu(layer51)
            print 'after fc shape is ', layer51.shape
        return layer51


def compare2onedimension(x):
    shape = x.get_shape().as_list()
    print shape
    print type(x)
    compared = tf.reshape(
        x, [shape[0] * shape[1] * shape[2]]
    )
    return compared


def train(dataset):
    x = tf.placeholder(
        tf.float32,
        [
            FLAGS.batch_size,
            IMAGE_SIZE,
            IMAGE_SIZE,
            IMAGE_CHANNAL
        ],
        name='input-x'
    )
    tf.summary.image(
        'input/image',
        x,
        FLAGS.batch_size
    )
    y_ = tf.placeholder(
        tf.float32,
        [
            FLAGS.batch_size,
            IMAGE_SIZE,
            IMAGE_SIZE,
            1
        ]
    )
    tf.summary.image(
        'input/annotation',
        y_,
        FLAGS.batch_size
    )
    y = inference(x, 0.5)
    tf.summary.image(
        'output/annotation',
        y,
        FLAGS.batch_size
    )
    print 'y shape is ', y.shape
    print 'y_ shape is ', y_.shape
    print type(y)
    print type(y_)
    cross_entropy = -tf.reduce_mean(
        tf.cast(y_, tf.float32) * tf.log(tf.clip_by_value(
            y, 1e-10, 1.0
        ))
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean
    # loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,
    #                                                                       labels=y_,
    #                                                                       name="entropy")))
    # add to scalar
    tf.summary.scalar(
        'loss',
        loss
    )
    # optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    # grads = optimizer.compute_gradients(loss, var_list=tf.trainable_variables())
    train_op = tf.train.GradientDescentOptimizer(
        learning_rate=FLAGS.learning_rate
    ).minimize(
        loss=loss
    )
    # 计算准确率
    with tf.name_scope('accuracy'):
        correct_predict = tf.equal(
            tf.cast(y, tf.int32),
            tf.cast(y_, tf.int32)
        )
        accuracy_tensor = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
        tf.summary.scalar(
            'accuracy',
            accuracy_tensor
        )
    merged = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        save_path = '/home/give/PycharmProjects/MyFCN/log'
        writer = tf.summary.FileWriter(save_path, tf.get_default_graph())
        for i in range(MAX_ITERATION):
            train_image, train_annoation, flag = dataset.next_batch(FLAGS.batch_size)
            train_annoation = np.reshape(
                train_annoation,
                [
                    FLAGS.batch_size,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                    1
                ]
            )
            feed_dict = {
                x: train_image,
                y_: train_annoation
            }
            _, _, _, summary = sess.run(
                [train_op, loss, accuracy_tensor, merged],
                feed_dict=feed_dict
            )
            writer.add_summary(summary, i)
            if (i % 20) == 0:
                loss_value, accuracy_value = sess.run(
                    [loss, accuracy_tensor],
                    feed_dict=feed_dict
                )
                print 'loss value is %g accuracy is %g' % (loss_value, accuracy_value)
        writer.close()
if __name__ == '__main__':
    # input_image = tf.placeholder(
    #     tf.float32,
    #     [
    #         10,
    #         IMAGE_SIZE,
    #         IMAGE_SIZE,
    #         IMAGE_CHANNAL
    #     ]
    # )
    # print 'input image shape is ', tf.shape(input_image), input_image.shape
    # output_image = inference(input_image, 0.5)
    # output_image.shape
    dataset = DataSet(FLAGS.data_dir)
    train(dataset)