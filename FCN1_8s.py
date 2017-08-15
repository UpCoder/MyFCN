# -*- coding: utf-8 -*-
# 上采样之后得到的得分，然后通过argmax来得到最后的分类结果
import tensorflow as tf
from DataSet import DataSet
import numpy as np
import gc

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

UPSAMPLE2_1_SIZE = 5
UPSAMPLE2_1_DEEP = CONV3_3_DEEP
UPSAMPLE2_2_SIZE = 5
UPSAMPLE2_2_DEEP = 150


FC1_SIZE = 1
FC1_DEEP = 1

CATEGORY_NUM = 150

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "50", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "'/home/give/Documents/dataset/ADEChallengeData2016'", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-2", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")
MAX_ITERATION = int(1e5 + 1)
DECAY_LEARNING_RATE = 0.1


def do_conv(name, weight_shape, bias_shape, input_tensor):
    with tf.variable_scope(name):
        weight = tf.get_variable(
            'weight',
            shape=weight_shape,
            initializer=tf.truncated_normal_initializer(stddev=0.02)
        )
        bias = tf.get_variable(
            'bias',
            shape=bias_shape,
            initializer=tf.constant_initializer(0.0)
        )
        conv = tf.nn.conv2d(
            input_tensor,
            weight,
            strides=[1, 1, 1, 1],
            padding='SAME',
        )
        layer = tf.nn.bias_add(conv, bias)
        return tf.nn.relu(layer)


def inference(image, keep_prob):
    with tf.variable_scope('inference'):
        layer11 = do_conv(
            'conv1_1',
            weight_shape=[
                    CONV1_1_SIZE,
                    CONV1_1_SIZE,
                    IMAGE_CHANNAL,
                    CONV1_1_DEEP
                ],
            bias_shape=[
                CONV1_1_DEEP
            ],
            input_tensor=image
        )
        print layer11.shape
        with tf.variable_scope('pooling1'):
            pooling1 = tf.nn.max_pool(
                layer11,
                strides=[1, 2, 2, 1],
                padding='SAME',
                ksize=[1, 2, 2, 1]
            )
            print pooling1.shape
        layer21 = do_conv(
            'conv2_1',
            weight_shape=[
                CONV2_1_SIZE,
                CONV2_1_SIZE,
                CONV1_1_DEEP,
                CONV2_1_DEEP
            ],
            bias_shape=[
                CONV2_1_DEEP
            ],
            input_tensor=pooling1
        )
        print layer21.shape
        with tf.variable_scope('pooling2'):
            pooling2 = tf.nn.max_pool(
                layer21,
                strides=[1, 2, 2, 1],
                padding='SAME',
                ksize=[1, 2, 2, 1]
            )
            print pooling2.shape
        layer31 = do_conv(
            'conv3_1',
            weight_shape=[
                CONV3_1_SIZE,
                CONV3_1_SIZE,
                CONV2_1_DEEP,
                CONV3_1_DEEP
            ],
            bias_shape=[
                CONV3_1_DEEP
            ],
            input_tensor=pooling2
        )
        print layer31.shape
        layer32 = do_conv(
            'conv3_2',
            weight_shape=[
                CONV3_2_SIZE,
                CONV3_2_SIZE,
                CONV3_1_DEEP,
                CONV3_2_DEEP
            ],
            bias_shape=[
                CONV3_2_DEEP
            ],
            input_tensor=layer31
        )
        print layer32.shape
        layer33 = do_conv(
            'conv3_3',
            weight_shape=[
                CONV3_3_SIZE,
                CONV3_3_SIZE,
                CONV3_2_DEEP,
                CONV3_3_DEEP
            ],
            bias_shape=[
                CONV3_3_DEEP
            ],
            input_tensor=layer32
        )
        print layer33.shape
        with tf.variable_scope('pooling3'):
            pooling3 = tf.nn.max_pool(
                layer33,
                strides=[1, 2, 2, 1],
                padding='SAME',
                ksize=[1, 2, 2, 1]
            )
            print pooling3.shape
        layer41 = do_conv(
            'conv4_1',
            weight_shape=[
                CONV4_1_SIZE,
                CONV4_1_SIZE,
                CONV3_3_DEEP,
                CONV4_1_DEEP
            ],
            bias_shape=[
                CONV4_1_DEEP
            ],
            input_tensor=pooling3
        )
        print layer41.shape
        layer42 = do_conv(
            'conv4_2',
            weight_shape=[
                CONV4_2_SIZE,
                CONV4_2_SIZE,
                CONV4_1_DEEP,
                CONV4_2_DEEP
            ],
            bias_shape=[
                CONV4_2_DEEP
            ],
            input_tensor=layer41
        )
        print layer42.shape
        layer43 = do_conv(
            'conv4_3',
            weight_shape=[
                CONV4_3_SIZE,
                CONV4_3_SIZE,
                CONV4_2_DEEP,
                CONV4_3_DEEP
            ],
            bias_shape=[
                CONV4_3_DEEP
            ],
            input_tensor=layer42
        )
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
        with tf.variable_scope('upsample2'):
            weight = tf.get_variable(
                'weight',
                shape=[
                    UPSAMPLE2_1_SIZE,
                    UPSAMPLE2_1_SIZE,
                    UPSAMPLE2_1_DEEP,
                    CONV4_3_DEEP
                ]
            )
            upsample2_1 = tf.nn.conv2d_transpose(
                layer43,
                weight,
                output_shape=[FLAGS.batch_size, IMAGE_SIZE/8, IMAGE_SIZE/8, UPSAMPLE2_1_DEEP],
                strides=[1, 1, 1, 1]
            )
            input_tensor = upsample2_1 + pooling3
            weight2 = tf.get_variable(
                'weight2',
                shape=[
                    UPSAMPLE2_2_SIZE,
                    UPSAMPLE2_2_SIZE,
                    UPSAMPLE2_2_DEEP,
                    UPSAMPLE2_1_DEEP
                ]
            )
            upsample2_2 = tf.nn.conv2d_transpose(
                input_tensor,
                weight2,
                output_shape=[FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, UPSAMPLE2_2_DEEP],
                strides=[1, 8, 8, 1]
            )
        return upsample2_2


def compare2onedimension(x):
    shape = x.get_shape().as_list()
    print shape
    print type(x)
    compared = tf.reshape(
        x, [shape[0] * shape[1] * shape[2]]
    )
    return compared


def conver2onehot(annotation):
    shape = np.shape(annotation)
    res_shape = []
    res_shape.extend(shape)
    res_shape.append(150)
    res = np.zeros(res_shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for z in range(shape[2]):
                res[i, j, z, annotation[i, j, z]-1] = 1
    return res


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
            CATEGORY_NUM
        ]
    )
    tf.summary.image(
        'input/annotation',
        tf.reshape(
            tf.cast(tf.argmax(y_, dimension=3), tf.uint8),
            [
                FLAGS.batch_size,
                IMAGE_SIZE,
                IMAGE_SIZE,
                1
            ]
        ),
        FLAGS.batch_size
    )
    y = inference(x, 0.5)
    tf.summary.image(
        'output/annotation',
        tf.reshape(
            tf.cast(tf.argmax(y, dimension=3), tf.uint8),
            [
                FLAGS.batch_size,
                IMAGE_SIZE,
                IMAGE_SIZE,
                1
            ]
        ),
        FLAGS.batch_size
    )
    print 'y shape is ', y.shape
    print 'y_ shape is ', y_.shape
    print type(y)
    print type(y_)
    print type(y)
    # y = tf.nn.softmax(y)
    # cross_entropy = -tf.reduce_mean(
    #     tf.cast(y_, tf.float32) * tf.log(y)
    # )
    # cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # loss = cross_entropy

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=y,
        labels=y_,
        name="entropy"
    ))
    # add to scalar
    tf.summary.scalar(
        'loss',
        loss
    )
    # optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    # grads = optimizer.compute_gradients(loss, var_list=tf.trainable_variables())
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        learning_rate=FLAGS.learning_rate,
        global_step=global_step,
        decay_rate=DECAY_LEARNING_RATE,
        decay_steps=len(dataset.train_image) / FLAGS.batch_size,
        staircase=True
    )
    train_op = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate
    ).minimize(
        loss=loss,
        global_step=global_step
    )
    # 计算准确率
    with tf.name_scope('accuracy'):
        correct_predict = tf.equal(
            tf.cast(tf.argmax(y, dimension=3), tf.int32),
            tf.cast(tf.argmax(y_, dimension=3), tf.int32)
        )
        accuracy_tensor = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
        tf.summary.scalar(
            'accuracy',
            accuracy_tensor
        )
    merged = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        save_path = '/home/give/PycharmProjects/MyFCN/log/FCN1_8s_decay_learning_rate'
        writer = tf.summary.FileWriter(save_path, tf.get_default_graph())
        for i in range(MAX_ITERATION):
            train_image, train_annoation, flag = dataset.next_batch(FLAGS.batch_size)
            # print 'before eye shape is ', np.shape(train_annoation)
            train_annoation_adjust = conver2onehot(np.array(train_annoation))
            # print 'after eye shape is ', np.shape(train_annoation_adjust)
            feed_dict = {
                x: train_image,
                y_: train_annoation_adjust
            }
            _, _, _, step, summary = sess.run(
                [train_op, loss, accuracy_tensor, global_step,  merged],
                feed_dict=feed_dict
            )
            writer.add_summary(summary, i)
            if (i % 20) == 0:
                loss_value, accuracy_value, learning_rate_value, global_step_value = sess.run(
                    [loss, accuracy_tensor, learning_rate, global_step],
                    feed_dict=feed_dict
                )
                print 'loss value is %g accuracy is %g learning rate is %g global_step_value is %g'\
                      % (loss_value, accuracy_value, learning_rate_value, global_step_value)
            del train_annoation_adjust
            gc.collect()
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