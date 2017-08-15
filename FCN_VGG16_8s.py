# -*- coding: utf-8 -*-
# 上采样之后得到的得分，然后通过argmax来得到最后的分类结果
import tensorflow as tf
from DataSet import DataSet
import numpy as np
import gc

IMAGE_SIZE = 224
IMAGE_CHANNAL = 3

CONV1_1_SIZE = 3
CONV1_1_DEEP = 64
CONV1_2_SIZE = 3
CONV1_2_DEEP = 64

CONV2_1_SIZE = 3
CONV2_1_DEEP = 128
CONV2_2_SIZE = 3
CONV2_2_DEEP = 128

CONV3_1_SIZE = 3
CONV3_1_DEEP = 256
CONV3_2_SIZE = 3
CONV3_2_DEEP = 256
CONV3_3_SIZE = 1
CONV3_3_DEEP = 256

CONV4_1_SIZE = 3
CONV4_1_DEEP = 512
CONV4_2_SIZE = 3
CONV4_2_DEEP = 512
CONV4_3_SIZE = 1
CONV4_3_DEEP = 512

CONV5_1_SIZE = 3
CONV5_1_DEEP = 512
CONV5_2_SIZE = 3
CONV5_2_DEEP = 512
CONV5_3_SIZE = 1
CONV5_3_DEEP = 512

UPSAMPLE1_1_SIZE = 4
UPSAMPLE1_1_DEEP = 150

UPSAMPLE2_1_SIZE = 4
UPSAMPLE2_1_DEEP = CONV4_3_DEEP
UPSAMPLE2_2_SIZE = 4
UPSAMPLE2_2_DEEP = 150

UPSAMPLE3_1_SIZE = 16
UPSAMPLE3_1_DEEP = CONV3_3_DEEP
UPSAMPLE3_2_SIZE = 16
UPSAMPLE3_2_DEEP = 150


FC1_SIZE = 1
FC1_DEEP = 1

CATEGORY_NUM = 150

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "50", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "'/home/give/Documents/dataset/ADEChallengeData2016'", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-1", "Learning rate for Adam Optimizer")
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
        layer12 = do_conv(
            'conv1_2',
            weight_shape=[
                CONV1_2_SIZE,
                CONV1_2_SIZE,
                CONV1_1_DEEP,
                CONV1_2_DEEP
            ],
            bias_shape=[
                CONV1_2_DEEP
            ],
            input_tensor=layer11
        )
        print layer12.shape


        with tf.variable_scope('pooling1'):
            pooling1 = tf.nn.max_pool(
                layer12,
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
                CONV1_2_DEEP,
                CONV2_1_DEEP
            ],
            bias_shape=[
                CONV2_1_DEEP
            ],
            input_tensor=pooling1
        )
        layer22 = do_conv(
            'conv2_2',
            weight_shape=[
                CONV2_2_SIZE,
                CONV2_2_SIZE,
                CONV2_1_DEEP,
                CONV2_2_DEEP
            ],
            bias_shape=[
                CONV2_2_DEEP
            ],
            input_tensor=layer21
        )


        with tf.variable_scope('pooling2'):
            pooling2 = tf.nn.max_pool(
                layer22,
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
                CONV2_2_DEEP,
                CONV3_1_DEEP
            ],
            bias_shape=[
                CONV3_1_DEEP
            ],
            input_tensor=pooling2
        )
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
        with tf.variable_scope('pooling4'):
            pooling4 = tf.nn.max_pool(
                layer43,
                strides=[1, 2, 2, 1],
                padding='SAME',
                ksize=[1, 2, 2, 1]
            )
            print pooling4.shape
        layer51 = do_conv(
            'conv5_1',
            weight_shape=[
                CONV5_1_SIZE,
                CONV5_1_SIZE,
                CONV4_3_DEEP,
                CONV5_1_DEEP
            ],
            bias_shape=[
                CONV5_1_DEEP
            ],
            input_tensor=pooling4
        )
        layer52 = do_conv(
            'conv5_2',
            weight_shape=[
                CONV5_2_SIZE,
                CONV5_2_SIZE,
                CONV5_1_DEEP,
                CONV5_2_DEEP
            ],
            bias_shape=[
                CONV5_2_DEEP
            ],
            input_tensor=layer51
        )
        layer53 = do_conv(
            'conv5_3',
            weight_shape=[
                CONV5_3_SIZE,
                CONV5_3_SIZE,
                CONV5_2_DEEP,
                CONV5_3_DEEP
            ],
            bias_shape=[
                CONV5_3_DEEP
            ],
            input_tensor=layer52
        )
        with tf.variable_scope('pooling5'):
            pooling5 = tf.nn.max_pool(
                layer53,
                strides=[1, 2, 2, 1],
                padding='SAME',
                ksize=[1, 2, 2, 1]
            )
            print pooling5.shape
        with tf.variable_scope('upsample_32s'):
            weight = tf.get_variable(
                'weight',
                shape=[
                    UPSAMPLE1_1_SIZE,
                    UPSAMPLE1_1_SIZE,
                    UPSAMPLE1_1_DEEP,
                    CONV5_3_DEEP
                ],
                initializer=tf.truncated_normal_initializer(stddev=0.02)
            )
            upsample_32s = tf.nn.conv2d_transpose(
                pooling5,
                weight,
                output_shape=[FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, UPSAMPLE1_1_DEEP],
                strides=[1, 32, 32, 1]
            )
            print 'upsampe32s shape is ', upsample_32s.shape
        global upsample_16s_input
        upsample_16s_input = None
        with tf.variable_scope('upsample_16s'):
            weight = tf.get_variable(
                'upsample_16s_weight',
                shape=[
                    UPSAMPLE2_1_SIZE,
                    UPSAMPLE2_1_SIZE,
                    UPSAMPLE2_1_DEEP,
                    CONV5_3_DEEP
                ]
            )
            upsample2x = tf.nn.conv2d_transpose(
                pooling5,
                weight,
                output_shape=[FLAGS.batch_size, IMAGE_SIZE/16, IMAGE_SIZE/16, UPSAMPLE2_1_DEEP],
                strides=[1, 2, 2, 1]
            )
            upsample_16s_input = upsample2x + pooling4
            weight2 = tf.get_variable(
                'upsample_16s_weight2',
                shape=[
                    UPSAMPLE2_2_SIZE,
                    UPSAMPLE2_2_SIZE,
                    UPSAMPLE2_2_DEEP,
                    UPSAMPLE2_1_DEEP
                ],
                initializer=tf.truncated_normal_initializer(stddev=0.02)
            )
            upsample_16s = tf.nn.conv2d_transpose(
                upsample_16s_input,
                weight2,
                output_shape=[FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, UPSAMPLE2_2_DEEP],
                strides=[1, 16, 16, 1]
            )
            print 'upsample_16s shape is ', upsample_16s.shape
        with tf.variable_scope('upsample_8s'):
            if upsample_16s_input is None:
                print 'upsample 16s input is None'
                return upsample_16s
            weight = tf.get_variable(
                'weight1',
                shape=[
                    UPSAMPLE3_1_SIZE,
                    UPSAMPLE3_1_SIZE,
                    UPSAMPLE3_1_DEEP,
                    CONV4_3_DEEP
                ]
            )
            print 'upsample 16s input shape is ', upsample_16s_input.shape
            print 'weight shape is ', weight.shape
            upsample2x = tf.nn.conv2d_transpose(
                upsample_16s_input,
                weight,
                output_shape=[FLAGS.batch_size, IMAGE_SIZE / 8, IMAGE_SIZE / 8, UPSAMPLE3_1_DEEP],
                strides=[1, 2, 2, 1]
            )
            upsample_8s_input = upsample2x + pooling3
            weight2 = tf.get_variable(
                'weight2',
                shape=[
                    UPSAMPLE3_2_SIZE,
                    UPSAMPLE3_2_SIZE,
                    UPSAMPLE3_2_DEEP,
                    UPSAMPLE3_1_DEEP
                ],
                initializer=tf.truncated_normal_initializer(stddev=0.02)
            )
            upsample_8s = tf.nn.conv2d_transpose(
                upsample_8s_input,
                weight2,
                output_shape=[FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, UPSAMPLE3_2_DEEP],
                strides=[1, 8, 8, 1]
            )
        return upsample_8s


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
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        learning_rate=FLAGS.learning_rate,
        global_step=global_step,
        decay_steps=1,
        decay_rate=DECAY_LEARNING_RATE,
        staircase=True
    )

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
    train_op = tf.train.AdamOptimizer(
        learning_rate=learning_rate
    ).minimize(loss=loss)
    # train_op = tf.train.GradientDescentOptimizer(
    #     learning_rate=FLAGS.learning_rate
    # ).minimize(
    #     loss=loss
    # )
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
        save_path = '/home/give/PycharmProjects/MyFCN/log/FCN_VGG'
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
            _, _, _, summary = sess.run(
                [train_op, loss, accuracy_tensor, merged],
                feed_dict=feed_dict
            )
            writer.add_summary(summary, i)
            if (i % 20) == 0:
                loss_value, accuracy_value, learning_rate_value = sess.run(
                    [loss, accuracy_tensor, learning_rate],
                    feed_dict=feed_dict
                )
                print 'loss value is %g accuracy is %g learning rate is %g ' \
                      % (loss_value, accuracy_value, learning_rate_value)
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