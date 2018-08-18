# -*- coding:utf-8

import tensorflow as tf
import os
import time

# 标签为1字节
LABEL_BYTES = 1
# 图像大小
IMAGE_SIZE = 32
# 通道数
IMAGE_DEPTH = 3
# 图片数据为 32*32*3=3072 字节
IMAGE_BYTES = IMAGE_SIZE * IMAGE_SIZE * IMAGE_DEPTH
# 标签数
NUM_CLASSES = 10


def read_cifar10(filename_list, batch_size=64, num_epochs=10):
    """

    :param data_file:
    :param batch_size:
    :return:
        - the batch image glove_wv, shape is [batch_size, image_size, image_size, 3]
        - the batch labels glove_wv, shape is [batch_data, NUM_CLAEESE]
    """
    # 文件名队列
    file_queue = tf.train.string_input_producer(filename_list, num_epochs=num_epochs, shuffle=True)
    
    # 逐个文件读取数据
    record_bytes = IMAGE_BYTES + LABEL_BYTES
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    key,value = reader.read(file_queue)

    # 将字节信息转换成张量
    record = tf.reshape(tf.decode_raw(value, out_type=tf.uint8), shape=[record_bytes])
    
    # tf.slice 的输入是 tensor
    # 需要提前了解 cifar 中数据的组织形式，获取image和label张量
    label = tf.cast(tf.slice(record, begin=[0], size=[LABEL_BYTES]), tf.int32)
    image_depth_major = tf.reshape(tf.slice(record, [1], [IMAGE_BYTES]), [IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(tf.transpose(image_depth_major,[1,2,0]), tf.float32)
    
    # 批量读取数据
    min_after_dequeue = 5000 # min_after_dequeue defines how big a buffer we will randomly sample
    num_threads = 8
    capacity = min_after_dequeue + num_threads * batch_size
    
    # 这里得到的只是一个batch的数据,怎么设置成迭代器呢?
    image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                batch_size=batch_size,
                                                capacity=capacity,
                                                num_threads=num_threads,
                                                min_after_dequeue=min_after_dequeue)
    return image_batch, label_batch



if __name__ == "__main__":
    # 文件名列表, 这里只用一个文件来测试,epoch也设置为1
    batch_size = 64
    num_epochs = 2

    filename_list = [os.path.join("dataset", "data_batch_{}".format(i)) for i in range(1, 2)]
    image_batch, label_batch = read_cifar10(filename_list, batch_size, num_epochs)
    sess = tf.Session()
    init_op = tf.group(tf.global_variables_initializer(),
                      tf.local_variables_initializer())   # 在使用协调器管理多线程之前，先对其初始化
    sess.run(init_op)
    num_step_each_epoch = (1000-1)//batch_size + 1
    for i in range(num_epochs):
        for step in range(num_step_each_epoch):
            print(step)
            print(sess.run(image_batch[0,0,0,:]))



    
    
    
    
    