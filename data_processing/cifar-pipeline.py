# -*- coding:utf-8

import tensorflow as tf

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


def read_cifar10(data_file, batch_size):
    """

    :param data_file:
    :param batch_size:
    :return:
        - the batch image data, shape is [batch_size, image_size, image_size, 3]
        - the batch labels data, shape is [batch_data, NUM_CLAEESE]
    """

    # 单条数据记录大小为 1+32*32*3 = 3073
    record_bytes = LABEL_BYTES + IMAGE_BYTES

    # 创建文件名列表
    # Returns a list of files that match the given pattern(s)
    data_files = tf.gfile.Glob(data_file)  # <class 'list'>: ['cifar-10-batches-py/data_batch_1']

    # 创建文件名队列
    file_queue = tf.train.string_input_producer(data_files,   # {FIFOQueue}
                                                num_epochs=5,
                                                shuffle=True,
                                                capacity=32) # capacity 暂定
    # 创建二进制文件的 Reader 实例,按照记录大小从文件名队列中读取样例
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)  # 单个样本的数据的字节数
    _, value = reader.read(file_queue) # 表示从文件中每次读取这个多字节的信息吧?

    # 将样例拆分为标签和图片
    # tf.decode_raw() 将字符串字节转换为向量
    # r"""Reinterpret the bytes of a string as a vector of numbers
    record = tf.reshape(tf.decode_raw(value, out_type=tf.uint8),shape=[record_bytes]) # 图像的字节信息转换为向量,然后reshape=(3073,),dtype=uint8
    label = tf.cast(tf.slice(record,[0] ,[LABEL_BYTES]), tf.int32)    # Tensor("Cast:0", shape=(1,), dtype=int32)

    # 将长度为[depth*height*width] 的字符串转换为形如 [depth, height, width] 的图片张量
    depth_major = tf.reshape(tf.slice(record, [LABEL_BYTES], [IMAGE_BYTES]), [IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE]) # Tensor("Reshape_1:0", shape=(3, 32, 32), dtype=uint8)
    # 改变图片张量各维度顺序,从 [depth, height, width] 转换为 [height, width, depth]
    image = tf.cast(tf.transpose(depth_major, [1,2,0]), tf.float32) # [32,32,3],tf.float32

    # 创建样例队列
    # class RandomShuffleQueue(QueueBase)
    example_queue = tf.RandomShuffleQueue(   # tf.train.shuffle_batch 有啥区别
        capacity= 8 * batch_size,
        min_after_dequeue= 4 * batch_size,
        dtypes=[tf.float32, tf.int32],
        shapes=[[IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH], [1]]) # 随即选出的队列 <class 'list'>: [tf.float32, tf.int32]

    num_threads = 8

    # 创建样例队列的入队操作
    # enqueue 是 QueueBase 的方法
    example_enqueue_op = example_queue.enqueue([image, label]) # 入队操作
    # 将定义的 16 个线程全部添加到 queue runner中
    # def add_queue_runner: Adds a `QueueRunner` to a collection in the graph 加入默认的graph中的集合中
    # class QueueRunner(object): Holds a list of enqueue operations for a queue, each to be run in a thread.
    #    - queue: A `Queue` 一个队列,元素是样例
    #    - List of enqueue ops to run in threads later. 入队操作的list
    tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(example_queue, [example_enqueue_op]*num_threads))

    # 从样例中读取批样例图片和标签
    # Dequeues and concatenates `n` elements from this queue.
    images, labels = example_queue.dequeue_many(batch_size)  # 出队 images [32,32,32,3]  labels [32,1]
    # labels 的这一系列操作有点看不懂.....
    labels = tf.reshape(labels, [batch_size,1])
    # indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
    # labels = tf.sparse_to_dense(tf.concat(values=[0, labels], axis=1), [batch_size, NUM_CLASSES], 1.0, 0.0)
    labels = tf.one_hot(indices=labels, depth=NUM_CLASSES) # [32,1,10]
    labels = tf.squeeze(labels, axis=1)   # [32,10]

    # 展示 images 和 labels 的数据结构
    assert len(images.get_shape()) == 4
    assert images.get_shape()[0] == batch_size
    assert images.get_shape()[-1] == 3
    assert len(labels.get_shape()) == 2
    assert labels.get_shape()[0] == batch_size
    assert labels.get_shape()[1] == NUM_CLASSES

    return images, labels


if __name__ == "__main__":
    images, labels = read_cifar10("cifar-10-batches-py/data_batch_1", 5)
    print(images, labels)


