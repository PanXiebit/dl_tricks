import tensorflow as tf
import os

def read_and_decode_single_example(filename_queue):
    # 定义一个空的类对象，类似于c语言里面的结构体定义
    class Image:
        pass

    image = Image()
    image.height = 32
    image.width = 32
    image.depth = 3
    label_bytes = 1

    Bytes_to_read = label_bytes + image.height * image.width * 3
    # A Reader that outputs fixed-length records from a file
    reader = tf.FixedLengthRecordReader(record_bytes=Bytes_to_read)
    # Returns the next record (key, value) pair produced by a reader, key 和value都是字符串类型的tensor
    # Will dequeue a work unit from queue if necessary (e.g. when the
    # Reader needs to start reading from a new file since it has
    # finished with the previous file).
    _, value = reader.read(filename_queue)
    # Reinterpret the bytes of a string as a vector of numbers,每一个数值占用一个字节,在[0, 255]区间内，因此out_type要取uint8类型
    value = tf.decode_raw(bytes=value, out_type=tf.uint8)
    # Extracts a slice from a tensor， value中包含了label和feature，故要对向量类型tensor进行'parse'操作
    image.label = tf.slice(input_=value, begin=[0], size=[1])
    value = tf.slice(input_=value, begin=[1], size=[-1])
    value = tf.reshape(value, (image.depth, image.height, image.width))
    transposed_value = tf.transpose(value, perm=[1,2,0])
    image.mat = transposed_value
    return image

data_dir = "dataset/"
filenames =[os.path.join(data_dir, 'data_batch_1')]
# Output strings (e.g. filenames) to a queue for an input pipeline
filename_queue = tf.train.string_input_producer(string_tensor=filenames)
# returns symbolic label and image
img_obj = read_and_decode_single_example(filename_queue)
Label = img_obj.label
Image = img_obj.mat
sess = tf.Session()
# 初始化tensorflow图中的所有状态，如待读取的下一个记录tfrecord的位置，variables等
init = tf.initialize_all_variables()
sess.run(init)
tf.train.start_queue_runners(sess=sess)
# grab examples back.
# first example from file
#label_val_1, image_val_1 = sess.run([Label, Image])
#print(label_val_1, image_val_1)
# second example from file
#label_val_2, image_val_2 = sess.run([Label, Image])
#print(label_val_2, image_val_2)

batch_size = 10
min_samples_in_queue = 100
image_batch, label_batch = tf.train.shuffle_batch(tensors=[Image, Label],
                                                  batch_size=batch_size,
                                                  num_threads=8,
                                                  min_after_dequeue=min_samples_in_queue,
                                                  capacity=min_samples_in_queue+3*batch_size)

print(image_batch, label_batch)
print(image_batch, label_batch)
print(image_batch, label_batch)
