import tensorflow as tf
from data_tools import  data_aug_rot,data_aug_flip_ud,data_aug_flip_lr 

def decode_from_tfrecords(filename, batch_size,image_height,image_width):
    filename_queue = tf.train.string_input_producer(filename)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label' : tf.FixedLenFeature([],tf.string),
                                           'img_raw' : tf.FixedLenFeature([], tf.string)
                                       })

    image = tf.decode_raw(features['img_raw'],tf.uint8)
    label = tf.decode_raw(features['label'],tf.uint8)
    image,label = data_aug_rot(image,label)
    image,label = data_aug_flip_ud(image,label)
    image,label = data_aug_flip_lr(image,label)
    image = tf.reshape(image,[int(image_height/2),int(image_width/2),1])
    label = tf.reshape(label,[image_height,image_width,1])
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32) 
    min_after_dequeue = 1000
    capacity = min_after_dequeue+3*batch_size
    image,label = tf.train.shuffle_batch([image,label],
                             batch_size=batch_size,
                             num_threads=3,
                             capacity=capacity,
                             min_after_dequeue=1000)
    return image,label
