from datetime import datetime
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops
import os
import random

from PIL import Image
from math import log10
from data_tools import cv_resize,read_img
from decode_tools import decode_from_tfrecords 
from net import inference

max_iters = 10000000
batch_size = 64
image_height = 48
image_width = 48

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(is_ft=False):
    with tf.Graph().as_default():
        with tf.variable_scope("model") as scope:
            root_path = "tfData/part"
            train_queue = list()
            for part_index in range(1,10):
                train_queue.append(root_path+str(part_index)+'.tfrecords')
            images,label = decode_from_tfrecords(train_queue,batch_size,image_height,image_width)
            images = tf.py_func(cv_resize,[images,image_height,image_width],tf.float32)
            images = tf.reshape(images,[batch_size,image_height,image_width,1])
            logits = inference(images)+images
            logits = tf.clip_by_value(logits,0,255)
            loss = tf.losses.mean_squared_error(logits,label)
            reg_loss = tf.add_n(tf.losses.get_regularization_losses())
            total_loss = loss
            
            opt = tf.train.AdamOptimizer(1e-4)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            train_op = slim.learning.create_train_op(total_loss, opt, global_step=global_step)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            if update_ops:
                updates = tf.group(*update_ops)
                total_loss = control_flow_ops.with_dependencies([updates], total_loss)
        
            saver = tf.train.Saver(tf.all_variables())
            init = tf.initialize_all_variables()

        
            sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
            sess.run(init)

        
            tf.train.start_queue_runners(sess=sess)

            if is_ft:
                model_file=tf.train.latest_checkpoint('./model')
                saver.restore(sess,model_file)
            tf.logging.set_verbosity(tf.logging.INFO)    
            loss_cnt = 0.0
            
            for step in range(max_iters):
                _, loss_value,l = sess.run([train_op, loss,logits])
                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                if step%10==0:
                    print l[0]
               
                loss_cnt+=loss_value
                if step % 100 == 0:
                    format_str = ('%s: step %d, loss = %.2f')
                    print(format_str % (datetime.now(), step, loss_cnt/10.0))
                    loss_cnt = 0.0
                if step % 500 == 0 or (step + 1) == max_iters:
                    checkpoint_path = os.path.join('../model', 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
             

train()
