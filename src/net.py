import time
import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from PIL import Image

def myscope(is_training=True,weight_decay=1e-7,stddev=0.09):
    batch_norm_params = {
       'is_training': is_training,
      'decay': 0.9,
      'updates_collections':None
    }
    #weights_init = tf.contrib.layers.xavier_initializer()
    weights_init = tf.truncated_normal_initializer(stddev=stddev)
    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    with slim.arg_scope([slim.conv2d],
                        weights_initializer=weights_init,
                        activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm,padding='SAME'):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer)as sc:
                return sc


def inference(data,is_training=True):
    arg_scope=myscope(is_training)
    filters = 64
    with slim.arg_scope(arg_scope):
        net = data*(1./255)-0.5
        net = slim.conv2d(net,filters,[3,3],scope='conv1')

        net_r1 = slim.conv2d(net,filters,[3,3],scope='conv2-1')
        net_r2 = slim.conv2d(net_r1,filters,[3,3],scope='conv2-2',activation_fn=None)
        net = net_r2+net
        net_r1 = slim.conv2d(net,filters,[3,3],scope='conv3-1')
        net_r2 = slim.conv2d(net_r1,filters,[3,3],scope='conv3-2',activation_fn=None)
        net = net_r2+net
        net_r1 = slim.conv2d(net,filters,[3,3],scope='conv4-1')
        net_r2 = slim.conv2d(net_r1,filters,[3,3],scope='conv4-2',activation_fn=None)
        net = net_r2+net
        net_r1 = slim.conv2d(net,filters,[3,3],scope='conv5-1')
        net_r2 = slim.conv2d(net_r1,filters,[3,3],scope='conv5-2',activation_fn=None)
        net = net_r2+net
        net_r1 = slim.conv2d(net,filters,[3,3],scope='conv6-1')
        net_r2 = slim.conv2d(net_r1,filters,[3,3],scope='conv6-2',activation_fn=None)
        net = net_r2+net
        net_r1 = slim.conv2d(net,filters,[3,3],scope='conv7-1')
        net_r2 = slim.conv2d(net_r1,filters,[3,3],scope='conv7-2',activation_fn=None)
        net = net_r2+net
        net_r1 = slim.conv2d(net,filters,[3,3],scope='conv8-1')
        net_r2 = slim.conv2d(net_r1,filters,[3,3],scope='conv8-2',activation_fn=None)
        net = net_r2+net
        net_r1 = slim.conv2d(net,filters,[3,3],scope='conv9-1')
        net_r2 = slim.conv2d(net_r1,filters,[3,3],scope='conv9-2',activation_fn=None)
        net = net_r2+net
        net_r1 = slim.conv2d(net,filters,[3,3],scope='conv10-1')
        net_r2 = slim.conv2d(net_r1,filters,[3,3],scope='conv10-2',activation_fn=None)
        net = net_r2+net
        net_r1 = slim.conv2d(net,filters,[3,3],scope='conv11-1')
        net_r2 = slim.conv2d(net_r1,filters,[3,3],scope='conv11-2',activation_fn=None)
        net = net_r2+net
        net_r1 = slim.conv2d(net,filters,[3,3],scope='conv12-1')
        net_r2 = slim.conv2d(net_r1,filters,[3,3],scope='conv12-2',activation_fn=None)
        net = net_r2+net
        net_r1 = slim.conv2d(net,filters,[3,3],scope='conv13-1')
        net_r2 = slim.conv2d(net_r1,filters,[3,3],scope='conv13-2',activation_fn=None)
        net = net_r2+net
        net_r1 = slim.conv2d(net,filters,[3,3],scope='conv14-1')
        net_r2 = slim.conv2d(net_r1,filters,[3,3],scope='conv14-2',activation_fn=None)
        net = net_r2+net
        net_r1 = slim.conv2d(net,filters,[3,3],scope='conv15-1')
        net_r2 = slim.conv2d(net_r1,filters,[3,3],scope='conv15-2',activation_fn=None)
        net = net_r2+net
        net_r1 = slim.conv2d(net,filters,[3,3],scope='conv16-1')
        net_r2 = slim.conv2d(net_r1,filters,[3,3],scope='conv16-2',activation_fn=None)
        net = net_r2+net
        net_r1 = slim.conv2d(net,filters,[3,3],scope='conv17-1')
        net_r2 = slim.conv2d(net_r1,filters,[3,3],scope='conv17-2',activation_fn=None)
        net = net_r2+net

        net = slim.conv2d(net,1,[3,3],activation_fn=None,normalizer_fn=None,scope='conv20')
        return net
