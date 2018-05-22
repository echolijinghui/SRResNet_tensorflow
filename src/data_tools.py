import numpy as np
import tensorflow as tf
import cv2
import random
import os
from tensorflow.python.ops import control_flow_ops
from PIL import Image



def cv_resize(img,w,h,train=True):
    img = np.squeeze(img)
    res = list()
    if train == False:
        img = np.expand_dims(img,0)
    batch = img.shape[0]
    for index in range(batch):
        tmp = img[index,:,:]
        if train == True:
            res.append(cv2.resize(tmp,(h,w),interpolation = cv2.INTER_CUBIC))
        else:
            res.append(cv2.resize(tmp,(h,w),interpolation = cv2.INTER_CUBIC))
    res = np.array(res)
    return res


def data_aug_rot(image,label):
    image = tf.reshape(image,[24,24,1])
    label = tf.reshape(label,[48,48,1])
    img_concat = tf.concat([image,image,image],axis=2)
    label_concat = tf.concat([label,label,label],axis=2)
    index = random.randint(1,4)
    img_concat = tf.image.rot90(img_concat,index)
    label_concat = tf.image.rot90(label_concat,index)
    image = img_concat[:,:,0]
    label = label_concat[:,:,0]
    return image,label

def data_aug_flip_lr(image,label):
    image = tf.reshape(image,[24,24,1])
    label = tf.reshape(label,[48,48,1])
    img_concat = tf.concat([image,image,image],axis=2)
    label_concat = tf.concat([label,label,label],axis=2)
    index = random.randint(1,2)
    if index == 1:
        image = tf.image.flip_left_right(image)
        label = tf.image.flip_left_right(label)
    image = img_concat[:,:,0]
    label = label_concat[:,:,0]
    return image,label

def data_aug_flip_ud(image,label):
    image = tf.reshape(image,[24,24,1])
    label = tf.reshape(label,[48,48,1])
    img_concat = tf.concat([image,image,image],axis=2)
    label_concat = tf.concat([label,label,label],axis=2)
    index = random.randint(1,2)
    if index == 1:
        image = tf.image.flip_up_down(image)
        label = tf.image.flip_up_down(label)
    image = img_concat[:,:,0]
    label = label_concat[:,:,0]
    return image,label



def random_crop_and_pad_image_and_labels(image, labels, size):
    combined = tf.concat([image, labels], axis=2)
    image_shape = tf.shape(image)
    combined_pad = tf.image.pad_to_bounding_box(
      combined, 0, 0,
      tf.maximum(size[0], image_shape[0]),
      tf.maximum(size[1], image_shape[1]))
    last_label_dim = tf.shape(labels)[-1]
    last_image_dim = tf.shape(image)[-1]
    combined_crop = tf.random_crop(
            combined_pad,
            size=tf.concat([size, [last_label_dim + last_image_dim]],
                     axis=0))
    return combined_crop[:, :, :last_image_dim],combined_crop[:, :, last_image_dim:]




def read_img(index):
    path = 'test_data/Set5_Y/img_2/' 
    images_name= os.listdir(path)
   # print images_name
    image_name =sorted(images_name)[index]
    print image_name  
    img_path = path+ image_name
    path_label = 'test_data/Set5_Y/img_raw/' 
    label_path = path_label+ image_name.replace('_2.bmp','.bmp')
    print img_path, label_path
    img = Image.open(img_path)
    label = Image.open(label_path)
    img = Image.open(img_path)
    label = Image.open(label_path)
    img = np.array(img)
    label = np.array(label)
    img = np.expand_dims(img,0)
    label = np.expand_dims(label,0)
    img = np.expand_dims(img,-1)
    label = np.expand_dims(label,-1)
    return img,label

def read_img_b(index):
    path = 'test_data/Set5_Bicubic/img_2/' 
    bicubics_name= os.listdir(path)
    bicubic_name =sorted(bicubics_name)[index]
    print bicubic_name  
    bicubic_path = path+ bicubic_name
    path_label = 'test_data/Set5_Bicubic/img_gt/' 
    label_path = path_label+ bicubic_name.replace('_2.bmp','.bmp')
    print bicubic_path, label_path
    bicubic = Image.open(bicubic_path)
    label = Image.open(label_path)
    bicubic = Image.open(bicubic_path)
    label = Image.open(label_path)
    bicubic = np.array(bicubic)
    label = np.array(label)
    bicubic = np.expand_dims(bicubic,0)
    label = np.expand_dims(label,0)
    bicubic = np.expand_dims(bicubic,-1)
    label = np.expand_dims(label,-1)
    return bicubic,label
