#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 17:03:18 2018

@author: anandkadumberi
"""
#%% Import required libraries
import cv2
import os
import pandas as pd
import numpy as np
import tensorflow as tf

#%% Read the training data paths
data_dir='/Users/anandkadumberi/Anand/projects/nucleus/data'
train_dir=os.path.join(data_dir,'stage1_train')
label_csv=os.path.join(data_dir,'stage1_train_labels.csv')

train_df=pd.read_csv(label_csv)

encoded_pixels=train_df['EncodedPixels']
nuc_img_files=train_df['ImageId']
#%% Classical numpy opencv approach of loading images
x_images,y_images=[],[]
for files in nuc_img_files:
    img_path=os.path.join(train_dir,files,'images',files+'.png')
    x_images.append(img_path)
    mask_filename=os.path.join(train_dir,files,'images',files+'_mask.png')
    y_images.append(mask_filename)
#%%Tensorflow unet
def get_variable(name,shape):
    return tf.get_variable(name, shape, initializer = tf.contrib.layers.xavier_initializer())

def UNet(X):
    ### Unit 1 ###
    with tf.name_scope('Unit1'):
        W1_1 =   get_variable("W1_1", [3,3,3,16] )
        Z1 = tf.nn.conv2d(X,W1_1, strides = [1,1,1,1], padding = 'SAME')
        A1 = tf.nn.relu(Z1)
        W1_2 =   get_variable("W1_2", [3,3,16,16] )
        Z2 = tf.nn.conv2d(A1,W1_2, strides = [1,1,1,1], padding = 'SAME')
        A2 = tf.nn.relu(Z2) 
        P1 = tf.nn.max_pool(A2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
    ### Unit 2 ###
    with tf.name_scope('Unit2'):
        W2_1 =   get_variable("W2_1", [3,3,16,32] )
        Z3 = tf.nn.conv2d(P1,W2_1, strides = [1,1,1,1], padding = 'SAME')
        A3 = tf.nn.relu(Z3)
        W2_2 =   get_variable("W2_2", [3,3,32,32] )
        Z4 = tf.nn.conv2d(A3,W2_2, strides = [1,1,1,1], padding = 'SAME')
        A4 = tf.nn.relu(Z4) 
        P2 = tf.nn.max_pool(A4, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
    ### Unit 3 ###
    with tf.name_scope('Unit3'):
        W3_1 =   get_variable("W3_1", [3,3,32,64] )
        Z5 = tf.nn.conv2d(P2,W3_1, strides = [1,1,1,1], padding = 'SAME')
        A5 = tf.nn.relu(Z5)
        W3_2 =   get_variable("W3_2", [3,3,64,64] )
        Z6 = tf.nn.conv2d(A5,W3_2, strides = [1,1,1,1], padding = 'SAME')
        A6 = tf.nn.relu(Z6) 
        P3 = tf.nn.max_pool(A6, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
    ### Unit 4 ###
    with tf.name_scope('Unit4'):
        W4_1 =   get_variable("W4_1", [3,3,64,128] )
        Z7 = tf.nn.conv2d(P3,W4_1, strides = [1,1,1,1], padding = 'SAME')
        A7 = tf.nn.relu(Z7)
        W4_2 =   get_variable("W4_2", [3,3,128,128] )
        Z8 = tf.nn.conv2d(A7,W4_2, strides = [1,1,1,1], padding = 'SAME')
        A8 = tf.nn.relu(Z8) 
        P4 = tf.nn.max_pool(A8, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
    ### Unit 5 ###
    with tf.name_scope('Unit5'):
        W5_1 =   get_variable("W5_1", [3,3,128,256] )
        Z9 = tf.nn.conv2d(P4,W5_1, strides = [1,1,1,1], padding = 'SAME')
        A9 = tf.nn.relu(Z9)
        W5_2 =   get_variable("W5_2", [3,3,256,256] )
        Z10 = tf.nn.conv2d(A9,W5_2, strides = [1,1,1,1], padding = 'SAME')
        A10 = tf.nn.relu(Z10) 
    ### Unit 6 ###
    with tf.name_scope('Unit6'):
        W6_1 =   get_variable("W6_1", [3,3,256,128] )
        U1 = tf.layers.conv2d_transpose(A10, filters = 128, kernel_size = 2, strides = 2, padding = 'SAME')
        U1 = tf.concat([U1, A8],3)
        W6_2 =   get_variable("W6_2", [3,3,128,128] )
        Z11 = tf.nn.conv2d(U1,W6_1, strides = [1,1,1,1], padding = 'SAME')
        A11 = tf.nn.relu(Z11)
        Z12 = tf.nn.conv2d(A11,W6_2, strides = [1,1,1,1], padding = 'SAME')
        A12 = tf.nn.relu(Z12)
    ### Unit 7 ###
    with tf.name_scope('Unit7'):
        W7_1 =   get_variable("W7_1", [3,3,128,64] )
        U2 = tf.layers.conv2d_transpose(A12, filters = 64, kernel_size = 2, strides = 2, padding = 'SAME')
        U2 = tf.concat([U2, A6],3)
        Z13 = tf.nn.conv2d(U2,W7_1, strides = [1,1,1,1], padding = 'SAME')
        A13 = tf.nn.relu(Z13)
        W7_2 =   get_variable("W7_2", [3,3,64,64] )
        Z14 = tf.nn.conv2d(A13,W7_2, strides = [1,1,1,1], padding = 'SAME')
        A14 = tf.nn.relu(Z14)
    ### Unit 8 ###
    with tf.name_scope('Unit8'):
        W8_1 =   get_variable("W8_1", [3,3,64,32] )
        U3 = tf.layers.conv2d_transpose(A14, filters = 32, kernel_size = 2, strides = 2, padding = 'SAME')
        U3 = tf.concat([U3, A4],3)
        Z15 = tf.nn.conv2d(U3,W8_1, strides = [1,1,1,1], padding = 'SAME')
        A15 = tf.nn.relu(Z15)
        W8_2 =   get_variable("W8_2", [3,3,32,32] )
        Z16 = tf.nn.conv2d(A15,W8_2, strides = [1,1,1,1], padding = 'SAME')
        A16 = tf.nn.relu(Z16)
    ### Unit 9 ###
    with tf.name_scope('Unit9'):
        W9_1 =   get_variable("W9_1", [3,3,32,16] )
        U4 = tf.layers.conv2d_transpose(A16, filters = 16, kernel_size = 2, strides = 2, padding = 'SAME')
        U4 = tf.concat([U4, A2],3)
        Z17 = tf.nn.conv2d(U4,W9_1, strides = [1,1,1,1], padding = 'SAME')
        A17 = tf.nn.relu(Z17)
        W9_2 =   get_variable("W9_2", [3,3,16,16] )
        Z18 = tf.nn.conv2d(A17,W9_2, strides = [1,1,1,1], padding = 'SAME')
        A18 = tf.nn.relu(Z18)
    ### Unit 10 ###
    with tf.name_scope('out_put'):
        W10 =    get_variable("W10", [1,1,16,1] )
        Z19 = tf.nn.conv2d(A18,W10, strides = [1,1,1,1], padding = 'SAME')
        A19 = tf.nn.sigmoid(Z19)
        Y_pred = A19
    return Y_pred

def loss_function(y_pred, y_true):
    cost = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true,y_pred))
    return cost

def mean_iou(y_pred,y_true):
    y_pred_ = tf.to_int64(y_pred > 0.5)
    y_true_ = tf.to_int64(y_true > 0.5)
    score, up_opt = tf.metrics.mean_iou(y_true_, y_pred_, 2)
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score
#%%
IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS=256,256,3
# build the graph as a dictionary
def build_graph():
    with tf.Graph().as_default() as g:
        with tf.device("/cpu:0"):
            with tf.name_scope('input'):
                x_ = tf.placeholder(tf.float32, shape=(None,IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
                y_ = tf.placeholder(tf.float32, shape=(None,IMG_HEIGHT, IMG_WIDTH, 1))
            y_pred = UNet(x_)
            with tf.name_scope('loss'):
                loss = loss_function(y_pred,y_)
        with tf.device("/cpu:0"):
            with tf.name_scope("metrics"):
                iou = mean_iou(y_pred,y_)
        model_dict = {'graph': g, 'inputs': [x_, y_],'Iou':iou,'Loss':loss, 'y_pred':y_pred}
    return model_dict

#%%
writer = tf.summary.FileWriter('/Users/anandkadumberi/Anand/projects/nucleus', graph=model_dict['graph'])
#%% Tensorflow Dataset API application
x=tf.constant(x_images[0:1000])
y=tf.constant(y_images[0:1000])

def image_reader(x,y):
    x_fp,y_fp= tf.read_file(x),tf.read_file(y)
    x_decoded,y_decoded=tf.image.decode_png(x_fp),tf.image.decode_png(y_fp)
    x_img,y_img=tf.image.resize_images(x_decoded,[200,200]),tf.image.resize_images(y_decoded,[200,200])
    return x_img,y_img

dataset=tf.data.Dataset.from_tensor_slices((x,y))
dataset= dataset.map(image_reader)
batched_dataset=dataset.batch(1)
iterator= batched_dataset.make_one_shot_iterator()
next_el=iterator.get_next()

with tf.Session() as sess:
    print sess.run(next_el)
    print sess.run(next_el)

#%% Tensorflow approach of loading images
'''#filename_queue= tf.train.string_input_producer()
msk_lst=[] 
for m in masks:
    msk_fp= os.path.join(test_mask_dir,m)
    msk_lst.append(msk_fp)
#%%
filenames=tf.constant(msk_lst)
def mask_parser(filename):
    image_string=tf.WholeFileReader(filename)
    image_decoded=tf.image.decode_png(image_string)
    return image_decoded

dataset=tf.data.Dataset.from_tensors((filenames))
dataset=dataset.map(mask_parser)
'''