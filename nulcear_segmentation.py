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

x_images,y_images=[],[]

#%%
test_dir='/Users/anandkadumberi/Anand/projects/nucleus/data/test/00071198d059ba7f5914a526d124d28e6d010c92466da21d4a04cd5413362552'
test_img_dir=os.path.join(test_dir,'images')
test_mask_dir=os.path.join(test_dir,'masks')
masks=os.listdir(test_mask_dir)
#%% Classical numpy opencv approach of loading images
for files in nuc_img_files[0:2000]:
    img_path=os.path.join(train_dir,files,'images',files+'.png')
    image=cv2.imread(img_path)
    x_images.append(image)
    mask_file=os.path.join(train_dir,files,'masks')
    mask_filenames=os.listdir(mask_file)
    
    mask_array=np.asarray(image)
    for masks in mask_filenames:
        mask_array += cv2.imread(os.path.join(mask_file,masks))
    y_images.append(mask_array)

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