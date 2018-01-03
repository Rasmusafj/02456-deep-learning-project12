#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 09:07:19 2017

@author: tmpethick
"""
import time
import os
from types import SimpleNamespace

import numpy as np
import keras
import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Flatten, Reshape,\
    Activation, Dropout, Conv2D, Conv2DTranspose,\
    MaxPooling2D, BatchNormalization, UpSampling2D, Lambda
from keras.models import Model, Sequential
import pescador
import matplotlib.pyplot as plt

from layers import DePool2D, Upsampling2DPadded

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


#%%

# =============================================================================
# Time decorator
# =============================================================================

def timer(f):
    def wrapper(*args, **kwargs):
        start = time.time()
        y = f(*args, **kwargs)
        done = time.time()
        elapsed = done - start
        print("Elapsed: ", elapsed)
        return y
    return wrapper

# =============================================================================
# Config
# =============================================================================

def init_config(img_shape, conf=None, extra_conf=None):
    """
    Arguments:
        X_train: batch of training examples (batches, channels, rows, cols)
    """
    img_chns, img_rows, img_cols = img_shape
    extra_conf = extra_conf if extra_conf is not None else {}

    conf.__dict__.update({
        # Using Keras 'channels_first':
        'uuid': extra_conf['lambda_uuid'](conf.uuid) if extra_conf.get('lambda_uuid') else conf.uuid,
        'original_img_size': (img_chns, img_rows, img_cols),
        'img_chns': img_chns,
        'img_rows': img_rows,
        'img_cols': img_cols,
    })
    conf.__dict__.update(extra_conf)
    return conf

   
def layer(f):
    def wrapper(*arg, **kwargs):
        return lambda x: f(x, *arg, **kwargs)
    return wrapper


@layer
def conv2d(x, *args, use_batch_normalization=False, **kwargs):
    y = Conv2D(*args, padding='same', **kwargs)(x)
    if use_batch_normalization:
        y = BatchNormalization(axis=1)(y)
    y = Activation('relu')(y)
    return y

@layer
def deconv2d(x, *args, use_batch_normalization=False, activation='relu', **kwargs):
    y = Conv2DTranspose(*args, padding='same', **kwargs)(x)
    if use_batch_normalization:
        y = BatchNormalization(axis=1)(y)
    if activation is not None:
        y = Activation(activation)(y)
    return y
