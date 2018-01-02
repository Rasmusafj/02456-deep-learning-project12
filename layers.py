#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 08:43:27 2017

@author: tmpethick
"""
from keras.layers import Lambda, multiply
from keras.layers.convolutional import UpSampling2D

def ZeroPad2D(**kwargs):
    def call(inputs, data_format=None):
        from keras import backend as K
        
        y, goal_y = inputs
        shape = K.int_shape(y)
        goal_shape = K.int_shape(goal_y)
        a, b = goal_shape[-2] - shape[-2], goal_shape[-1] - shape[-1]
        y_pad = K.spatial_2d_padding(y, padding=((0, a), (0, b)), data_format=data_format)
        return y_pad

    return Lambda(call, output_shape=lambda x: x[1], arguments=kwargs)

def Where():
    def getwhere(inputs):
        """
        Calculate the 'where' mask that contains switches indicating which
        index contained the max value when MaxPool2D was applied.  Using the
        gradient of the sum is a nice trick to keep everything high level.
        """
        from keras import backend as K
        y_prepool, y_postpool = inputs
        return K.gradients(K.sum(y_postpool), y_prepool)
    
    return Lambda(getwhere, output_shape=lambda x: x[0])

def DePool2D(*args, data_format=None, **kwargs):
    """
    Uses a switch variable as described in https://arxiv.org/pdf/1311.2901.pdf
    Can only invert maxpooling where `size = stride`.
    """

    def call(inputs):
        y_prepool, y_postpool, x = inputs
        where = Where()([y_prepool, y_postpool])
        y = UpSampling2D(*args, data_format=data_format, **kwargs)(x)
        y = ZeroPad2D(data_format=data_format)([y, where])
        return multiply([y, where])
    
    return call

def Upsampling2DPadded(*args, data_format=None, **kwargs):
    def call(inputs):
        prepool, postpool, x = inputs
        y = UpSampling2D(*args, data_format=data_format, **kwargs)(x)
        y = ZeroPad2D(data_format=data_format)([y, prepool])
        return y

    return call
