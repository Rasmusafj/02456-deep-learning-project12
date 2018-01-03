# -*- coding: utf-8 -*-

import keras.backend as K
from keras.models import Model
from keras.layers import Input, MaxPooling2D, multiply
import numpy as np
from numpy.testing import assert_array_equal

from layers import DePool2D, ZeroPad2D

def test_depool():
    size = (2, 2)
    x = Input(batch_shape=(1, 1, 2, 2))
    mx = MaxPooling2D(pool_size=size, strides=size, data_format='channels_first')(x)
    dp = DePool2D(size=size, data_format='channels_first')([x, mx, mx])
    model = Model(inputs=x, outputs=dp)

    input_ = np.array([[[[1,2],
                         [3,4]]]])

    expected = np.array([[[[0,0],
                           [0,4]]]])

    assert_array_equal(model.predict(input_), expected)

def test_depool_auto_batch_size():
    size = (2, 2)
    x = Input(shape=(1, 2, 2))
    mx = MaxPooling2D(pool_size=size, strides=size, data_format='channels_first')(x)
    dp = DePool2D(size=size, data_format='channels_first')([x, mx, mx])
    model = Model(inputs=x, outputs=dp)

    input_ = np.array([[[[1,2],
                         [3,4]]]])

    expected = np.array([[[[0,0],
                           [0,4]]]])
    assert_array_equal(model.predict(input_, batch_size=1), expected)

def test_depool_pad():
    size = (1, 2)
    x = Input(batch_shape=(1, 1, 2, 3))
    mx = MaxPooling2D(pool_size=size, strides=size, data_format='channels_first')(x)
    dp = DePool2D(size=size, data_format='channels_first')([x, mx, mx])
    model = Model(inputs=x, outputs=dp)

    input_ = np.array([[[[1,2,3],
                         [3,4,3]]]])

    expected = np.array([[[[0,2,0],
                           [0,4,0]]]])

    assert_array_equal(model.predict(input_), expected)
    
    model.compile(optimizer='sgd', loss='categorical_crossentropy')
    model.fit(
        input_, expected,
        batch_size=1,
        epochs=1,
        verbose=0
    )
