#  Copyright (c) 2022 Andrew
#  Email: andrewlee1807@gmail.com

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class StrideLayer(layers.Layer):
    def __init__(self,
                 nb_stride=3,
                 nb_filters=64,
                 kernel_size=3,
                 padding='causal',
                 dropout_rate=0.0,
                 init=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
                 name="StrideLayer", **kwargs):
        super(StrideLayer, self).__init__(name=name, **kwargs)

        self.conv1 = layers.Conv1D(filters=64,
                                   kernel_size=kernel_size,
                                   strides=nb_stride,
                                   padding=padding,
                                   name='conv1D',
                                   kernel_initializer=init)

        self.conv2 = layers.Conv1D(filters=64,
                                   kernel_size=kernel_size,
                                   strides=nb_stride,
                                   padding=padding,
                                   name='conv1D',
                                   kernel_initializer=init)

        self.batch1 = layers.BatchNormalization(axis=-1)
        self.ac1 = layers.Activation('relu')
        self.drop1 = layers.Dropout(rate=dropout_rate)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.batch1(x)
        x = self.ac1(x)
        x = self.drop1(x)
        return x


# Idea 1: Preprocessing input, stride connection apply then
class Model1(tf.keras.Model):
    def __init__(self,
                 list_stride=(7, 1),
                 nb_filters=64,
                 kernel_size=3,
                 padding='causal',
                 target_size=24,
                 dropout_rate=0.0):
        self.nb_filters = nb_filters
        self.list_stride = list_stride
        self.kernel_size = kernel_size

        super(Model1, self).__init__()
        init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        assert padding in ['causal', 'same']

        for i, d in enumerate(self.list_stride):
            stride_block_filters = self.nb_filters
            self.stride_blocks.append(
                StrideLayer(nb_stride=self.list_stride[i],
                            nb_filters=stride_block_filters,
                            kernel_size=self.kernel_size,
                            padding=self.padding,
                            init=init,
                            dropout_rate=dropout_rate,
                            name=f"DilatedLayer_{i}")
            )

        self.slicer_layer = layers.Lambda(lambda tt: tt[:, -1, :], name='Slice_Output')

        self.dense = layers.Dense(units=target_size)

    def call(self, inputs, training=True):
        x = self.dilation1(inputs)
        x = self.dilation2(x)
        x = self.slicer_layer(x)
        x = self.dense(x)
        return x
