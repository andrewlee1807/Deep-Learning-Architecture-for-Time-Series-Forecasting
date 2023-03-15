#  Copyright (c) 2022 Andrew
#  Email: andrewlee1807@gmail.com

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import inspect


class StrideLayer(layers.Layer):
    def __init__(self,
                 nb_stride=3,
                 nb_filters=64,
                 kernel_size=3,
                 dilation_rate=1,
                 padding='causal',
                 dropout_rate=0.0,
                 init=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
                 name="StrideLayer", **kwargs):
        super(StrideLayer, self).__init__(name=name, **kwargs)

        self.nb_filters = nb_filters
        self.dilation_rate = dilation_rate
        self.layers = []

        for k in range(2):
            name = 'conv1D_{}'.format(k)
            conv = layers.Conv1D(filters=self.nb_filters,
                                 kernel_size=kernel_size,
                                 dilation_rate=self.dilation_rate,
                                 strides=nb_stride,
                                 padding=padding,
                                 name=name,
                                 kernel_initializer=init)

            self.layers.append(conv)

        #
        # self.batch1 = layers.BatchNormalization(axis=-1)
        self.ac1 = layers.Activation('relu')
        self.drop1 = layers.Dropout(rate=dropout_rate)

    def call(self, inputs, training=None, **kwargs):
        # x = self.conv1(inputs)
        # x = self.batch1(x)
        # x = self.ac1(x)
        # x = self.drop1(x)
        x1 = inputs
        for layer in self.layers:
            training_flag = 'training' in dict(inspect.signature(layer.call).parameters)
            x1 = layer(x1, training=training) if training_flag else layer(x1)
        x = self.ac1(x1)
        return x


# Idea 1: Preprocessing input, stride connection apply then
class Model1(tf.keras.Model):
    def __init__(self,
                 list_stride=(7, 1),
                 list_dilation=(1, 1),
                 nb_filters=64,
                 kernel_size=6,
                 padding='causal',
                 target_size=24,
                 dropout_rate=0.0):
        self.nb_filters = nb_filters
        self.list_stride = list_stride
        self.list_dilation = list_dilation
        self.kernel_size = kernel_size
        self.padding = padding

        super(Model1, self).__init__()
        init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        assert padding in ['causal', 'same']

        self.stride_blocks = []

        for i in range(len(self.list_stride)):
            stride_block_filters = self.nb_filters
            self.stride_blocks.append(
                StrideLayer(nb_stride=self.list_stride[i],
                            dilation_rate=self.list_dilation[i],
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
        x = inputs
        for stride_block in self.stride_blocks:
            x = stride_block(x)
        x = self.slicer_layer(x)
        x = self.dense(x)
        return x


# Idea 2: 2 kernel apply
class Model2(tf.keras.Model):
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
        self.padding = padding

        super(Model2, self).__init__()
        init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        assert padding in ['causal', 'same']

        self.stride_blocks = []

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
        x = inputs
        for stride_block in self.stride_blocks:
            x = stride_block(x)
        x = self.slicer_layer(x)
        x = self.dense(x)
        return x

