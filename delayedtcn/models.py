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
                                 name=name,)
                                 # kernel_initializer=init)

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


class StrideLayer2(layers.Layer):
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

        self.layers_con1 = []
        self.layers_con2 = []

        for k in range(1):
            name1 = 'conv1D_{}'.format(k) + '_con1'
            conv1 = layers.Conv1D(filters=nb_filters,
                                  kernel_size=kernel_size,
                                  dilation_rate=dilation_rate,
                                  strides=nb_stride,
                                  padding=padding,
                                  name=name1,
                                  kernel_initializer=init)
            self.layers_con1.append(conv1)

            name2 = 'conv1D_{}'.format(k) + '_con2'
            conv2 = layers.Conv1D(filters=nb_filters,
                                  kernel_size=kernel_size,
                                  dilation_rate=dilation_rate,
                                  strides=nb_stride,
                                  padding=padding,
                                  name=name2,
                                  kernel_initializer=init)
            self.layers_con2.append(conv2)

        self.ac1 = layers.Activation('relu')
        self.drop1 = layers.Dropout(rate=dropout_rate)

    def call(self, inputs, training=None, **kwargs):
        x1 = inputs
        for i in len(self.layers_con1):
            # training_flag = 'training' in dict(inspect.signature(layer.call).parameters)
            # x1 = layer(x1, training=training) if training_flag else layer(x1)
            x1 = self.layers_con1[i](x1)
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

        # Flatten the input data, as MLPs require a 1D input
        # self.flatten = layers.Flatten(input_shape=(None, 168))

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

        self.dense1 = layers.Dense(units=128)
        self.dense = layers.Dense(units=target_size)
        # self.dense = layers.Dense(units=(2), activation='softmax')
        # self.reshape = layers.Reshape(target_shape=(1, 2))

    def call(self, inputs, training=True):
        x = inputs
        # x = self.flatten(x)
        for stride_block in self.stride_blocks:
            x = stride_block(x)
        x = self.slicer_layer(x)
        x = self.dense1(x)
        x = self.dense(x)
        # x = self.reshape(x)
        return x

    def summary(self, x):
        # x = layers.Input(shape=(24, 24, 3))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x), name="DelayNet1")
        return model.summary()


def shift_sequence(input_seq, kernel_size):
    # Shift the elements of the input sequence to the left by 3 positions
    shifted_seq = tf.roll(input_seq, shift=-kernel_size, axis=1)
    # Set the first 3 elements of the shifted sequence to zero
    shifted_seq = shifted_seq[:, :-kernel_size, :]

    # Add 3 zero elements to the end of the shifted sequence
    zero_seq = tf.zeros((tf.shape(input_seq)[0], kernel_size, 1))
    padded_seq = tf.keras.layers.Concatenate(axis=1)([shifted_seq, zero_seq])
    return padded_seq


# Idea 2: 2 kernel apply
class Model2(tf.keras.Model):
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
        self.kernel_size = kernel_size // 2  # 2 kernel apply; kernel_size=12 -> kernel_size1=6 and kernel_size2=6
        self.padding = padding

        super(Model2, self).__init__()
        init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        assert padding in ['causal', 'same']

        self.ac1 = layers.Activation('relu')

        # for i, d in enumerate(self.list_stride):
        stride_block_filters = self.nb_filters

        self.layers_1_conv1 = None
        self.layers_1_conv2 = None
        self.layers_2_conv1 = None
        self.layers_2_conv2 = None

        for k in range(1):
            name1 = 'Layer1-' + 'conv1D_{}'.format(k) + '_con1'
            conv1 = layers.Conv1D(filters=stride_block_filters,
                                  kernel_size=self.kernel_size,
                                  dilation_rate=1,
                                  strides=self.list_stride[0],
                                  padding=padding,
                                  name=name1,
                                  kernel_initializer=init)
            self.layers_1_conv1 = conv1

            name2 = 'Layer1-' + 'conv1D_{}'.format(k) + '_con2'
            conv2 = layers.Conv1D(filters=stride_block_filters,
                                  kernel_size=self.kernel_size,
                                  dilation_rate=1,
                                  strides=self.list_stride[0],
                                  padding=padding,
                                  name=name2,
                                  kernel_initializer=init)
            self.layers_1_conv2 = conv2

            self.concat = layers.Concatenate(name="Concatenate-layer", axis=1)

        self.conv_middle1 = layers.Conv1D(filters=64,
                                          kernel_size=3,
                                          strides=2,
                                          padding='same',
                                          activation='relu')

        self.conv_middle2 = layers.Conv1D(filters=64,
                                          kernel_size=3,
                                          strides=2,
                                          padding='same',
                                          activation='relu')

        for k in range(1):
            name1 = 'Layer2-' + 'conv1D_{}'.format(k) + '_con1'
            conv1 = layers.Conv1D(filters=stride_block_filters,
                                  kernel_size=self.kernel_size,
                                  dilation_rate=1,
                                  strides=self.list_stride[1],
                                  padding=padding,
                                  name=name1,
                                  kernel_initializer=init)
            self.layers_2_conv1 = conv1

            name2 = 'Layer2-' + 'conv1D_{}'.format(k) + '_con2'
            conv2 = layers.Conv1D(filters=stride_block_filters,
                                  kernel_size=self.kernel_size,
                                  dilation_rate=1,
                                  strides=self.list_stride[1],
                                  padding=padding,
                                  name=name2,
                                  kernel_initializer=init)
            self.layers_2_conv2 = conv2

        self.slicer_layer = layers.Lambda(lambda tt: tt[:, -1, :], name='Slice_Output')

        self.dense = layers.Dense(units=target_size)

    def call(self, inputs, training=True):
        x = inputs
        # for i, d in enumerate(self.list_stride):
        x1 = self.layers_1_conv1(x)
        # Skip the first kernel_size elements of the input
        # example: x[:, 6:]: it'll be calculated from 7th element to the end
        input_shifted = shift_sequence(x, self.kernel_size)
        # input_shifted = tf.keras.layers.Lambda(shift_sequence)([x, self.kernel_size])

        x2 = self.layers_1_conv2(input_shifted)
        # Concatenate the output of the two convolution layers
        # layer1_concat = layers.Concatenate(axis=2)([x1, x2])
        x1x2 = self.concat([x1, x2])

        xxx = self.conv_middle1(x1x2)
        xxx = self.conv_middle1(xxx)

        x3 = self.layers_2_conv1(xxx)
        x4 = self.layers_2_conv2(x3)

        x = self.slicer_layer(x4)
        x = self.dense(x)
        return x

    def summary(self, x):
        # x = layers.Input(shape=(24, 24, 3))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x), name="DelayNet2")
        return model.summary()
