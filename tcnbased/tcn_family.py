#  Copyright (c) 2022 Andrew
#  Email: andrewlee1807@gmail.com
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tcn import TCN


class TCN_Vanilla(tf.keras.Model):
    def __init__(self,
                 input_width,
                 dilations=(1, 2),
                 nb_filters=64,
                 kernel_size=12,
                 padding='causal',
                 use_skip_connections=True,
                 use_batch_norm=False,
                 num_features=1,
                 target_size=24,
                 dropout_rate=0.0):
        self.input_width = input_width
        self.nb_filters = nb_filters
        self.dilations = dilations
        self.kernel_size = kernel_size
        self.padding = padding

        super(TCN_Vanilla, self).__init__()
        init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        assert padding in ['causal', 'same']

        # inputs = Input(shape=(input_width, num_features))

        self.tcn_block = TCN(input_shape=(self.input_width, num_features),
                             kernel_size=kernel_size,
                             nb_filters=nb_filters,
                             dilations=self.dilations,
                             dropout_rate=dropout_rate,
                             use_skip_connections=use_skip_connections,
                             use_batch_norm=use_batch_norm,
                             use_weight_norm=False,
                             return_sequences=False
                             )

        self.dense = Dense(units=target_size)

    def call(self, inputs, training=True):
        x = inputs
        x = self.tcn_block(x)
        x = self.dense(x)
        return x

    def summary(self, x):
        # x = layers.Input(shape=(24, 24, 3))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x), name="TCN")
        return model.summary()
