#  Copyright (c) 2023 Andrew
#  Email: andrewlee1807@gmail.com

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.losses import Huber


class LSTMModel:
    def __init__(self,
                 input_width,
                 num_hidden_layer=(64, 32),
                 num_features=1,
                 output_length=1
                 ):
        self.model = Sequential()
        self.model.add(LSTM(num_hidden_layer[0], input_shape=(input_width, num_features), return_sequences=True))
        for i in range(1, len(num_hidden_layer)):
            self.model.add(LSTM(num_hidden_layer[i], return_sequences=True))
        # self.model.add(LSTM(num_hidden_layer[1], return_sequences=True))
        # self.model.add(LSTM(num_hidden_layer[2]))
        self.model.add(Dense(output_length))

    def compile_model(self, optimizer, metrics):
        self.model.compile(optimizer=optimizer,
                           loss=Huber(),
                           metrics=metrics)
        self.model.summary()
