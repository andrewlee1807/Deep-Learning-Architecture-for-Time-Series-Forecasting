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
        self.model.add(LSTM(num_hidden_layer[1]))
        self.model.add(Dense(output_length))

    def compile_model(self, optimizer, metrics):
        self.model.compile(optimizer=optimizer,
                           loss=Huber(),
                           metrics=metrics)
        self.model.summary()

    def train_model(self, X_train, y_train, epochs):
        y_train = y_train.reshape(-1, 3, 4)
        self.model.fit(X_train, y_train, epochs=epochs)

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred.reshape(-1, 3, 4)
