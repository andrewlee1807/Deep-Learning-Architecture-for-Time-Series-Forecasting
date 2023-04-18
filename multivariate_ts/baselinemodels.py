#  Copyright (c) 2023 Andrew
#  Email: andrewlee1807@gmail.com

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class LSTMModel:
    def __init__(self):
        self.model = Sequential()
        self.model.add(LSTM(64, input_shape=(168, 4), return_sequences=True))
        self.model.add(LSTM(32))
        self.model.add(Dense(12))

    def compile_model(self):
        self.model.compile(optimizer='adam', loss='mse')

    def train_model(self, X_train, y_train, epochs):
        y_train = y_train.reshape(-1, 3, 4)
        self.model.fit(X_train, y_train, epochs=epochs)

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred.reshape(-1, 3, 4)
